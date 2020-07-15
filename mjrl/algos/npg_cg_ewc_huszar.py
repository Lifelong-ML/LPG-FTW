import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce_ewc2 import BatchREINFORCEEWC


class NPGEWC(BatchREINFORCEEWC):
    def __init__(self, all_env, policy, all_baseline,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None,
                 ewc_lambda=1e-3,
                 scaled_lambda=False):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.all_env = all_env
        self.policy = policy
        self.all_baseline = all_baseline
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None

        self.ewc_lambda = ewc_lambda
        self.scaled_lambda = scaled_lambda
        self.d = (self.policy.n+1) * self.policy.m
        self.Fsum = np.zeros((self.d, self.d))
        self.b = np.zeros((self.d, 1))
        self.observed_tasks = set()
        self.theta = {}
        self.hess = {}
        self.grad = {}
        
        self.T = 0
        if save_logs: self.logger = {}

    def HVP(self, observations, actions, vector, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        vec = Variable(torch.from_numpy(vector).float(), requires_grad=False)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        hvp_flat = np.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])
        return hvp_flat + regu_coef*vector

    def build_Hvp_eval(self, inputs, regu_coef=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval

    def grad_and_hess(self, observations, actions, advantages, theta):
           

        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        objective = cpi_surr 

        vpg_grad = torch.autograd.grad(objective, theta, create_graph=True)
        vpg_grad = torch.cat((vpg_grad[0], torch.unsqueeze(vpg_grad[1],1)), 1)
        vpg_grad = vpg_grad.reshape((-1,1))

        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)

        kl_grad = torch.autograd.grad(mean_kl, theta, create_graph=True)
        kl_grad = torch.cat((kl_grad[0], torch.unsqueeze(kl_grad[1],1)), 1)
        kl_grad = kl_grad.reshape((-1,1))
        # print(kl_grad, kl_grad.shape)
        # exit()
        hess_cols = []
        for i in range(kl_grad.numel()):
            tmp = torch.autograd.grad(kl_grad[i], theta, retain_graph=(i < (kl_grad.numel()-1)))
            tmp = torch.cat((tmp[0], torch.unsqueeze(tmp[1], 1)), 1)
            tmp = tmp.reshape((-1,1))
            hess_cols.append(tmp)
        FIM = torch.cat(hess_cols, dim=1)
        
        vpg_grad = vpg_grad.data.numpy()
        FIM = FIM.data.numpy()
        lagrange = np.sqrt(vpg_grad.T.dot(np.linalg.inv(FIM)).dot(vpg_grad) / self.n_step_size * 2)
        vpg_hess = -FIM * lagrange / 2    # divide by 2 to match LPG-FTW assumptions 

        self.grad[self.task_id] = vpg_grad
        self.hess[self.task_id] = vpg_hess

    def add_approximate_cost(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max'):

        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, env_name,
                                                             self.seed, num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger[self.task_id].log_kv('time_sampling_hess', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # Change advantage whitening to making advantages \in [-1, 0]
        # advantages = advantages - np.max(advantages)
        # advantages = advantages / (-np.min(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]

        if self.save_logs: self.log_rollout_statistics(paths)

        theta = list(self.policy.model.parameters())
        
        self.grad_and_hess(observations, actions, advantages, theta)

        theta = torch.cat((theta[0], torch.unsqueeze(theta[1], 1)), 1)
        theta = theta.reshape((-1,1)).data.numpy()
        self.theta[self.task_id] = theta

        self.Fsum += self.hess[self.task_id]

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # advantages = advantages - np.max(advantages)
        # advantages = advantages / (-np.min(advantages) + 1e-6)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling, 
        # but scaling can help with least squares

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger[self.task_id].log_kv('alpha', alpha)
            self.logger[self.task_id].log_kv('delta', n_step_size)
            self.logger[self.task_id].log_kv('time_vpg', t_gLL)
            self.logger[self.task_id].log_kv('time_npg', t_FIM)
            self.logger[self.task_id].log_kv('kl_dist', kl_dist)
            self.logger[self.task_id].log_kv('surr_improvement', surr_after - surr_before)
            self.logger[self.task_id].log_kv('running_score', self.running_score)
            try:
                success_rate = self.env.env.env.evaluate_success(paths)
                self.logger[self.task_id].log_kv('success_rate', success_rate)
            except:
                pass

        return base_stats

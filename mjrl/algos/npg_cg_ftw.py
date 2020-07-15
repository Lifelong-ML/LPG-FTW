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
from mjrl.algos.batch_reinforce_ftw import BatchREINFORCEFTW


class NPGFTW(BatchREINFORCEFTW):
    def __init__(self, all_env, policy, all_baseline,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None,
                 new_col_mode='regularize'):
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
        self.running_score = {}
        self.save_logs = save_logs
        if save_logs: self.logger = {}
        self.d = policy.model.L.shape[0] 
        self.A = np.zeros(( self.d * self.policy.k, self.d * self.policy.k))
        self.B = np.zeros((self.d * self.policy.k, 1))
        self.observed_tasks = set()
        self.theta = {}
        self.grad = {}
        self.hess = {}
        self.new_col_mode = new_col_mode

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
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True, allow_unused=True)#, retain_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)#, retain_graph=True)
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

        vpg_grad = torch.autograd.grad(objective, [theta], create_graph=True)[0]

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

        kl_grad = torch.autograd.grad(mean_kl, [theta], create_graph=True)[0]
        FIM = torch.cat([torch.autograd.grad(kl_grad[i], [theta], retain_graph=(i < (kl_grad.numel()-1)))[0] for i in range(kl_grad.numel())], dim=1)
        vpg_grad = vpg_grad.data.numpy()
        const = -min(np.linalg.eigvalsh(FIM)) + 1e-3
        FIM = FIM.data.numpy() + const * np.eye(FIM.shape[0])
        lagrange = np.sqrt(vpg_grad.T.dot(np.linalg.inv(FIM)).dot(vpg_grad) / self.n_step_size * 2)
        vpg_hess = -FIM * lagrange / 2    # divide by 2 to match LPG-FTW assumptions 

        return vpg_grad, vpg_hess

    # ----------------------------------------------------------
    def train_from_paths(self, paths, task_id):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
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
        if task_id in self.running_score:
            self.running_score[task_id] = 0.9*self.running_score[task_id] + 0.1*mean_return
        else:
            self.running_score[task_id] = mean_return
        if self.save_logs: self.log_rollout_statistics(paths, task_id)

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
        curr_params = self.policy.get_param_values(task_id)
        new_params = curr_params + alpha * npg_grad
        if task_id >= self.policy.k:
            print(npg_grad.shape, new_params.shape)
        self.policy.set_param_values(new_params, task_id, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, task_id, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger[task_id].log_kv('alpha', alpha)
            self.logger[task_id].log_kv('delta', n_step_size)
            self.logger[task_id].log_kv('time_vpg', t_gLL)
            self.logger[task_id].log_kv('time_npg', t_FIM)
            self.logger[task_id].log_kv('kl_dist', kl_dist)
            self.logger[task_id].log_kv('surr_improvement', surr_after - surr_before)
            self.logger[task_id].log_kv('running_score', self.running_score[task_id])
            try:
                success_rate = self.env.env.env.evaluate_success(paths)
                self.logger[task_id].log_kv('success_rate', success_rate)
            except:
                pass

        return base_stats

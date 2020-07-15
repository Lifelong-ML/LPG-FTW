import logging
logging.disable(logging.CRITICAL)
import numpy as np
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog

import pickle
import os


class BatchREINFORCEFTW:
    def __init__(self, all_env, policy, all_baseline,
                 learn_rate=0.01,
                 seed=None,
                 save_logs=False,
                 new_col_mode='regularize'):

        self.all_env = all_env
        self.policy = policy
        self.all_baseline = all_baseline
        self.theta = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = {}
        self.save_logs = save_logs
        if save_logs: self.logger = {}
        self.d = policy.model.L.shape[0] 
        self.A = np.zeros(( self.d * self.policy.k, self.d * self.policy.k))
        self.B = np.zeros((self.d * self.policy.k, 1))
        self.theta = {}
        self.grad = {}
        self.hess = {}
        self.new_col_mode = new_col_mode

    def set_task(self, task_id):
        self.env = self.all_env[task_id]
        self.policy.set_task(task_id)
        self.baseline = self.all_baseline[task_id]
        if task_id not in self.observed_tasks:
            if self.save_logs: self.logger[task_id] = DataLog()
            self.observed_tasks.add(task_id)

    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)

        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        A = Variable(torch.from_numpy(self.A).float(), requires_grad=False)
        B = Variable(torch.from_numpy(self.B).float(), requires_grad=False)
        vecLT = self.policy.model.L.reshape((-1,1))  # because of the way torch orders upon reshape, this is equivalent to vec(L^\top)
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)

        if self.policy.model.T > self.policy.k:     # regularize S
            objective = cpi_surr - 1e-5*torch.norm(self.policy.trainable_params[1], 1)
        else:   # regularize nothing (equivalent to training STL)
            objective = cpi_surr   


        vpg_grad = torch.autograd.grad(objective, self.policy.trainable_params)#, retain_graph=True)

        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    def grad_and_hess(self, observations, actions, advantages, theta):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        objective = cpi_surr 

        vpg_grad = torch.autograd.grad(objective, [theta], create_graph=True)[0]
        
        log_std = self.policy.log_std
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False).reshape(-1,1)
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        obs_var = torch.cat([obs_var, torch.ones(obs_var.shape[0],1)], dim=1)

        vpg_grad = vpg_grad.data.numpy()

        # delete this
        adv_var = adv_var - adv_var.max()
        hess_tmp = torch.mm(torch.t(obs_var * adv_var), obs_var).data.numpy() / adv_var.numel()
        log_std = log_std.data.numpy()
        hess_tmp = np.kron(np.diag(1/np.exp(log_std)), hess_tmp)

        vpg_hess = hess_tmp
        vpg_hess = (vpg_hess + vpg_hess.T) / 2

        return vpg_grad, vpg_hess

    # ----------------------------------------------------------
    def train_step(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max',
                   task_id=0):

        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            policy_copy = self.copy_policy_for_detach()
            paths = trajectory_sampler.sample_paths_parallel(N, policy_copy, T, env_name,
                                                           self.seed, num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger[task_id].log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths, task_id)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger[task_id].log_kv('time_VF', timer.time()-ts)
            self.logger[task_id].log_kv('VF_error_before', error_before)
            self.logger[task_id].log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths, task_id):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

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

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values(task_id)
        new_params = curr_params + self.theta * vpg_grad
        
        self.policy.set_param_values(new_params, task_id, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, task_id, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger[task_id].log_kv('alpha_{}'.format(task_id), self.theta)
            self.logger[task_id].log_kv('time_vpg_{}'.format(task_id), t_gLL)
            self.logger[task_id].log_kv('kl_dist_{}'.format(task_id), kl_dist)
            self.logger[task_id].log_kv('surr_improvement_{}'.format(task_id), surr_after - surr_before)
            self.logger[task_id].log_kv('running_score_{}'.format(task_id), self.running_score[task_id])

        return base_stats

    def test_tasks(self, task_ids=None, 
                    test_rollouts=10,
                    num_cpu=1,
                    update_s=False):
        if task_ids is None:
            task_ids = list(self.observed_tasks)

        mean_pol_perf = {}
        for task_id in task_ids:
            self.set_task(task_id)
            policy_copy = self.copy_policy_for_detach()
            eval_paths = trajectory_sampler.sample_paths_parallel(N=test_rollouts, policy=policy_copy, num_cpu=num_cpu,
                                               env_name=self.env.env_id, mode='evaluation', pegasus_seed=self.seed)
            mean_pol_perf[task_id] = np.mean([np.sum(path['rewards']) for path in eval_paths])
            self.seed = self.seed + test_rollouts if self.seed is not None else self.seed

        return mean_pol_perf


    def data_for_grad(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max',
                   task_id=0,
                   returns=False):
        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            policy_copy = self.copy_policy_for_detach()
            paths = trajectory_sampler.sample_paths_parallel(N, policy_copy, T, env_name,
                                                             self.seed, num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger[task_id].log_kv('time_sampling_hess', timer.time() - ts)

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

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]

        if returns:
            return observations, actions, advantages, mean_return
        return observations, actions, advantages

    def add_approximate_cost(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max',
                   task_id=0):

        # Keep track of times for various computations
        t_gLL = 0.0

        # Add new column
        if self.policy.model.T > self.policy.k and self.policy.model.dict_dim < self.policy.model.max_dict_dim:
            avg_L_norm =  torch.norm(self.policy.model.L, 2, 0).mean()   # get the column-wise norm average of L
            epsilon_norm = torch.norm(self.policy.model.epsilon_col, 2)
            s_scalar = torch.reshape(epsilon_norm / avg_L_norm, (1,1))  # this will be the scale (so the s scalar)
            print(s_scalar)
            self.policy.model.epsilon_col.data = self.policy.model.epsilon_col.data * avg_L_norm / epsilon_norm
            self.policy.old_model.epsilon_col.data = self.policy.old_model.epsilon_col.data * avg_L_norm / epsilon_norm
            self.add_column(s_scalar)

        self.policy.model.set_use_theta(True) # make sure theta is a differentiable variable
        theta_2 = self.policy.model.theta
        observations, actions, advantages = self.data_for_grad(N, sample_mode, env_name, T, gamma, gae_lambda, num_cpu, task_id)
        self.grad[task_id], self.hess[task_id] = self.grad_and_hess(observations, actions, advantages, self.policy.model.theta)
        self.theta[task_id] = self.policy.model.theta.data.numpy()

        # A, b
        s = self.policy.model.S[task_id].data.numpy()
        self.A += 2 * np.kron(np.outer(s,s), self.hess[task_id])
        self.B += np.kron(s.T, - self.grad[task_id].T + 2 * self.theta[task_id].T.dot(self.hess[task_id])).T

        # Update L
        if self.policy.model.T > self.policy.k: # update L only for tasks k+1 on
            T = self.policy.model.T
            vals = np.linalg.inv(1 / T *  self.A - 1e-5 * np.eye(self.policy.model.L.data.numpy().size)).dot(1 / T * self.B)
            vals = vals.reshape(self.policy.model.L.data.numpy().shape, order='F')
            self.policy.model.L.data = torch.from_numpy(vals).float()

        self.policy.model.set_use_theta(False) # make sure Ls is ran through the graph

    def add_column(self, s_scalar=torch.ones(1,1)):
        self.policy.model.L = torch.cat((self.policy.model.L, self.policy.model.epsilon_col), 1)
        self.policy.old_model.L = torch.cat((self.policy.old_model.L, self.policy.old_model.epsilon_col), 1)

        self.policy.model.dict_dim += 1
        for t in range(self.policy.model.T - 1):
            self.policy.model.S[t] = torch.cat((self.policy.model.S[t], torch.zeros(1,1,requires_grad=True)), 0)
            self.policy.old_model.S[t] = torch.cat((self.policy.old_model.S[t], torch.zeros(1,1,requires_grad=True)), 0)
        self.A = np.c_[np.r_[self.A, np.zeros((self.d, self.d*(self.policy.model.dict_dim-1)))], np.zeros((self.policy.model.dict_dim*self.d, self.d))]   # add zeros
        self.B = np.r_[self.B, np.zeros((self.d, 1))]    # add zeros
        self.policy.model.S[self.policy.model.T - 1] = torch.cat((self.policy.model.S[self.policy.model.T - 1], Variable(s_scalar, requires_grad=True)), 0)
        self.policy.old_model.S[self.policy.model.T - 1] = torch.cat((self.policy.old_model.S[self.policy.model.T - 1], Variable(s_scalar, requires_grad=True)), 0)
        self.policy.model.epsilon_col = torch.zeros_like(self.policy.model.epsilon_col)
        self.policy.old_model.epsilon_col = torch.zeros_like(self.policy.old_model.epsilon_col)

    def copy_policy_for_detach(self):
        policy_copy = copy.copy(self.policy)
        policy_copy.model = copy.copy(self.policy.model)
        policy_copy.old_model =copy.copy(self.policy.old_model)
        policy_copy.trainable_params = copy.copy(self.policy.trainable_params)
        policy_copy.old_params = copy.copy(self.policy.old_params)
        policy_copy.log_std = copy.copy(self.policy.log_std)
        policy_copy.old_log_std = copy.copy(self.policy.old_log_std)
        policy_copy.model.S = copy.copy(self.policy.model.S)
        policy_copy.old_model.S = copy.copy(self.policy.old_model.S)
        policy_copy.model.L = policy_copy.model.L.detach()
        policy_copy.old_model.L = policy_copy.old_model.L.detach()
        for tid in policy_copy.model.S:
            policy_copy.model.S[tid] = policy_copy.model.S[tid].detach()
            policy_copy.old_model.S[tid] = policy_copy.old_model.S[tid].detach()
            policy_copy.log_std[tid] = policy_copy.log_std[tid].detach()
            policy_copy.old_log_std[tid] = policy_copy.old_log_std[tid].detach()
            policy_copy.model.epsilon_col = policy_copy.model.epsilon_col.detach()
            policy_copy.old_model.epsilon_col = policy_copy.old_model.epsilon_col.detach()
        for i in range(len(policy_copy.trainable_params)):
            policy_copy.trainable_params[i] = policy_copy.trainable_params[i].detach()
            policy_copy.old_params[i] = policy_copy.old_params[i].detach()

        if 'theta' in policy_copy.model.__dict__:
            policy_copy.model.theta = policy_copy.model.theta.detach()
            policy_copy.old_model.theta = policy_copy.old_model.theta.detach()

        return policy_copy

    def log_rollout_statistics(self, paths, task_id):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger[task_id].log_kv('stoc_pol_mean', mean_return)
        self.logger[task_id].log_kv('stoc_pol_std', std_return)
        self.logger[task_id].log_kv('stoc_pol_max', max_return)
        self.logger[task_id].log_kv('stoc_pol_min', min_return)

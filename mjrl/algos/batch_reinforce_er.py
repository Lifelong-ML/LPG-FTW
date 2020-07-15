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
import random

class BatchREINFORCEER:
    
    class ReplayMemory():

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.observed_so_far = 0

        def push(self, paths):
            """Saves a transition."""
            for j, path in enumerate(paths):
                if len(self.memory) < self.capacity:
                    self.memory.append(path)
                else:
                    i = np.random.randint(0, self.observed_so_far + j + 1)
                    if i < self.capacity:
                        self.memory[i] = path
            self.observed_so_far += j

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, all_env, policy, all_baseline,
                 learn_rate=0.01,
                 seed=None,
                 save_logs=False,
                 capacity=10,
                 scaled_lambda=False,
                 gamma=0.995,
                 gae_lambda=0.98):

        self.all_env = all_env
        self.policy = policy
        self.all_baseline = all_baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None

        self.observed_tasks = set()
        self.replay_buffer = {}
        self.theta = {}
        self.capacity = capacity

        self.T = 0

        if save_logs: self.logger = {}

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def set_task(self, task_id):
        self.task_id = task_id
        self.env = self.all_env[task_id]
        self.policy.set_task(task_id)
        self.baseline = self.all_baseline[task_id]
        if task_id not in self.observed_tasks:
            if self.save_logs: self.logger[task_id] = DataLog()
            self.observed_tasks.add(task_id)
            self.replay_buffer[task_id] = BatchREINFORCEER.ReplayMemory(self.capacity)
            self.T += 1

    def CPI_surrogate(self, observations, actions, advantages, N):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.sum(LR*adv_var)
        n = len(LR)
        
        ########################################
        current_task = self.task_id
        replay_batch_all = N  # 50% on- 50% off-policy as per Rolnick et al.
        if self.T > 1:
            replay_batch_per_task = replay_batch_all // (self.T - 1)
            for task in self.observed_tasks:
                if task == current_task:
                    continue
                self.set_task(task)      # need to use this policy's task, but not change the trainable parameters
                paths = self.replay_buffer[task].sample(replay_batch_per_task)
                process_samples.compute_advantages(paths, self.baseline, self.gamma, self.gae_lambda)       # compute the paths' advantages with updated baseline

                observations_er = np.concatenate([path["observations"] for path in paths])
                actions_er = np.concatenate([path["actions"] for path in paths])
                advantages_er = np.concatenate([path["advantages"] for path in paths])
                # Advantage whitening
                advantages_er = (advantages_er - np.mean(advantages_er)) / (np.std(advantages_er) + 1e-6)
                dist_info_er = [torch.cat([path["LL"] for path in paths]).detach()]

                adv_var = Variable(torch.from_numpy(advantages_er).float(), requires_grad=False)
                old_dist_info = self.policy.old_dist_info(observations_er, actions_er)
                new_dist_info = self.policy.new_dist_info(observations_er, actions_er)
                LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
                LR_IS = self.policy.likelihood_ratio(new_dist_info, dist_info_er)

                surr += torch.sum(LR_IS*LR*adv_var)
                n += len(LR)

            self.set_task(current_task)      # this should take care of setting the trainable parameters correctly
        ########################################

        surr /= n
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages, N):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages, N)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   num_cpu='max',
                   task_id=None,
                   gamma=0.995,
                   gae_lambda=0.98):

        assert gamma == self.gamma and gae_lambda == self.gae_lambda

        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()
        with torch.no_grad():
            if sample_mode == 'trajectories':
                paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, env_name,
                                                                 self.seed, num_cpu)
            elif sample_mode == 'samples':
                paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                                   pegasus_seed=self.seed, num_cpu=num_cpu)

        if self.save_logs:
            self.logger[self.task_id].log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, self.gamma)

        ##############################
        process_samples.compute_LL(paths, self.policy)
        self.replay_buffer[self.task_id].push(paths)    # store paths for replay without advantage (will be re-computed)
        ##############################

        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, self.gamma, self.gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger[self.task_id].log_kv('time_VF', timer.time()-ts)
            self.logger[self.task_id].log_kv('VF_error_before', error_before)
            self.logger[self.task_id].log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

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
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages, len(paths)).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages, len(paths))
        t_gLL += timer.time() - ts

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + self.alpha * vpg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages, len_paths).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger[self.task_id].log_kv('alpha', self.alpha)
            self.logger[self.task_id].log_kv('time_vpg', t_gLL)
            self.logger[self.task_id].log_kv('kl_dist', kl_dist)
            self.logger[self.task_id].log_kv('surr_improvement', surr_after - surr_before)
            self.logger[self.task_id].log_kv('running_score', self.running_score)

        return base_stats

    def test_tasks(self, task_ids=None, 
                    test_rollouts=10,
                    num_cpu=1):
        if task_ids is None:
            task_ids = list(self.observed_tasks)

        mean_pol_perf = {}
        for task_id in task_ids:
            self.set_task(task_id)
            eval_paths = trajectory_sampler.sample_paths_parallel(N=test_rollouts, policy=self.policy, num_cpu=num_cpu,
                                               env_name=self.env.env_id, mode='evaluation', pegasus_seed=self.seed)
            mean_pol_perf[task_id] = np.mean([np.sum(path['rewards']) for path in eval_paths])
            self.seed = self.seed + test_rollouts if self.seed is not None else self.seed

        return mean_pol_perf

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger[self.task_id].log_kv('stoc_pol_mean', mean_return)
        self.logger[self.task_id].log_kv('stoc_pol_std', std_return)
        self.logger[self.task_id].log_kv('stoc_pol_max', max_return)
        self.logger[self.task_id].log_kv('stoc_pol_min', min_return)

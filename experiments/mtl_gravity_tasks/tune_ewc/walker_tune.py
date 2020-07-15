from mjrl.utils.gym_env import GymEnv
# from mjrl.policies.gaussian_linear_ewc import LinearPolicyEWC   # use this instead for not shared version
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_ewc import NPGEWC
from mjrl.utils.train_agent import train_agent
import time as timer
import numpy as np
import gym
import pickle
import torch
from mjrl.utils.make_train_plots import make_multitask_train_plots, make_multitask_test_plots

SEED = 0    # use different tasks for tuning
torch.set_num_threads(5)

# MTL policy
# ==================================
lambda_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
e_unshuffled = {}

num_tasks = 5
num_cpu = 5
num_seeds = 5

np.random.seed(SEED)
gravity_factors = []
for task_id in range(num_tasks):
      try_again = True
      while try_again:
            gravity_factor = np.random.rand() + 0.5
            env_id = 'Walker{:.4f}g-v0'.format(gravity_factor)
            try:
                  gym.envs.register(
                        id=env_id,
                        entry_point='gym_extensions.continuous.mujoco.modified_walker2d:Walker2dGravityEnv',
                        max_episode_steps=1000,
                        reward_threshold=3800.0,
                        kwargs=dict(gravity=-gravity_factor*9.81)
                        )
                  try_again = False
                  gravity_factors.append(gravity_factor)
            except:
                  try_again = True

      e_unshuffled[task_id] = GymEnv(env_id)   # only do the environment here, so different files can create the same tasks

job_name_ewc = 'tuning/ewc_std_shared_notscaled/walker_ewc_gravity_tuning'

for i in range(num_seeds):
      np.random.seed(SEED)
      torch.manual_seed(SEED)

      job_name_ewc_seed = job_name_ewc + '/seed_{}'.format(i)

      e = {}
      task_order = np.random.permutation(num_tasks)
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]

      for ewc_lambda in lambda_range:   
            baseline_ewc = {}
            for task_id in range(num_tasks):
                  baseline_ewc[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3, use_gpu=True)
                  
            policy_ewc = LinearPolicy(e[0].spec, seed=SEED)
            agent_ewc = NPGEWC(e, policy_ewc, baseline_ewc, ewc_lambda=ewc_lambda, scaled_lambda=False, normalized_step_size=0.1, seed=SEED, save_logs=True)


            # agent = BatchREINFORCE(e, policy, baseline, learn_rate=0.0001, seed=SEED, save_logs=True)
            job_name_ewc_seed_lambda = job_name_ewc_seed + '/lambda{}'.format(ewc_lambda)
            for task_id in range(num_tasks):
                  ts = timer.time()
                  train_agent(job_name=job_name_ewc_seed_lambda,
                              agent=agent_ewc,
                              seed=SEED,
                              niter=50,
                              gamma=0.995,  
                              gae_lambda=0.97,
                              num_cpu=num_cpu,
                              sample_mode='trajectories',
                              num_traj=50,
                              save_freq=5,
                              evaluation_rollouts=0,
                              task_id=task_id)
                  agent_ewc.add_approximate_cost(N=10, 
                        num_cpu=1)

                  print("time taken for linear policy training = %f" % (timer.time()-ts))

            f = open(job_name_ewc_seed_lambda+'/trained_ewc_policy.pickle', 'wb')
            pickle.dump(policy_ewc, f)
            f.close()
            f = open(job_name_ewc_seed_lambda+'/trained_ewc_baseline.pickle', 'wb')
            pickle.dump(baseline_ewc, f)
            f.close()
            f = open(job_name_ewc_seed_lambda+'/trained_ewc_alphas.pickle', 'wb')
            pickle.dump(agent_ewc.theta, f)
            f.close()
            f = open(job_name_ewc_seed_lambda+'/trained_ewc_grads.pickle', 'wb')
            pickle.dump(agent_ewc.grad, f)
            f.close()
            f = open(job_name_ewc_seed_lambda+'/trained_ewc_hess.pickle', 'wb')
            pickle.dump(agent_ewc.hess, f)
            f.close()
            f = open(job_name_ewc_seed_lambda+'/env_factors.pickle', 'wb')
            pickle.dump(gravity_factors, f)
            f.close()

            make_multitask_train_plots(loggers=agent_ewc.logger, keys=['stoc_pol_mean'], save_loc=job_name_ewc_seed_lambda+'/logs/')

            mean_test_perf = agent_ewc.test_tasks(test_rollouts=10,
                                num_cpu=num_cpu)
            result = np.mean(list(mean_test_perf.values()))
            print(result)
            make_multitask_test_plots(mean_test_perf, save_loc=job_name_ewc_seed_lambda+'/')

            result_file = open(job_name_ewc_seed_lambda + '/results.txt', 'w')
            result_file.write(str(mean_test_perf))
            result_file.close()

      SEED += 100


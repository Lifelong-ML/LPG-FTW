from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import time as timer
import numpy as np
import gym
import pickle
import torch
import os
from mjrl.utils.make_train_plots import make_multitask_train_plots

SEED = 50   # initial value, 10 will be added for every iteration
job_name_stl = 'results/hopper_stl_gravity_exp'
job_name_mtl = 'results/hopper_mtl_gravity_exp'
torch.set_num_threads(5)

# STL policy
# ==================================

num_tasks = 20
num_seeds = 5
num_cpu = 5


f = open(job_name_mtl+'/env_factors.pickle', 'rb')
gravity_factors = pickle.load(f)
f.close()
f = open(job_name_mtl+'/env_ids.pickle','rb')
env_ids = pickle.load(f)
f.close() 
e_unshuffled = {}
for task_id in range(num_tasks): 
      gravity_factor = gravity_factors[task_id] 
      env_id = env_ids[task_id]    
      gym.envs.register(
            id=env_id,
            entry_point='gym_extensions.continuous.mujoco.modified_hopper:HopperGravityEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs=dict(gravity=-gravity_factor*9.81)
            )
      e_unshuffled[task_id] = GymEnv(env_id)   # only do the environment here, so different files can create the same tasks

for i in range(num_seeds):
      np.random.seed(SEED)
      torch.manual_seed(SEED)

      job_name_stl_seed = job_name_stl + '/seed_{}'.format(i)

      e = {}
      baseline_stl = {}
      policy_stl = {}
      agent_stl = {}
      task_order = np.random.permutation(num_tasks)
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]
            baseline_stl[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=10, learn_rate=1e-4, use_gpu=True)
            policy_stl[task_id] = LinearPolicy(e[task_id].spec, seed=SEED)
            agent_stl[task_id] = NPG(e[task_id], policy_stl[task_id], baseline_stl[task_id], normalized_step_size=0.01, seed=SEED, save_logs=True)

      loggers_stl = {}
      grads_stl = {}
      hess_stl = {}
      for task_id in range(num_tasks):
            ts = timer.time()

            train_agent(job_name=job_name_stl_seed,
                        agent=agent_stl[task_id],
                        seed=SEED,
                        niter=100,
                        gamma=0.995,  
                        gae_lambda=0.97,
                        num_cpu=num_cpu,
                        sample_mode='trajectories',
                        num_traj=50,
                        save_freq=5,
                        evaluation_rollouts=0,
                        task_id=task_id)
            agent_stl[task_id].add_approximate_cost(N=10, 
                  num_cpu=num_cpu)
            grads_stl[task_id] = agent_stl[task_id].grad
            hess_stl[task_id] = agent_stl[task_id].hess
            loggers_stl[task_id] = agent_stl[task_id].logger
            print("time taken for linear policy training = %f" % (timer.time()-ts))

      f = open(job_name_stl_seed+'/trained_stl_policy.pickle', 'wb')
      pickle.dump(policy_stl, f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_baseline.pickle', 'wb')
      pickle.dump(baseline_stl, f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_grads.pickle', 'wb')
      pickle.dump(grads_stl, f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_hess.pickle', 'wb')
      pickle.dump(hess_stl, f)
      f.close()
      f = open(job_name_stl_seed+'/task_order.pickle', 'wb')
      pickle.dump(task_order, f)
      f.close()

      make_multitask_train_plots(loggers=loggers_stl, keys=['stoc_pol_mean'], save_loc=job_name_stl_seed+'/logs/')

      SEED += 10



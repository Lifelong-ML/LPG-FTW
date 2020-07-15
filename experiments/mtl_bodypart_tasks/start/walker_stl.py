'''
Measure performance after updating L at the end
of training each task
'''

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.algos.npg_cg import NPG
import numpy as np
import gym
import pickle
import torch
import os

SEED = 50  # initial value, 10 will be added for every iteration
job_name_mtl = 'results/walker_mtl_bodyparts_exp'
job_name_stl = 'results/walker_stl_bodyparts_exp'

# MTL policy
# ==================================

num_tasks = 50
num_seeds = 5
num_cpu = 5

f = open(job_name_mtl+'/env_factors.pickle', 'rb')
size_factors_list = pickle.load(f)
f.close()
f = open(job_name_mtl+'/env_ids.pickle','rb')
env_ids = pickle.load(f)
f.close() 
e_unshuffled = {}
for task_id in range(num_tasks): 
      size_factors = size_factors_list[task_id] 
      env_id = env_ids[task_id]    
      gym.envs.register(
            id=env_id,
            entry_point='gym_extensions.continuous.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs=dict(body_parts=['torso_geom','thigh_geom','leg_geom','foot_geom'], size_scales=size_factors)
            )
      e_unshuffled[task_id] = GymEnv(env_id)   # only do the environment here, so different files can create the same tasks

for i in range(num_seeds):
      np.random.seed(SEED)
      torch.manual_seed(SEED)

      job_name_stl_seed = job_name_stl + '/seed_{}'.format(i)

      f = open(job_name_stl_seed+'/task_order.pickle', 'rb')
      task_order = pickle.load(f)
      f.close()
      e = {}
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]

      forward_transfer_results = {}
      for task_id in range(num_tasks):            
            f = open(job_name_stl_seed + '/iterations/task_{}/'.format(task_id) +  'policy_0.pickle', 'rb')
            policy_stl = pickle.load(f)
            f.close()
            f = open(job_name_stl_seed + '/iterations/task_{}/'.format(task_id) +  'baseline_0.pickle', 'rb')
            baseline_stl = pickle.load(f)
            f.close()  

            agent_stl = NPG(e[task_id], policy_stl, baseline_stl, normalized_step_size=0.01, seed=SEED, save_logs=False)
            agent_stl.set_task(task_id)
            eval_paths = trajectory_sampler.sample_paths_parallel(N=10,
                  policy=policy_stl,
                  num_cpu=num_cpu,
                  env_name=e[task_id].env_id,
                  mode='evaluation',
                  pegasus_seed=SEED)

            forward_transfer_results[task_id] = np.mean([np.sum(path['rewards']) for path in eval_paths])

      result_file = open(job_name_stl_seed + '/start_results.txt', 'w')
      result_file.write(str(forward_transfer_results))
      result_file.close()

      SEED += 10




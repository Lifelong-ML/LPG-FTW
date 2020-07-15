import numpy as np
import gym
import pickle
import os

SEED = 700  # 500 for halfcheetah, 600 for hopper, 700 for walker 

num_tasks = 50

job_name_mtl = 'results/walker_mtl_gravity_exp'
os.makedirs(job_name_mtl, exist_ok=True)

np.random.seed(SEED)
gravity_factors = []
env_ids = []
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
                  env_ids.append(env_id)
            except:
                  try_again = True


pickle.dump(gravity_factors, open(job_name_mtl+'/env_factors.pickle', 'wb'))
pickle.dump(env_ids, open(job_name_mtl+'/env_ids.pickle', 'wb'))






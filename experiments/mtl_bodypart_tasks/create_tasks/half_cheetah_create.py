import numpy as np
import gym
import pickle
import os

SEED = 500  # 500 for halfcheetah, 600 for hopper, 700 for walker 

num_tasks = 20

job_name_mtl = 'results/halfcheetah_mtl_bodyparts_exp'
os.makedirs(job_name_mtl, exist_ok=True)

np.random.seed(SEED)
size_factors_list = []
env_ids = []

for task_id in range(num_tasks):
      try_again = True
      while try_again:
            # size_factors = np.random.rand(4) * 0.5 + 0.75    # between 0.75 and 1.25
            size_factors = np.random.rand(4) + 0.5             # between 0.5 and 1.5
            size_factors = np.floor(size_factors * 10000) / 10000       # 4 decimals precision 
            size_factors = list(size_factors)
            factors_str = str(size_factors).translate({ord(i):None for i in ', []'})  # remove spaces, commas, and [] from string
            env_id = 'HalfCheetah'+factors_str+'_sizes-v0'
            try:
                  gym.envs.register(
                        id=env_id,
                        entry_point='gym_extensions.continuous.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                        max_episode_steps=1000,
                        reward_threshold=3800.0,
                        kwargs=dict(body_parts=['torso','fthigh','fshin','ffoot'], size_scales=size_factors)
                        )
                  try_again = False
                  size_factors_list.append(size_factors)
                  env_ids.append(env_id)
            except:
                  try_again = True

pickle.dump(size_factors_list, open(job_name_mtl+'/env_factors.pickle', 'wb'))
pickle.dump(env_ids, open(job_name_mtl+'/env_ids.pickle', 'wb'))


from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.algos.npg_cg import NPG
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import numpy as np
import gym
import pickle
import torch
import os

SEED = 50   # initial value, 10 will be added for every iteration
job_name_stl = 'results/metaworld_stl_exp'
torch.set_num_threads(5)

# MTL policy
# ==================================

num_tasks = 10
num_seeds = 5
num_cpu = 5

env_dict = {
      'reach-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
      'push-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
      'pick-place-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
      'door-v1': 'sawyer_door:SawyerDoorEnv',
      'drawer-open-v1': 'sawyer_drawer_open:SawyerDrawerOpenEnv',
      'drawer-close-v1': 'sawyer_drawer_close:SawyerDrawerCloseEnv',
      'button-press-topdown-v1': 'sawyer_button_press_topdown:SawyerButtonPressTopdownEnv',
      'peg-insert-side-v1': 'sawyer_peg_insertion_side:SawyerPegInsertionSideEnv',
      'window-open-v1': 'sawyer_window_open:SawyerWindowOpenEnv',
      'window-close-v1': 'sawyer_window_close:SawyerWindowCloseEnv',
}

e_unshuffled = {}

for task_id, (env_id, entry_point) in enumerate(env_dict.items()):
      kwargs = {'obs_type': 'plain'}
      if env_id == 'reach-v1':
            kwargs['task_type'] = 'reach'
      elif env_id == 'push-v1':
            kwargs['task_type'] = 'push'
      elif env_id == 'pick-place-v1':
            kwargs['task_type'] = 'pick_place'
      gym.envs.register(
            id=env_id,
            entry_point='metaworld.envs.mujoco.sawyer_xyz.' + entry_point,
            max_episode_steps=150,
            kwargs=kwargs
            )
      e_unshuffled[task_id] = GymEnv(env_id)

for i in range(num_seeds):
      np.random.seed(SEED)
      torch.manual_seed(SEED)
      job_name_stl_seed = job_name_stl + '/seed_{}'.format(i)
      e = {}
      task_order = np.random.permutation(num_tasks)
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]
            
      baseline_stl = {} 
      forward_transfer_results = {}
      for task_id in range(num_tasks):            
            iterdir = job_name_stl_seed + '/iterations/task_{}/'.format(task_id)
            f = open(iterdir + 'policy_199.pickle','rb')
            policy_stl = pickle.load(f)
            f.close()
            f = open(iterdir +  'baseline_199.pickle', 'rb')
            baseline_stl[task_id] = pickle.load(f)
            f.close()

            agent_stl = NPG(e[task_id], policy_stl, baseline_stl[task_id], normalized_step_size=0.01, seed=SEED, save_logs=False)
            agent_stl.set_task(task_id)
            eval_paths = trajectory_sampler.sample_paths_parallel(N=10,
                  policy=policy_stl,
                  num_cpu=num_cpu,
                  env_name=e[task_id].env_id,
                  mode='evaluation',
                  pegasus_seed=SEED)

            forward_transfer_results[task_id] = np.mean([np.sum(path['rewards']) for path in eval_paths])

      result_file = open(job_name_stl_seed + '/finetune_results.txt', 'w')
      result_file.write(str(forward_transfer_results))
      result_file.close()

      SEED += 10




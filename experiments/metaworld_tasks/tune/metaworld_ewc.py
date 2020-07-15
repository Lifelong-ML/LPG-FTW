from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_ewc import MLPEWC
from mjrl.algos.npg_cg_ewc_mlp import NPGEWC
import numpy as np
import gym
import pickle
import torch
import os

SEED = 50   # initial value, 10 will be added for every iteration
job_name_ewc = 'results/metaworld_ewc_exp'
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
      job_name_ewc_seed = job_name_ewc + '/seed_{}'.format(i)
      e = {}
      task_order = np.random.permutation(num_tasks)
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]
            
      baseline_mtl = {} 
      forward_transfer_results = {}
      for task_id in range(num_tasks):            
            iterdir = job_name_ewc_seed + '/iterations/task_{}/'.format(task_id)
            f = open(iterdir + 'policy_199.pickle','rb')
            policy_mtl = pickle.load(f)
            f.close()
            f = open(iterdir +  'baseline_199.pickle', 'rb')
            baseline_mtl[task_id] = pickle.load(f)
            f.close()

            agent_mtl = NPGEWC(e, policy_mtl, baseline_mtl, ewc_lambda=1e-7, scaled_lambda=False, normalized_step_size=0.01, seed=SEED, save_logs=True)
            mean_test_perf = agent_mtl.test_tasks(test_rollouts=10,
                                num_cpu=num_cpu,
                                task_ids=np.array([task_id]))
            forward_transfer_results = {**forward_transfer_results, **mean_test_perf}

      result_file = open(job_name_ewc_seed + '/finetune_results.txt', 'w')
      result_file.write(str(forward_transfer_results))
      result_file.close()

      SEED += 10




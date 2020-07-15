from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_ewc import MLPEWC
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_er import NPGER
from mjrl.utils.train_agent import train_agent
import time as timer
import numpy as np
import gym
import pickle
import torch
import os
from mjrl.utils.make_train_plots import make_multitask_train_plots, make_multitask_test_plots

import argparse
parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')

parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=5, type=int)
parser.add_argument('-i', '--initial_seed', dest='initial_seed', default=0, type=int)

args = parser.parse_args()

SEED = 50 + 10 * args.initial_seed   # use different orders for tuning
job_name_er = 'results/metaworld_er_exp'
torch.set_num_threads(5)

# MTL policy
# ==================================

num_tasks = 10
num_seeds = args.num_seeds
initial_seed = args.initial_seed
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

for i in range(initial_seed, num_seeds + initial_seed):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    job_name_er_seed = job_name_er + '/seed_{}'.format(i)

    e = {}
    baseline_er = {}   
    task_order = np.random.permutation(num_tasks)
    for task_id in range(num_tasks):
        e[task_id] = e_unshuffled[task_order[task_id]]
        baseline_er[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=10, learn_rate=1e-3, use_gpu=True)

    policy_er = MLPEWC(e[0].spec, hidden_sizes=(32,32), seed=SEED)
    agent_er = NPGER(e, policy_er, baseline_er, capacity=500, normalized_step_size=0.01, seed=SEED, save_logs=True, gamma=0.995, gae_lambda=0.97)

    for task_id in range(num_tasks):
        ts = timer.time()
        train_agent(job_name=job_name_er_seed,
                agent=agent_er,
                seed=SEED,
                niter=200,
                gamma=0.995,  
                gae_lambda=0.97,
                num_cpu=num_cpu,
                sample_mode='trajectories',
                num_traj=50,
                save_freq=5,
                evaluation_rollouts=0,
                task_id=task_id)
        iterdir = job_name_er_seed + '/iterations/task_{}/'.format(task_id)
        os.makedirs(iterdir, exist_ok=True)
        policy_file = open(iterdir + 'policy_updated.pickle', 'wb')
        pickle.dump(agent_er.policy, policy_file)
        policy_file.close()

        print("time taken for linear policy training = %f" % (timer.time()-ts))

    f = open(job_name_er_seed+'/trained_mtl_policy.pickle', 'wb')
    pickle.dump(policy_er, f)
    f.close()
    f = open(job_name_er_seed+'/trained_mtl_baseline.pickle', 'wb')
    pickle.dump(baseline_er, f)
    f.close()
    f = open(job_name_er_seed+'/task_order.pickle', 'wb')
    pickle.dump(task_order, f)
    f.close()


    make_multitask_train_plots(loggers=agent_er.logger, keys=['stoc_pol_mean'], save_loc=job_name_er_seed+'/logs/')

    mean_test_perf = agent_er.test_tasks(test_rollouts=10,
                  num_cpu=num_cpu)
    result = np.mean(list(mean_test_perf.values()))
    print(result)
    make_multitask_test_plots(mean_test_perf, save_loc=job_name_er_seed+'/')

    result_file = open(job_name_er_seed + '/results.txt', 'w')
    result_file.write(str(mean_test_perf))
    result_file.close()

    SEED += 10




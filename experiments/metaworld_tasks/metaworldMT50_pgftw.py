from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_lpg_ftw import MLPLPGFTW
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_ftw import NPGFTW
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

SEED = 50 + 10 * args.initial_seed  # initial value, 10 will be added for every iteration
job_name_lpgftw = 'results/metaworldMT50_lpgftw_exp'
torch.set_num_threads(5)

# MTL policy
# ==================================

num_tasks = 48
num_seeds = args.num_seeds
initial_seed = args.initial_seed
num_cpu = 5

env_dict = {
    'reach-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
    'push-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
    'pick-place-v1': 'sawyer_reach_push_pick_place:SawyerReachPushPickPlaceEnv',
    'reach-wall-v1': 'sawyer_reach_push_pick_place_wall:SawyerReachPushPickPlaceWallEnv',
    'pick-place-wall-v1': 'sawyer_reach_push_pick_place_wall:SawyerReachPushPickPlaceWallEnv',
    'push-wall-v1': 'sawyer_reach_push_pick_place_wall:SawyerReachPushPickPlaceWallEnv',
    'door-open-v1': 'sawyer_door:SawyerDoorEnv',
    'door-close-v1': 'sawyer_door_close:SawyerDoorCloseEnv',
    'drawer-open-v1': 'sawyer_drawer_open:SawyerDrawerOpenEnv',
    'drawer-close-v1': 'sawyer_drawer_close:SawyerDrawerCloseEnv',
    'button-press-topdown-v1': 'sawyer_button_press_topdown:SawyerButtonPressTopdownEnv',
    'button-press-v1': 'sawyer_button_press:SawyerButtonPressEnv',
    'button-press-topdown-wall-v1': 'sawyer_button_press_topdown_wall:SawyerButtonPressTopdownWallEnv',
    'button-press-wall-v1': 'sawyer_button_press_wall:SawyerButtonPressWallEnv',
    'peg-insert-side-v1': 'sawyer_peg_insertion_side:SawyerPegInsertionSideEnv',
    'peg-unplug-side-v1': 'sawyer_peg_unplug_side:SawyerPegUnplugSideEnv',
    'window-open-v1': 'sawyer_window_open:SawyerWindowOpenEnv',
    'window-close-v1': 'sawyer_window_close:SawyerWindowCloseEnv',
    'dissassemble-v1': 'sawyer_disassemble_peg:SawyerNutDisassembleEnv',
    'hammer-v1': 'sawyer_hammer:SawyerHammerEnv',
    'plate-slide-v1': 'sawyer_plate_slide:SawyerPlateSlideEnv',
    'plate-slide-side-v1': 'sawyer_plate_slide_side:SawyerPlateSlideSideEnv',
    'plate-slide-back-v1': 'sawyer_plate_slide_back:SawyerPlateSlideBackEnv',
    'plate-slide-back-side-v1': 'sawyer_plate_slide_back_side:SawyerPlateSlideBackSideEnv',
    'handle-press-v1': 'sawyer_handle_press:SawyerHandlePressEnv',
    'handle-pull-v1': 'sawyer_handle_pull:SawyerHandlePullEnv',
    'handle-press-side-v1': 'sawyer_handle_press_side:SawyerHandlePressSideEnv',
    'handle-pull-side-v1': 'sawyer_handle_pull_side:SawyerHandlePullSideEnv',
    # 'stick-push-v1': 'sawyer_stick_push:SawyerStickPushEnv',
    # 'stick-pull-v1': 'sawyer_stick_pull:SawyerStickPullEnv',
    'basketball-v1': 'sawyer_basketball:SawyerBasketballEnv',
    'soccer-v1': 'sawyer_soccer:SawyerSoccerEnv',
    'faucet-open-v1': 'sawyer_faucet_open:SawyerFaucetOpenEnv',
    'faucet-close-v1': 'sawyer_faucet_close:SawyerFaucetCloseEnv',
    'coffee-push-v1': 'sawyer_coffee_push:SawyerCoffeePushEnv',
    'coffee-pull-v1': 'sawyer_coffee_pull:SawyerCoffeePullEnv',
    'coffee-button-v1': 'sawyer_coffee_button:SawyerCoffeeButtonEnv',
    'sweep-v1': 'sawyer_sweep:SawyerSweepEnv',
    'sweep-into-v1': 'sawyer_sweep_into_goal:SawyerSweepIntoGoalEnv',
    'pick-out-of-hole-v1': 'sawyer_pick_out_of_hole:SawyerPickOutOfHoleEnv',
    'assembly-v1': 'sawyer_assembly_peg:SawyerNutAssemblyEnv',
    'shelf-place-v1': 'sawyer_shelf_place:SawyerShelfPlaceEnv',
    'push-back-v1': 'sawyer_push_back:SawyerPushBackEnv',
    'lever-pull-v1': 'sawyer_lever_pull:SawyerLeverPullEnv',
    'dial-turn-v1': 'sawyer_dial_turn:SawyerDialTurnEnv',
    'bin-picking-v1': 'sawyer_bin_picking:SawyerBinPickingEnv',
    'box-close-v1': 'sawyer_box_close:SawyerBoxCloseEnv',
    'hand-insert-v1': 'sawyer_hand_insert:SawyerHandInsertEnv',
    'door-lock-v1': 'sawyer_door_lock:SawyerDoorLockEnv',
    'door-unlock-v1': 'sawyer_door_unlock:SawyerDoorUnlockEnv'
}

e_unshuffled = {}

for task_id, (env_id, entry_point) in enumerate(env_dict.items()):
    kwargs = {'obs_type': 'plain', 'random_init': False}
    if env_id == 'reach-v1' or env_id == 'reach-wall-v1':
        kwargs['task_type'] = 'reach'
    elif env_id == 'push-v1' or env_id == 'push-wall-v1':
        kwargs['task_type'] = 'push'
    elif env_id == 'pick-place-v1' or env_id == 'pick-place-wall-v1':
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

    job_name_lpgftw_seed = job_name_lpgftw + '/seed_{}'.format(i)

    e = {}
    baseline_mtl = {}   
    task_order = np.random.permutation(num_tasks)
    for task_id in range(num_tasks):
        e[task_id] = e_unshuffled[task_order[task_id]]
        baseline_mtl[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=10, learn_rate=1e-3, use_gpu=True)
        
    policy_mtl = MLPLPGFTW(e[0].spec, hidden_sizes=(32,32), k=1, max_k=3, seed=SEED)
    agent_mtl = NPGFTW(e, policy_mtl, baseline_mtl, normalized_step_size=0.01, seed=SEED, save_logs=True, new_col_mode='max_k')


    for task_id in range(num_tasks):
        ts = timer.time()
        train_agent(job_name=job_name_lpgftw_seed,
                agent=agent_mtl,
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
        agent_mtl.add_approximate_cost(N=10, 
            task_id=task_id, 
            num_cpu=num_cpu)
        iterdir = job_name_lpgftw_seed + '/iterations/task_{}/'.format(task_id)
        os.makedirs(iterdir, exist_ok=True)
        policy_file = open(iterdir + 'policy_updated.pickle', 'wb')
        pickle.dump(agent_mtl.policy, policy_file)
        policy_file.close()

        print("time taken for linear policy training = %f" % (timer.time()-ts))

    f = open(job_name_lpgftw_seed+'/trained_mtl_policy.pickle', 'wb')
    pickle.dump(policy_mtl, f)
    f.close()
    f = open(job_name_lpgftw_seed+'/trained_mtl_baseline.pickle', 'wb')
    pickle.dump(baseline_mtl, f)
    f.close()
    f = open(job_name_lpgftw_seed+'/trained_mtl_alphas.pickle', 'wb')
    pickle.dump(agent_mtl.theta, f)
    f.close()
    f = open(job_name_lpgftw_seed+'/trained_mtl_grads.pickle', 'wb')
    pickle.dump(agent_mtl.grad, f)
    f.close()
    f = open(job_name_lpgftw_seed+'/trained_mtl_hess.pickle', 'wb')
    pickle.dump(agent_mtl.hess, f)
    f.close()
    f = open(job_name_lpgftw_seed+'/task_order.pickle', 'wb')
    pickle.dump(task_order, f)
    f.close()


    make_multitask_train_plots(loggers=agent_mtl.logger, keys=['stoc_pol_mean'], save_loc=job_name_lpgftw_seed+'/logs/')

    mean_test_perf = agent_mtl.test_tasks(test_rollouts=10,
                  num_cpu=num_cpu)
    result = np.mean(list(mean_test_perf.values()))
    print(result)
    make_multitask_test_plots(mean_test_perf, save_loc=job_name_lpgftw_seed+'/')

    result_file = open(job_name_lpgftw_seed + '/results.txt', 'w')
    result_file.write(str(mean_test_perf))
    result_file.close()

    SEED += 10


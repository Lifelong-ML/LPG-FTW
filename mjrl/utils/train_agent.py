import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.trajectory_sampler import sample_paths_parallel
import numpy as np
import pickle
import time as timer
import os
import copy

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                task_id = None,     # only used for MTL 
                ):

    if task_id is not None:
        agent.set_task(task_id)
        try:
            logger = agent.logger[task_id]
        except:
            logger = agent.logger
    else:
        logger = agent.logger

    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.makedirs(job_name, exist_ok=True)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.copy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    e = GymEnv(agent.env.env_id)
    for i in range(niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)
        if train_curve[i-1] > best_perf:
            best_policy = copy.copy(agent.policy)
            best_perf = train_curve[i-1]
        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)
        if task_id is not None:
            args['task_id'] = task_id
        stats = agent.train_step(**args)
        train_curve[i] = stats[0]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths_parallel(N=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                               env_name=e.env_id, mode='evaluation', pegasus_seed=seed)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                logger.log_kv('eval_score', mean_pol_perf)
        if i % save_freq == 0  or i == niter - 1: # and i > 0
            if agent.save_logs:
                if task_id is None:
                    logdir = 'logs/'
                else:
                    logdir = 'logs/task_{}/'.format(task_id)
                os.makedirs(logdir, exist_ok=True)
                logger.save_log(logdir)
                make_train_plots(log=logger.log, keys=plot_keys, save_loc=logdir)
            if task_id is None:
                iterdir = 'iterations/'
            else:
                iterdir = 'iterations/task_{}/'.format(task_id)
            os.makedirs(iterdir, exist_ok=True)

            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            f = open(iterdir + policy_file, 'wb')
            pickle.dump(agent.policy, f)
            f.close()
            f = open(iterdir + baseline_file, 'wb')
            pickle.dump(agent.baseline, f)
            f.close()
            f = open(iterdir + 'best_policy.pickle', 'wb')
            pickle.dump(best_policy, f)
            f.close()
        # print results to console
        if i == 0:
            if task_id is None:
                result_file = open('results.txt', 'w')
            else:
                result_file = open('results_{}.txt'.format(task_id), 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, best_perf))

        if task_id is None:
            result_file = open('results.txt', 'a')
        else:
            result_file = open('results_{}.txt'.format(task_id), 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    if task_id is None:
        best_policy_file = open('iterations/best_policy.pickle', 'wb')
    else:
        best_policy_file = open('iterations/task_{}/best_policy.pickle'.format(task_id), 'wb')
    pickle.dump(best_policy, best_policy_file)
    best_policy_file.close()
    if agent.save_logs:
        if task_id is None:
            logger.save_log('logs/')
            make_train_plots(log=logger.log, keys=plot_keys, save_loc='logs/')
        else:
            logger.save_log('logs/task_{}/'.format(task_id))
            make_train_plots(log=logger.log, keys=plot_keys, save_loc='logs/task_{}/'.format(task_id))
    os.chdir(previous_dir)

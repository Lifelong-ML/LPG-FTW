from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_lpg_ftw import MLPLPGFTW
from mjrl.algos.npg_cg import NPG
from mjrl.algos.npg_cg_ftw import NPGFTW
import numpy as np
import gym
import pickle
import torch
import os
import scipy.linalg
from sklearn.linear_model import Lasso
from torch.autograd import Variable

SEED = 50   # initial value, 10 will be added for every iteration
job_name_stl = 'results/metaworld_stl_exp'
job_name_pgella = 'results/metaworld_pgella_exp'
torch.set_num_threads(5)

# STL policy
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
      job_name_pgella_seed = job_name_pgella + '/seed_{}'.format(i)

      f = open(job_name_stl_seed+'/trained_stl_policy.pickle', 'rb')
      policy_stl = pickle.load(f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_baseline.pickle', 'rb')
      baseline_stl = pickle.load(f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_grads.pickle', 'rb')
      grads_stl = pickle.load(f)
      f.close()
      f = open(job_name_stl_seed+'/trained_stl_hess.pickle', 'rb')
      hess_stl = pickle.load(f)
      f.close()
      f = open(job_name_stl_seed+'/task_order.pickle', 'rb')
      task_order = pickle.load(f)
      f.close()

      e = {}
      agent_stl = {}
      for task_id in range(num_tasks):
            e[task_id] = e_unshuffled[task_order[task_id]]
            agent_stl[task_id] = NPG(e[task_id], policy_stl[task_id], baseline_stl[task_id], normalized_step_size=0.01, seed=SEED, save_logs=True)
            agent_stl[task_id].grad = grads_stl[task_id]
            agent_stl[task_id].hess = hess_stl[task_id]
      
      k = 3  

      n = policy_stl[0].n
      m = policy_stl[0].m 
      d = ((n + 1) * policy_stl[0].model.hidden_sizes[0]
            + (policy_stl[0].model.hidden_sizes[0] + 1) * policy_stl[0].model.hidden_sizes[1])
      A = np.zeros((d*k, d*k))
      b = np.zeros((d*k, 1))   
      S = np.zeros((k, num_tasks))
      L = np.zeros((d, k))
      Theta = np.zeros((d, num_tasks))
      policy_mtl = MLPLPGFTW(e[0].spec, hidden_sizes=(32,32), k=k, max_k=k,seed=SEED)
      agent_mtl = NPGFTW(e, policy_mtl, baseline_stl, normalized_step_size=0.01, seed=SEED, save_logs=True, new_col_mode='max_k')

      lasso_solver = Lasso(alpha=1e-5, fit_intercept=False)
      forward_transfer_results = {}
      for task_id in range(num_tasks):
            theta = torch.cat([
                policy_stl[task_id].model.fc0.weight.contiguous().view(-1,1).data,
                policy_stl[task_id].model.fc0.bias.contiguous().view(-1,1).data,
                policy_stl[task_id].model.fc1.weight.contiguous().view(-1,1).data,
                policy_stl[task_id].model.fc1.bias.contiguous().view(-1,1).data
              ]).data.numpy()

            agent_mtl.theta[task_id] = theta
            
            Theta[:, task_id] = theta.squeeze()
            D = -agent_stl[task_id].hess
            T = task_id + 1
            agent_mtl.hess[task_id] = D
            agent_mtl.grad[task_id] = np.zeros((D.shape[1],1))
            
            if T <= k:
                  L[:, task_id] = theta.squeeze()
                  s = np.zeros((k,1))
                  s[task_id] = 1
            else:
                  # Find s
                  Dsqrt = np.real(scipy.linalg.sqrtm(D))
                  target = Dsqrt.dot(theta);
                  feat = Dsqrt.dot(L)

                  lasso_solver.fit(feat, target)
                  s = lasso_solver.coef_.reshape(-1,1)
            S[:, task_id] = s.squeeze()
            A += np.kron(s.dot(s.T), D)
            b += np.kron(s.T, theta.T.dot(D)).T

            vals = np.linalg.inv(A / T + 1e-5 * np.eye(d*k)).dot(b / T)
            L = vals.reshape(L.shape, order='F')

            policy_mtl.model.L.data = torch.from_numpy(L).float()
            policy_mtl.model.S[task_id] = Variable(torch.from_numpy(S[:, task_id]).float(), requires_grad=False).unsqueeze(1)
            policy_mtl.old_model.S[task_id] = Variable(torch.from_numpy(S[:, task_id]).float(), requires_grad=False).unsqueeze(1)
            policy_mtl.log_std[task_id] = policy_stl[task_id].log_std
            policy_mtl.old_log_std[task_id] = policy_stl[task_id].log_std
            policy_mtl.model.epsilon_col = torch.zeros(L.shape[0], 1, requires_grad=False)
            policy_mtl.old_model.epsilon_col = torch.zeros(L.shape[0], 1, requires_grad=False)
            policy_mtl.model.fc_out[task_id] = policy_stl[task_id].model.fc2
            policy_mtl.old_model.fc_out[task_id] = policy_stl[task_id].model.fc2

            mean_test_perf = agent_mtl.test_tasks(test_rollouts=10,
                  num_cpu=num_cpu,
                  task_ids=np.array([task_id]))
            forward_transfer_results = {**forward_transfer_results, **mean_test_perf}

      agent_mtl.observed_tasks = set(np.arange(num_tasks))
      

      os.makedirs(job_name_pgella_seed, exist_ok=True)
      f = open(job_name_pgella_seed+'/trained_mtl_policy.pickle', 'wb')
      pickle.dump(policy_mtl, f)
      f.close()
      f = open(job_name_pgella_seed+'/task_order.pickle', 'wb')
      pickle.dump(task_order, f)
      f.close()

      mean_test_perf = agent_mtl.test_tasks(test_rollouts=10,
                          num_cpu=num_cpu)
      result = np.mean(list(mean_test_perf.values()))
      print(result)
      result_file = open(job_name_pgella_seed + '/results.txt', 'w')
      result_file.write(str(mean_test_perf))
      result_file.close()

      forward_transfer_result_file = open(job_name_pgella_seed + '/forward_transfer_results.txt', 'w')
      forward_transfer_result_file.write(str(forward_transfer_results))
      forward_transfer_result_file.close()

      SEED += 10



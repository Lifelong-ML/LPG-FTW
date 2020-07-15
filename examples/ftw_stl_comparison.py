from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.policies.gaussian_linear_lpg_ftw import LinearPolicyLPGFTW
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.npg_cg_ftw import NPGFTW
from mjrl.utils.train_agent import train_agent
import time as timer
import numpy as np
import gym

from mjrl.utils.make_train_plots import make_multitask_train_plots, make_multitask_test_plots

SEED = 500

# MTL policy
# ==================================
e = {}
baseline_mtl = {}
baseline_stl = {}
policy_stl = {}
agent_stl = {}
num_tasks = 20
num_cpu = 1

np.random.seed(SEED)
for task_id in range(num_tasks):
      try_again = True
      while try_again:
            gravity_factor = np.random.rand() + 0.5
            env_id = 'HalfCheetah{:.4f}g-v0'.format(gravity_factor)
            try:
                  gym.envs.register(
                        id=env_id,
                        entry_point='gym_extensions.continuous.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                        max_episode_steps=1000,
                        reward_threshold=3800.0,
                        kwargs=dict(gravity=-gravity_factor*9.81)
                        )
                  try_again = False
            except:
                  try_again = True

      # e[task_id] =  GymEnv('mjrl_swimmer-v0')
      e[task_id] = GymEnv(env_id)   # only do the environment here, so different files can create the same tasks
      print(env_id)

for task_id in range(num_tasks):
      # e = GymEnv('mjrl_point_mass-v0')
      # e = GymEnv('Swimmer-v2')
      baseline_mtl[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
      
      baseline_stl[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
      policy_stl[task_id] = LinearPolicy(e[task_id].spec, seed=SEED)
      agent_stl[task_id] = NPG(e[task_id], policy_stl[task_id], baseline_stl[task_id], normalized_step_size=1, seed=SEED, save_logs=True)

policy_mtl = LinearPolicyLPGFTW(e[0].spec, k=1, max_k=2, seed=SEED)
agent_mtl = NPGFTW(e, policy_mtl, baseline_mtl, normalized_step_size=1, seed=SEED, save_logs=True, new_col_mode='max_k')


# agent = BatchREINFORCE(e, policy, baseline, learn_rate=0.0001, seed=SEED, save_logs=True)
job_name_mtl = 'halfcheetah_lpgftw_gravity_exp1'
job_name_stl = 'halfcheetah_stl_gravity_exp1'
loggers_stl = {}
for task_id in range(num_tasks):
      ts = timer.time()
      train_agent(job_name=job_name_mtl,
                  agent=agent_mtl,
                  seed=SEED,
                  niter=50,
                  gamma=0.995,  
                  gae_lambda=0.97,
                  num_cpu=num_cpu,
                  sample_mode='trajectories',
                  num_traj=10,
                  save_freq=5,
                  evaluation_rollouts=5,
                  task_id=task_id)
      agent_mtl.add_approximate_cost(N=10, 
            task_id=task_id, 
            num_cpu=1)

      train_agent(job_name=job_name_stl,
                  agent=agent_stl[task_id],
                  seed=SEED,
                  niter=50,
                  gamma=0.995,  
                  gae_lambda=0.97,
                  num_cpu=num_cpu,
                  sample_mode='trajectories',
                  num_traj=10,
                  save_freq=5,
                  evaluation_rollouts=5,
                  task_id=task_id)
      loggers_stl[task_id] = agent_stl[task_id].logger
      print("time taken for linear policy training = %f" % (timer.time()-ts))

make_multitask_train_plots(loggers=agent_mtl.logger, keys=['stoc_pol_mean'], save_loc=job_name_mtl+'/logs/')
make_multitask_train_plots(loggers=loggers_stl, keys=['stoc_pol_mean'], save_loc=job_name_stl+'/logs/')

mean_test_perf = agent_mtl.test_tasks(test_rollouts=10,
                    num_cpu=num_cpu)
result = np.mean(list(mean_test_perf.values()))
print(result)
make_multitask_test_plots(mean_test_perf, save_loc=job_name_mtl+'/')





from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import time as timer

SEED = 500


e = GymEnv('Walker2d-v2')
policy = LinearPolicy(e.spec, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='walker_nominal',
            agent=agent,
            seed=SEED,
            niter=500,
            gamma=0.995,  
            gae_lambda=0.97,
            num_cpu=4,
            sample_mode='trajectories',
            num_traj=50,
            save_freq=5,
            evaluation_rollouts=5)
print("time taken for linear policy training = %f" % (timer.time()-ts))

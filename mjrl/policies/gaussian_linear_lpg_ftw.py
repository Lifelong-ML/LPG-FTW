import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from warnings import warn


class LinearPolicyLPGFTW:
    def __init__(self, env_spec,
                 k=3,
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None,
                 max_k=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std
        self.k = k      # number of elements in the dictionary (LPG-FTW)
        self.max_k = max_k

        self.init_log_std = init_log_std
        
        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = LinearModel(self.n, self.m, self.k, self.max_k)
        self.log_std = {}
        self.old_log_std = {}

        # # Old Policy network
        # # ------------------------
        self.old_model = LinearModel(self.n, self.m, self.k, self.max_k)
        self.old_model.L.data = self.model.L.data.clone()

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    def set_task(self, task_id):
        self.task_id = task_id
        first_time = False
        if self.task_id not in self.log_std:
            first_time = True

        self.model.set_task(task_id)
        self.old_model.set_task(task_id)
        self.old_model.S[task_id] = Variable(self.model.S[task_id])
        self.old_model.epsilon_col = Variable(self.model.epsilon_col)

        if first_time:
            self.log_std[task_id] = Variable(torch.ones(self.m) * self.init_log_std, requires_grad=True)
        # Old Policy network
        # ------------------------
        if first_time:
            self.old_log_std[task_id] = Variable(torch.ones(self.m) * self.init_log_std)
            self.old_model.L.data = self.model.L.data.clone()
        
        if self.model.T <= self.k:
            self.trainable_params = [self.log_std[task_id], self.model.L]
            self.old_params = [self.old_log_std[task_id], self.old_model.L]
        elif self.model.T <= self.max_k:
            self.trainable_params = [self.log_std[task_id], self.model.S[task_id], self.model.epsilon_col]
            self.old_params = [self.old_log_std[task_id], self.old_model.S[task_id], self.old_model.epsilon_col]
        else:       # if self.model.T > self.k
            self.trainable_params = [self.log_std[task_id], self.model.S[task_id]]
            self.old_params = [self.old_log_std[task_id], self.old_model.S[task_id]]

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std[task_id].data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        
    # Utility functions
    # ============================================
    def get_param_values(self, task_id):
        print(self.model.S[task_id])
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, task_id, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[0].data = \
                torch.clamp(self.trainable_params[0], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std[task_id].data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[0].data = \
                torch.clamp(self.old_params[0], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std[self.task_id] if log_std is None else log_std
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std[self.task_id])
        return [LL, mean, self.old_log_std[self.task_id]]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std[self.task_id])
        return [LL, mean, self.log_std[self.task_id]]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)

        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)


class LinearModel(nn.Module):
    def __init__(self, obs_dim, act_dim, dict_dim,max_dict_dim=None,
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(LinearModel, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dict_dim = dict_dim
        self.max_dict_dim = max_dict_dim
        self.T = 0

        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        self.L = (torch.rand(act_dim*(obs_dim+1), dict_dim, requires_grad=True) - 0.5) * 2 * np.sqrt(1 / obs_dim) * 1e-2
        self.S = {} 

        self.use_theta = False

    def set_use_theta(self, use, add_epsilon=False):
        self.use_theta = use
        if use:
            if add_epsilon:
                self.theta = torch.autograd.Variable(torch.mm(self.L, self.S[self.task_id]) + self.epsilon_col, requires_grad=True)
            else:
                self.theta = torch.autograd.Variable(torch.mm(self.L, self.S[self.task_id]), requires_grad=True)
        else:
            self.theta = torch.autograd.Variable(torch.zeros(0))

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.in_shift  = Variable(self.in_shift, requires_grad=False)
        self.in_scale  = Variable(self.in_scale, requires_grad=False)
        self.out_shift = Variable(self.out_shift, requires_grad=False)
        self.out_scale = Variable(self.out_scale, requires_grad=False)

    def set_task(self, task_id):
        self.task_id = task_id

        if task_id not in self.S:
            if self.T < self.dict_dim:

                s = np.zeros((self.dict_dim, 1))
                s[self.T] = 1
                self.S[task_id] = Variable(torch.from_numpy(s).float(), requires_grad=True)
                self.epsilon_col = torch.zeros(self.act_dim*(self.obs_dim+1), 1, requires_grad=False)
            elif self.dict_dim < self.max_dict_dim:
                self.S[task_id] = Variable(torch.stack(list(self.S.values())).mean(0), requires_grad=True)    # mean of previous values
                self.epsilon_col = torch.zeros(self.act_dim*(self.obs_dim+1), 1, requires_grad=True)
            else:
                self.S[task_id] = Variable(torch.stack(list(self.S.values())).mean(0), requires_grad=True)    # mean of previous values
                self.epsilon_col = torch.zeros(self.act_dim*(self.obs_dim+1), 1, requires_grad=False)
            self.T += 1

    def forward(self, x):
        out = (x - self.in_shift)/(self.in_scale + 1e-8)
        out = torch.cat((out, torch.ones(out.shape[0], 1)), 1)

        if not self.use_theta:
            self.theta = torch.mm(self.L, self.S[self.task_id])

        out = (torch.mm(out, torch.t(self.theta.reshape((self.act_dim, self.obs_dim + 1)))) 
            + torch.mm(out, torch.t(self.epsilon_col.reshape((self.act_dim, self.obs_dim + 1)))))

        out = out * self.out_scale + self.out_shift
        return out
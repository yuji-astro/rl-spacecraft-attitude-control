import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from buffer import BasicBuffer
from noise import OUNoise
import numpy as np

#変数定義
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

#initialization 2021/1/11
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# neural network to approximate critic function
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        #Q1
        self.linear1 = nn.Linear(self.obs_dim + self.action_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        # self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

        #Q2
        self.linear5 = nn.Linear(self.obs_dim + self.action_dim, 1024)
        self.linear6 = nn.Linear(1024, 512)
        # self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear7 = nn.Linear(512, 300)
        self.linear8 = nn.Linear(300, 1)

        # #initialization
        # self.init_w=3e-3
        # self.init_weights(self.init_w)

    def init_weights(self, init_w): #2021/1/11
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.linear3.weight.data = fanin_init(self.linear3.weight.data.size())
        # self.linear4.weight.data.uniform_(-self.init_w, self.init_w)

        self.linear5.weight.data = fanin_init(self.linear5.weight.data.size())
        self.linear6.weight.data = fanin_init(self.linear6.weight.data.size())
        self.linear7.weight.data = fanin_init(self.linear7.weight.data.size())
        # self.linear8.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, x, a):
        x = torch.cat([x,a], 1)

        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = F.relu(self.linear3(q1))
        q1 = self.linear4(q1)

        q2 = F.relu(self.linear5(x))
        q2 = F.relu(self.linear6(q2))
        q2 = F.relu(self.linear7(q2))
        q2 = self.linear8(q2)

        return q1, q2
    
    def Q1(self, x, a):
        x = torch.cat([x,a], 1)
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = F.relu(self.linear3(q1))
        q1 = self.linear4(q1)
        return q1

# neural network to approximate actor function
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = 0.5
        # self.max_action = [1,1,1,1,1]

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)

        self.mean_linear = nn.Linear(128, self.action_dim)
        self.log_std_linear = nn.Linear(128, self.action_dim)

        # #initialization
        # self.init_w=1e-3
        # self.init_weights(self.init_w)

    def init_weights(self, init_w): #2021/1/11
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        # output of tanh is bounded between -1 and 1
        # multiply by maximum action (here: 10N) in order to scale the action appropriately
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)*self.max_action
        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True) #マイナス説あり
        mean = torch.tanh(mean)*self.max_action
        return action, log_prob, mean

# neural network to approximate actor function
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DeterministicPolicy, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = 0.5
        # self.max_action = [1,1,1,1,1]

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)

        self.mean = nn.Linear(128, self.action_dim)
        self.noise = torch.Tensor(self.action_dim)

        # #initialization
        # self.init_w=1e-3
        # self.init_weights(self.init_w)

    def init_weights(self, init_w): #2021/1/11
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.mean.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        # output of tanh is bounded between -1 and 1
        # multiply by maximum action (here: 10N) in order to scale the action appropriately
        mean = torch.tanh(self.mean(x))*self.max_action

        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

class SACAgent:
    def __init__(self, env, gamma, tau, alpha, buffer_maxlen, critic_learning_rate, actor_learning_rate, alpha_learning_rate, train, policy_type, self_entropy_tuning,
                    target_update_interval):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space.shape
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.train = train  # set to true if we want to train the agent, set to false to simulate agent
        self.policy_type = policy_type
        self.self_entropy_tuning = self_entropy_tuning
        self.max_action = 0.5
        self.lower_action = -0.5

        self.critic_loss_for_log = 0
        self.actor_loss_for_log = 0

        #ポリシーの更新頻度
        self.target_update_interval = target_update_interval
        self.total_it = 0

        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)

        if self.policy_type == "Gaussian":
            if self.self_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate) 
            
            self.actor = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)
        
        else:
            self.alpha = 0
            self.self_entropy_tuning = False
            self.actor = DeterministicPolicy(self.obs_dim, self.action_dim).to(self.device)

        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, obs):
        state = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
        if self.train is True:
            action,_,_ = self.actor.sample(state)
        else:
            _,_,action = self.actor.sample(state)
        action = action.squeeze(0).cpu().detach().numpy()
        action = np.clip(action, self.lower_action, self.max_action)
        
        return action

    def update(self, batch_size):
        self.total_it += 1
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # masks = torch.FloatTensor(masks).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_actions, next_log_pi, _ = self.actor.sample(next_state_batch)
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            # next_actions = self.actor_target.forward(next_state_batch)
            next_Q1_target, next_Q2_target = self.critic_target.forward(next_state_batch, next_actions.detach())
            min_next_Q_target = torch.min(next_Q1_target, next_Q2_target) - self.alpha * next_log_pi 
            # expected_Q = reward_batch + masks * self.gamma * next_Q
            expected_Q = reward_batch + self.gamma * min_next_Q_target

        
        curr_Q1, curr_Q2 = self.critic.forward(state_batch, action_batch)
        # update critic
        q_loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())
        self.critic_loss_for_log = q_loss.detach()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update actor
        pi, log_pi, _ = self.actor.sample(state_batch)
        Q1_pi, Q2_pi = self.critic(state_batch, pi)
        min_Q_pi = torch.min(Q1_pi, Q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_Q_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.actor_loss_for_log = policy_loss.detach()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        if self.self_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if self.total_it % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


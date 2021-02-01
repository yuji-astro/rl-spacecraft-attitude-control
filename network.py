import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffer import BasicBuffer
from noise import OUNoise


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

        #initialization
        self.init_w=1e-3
        # self.init_weights(self.init_w)
   
    def init_weights(self, init_w): #2021/1/11
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.linear3.weight.data = fanin_init(self.linear3.weight.data.size())
        self.linear4.weight.data.uniform_(-self.init_w, self.init_w)

        self.linear5.weight.data = fanin_init(self.linear5.weight.data.size())
        self.linear6.weight.data = fanin_init(self.linear6.weight.data.size())
        self.linear7.weight.data = fanin_init(self.linear7.weight.data.size())
        self.linear8.weight.data.uniform_(-self.init_w, self.init_w)

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
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = 0.5

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

        #initialization
        self.init_w=1e-3
        # self.init_weights(self.init_w)

    def init_weights(self, init_w): #2021/1/11
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.linear3.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        # output of tanh is bounded between -1 and 1
        # multiply by maximum action (here: 10N) in order to scale the action appropriately
        x = torch.tanh(self.linear3(x))
        return x


class TD3Agent:
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate, train, decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.train = train  # set to true if we want to train the agent, set to false to simulate agent
        self.decay = decay
        self.max_action = 1
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        #ポリシーの更新頻度
        self.policy_freq = 2 
        self.total_it = 0

        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)

        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)

        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Copy Actor target parameters 2021/1/11
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

        self.replay_buffer = BasicBuffer(buffer_maxlen)
        # use exploration noise only during training
        if self.train == True:
            self.noise = OUNoise(self.env.action_space, decay_period=self.decay)

    def get_action(self, obs, t=0):
        state = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        # add exploration noise only during training
        if self.train == True:
            action = self.noise.get_action(action, t=t)
        return action

    def update(self, batch_size):
        self.total_it += 1

        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # masks = torch.FloatTensor(masks).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target.forward(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
            # next_actions = self.actor_target.forward(next_state_batch)
            next_Q1, next_Q2 = self.critic_target.forward(next_state_batch, next_actions.detach())
            next_Q = torch.min(next_Q1, next_Q2)
            expected_Q = reward_batch + self.gamma * next_Q

        
        curr_Q1, curr_Q2 = self.critic.forward(state_batch, action_batch)
        # update critic
        q_loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update actor (deleyed)
        if self.total_it % self.policy_freq == 0:
            policy_loss = -self.critic.Q1(state_batch, self.actor.forward(state_batch)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

# # neural network to approximate critic function
# class Critic(nn.Module):
#     def __init__(self, obs_dim, action_dim):
#         super(Critic, self).__init__()

#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#         self.linear1 = nn.Linear(self.obs_dim, 1024)
#         # original paper adds action one layer before output layer, here we add action in first hidden layer
#         self.linear2 = nn.Linear(1024 + self.action_dim, 512)
#         self.linear3 = nn.Linear(512, 300)
#         self.linear4 = nn.Linear(300, 1)

#     def forward(self, x, a):
#         x = F.relu(self.linear1(x))
#         # original paper adds action one layer before output layer, here we add action in first hidden layer
#         xa_cat = torch.cat([x,a], 1)
#         xa = F.relu(self.linear2(xa_cat))
#         xa = F.relu(self.linear3(xa))
#         qval = self.linear4(xa)

#         return qval


# # neural network to approximate actor function
# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim):
#         super(Actor, self).__init__()

#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#         self.linear1 = nn.Linear(self.obs_dim, 512)
#         self.linear2 = nn.Linear(512, 128)
#         self.linear3 = nn.Linear(128, self.action_dim)

#     def forward(self, obs):
#         x = F.relu(self.linear1(obs))
#         x = F.relu(self.linear2(x))
#         # output of tanh is bounded between -1 and 1
#         # multiply by maximum action (here: 10N) in order to scale the action appropriately
#         x = torch.tanh(self.linear3(x))

#         return x


# class DDPGAgent:
#     def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate, train, decay):
#         self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#         print(self.device)
#         self.env = env
#         self.obs_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.shape[0]
#         # hyperparameters
#         self.env = env
#         self.gamma = gamma
#         self.tau = tau
#         self.train = train  # set to true if we want to train the agent, set to false to simulate agent
#         self.decay = decay

#         # initialize actor and critic networks
#         self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
#         self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)

#         self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
#         self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)

#         # Copy critic target parameters
#         for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#             target_param.data.copy_(param.data)

#         # optimizers
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

#         self.replay_buffer = BasicBuffer(buffer_maxlen)
#         # use exploration noise only during training
#         if self.train == True:
#             self.noise = OUNoise(self.env.action_space, decay_period=self.decay)

#     def get_action(self, obs, t=0):
#         state = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
#         action = self.actor.forward(state)
#         action = action.squeeze(0).cpu().detach().numpy()
#         # add exploration noise only during training
#         if self.train == True:
#             action = self.noise.get_action(action, t=t)

#         return action

#     def update(self, batch_size):
#         states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
#         state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
#         state_batch = torch.FloatTensor(state_batch).to(self.device)
#         action_batch = torch.FloatTensor(action_batch).to(self.device)
#         reward_batch = torch.FloatTensor(reward_batch).to(self.device)
#         next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
#         masks = torch.FloatTensor(masks).to(self.device)

#         curr_Q = self.critic.forward(state_batch, action_batch)
#         next_actions = self.actor_target.forward(next_state_batch)
#         next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
#         expected_Q = reward_batch + self.gamma * next_Q

#         # update critic
#         q_loss = F.mse_loss(curr_Q, expected_Q.detach())

#         self.critic_optimizer.zero_grad()
#         q_loss.backward()
#         self.critic_optimizer.step()

#         # update actor
#         policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()

#         self.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         self.actor_optimizer.step()

#         # update target networks
#         for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#             target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

#         for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#             target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor network for DA-AC with hybrid action space
class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action, min_std, max_std):
        super(Actor, self).__init__()
        # Shared layers
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # Discrete action output (logits for categorical distribution)
        self.discrete_logits = nn.Linear(256, discrete_action_dim)
        # Continuous parameter action outputs (mean and log_std for Gaussian distribution)
        self.mean_parameter = nn.Linear(256, parameter_action_dim)
        self.log_std_parameter = nn.Linear(256, parameter_action_dim)
        self.max_action = max_action
        self.discrete_action_dim = discrete_action_dim

        min_logstd = np.log(min_std * max_action)
        max_logstd = np.log(max_std * max_action)
        self.log_std_scale = (max_logstd - min_logstd) / 2.0
        self.log_std_shift = (max_logstd + min_logstd) / 2.0

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Discrete action logits
        discrete_logits = self.discrete_logits(a)
        discrete_probs = F.softmax(discrete_logits, dim=1)
        # Continuous parameter mean and log_std
        mean_parameter = torch.tanh(self.mean_parameter(a))
        log_std_parameter = torch.tanh(self.log_std_parameter(a))
        return discrete_probs, mean_parameter, log_std_parameter

    def scale_log_std(self, log_std_parameter):
        return log_std_parameter * self.log_std_scale + self.log_std_shift

    def to_delta(self, discrete_action_one_hot, parameter_action):
        mean_parameter = parameter_action / self.max_action
        log_std_parameter = torch.ones_like(mean_parameter) * -1
        return discrete_action_one_hot, mean_parameter, log_std_parameter


# Twin Q-network (Critic) for DA-AC
class Critic(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim):
        super(Critic, self).__init__()
        # Input concatenates state, one-hot discrete action, and continuous parameter action
        input_dim = state_dim + discrete_action_dim + 2 * parameter_action_dim
        
        # Q1 network
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 network
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, discrete_probs, mean_parameter, log_std_parameter):
        # Concatenate inputs
        sa = torch.cat([state, discrete_probs, mean_parameter, log_std_parameter], dim=1)

        # Q1 forward pass
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        # Q2 forward pass
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2

    def Q1(self, state, discrete_probs, mean_parameter, log_std_parameter):
        # Concatenate inputs
        sa = torch.cat([state, discrete_probs, mean_parameter, log_std_parameter], dim=1)
        
        # Q1 forward pass
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1


# DA-AC Agent
class DAAC(object):
    def __init__(
            self,
            state_dim,
            discrete_action_dim,
            parameter_action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_freq=2,
            min_std=0.05,
            max_std=0.2,
            interpolation=True,
            uniform_exploration_steps=0,  # Number of steps for uniform exploration
    ):
        self.discrete_action_dim = discrete_action_dim
        self.parameter_action_dim = parameter_action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.interpolation = interpolation
        self.uniform_exploration_steps = uniform_exploration_steps  # Uniform exploration threshold
        self.total_it = 0

        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action, min_std, max_std).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        if self.total_it < self.uniform_exploration_steps and not eval:
            logits = torch.randn(self.discrete_action_dim, device=device)
            discrete_probs = F.softmax(logits, dim=0)
            mean_parameter = 2 * torch.rand(self.parameter_action_dim, device=device) - 1
            log_std_parameter = 2 * torch.rand(self.parameter_action_dim, device=device) - 1
        else:
            discrete_probs, mean_parameter, log_std_parameter = self.actor(state)
        
        if eval:            
            # Greedy action selection
            discrete_action = torch.argmax(discrete_probs, dim=1).item()
            parameter_action = mean_parameter * self.max_action
            return discrete_action, parameter_action.cpu().data.numpy().flatten()

        # Sample actions for evaluation
        discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
        discrete_action = discrete_dist.sample().item()

        mean = mean_parameter * self.max_action
        log_std = self.actor.scale_log_std(log_std_parameter)
        normal = torch.distributions.Normal(0, log_std.exp())
        noise = normal.sample()
        parameter_action = mean + noise
        parameter_action = torch.clamp(parameter_action, -self.max_action, self.max_action)

        parameter_action = parameter_action.cpu().data.numpy().flatten()
        discrete_probs = discrete_probs.cpu().data.numpy().flatten()
        mean_parameter = mean_parameter.cpu().data.numpy().flatten()
        log_std_parameter = log_std_parameter.cpu().data.numpy().flatten()

        return discrete_action, parameter_action, discrete_probs, mean_parameter, log_std_parameter

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
    
        # Sample from replay buffer
        state, discrete_probs, mean_parameter, log_std_parameter, discrete_action_onehot, parameter_action, next_state, _, reward, not_done = replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            next_discrete_probs, next_mean_parameter, next_log_std_parameter = self.actor(next_state)

            if self.interpolation:
                discrete_probs2, mean_parameter2, log_std_parameter2 = self.actor.to_delta(discrete_action_onehot, parameter_action)
                random_weight = torch.rand(batch_size, 1, device=device)
                discrete_probs = (1 - random_weight) * discrete_probs + random_weight * discrete_probs2
                mean_parameter = (1 - random_weight) * mean_parameter + random_weight * mean_parameter2
                log_std_parameter = (1 - random_weight) * log_std_parameter + random_weight * log_std_parameter2

            target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_probs, next_mean_parameter, next_log_std_parameter)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Update critic
        current_Q1, current_Q2 = self.critic(state, discrete_probs, mean_parameter, log_std_parameter)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Update actor
            discrete_probs, mean_parameter, log_std_parameter = self.actor(state)
            actor_loss = -self.critic.Q1(state, discrete_probs, mean_parameter, log_std_parameter).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target critic network
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

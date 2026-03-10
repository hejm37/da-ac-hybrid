import numpy as np


def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * self.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class RingBuffer(object):
    """
    Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
    """
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  # unnecessary, not freeing any memory, could be slow


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    """
    Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
    """
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries-1, size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class Agent(object):
    """
    Defines a basic reinforcement learning agent for OpenAI Gym environments
    """

    NAME = "Abstract Agent"

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state):
        """
        Determines the action to take in the given state.

        :param state:
        :return:
        """
        raise NotImplementedError

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        """
        Performs a learning step given a (s,a,r,s',a') sample.

        :param state: previous observed state (s)
        :param action: action taken in previous state (a)
        :param reward: reward for the transition (r)
        :param next_state: the resulting observed state (s')
        :param next_action: action taken in next state (a')
        :param terminal: whether the episode is over
        :param time_steps: number of time steps the action took to execute (default=1)
        :return:
        """
        raise NotImplementedError

    def start_episode(self):
        """
        Perform any initialisation for the start of an episode.

        :return:
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Performs any cleanup before the next episode.

        :return:
        """
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc

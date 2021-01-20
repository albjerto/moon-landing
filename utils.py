import random
from collections import namedtuple
import torch
import gym
import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'new_state', 'done'))


def set_seed(seed, env):
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)


class ReplayMemory:
    def __init__(self, max_length, device):
        self.max_length = max_length
        self.experiences = []
        self.current_index = 0
        self.device = device

    def append(self, item):
        if len(self.experiences) < self.max_length:
            self.experiences.append(item)
        else:
            self.experiences[self.current_index % self.max_length] = item

        self.current_index += 1

    # code snippet taken, with some changes, from the function optimize_model()
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # explanation of the *zip operation: https://stackoverflow.com/a/19343/3343043
    def sample(self, batch_size):
        batch = Experience(*zip(*random.sample(self.experiences, batch_size)))

        states = torch.stack(batch.state).to(self.device)
        next_states = torch.stack(batch.new_state).to(self.device)
        actions = torch.LongTensor(batch.action).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).reshape(-1, 1).to(self.device)
        dones = torch.FloatTensor(batch.done).reshape(-1, 1).to(self.device)
        return states, next_states, actions, rewards, dones

    def can_sample(self, batch_size):
        return len(self.experiences) >= batch_size


class EnvWrapper:
    def __init__(self, env, device):
        self.env = None
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env

        self.state = None
        self.device = device
        self.unwrapped = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

    def process_state(self, state):
        return torch.tensor(state, dtype=torch.float, device=self.device)

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.process_state(next_state)
        self.state = next_state
        return next_state, reward, done

    def reset(self):
        self.state = self.process_state(self.env.reset())
        return self.state

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

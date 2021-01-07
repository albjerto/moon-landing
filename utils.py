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
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.times_pushed = 0

    def push(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.times_pushed % self.capacity] = item

        self.times_pushed += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size


class EnvWrapper:
    def __init__(self, env, device):
        self.env = None
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env

        self.state = None
        self.device = device

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

import torch
import random
from utils.common import Experience


# class taken, with minor changes, from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
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

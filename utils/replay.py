import torch
import random
from utils.common import Experience


# class taken, with minor changes, from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
class ReplayMemory:
    """ Class for experience replay """
    def __init__(self, max_length, device):
        self.max_length = max_length
        self.experiences = []
        self.current_index = 0
        self.device = device

    def append(self, item):
        """
        Append an Experience tuple to the buffer.
        If the buffer's length is equal to the maximum length,
        replaces the oldest Experience instead.
        :param item: Experience tuple
        """
        if len(self.experiences) < self.max_length:
            self.experiences.append(item)
        else:
            self.experiences[self.current_index % self.max_length] = item

        self.current_index += 1

    # code snippet taken, with some changes, from the function optimize_model()
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # explanation of the *zip operation: https://stackoverflow.com/a/19343/3343043
    def sample(self, batch_size):
        """
        Samples a batch of uniformly random Experiences.
        :param batch_size: size of the batch to be drawn
        :return: the batch
        """
        batch = Experience(*zip(*random.sample(self.experiences, batch_size)))

        states = torch.stack(batch.state).to(self.device)
        next_states = torch.stack(batch.new_state).to(self.device)
        actions = torch.LongTensor(batch.action).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).reshape(-1, 1).to(self.device)
        dones = torch.FloatTensor(batch.done).reshape(-1, 1).to(self.device)
        return states, next_states, actions, rewards, dones

    def can_sample(self, batch_size):
        """
        Check if the length of the buffer is enough to provide a sample.
        :param batch_size: size of the batch to be drawn
        :return: True if there the number of
            Experiences tuple is greater or equal batch_size
        """
        return len(self.experiences) >= batch_size

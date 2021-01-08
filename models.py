import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, linear1_units=64, linear2_units=64):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input_dim, linear1_units)
        self.linear2 = nn.Linear(linear1_units, linear2_units)
        self.out = nn.Linear(linear2_units, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.out(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, linear_units=64, advantage_units=32, value_units=32):
        super(DuelingDQN, self).__init__()

        self.linear = nn.Linear(input_dim, linear_units)

        self.advantage1 = nn.Linear(linear_units, advantage_units)
        self.advantage2 = nn.Linear(advantage_units, output_dim)

        self.value1 = nn.Linear(linear_units, value_units)
        self.value2 = nn.Linear(value_units, 1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        advantage = self.advantage2(F.relu(self.advantage1(x)))
        value = self.value2(F.relu(self.value1(x)))

        return value + advantage - advantage.mean()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

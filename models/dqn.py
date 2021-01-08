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

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Basic fully-connected neural network.
    """
    def __init__(self, input_dim, output_dim, linear1_units=64, linear2_units=64):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input_dim, linear1_units)
        self.linear2 = nn.Linear(linear1_units, linear2_units)
        self.out = nn.Linear(linear2_units, output_dim)

    def forward(self, x):
        """
        Perform a pass through the network.
        :param x: current state of the environment
        :return: predicted q-values for each action
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.out(x)

    def save(self, filename):
        """
        Save the network weights to file.
        :param filename: path
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename, device):
        """
        Loads the network weights from file.
        :param filename: path of the weight file
        :param device: CPU or GPU
        """
        self.load_state_dict(torch.load(filename, map_location=device))


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    """
    def __init__(self, input_dim, output_dim, linear_units=128, advantage_units=128, value_units=128):
        super(DuelingDQN, self).__init__()

        self.name = "DuelingDQN"
        self.linear = nn.Sequential(
            nn.Linear(input_dim, linear_units),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(linear_units, advantage_units),
            nn.ReLU(),
            nn.Linear(advantage_units, output_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(linear_units, value_units),
            nn.ReLU(),
            nn.Linear(value_units, 1)
        )
        #
        # self.advantage = nn.Sequential(
        #     nn.Linear(linear_units, advantage_units),
        #     nn.ReLU(),
        #     nn.Linear(advantage_units, output_dim)
        # )
        #
        # self.value = nn.Sequential(
        #     nn.Linear(linear_units, value_units),
        #     nn.ReLU(),
        #     nn.Linear(value_units, 1)
        # )

    def forward(self, x):
        """
        Perform a pass through the network.
        :param x: current state of the environment
        :return: predicted q-values for each action
        """
        x = self.linear(x)
        advantage = self.advantage(x)
        value = self.value(x)

        return value + (advantage - advantage.mean())

    def save(self, filename):
        """
        Save the network weights to file.
        :param filename: path
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename, device):
        """
        Loads the network weights from file.
        :param filename: path of the weight file
        :param device: CPU or GPU
        """
        self.load_state_dict(torch.load(filename, map_location=device))

import random
from collections import namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'new_state', 'done'))


def set_seed(seed, env):
    """
    Sets the random seed in the environment, pytorch, numpy and random.
    :param seed:
    :param env:
    :return:
    """
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

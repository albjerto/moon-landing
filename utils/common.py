import random
from collections import namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'new_state', 'done'))


def set_seed(seed, env):
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

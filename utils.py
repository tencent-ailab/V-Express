import random

import torch
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)

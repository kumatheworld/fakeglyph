import os
import random

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: int) -> None:
    """
    Not entirely sure if this works properly.
    See https://pytorch.org/docs/stable/notes/randomness.html
    and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    for details.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False

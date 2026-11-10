"""Deterministic random seed setup for Assignment 6."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix RNG seeds across Python, NumPy, and PyTorch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

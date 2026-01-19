"""
Reproducibility Utilities for TC-MMWM
-------------------------------------
Provides functions to set random seeds and deterministic computation
for consistent experiments across PyTorch, NumPy, and Python.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Global seed set to {seed}")


def get_device(cuda_preference=True):
    """
    Returns the torch device to use for training or inference.

    Args:
        cuda_preference (bool): If True, prefers CUDA if available.

    Returns:
        torch.device: Device object
    """
    if cuda_preference and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Reproducibility] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[Reproducibility] Using CPU")
    return device


def enable_deterministic_operations():
    """
    Ensures that all CUDA operations are deterministic.
    This is recommended for reproducible training results.
    """
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("[Reproducibility] Deterministic CUDA operations enabled.")


def reproducible_setup(seed: int = 42, cuda_preference=True):
    """
    Sets up the environment for fully reproducible experiments.

    Args:
        seed (int): Random seed
        cuda_preference (bool): Use GPU if available
    Returns:
        torch.device: The selected device
    """
    set_seed(seed)
    enable_deterministic_operations()
    device = get_device(cuda_preference)
    return device


# Example usage
if __name__ == "__main__":
    device = reproducible_setup(seed=123)
    print(f"Experiment running on device: {device}")

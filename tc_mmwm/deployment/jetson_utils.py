"""
Jetson-Specific Utilities for TC-MMWM Deployment
------------------------------------------------
Provides helper functions to optimize inference on NVIDIA Jetson platforms,
including device selection, mixed-precision execution, and memory management.
"""

import torch
import jetson.inference
import jetson.utils

def select_device(preferred="cuda"):
    """
    Selects the device for inference based on availability.

    Args:
        preferred (str): Preferred device ("cuda" or "cpu")

    Returns:
        device (torch.device)
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Jetson Utils] Using device: {device}")
    return device

def enable_mixed_precision(model):
    """
    Converts model to mixed precision for reduced memory usage and faster inference
    on Jetson GPUs supporting FP16.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        model_fp16 (torch.nn.Module)
    """
    model.half()  # Convert to FP16
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()  # Keep BatchNorm in FP32
    print("[Jetson Utils] Mixed precision enabled (FP16)")
    return model

def clear_cuda_cache():
    """
    Clears GPU cache to reduce memory fragmentation
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[Jetson Utils] CUDA cache cleared")

def get_available_memory():
    """
    Returns total and allocated memory on GPU

    Returns:
        total_mem (float): Total GPU memory in MB
        allocated_mem (float): Allocated GPU memory in MB
    """
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
        allocated_mem = torch.cuda.memory_allocated(0) / 1e6
        print(f"[Jetson Utils] Total GPU memory: {total_mem:.1f} MB")
        print(f"[Jetson Utils] Allocated GPU memory: {allocated_mem:.1f} MB")
        return total_mem, allocated_mem
    else:
        print("[Jetson Utils] CUDA not available, returning 0 memory")
        return 0.0, 0.0

if __name__ == "__main__":
    # Example usage
    device = select_device("cuda")
    import torch.nn as nn
    dummy_model = nn.Linear(128, 128).to(device)
    dummy_model = enable_mixed_precision(dummy_model)
    clear_cuda_cache()
    get_available_memory()

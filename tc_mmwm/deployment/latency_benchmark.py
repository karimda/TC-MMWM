"""
Latency Benchmarking for TC-MMWM Deployment
-------------------------------------------
Measures end-to-end inference time per control cycle including:
- Multimodal perception encoding (vision, language, sensors)
- Causal latent state update
- Counterfactual action evaluation
- Action selection

Supports benchmarking on both workstation GPUs and embedded Jetson platforms.
"""

import time
import torch
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.deployment.jetson_utils import select_device, enable_mixed_precision, clear_cuda_cache

def benchmark_latency(model, input_samples, device=None, num_iterations=100):
    """
    Measures average latency of TC-MMWM forward pass over a batch of input samples.

    Args:
        model (torch.nn.Module): TC-MMWM model
        input_samples (dict): Dictionary containing:
            - vision: Tensor [B, C, H, W]
            - language: Tensor [B, L]
            - sensors: Tensor [B, S]
        device (torch.device): Device for inference
        num_iterations (int): Number of iterations for averaging

    Returns:
        avg_latency_ms (float): Average latency in milliseconds
    """
    model.eval()
    if device is None:
        device = select_device("cuda")
    model.to(device)
    model = enable_mixed_precision(model)

    # Move inputs to device
    vision = input_samples["vision"].to(device)
    language = input_samples["language"].to(device)
    sensors = input_samples["sensors"].to(device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(vision, language, sensors)

    # Benchmarking
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(vision, language, sensors)
    end_time = time.time()

    avg_latency_ms = (end_time - start_time) / num_iterations * 1000
    print(f"[Latency Benchmark] Average inference latency: {avg_latency_ms:.2f} ms")
    return avg_latency_ms

if __name__ == "__main__":
    # Example usage with dummy input
    device = select_device("cuda")

    # Dummy TC-MMWM model
    model = TC_MMWM(latent_dim=128, vision_dim=64, language_dim=32, sensor_dim=16)

    # Dummy input
    input_samples = {
        "vision": torch.randn(1, 3, 224, 224),
        "language": torch.randint(0, 1000, (1, 20)),
        "sensors": torch.randn(1, 16)
    }

    benchmark_latency(model, input_samples, device=device)
    clear_cuda_cache()

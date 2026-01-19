"""
Benchmark Real-Time Inference Latency for TC-MMWM
-------------------------------------------------
This script measures end-to-end inference latency including:
- Multimodal perception
- Latent causal state update
- Counterfactual action evaluation
- Policy execution

Supports multiple hardware targets: workstation GPU, Jetson AGX Orin, Jetson Xavier NX
"""

import time
import torch
import argparse
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.preprocessing.build_dataset import MultiModalDataset
from tc_mmwm.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TC-MMWM Inference Latency")
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device for benchmarking')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to measure')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    return parser.parse_args()


def benchmark(model, dataloader, device, num_batches=100):
    model.eval()
    latencies = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            start_time = time.time()
            _ = model(batch)  # Forward pass including causal state & counterfactual reasoning
            end_time = time.time()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency


def main():
    args = parse_args()
    logger = setup_logger()
    logger.info(f"Benchmarking TC-MMWM on {args.device}")

    # Load dataset
    dataset = MultiModalDataset(config_path=args.config, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = TC_MMWM(config_path=args.config).to(args.device)
    if args.checkpoint:
        from tc_mmwm.utils.checkpointing import load_checkpoint
        load_checkpoint(model, args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided, benchmarking untrained model.")

    # Run benchmark
    avg_latency = benchmark(model, dataloader, device=args.device, num_batches=args.num_batches)
    logger.info(f"Average Inference Latency: {avg_latency:.2f} ms per batch (batch size={args.batch_size})")
    print(f"Average Inference Latency: {avg_latency:.2f} ms per batch (batch size={args.batch_size})")


if __name__ == "__main__":
    main()

"""
Evaluate Script for TC-MMWM
---------------------------
Evaluates the Temporal Causal Multimodal World Model (TC-MMWM)
across multiple metrics including:
- Task success rate
- Long-horizon stability
- Out-of-distribution (OOD) generalization
- Language constraint compliance
- Robustness to noise
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.evaluation.metrics import (
    task_success_rate,
    long_horizon_stability,
    ood_generalization,
    language_compliance,
    robustness_score
)
from tc_mmwm.preprocessing.build_dataset import MultiModalDataset
from tc_mmwm.utils.logging import setup_logger
from tc_mmwm.utils.checkpointing import latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TC-MMWM")
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml',
                        help='Path to evaluation configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting TC-MMWM evaluation...")

    # Load dataset
    dataset = MultiModalDataset(config_path=args.config, split=args.split)
    dataloader = DataLoader(dataset, batch_size=dataset.config.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = TC_MMWM(config_path=args.config).to(args.device)
    
    # Load checkpoint
    if args.checkpoint is None:
        ckpt_path = latest_checkpoint()
    else:
        ckpt_path = args.checkpoint
    
    if ckpt_path is None:
        logger.error("No checkpoint found for evaluation.")
        return
    
    load_checkpoint(model, ckpt_path)
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    
    # Evaluation metrics
    success = task_success_rate(model, dataloader, device=args.device)
    long_horizon = long_horizon_stability(model, dataloader, device=args.device)
    ood = ood_generalization(model, dataloader, device=args.device)
    language = language_compliance(model, dataloader, device=args.device)
    robustness = robustness_score(model, dataloader, device=args.device)
    
    # Print results
    logger.info("===== TC-MMWM Evaluation Results =====")
    logger.info(f"Task Success Rate       : {success:.2f}%")
    logger.info(f"Long-Horizon Stability  : {long_horizon:.2f}%")
    logger.info(f"OOD Generalization      : {ood:.2f}%")
    logger.info(f"Language Compliance     : {language:.2f}%")
    logger.info(f"Robustness to Noise     : {robustness:.2f}%")
    logger.info("======================================")
    
    print("\n===== TC-MMWM Evaluation Results =====")
    print(f"Task Success Rate       : {success:.2f}%")
    print(f"Long-Horizon Stability  : {long_horizon:.2f}%")
    print(f"OOD Generalization      : {ood:.2f}%")
    print(f"Language Compliance     : {language:.2f}%")
    print(f"Robustness to Noise     : {robustness:.2f}%")
    print("======================================")


if __name__ == "__main__":
    main()

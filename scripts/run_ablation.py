"""
Run Ablation Studies for TC-MMWM
--------------------------------
This script evaluates the impact of removing specific components of TC-MMWM:
- Language modality
- Vision modality
- Sensor modality
- Causal composition operator
- Counterfactual reasoning module

Each ablation configuration is defined in configs/ablation/*.yaml
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.preprocessing.build_dataset import MultiModalDataset
from tc_mmwm.evaluation.metrics import task_success_rate
from tc_mmwm.utils.logging import setup_logger
from tc_mmwm.utils.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation Studies for TC-MMWM")
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['no_language', 'no_vision', 'no_sensors', 'no_causal_operator', 'no_counterfactual'],
                        help='Which ablation study to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = setup_logger()
    logger.info(f"Running ablation study: {args.ablation}")

    # Map ablation choice to config path
    ablation_config_map = {
        'no_language': 'configs/ablation/no_language.yaml',
        'no_vision': 'configs/ablation/no_vision.yaml',
        'no_sensors': 'configs/ablation/no_sensors.yaml',
        'no_causal_operator': 'configs/ablation/no_causal_operator.yaml',
        'no_counterfactual': 'configs/ablation/no_counterfactual.yaml'
    }

    config_path = ablation_config_map[args.ablation]

    # Load dataset
    dataset = MultiModalDataset(config_path=config_path, split=args.split)
    dataloader = DataLoader(dataset, batch_size=dataset.config.batch_size, shuffle=False, num_workers=4)

    # Initialize model with ablation config
    model = TC_MMWM(config_path=config_path).to(args.device)

    # Load checkpoint
    if args.checkpoint is None:
        logger.warning("No checkpoint provided, evaluating untrained model.")
    else:
        load_checkpoint(model, args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate task success rate
    success_rate = task_success_rate(model, dataloader, device=args.device)

    logger.info(f"Ablation [{args.ablation}] - Task Success Rate: {success_rate:.2f}%")
    print(f"Ablation [{args.ablation}] - Task Success Rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()

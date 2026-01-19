"""
Train Script for TC-MMWM
------------------------
This script trains the Temporal Causal Multimodal World Model (TC-MMWM)
on either simulated or real robotic datasets. Includes support for
interventional multimodal training, constraint enforcement, and
checkpointing.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.training.trainer import Trainer
from tc_mmwm.preprocessing.build_dataset import MultiModalDataset
from tc_mmwm.utils.checkpointing import save_checkpoint, latest_checkpoint
from tc_mmwm.utils.logging import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train TC-MMWM")
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir=args.save_dir)
    logger.info("Starting TC-MMWM training...")
    
    # Load dataset
    dataset = MultiModalDataset(config_path=args.config, split='train')
    dataloader = DataLoader(dataset, batch_size=dataset.config.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = TC_MMWM(config_path=args.config).to(args.device)
    
    # Initialize trainer
    trainer = Trainer(model=model, dataloader=dataloader, config_path=args.config, device=args.device, logger=logger)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = latest_checkpoint(args.save_dir)
    
    if ckpt_path is not None:
        model, trainer.optimizer, start_epoch, _ = trainer.load_checkpoint(ckpt_path)
        logger.info(f"Resumed training from checkpoint {ckpt_path}, starting at epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, trainer.config.num_epochs):
        train_loss = trainer.train_one_epoch(epoch)
        logger.info(f"Epoch [{epoch+1}/{trainer.config.num_epochs}] - Training Loss: {train_loss:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % trainer.config.save_every == 0 or (epoch + 1) == trainer.config.num_epochs:
            ckpt_file = os.path.join(args.save_dir, f"tc_mmwm_epoch_{epoch+1}.pt")
            save_checkpoint(model, trainer.optimizer, epoch+1, train_loss, ckpt_file)
    
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()

"""
Trainer for TC-MMWM
-------------------
Handles training loop for the Temporal Causal Multimodal World Model,
including multi-term loss computation, backpropagation, optimizer steps,
learning rate scheduling, and checkpointing.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tc_mmwm.tc_mmwm import TC_MMWM
from tc_mmwm.losses.total_loss import TotalLoss
from tc_mmwm.utils.checkpointing import save_checkpoint, load_checkpoint
from tc_mmwm.utils.logging import Logger

class Trainer:
    """
    Trainer class for TC-MMWM
    """

    def __init__(self, model: TC_MMWM,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 loss_weights=(1.0, 1.0, 1.0),
                 log_dir='logs/',
                 checkpoint_dir='checkpoints/',
                 max_epochs=100):
        """
        Args:
            model: TC-MMWM model instance
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
            optimizer: torch.optim optimizer
            scheduler: learning rate scheduler (optional)
            device: 'cuda' or 'cpu'
            loss_weights: tuple of weights (alpha, beta, gamma)
            log_dir: directory to store training logs
            checkpoint_dir: directory to save model checkpoints
            max_epochs: number of training epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs

        # Initialize multi-term loss
        alpha, beta, gamma = loss_weights
        self.criterion = TotalLoss(alpha=alpha, beta=beta, gamma=gamma)

        # Logging and checkpointing
        self.logger = Logger(log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss_epoch = 0.0
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            # Move batch to device
            for k in batch:
                batch[k] = batch[k].to(self.device)

            # Forward pass
            z_pred = self.model(batch['observations'], batch['actions'])
            z_target = batch['latent_target']
            z_next_pred = self.model.predict_next(batch['observations'], batch['actions'])
            z_next_obs = batch['latent_next']
            action_pred = batch['predicted_actions']
            action_executed = batch['actions']
            constraint_mask = batch['constraint_mask']

            # Compute total loss
            total_loss, loss_dict = self.criterion(
                z_pred, z_target,
                z_next_pred, z_next_obs,
                action_pred, action_executed,
                constraint_mask
            )

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss_epoch += total_loss.item()
            self.logger.log_step(loss_dict)

        avg_loss = total_loss_epoch / len(self.train_loader)
        self.logger.log_epoch(epoch, avg_loss, phase='train')
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss_epoch = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                # Move batch to device
                for k in batch:
                    batch[k] = batch[k].to(self.device)

                # Forward pass
                z_pred = self.model(batch['observations'], batch['actions'])
                z_target = batch['latent_target']
                z_next_pred = self.model.predict_next(batch['observations'], batch['actions'])
                z_next_obs = batch['latent_next']
                action_pred = batch['predicted_actions']
                action_executed = batch['actions']
                constraint_mask = batch['constraint_mask']

                total_loss, loss_dict = self.criterion(
                    z_pred, z_target,
                    z_next_pred, z_next_obs,
                    action_pred, action_executed,
                    constraint_mask
                )
                total_loss_epoch += total_loss.item()
                self.logger.log_step(loss_dict, phase='val')

        avg_loss = total_loss_epoch / len(self.val_loader)
        self.logger.log_epoch(epoch, avg_loss, phase='val')
        return avg_loss

    def train(self):
        """
        Full training loop over all epochs, including validation and checkpointing.
        """
        best_val_loss = float('inf')
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, self.checkpoint_dir, 'best_model.pt')

            # Regular checkpoint every epoch
            save_checkpoint(self.model, self.optimizer, epoch, self.checkpoint_dir, f'epoch_{epoch}.pt')

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

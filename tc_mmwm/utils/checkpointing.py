"""
Checkpointing Utilities for TC-MMWM
-----------------------------------
Provides functions to save and load model states, optimizer states,
and training metadata to enable resuming experiments and reproducibility.
"""

import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, path, extra=None):
    """
    Save a checkpoint of the model, optimizer, and training metadata.

    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current training epoch
        loss (float): Latest training loss
        path (str): Path to save the checkpoint file
        extra (dict, optional): Additional metadata to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    if extra is not None:
        checkpoint.update(extra)
    
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved checkpoint at {path}")


def load_checkpoint(model, optimizer=None, path=None, device='cpu'):
    """
    Load a checkpoint and restore model and optimizer states.

    Args:
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore
        path (str): Path to checkpoint file
        device (str or torch.device): Device to

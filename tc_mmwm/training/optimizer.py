"""
Optimizer Setup for TC-MMWM
---------------------------
Defines helper functions to create optimizers for the TC-MMWM model.
Supports Adam and AdamW with optional weight decay.
"""

import torch

def build_optimizer(model, optimizer_type='adam', lr=1e-3, weight_decay=0.0):
    """
    Creates an optimizer for the TC-MMWM model.

    Args:
        model: The TC-MMWM model instance
        optimizer_type: 'adam' or 'adamw'
        lr: Learning rate
        weight_decay: Weight decay for regularization

    Returns:
        optimizer: torch.optim optimizer
    """
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

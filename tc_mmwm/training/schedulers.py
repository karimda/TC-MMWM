"""
Learning Rate Schedulers for TC-MMWM
------------------------------------
Provides functions to create learning rate schedulers compatible with PyTorch optimizers.
Supports step decay, cosine annealing, and exponential decay.
"""

import torch

def build_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Creates a learning rate scheduler for a given optimizer.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ['step', 'cosine', 'exponential']
        **kwargs: Additional arguments for the scheduler

    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'step':
        # StepLR reduces LR by gamma every step_size epochs
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'cosine':
        # CosineAnnealingLR schedules LR following a cosine curve
        T_max = kwargs.get('T_max', 50)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == 'exponential':
        # ExponentialLR decays LR exponentially
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler

"""
State Prediction Loss
---------------------
Encourages accurate prediction of the latent state evolution over time.
Corresponds to Equation (13) in the manuscript:

Loss_state = || z_pred - z_true ||^2

Where:
- z_pred: predicted latent state at timestep t
- z_true: ground-truth latent state at timestep t
"""

import torch
import torch.nn as nn

class StatePredictionLoss(nn.Module):
    """
    L2 loss between predicted and true latent states.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        """
        Compute state prediction loss.

        Args:
            z_pred: Predicted latent states (B, latent_dim)
            z_true: Ground-truth latent states (B, latent_dim)

        Returns:
            loss: scalar tensor
        """
        loss = self.loss_fn(z_pred, z_true)
        return loss

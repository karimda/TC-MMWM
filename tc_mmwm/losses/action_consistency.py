"""
Action-Conditioned Consistency Loss
-----------------------------------
Ensures that the effect of executed actions on the latent state aligns
with the model's predictions. Corresponds to Equation (14):

Loss_action = || f(z_{t-1}, a_{t-1}) - z_pred_t ||^2

Where:
- f(z_{t-1}, a_{t-1}): predicted next latent state from previous state and action
- z_pred_t: actual predicted latent state at timestep t
"""

import torch
import torch.nn as nn

class ActionConsistencyLoss(nn.Module):
    """
    L2 loss between predicted next state given action and predicted latent state.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, z_next_pred: torch.Tensor, z_action_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute action-conditioned consistency loss.

        Args:
            z_next_pred: Predicted latent state at timestep t (B, latent_dim)
            z_action_pred: Latent state predicted by applying action on z_{t-1} (B, latent_dim)

        Returns:
            loss: scalar tensor
        """
        loss = self.loss_fn(z_next_pred, z_action_pred)
        return loss

"""
Total Loss for TC-MMWM
----------------------
Combines the three main loss components:
1. State Prediction Loss
2. Action-Conditioned Consistency Loss
3. Constraint Violation Loss

Total Loss = alpha * L_state + beta * L_action + gamma * L_constraint
"""

import torch
import torch.nn as nn
from .state_prediction import StatePredictionLoss
from .action_consistency import ActionConsistencyLoss
from .constraint_violation import ConstraintViolationLoss

class TotalLoss(nn.Module):
    """
    Computes the total loss for TC-MMWM as a weighted sum of individual losses.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Args:
            alpha: Weight for state prediction loss
            beta: Weight for action consistency loss
            gamma: Weight for constraint violation loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Initialize individual losses
        self.state_loss = StatePredictionLoss()
        self.action_loss = ActionConsistencyLoss()
        self.constraint_loss = ConstraintViolationLoss()

    def forward(self, z_pred, z_target, z_next_pred, z_next_obs, action_pred, action_executed, constraint_mask):
        """
        Args:
            z_pred: Predicted latent states for state prediction loss (B, T, latent_dim)
            z_target: Ground-truth latent states for state prediction loss (B, T, latent_dim)
            z_next_pred: Predicted next latent states after actions (B, T, latent_dim)
            z_next_obs: Observed next latent states (B, T, latent_dim)
            action_pred: Predicted actions (B, T, action_dim)
            action_executed: Executed actions (B, T, action_dim)
            constraint_mask: Binary mask of constraint violations (B, T)

        Returns:
            total_loss: Weighted sum of the three losses
            loss_dict: Dictionary of individual loss values for logging
        """
        # Compute individual losses
        L_state = self.state_loss(z_pred, z_target)
        L_action = self.action_loss(z_next_pred, z_next_obs, action_pred, action_executed)
        L_constraint = self.constraint_loss(z_pred, constraint_mask)

        # Weighted sum
        total_loss = self.alpha * L_state + self.beta * L_action + self.gamma * L_constraint

        # Return total loss and components for logging
        loss_dict = {
            "state_loss": L_state.item(),
            "action_loss": L_action.item(),
            "constraint_loss": L_constraint.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, loss_dict

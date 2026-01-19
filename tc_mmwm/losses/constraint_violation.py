"""
Constraint Violation Loss
-------------------------
Penalizes latent state transitions that violate language or physical constraints.
Corresponds to Equation (15):

Loss_constraint = sum_t I[z_t violates constraints]

Where:
- I[.] is an indicator function that flags states incompatible with language or feasibility constraints
- z_t is the latent state at timestep t
"""

import torch
import torch.nn as nn

class ConstraintViolationLoss(nn.Module):
    """
    Computes the loss for violating language or physical constraints in latent states.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint violation loss.

        Args:
            z: Latent states over time (B, T, latent_dim)
            mask: Binary mask indicating constraint violations (B, T)
                  1 for violation, 0 for compliant

        Returns:
            loss: scalar tensor representing average violation penalty
        """
        # Ensure mask is float for multiplication
        mask = mask.float()

        # Sum violations across time and batch, then average
        loss = torch.sum(mask) / (z.size(0) * z.size(1) + 1e-8)
        return loss

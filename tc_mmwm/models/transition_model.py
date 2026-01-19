"""
Language-Conditioned Transition Module for TC-MMWM

Models latent state evolution constrained by language instructions
and physical feasibility.

Corresponds to:
- Section 2.2.4 Language-Conditioned Transition Module
- Equation (7): z_{t+1} = f(z_t, a_t | l_t)
- Equation (8): Constraint enforcement via masking/scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageConditionedTransition(nn.Module):
    """
    Implements a latent state transition function conditioned on:
      - previous latent state z_t
      - executed action a_t
      - language instruction embedding l_t
    """

    def __init__(self, latent_dim: int = 128, action_dim: int = 12, hidden_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Concatenated input: [z_t, a_t, l_t]
        self.transition_network = nn.Sequential(
            nn.Linear(latent_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Learned gating for constraint enforcement
        self.constraint_gate = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()  # scales transitions between 0 (blocked) and 1 (allowed)
        )

        self._initialize_weights()

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, l_t: torch.Tensor):
        """
        Args:
            z_t: previous latent state (B, latent_dim)
            a_t: action vector (B, action_dim)
            l_t: language instruction embedding (B, latent_dim)

        Returns:
            z_next: next latent state (B, latent_dim)
        """

        # Predict unconstrained next state
        x = torch.cat([z_t, a_t, l_t], dim=1)
        z_pred = self.transition_network(x)  # (B, latent_dim)

        # Constraint enforcement
        # Computes gating mask based on z_t and language l_t
        gate_input = torch.cat([z_t, l_t], dim=1)
        mask = self.constraint_gate(gate_input)  # (B, latent_dim)

        # Apply mask to predicted latent state
        z_next = z_t + mask * z_pred

        return z_next, mask

    def _initialize_weights(self):
        """
        Xavier initialization for stable training
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

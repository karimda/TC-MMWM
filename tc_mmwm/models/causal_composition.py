"""
Causal Composition Operator for TC-MMWM

Implements context-dependent causal aggregation of modality-specific latent
contributions. Unlike standard feature fusion, this module explicitly models
the causal influence of each modality on latent state evolution.

Corresponds to:
- Section 2.2.3 Causal Composition Operator
- Equation (5): Latent state update
- Equation (6): Context-dependent causal weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalCompositionOperator(nn.Module):
    """
    Causal composition operator.

    Given modality-specific latent contributions:
        z_v (vision), z_l (language), z_s (sensors)

    Produces:
        - aggregated causal latent state z_t
        - interpretable causal weights alpha
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_modalities: int = 3,
        temperature: float = 1.0
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        self.temperature = temperature

        # Learned scoring functions f_m(.) for each modality (Eq. 6)
        self.scorers = nn.ModuleList([
            nn.Linear(latent_dim, 1) for _ in range(num_modalities)
        ])

        self._initialize_weights()

    def forward(self, modality_latents: list):
        """
        Forward pass of causal composition operator.

        Args:
            modality_latents:
                List of modality-specific latent tensors:
                [z_v, z_l, z_s], each of shape (B, latent_dim)

        Returns:
            z_t:
                Aggregated causal latent state (B, latent_dim)

            alpha:
                Context-dependent causal weights (B, num_modalities)
        """

        assert len(modality_latents) == self.num_modalities, \
            "Number of modality latents must match num_modalities"

        # Stack modality latents: (B, M, D)
        z_stack = torch.stack(modality_latents, dim=1)

        # Compute unnormalized causal scores e_m = f_m(z_m)
        # Corresponds to Equation (6)
        scores = []
        for i, scorer in enumerate(self.scorers):
            score = scorer(modality_latents[i])  # (B, 1)
            scores.append(score)

        scores = torch.cat(scores, dim=1)  # (B, M)

        # Context-dependent causal weights alpha_m
        alpha = F.softmax(scores / self.temperature, dim=1)  # (B, M)

        # Weighted causal aggregation (Equation 5)
        alpha_expanded = alpha.unsqueeze(-1)  # (B, M, 1)
        z_t = torch.sum(alpha_expanded * z_stack, dim=1)  # (B, D)

        return z_t, alpha

    def _initialize_weights(self):
        """
        Xavier initialization ensures stable causal weighting
        and prevents early modality collapse.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

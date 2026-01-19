"""
Sensor Encoder for TC-MMWM

Encodes proprioceptive and tactile sensor readings into a causal latent embedding.
This modality captures physical interaction dynamics such as contact, force,
joint motion, and stability, which are critical for manipulation tasks.

Corresponds to:
- Section 2.2 (Multimodal Causal State Space)
- Section 3.7 (Interpretability and Modality Contributions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEncoder(nn.Module):
    """
    Sensor encoder for proprioceptive and tactile inputs.

    Input:
        - Sensor readings of shape (B, D_s)

    Output:
        - Sensor latent embedding z_s of shape (B, D_latent)
    """

    def __init__(
        self,
        sensor_dim: int = 32,
        latent_dim: int = 128,
        hidden_dims=(128, 256),
        dropout: float = 0.1
    ):
        super().__init__()

        self.sensor_dim = sensor_dim
        self.latent_dim = latent_dim

        layers = []
        input_dim = sensor_dim

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h

        layers.append(nn.Linear(input_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, sensor_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sensor encoder.

        Args:
            sensor_inputs:
                Tensor of shape (B, sensor_dim)
                containing normalized proprioceptive and tactile signals.

        Returns:
            z_s: Sensor latent representation (B, latent_dim)
        """
        z_s = self.mlp(sensor_inputs)
        return z_s

    def _initialize_weights(self):
        """
        Xavier initialization ensures smooth gradient flow and
        stable multimodal fusion.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

"""
Vision Encoder for TC-MMWM

Encodes visual observations into a compact causal latent representation.
Designed to extract task-relevant visual factors while discarding spurious correlations.

This module corresponds to:
- Section 2.2 (Multimodal Encoders)
- Visual pathway in Figure 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Vision encoder using a convolutional backbone followed by a causal projection head.

    Input:
        - RGB image tensor of shape (B, 3, H, W)

    Output:
        - Visual latent embedding z_v of shape (B, D_v)
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        image_size: int = 128,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Convolutional feature extractor
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Compute feature map size after convolutions
        conv_output_size = image_size // (2 ** 4)
        flattened_dim = 256 * conv_output_size * conv_output_size

        # Causal projection head
        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self._initialize_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision encoder.

        Args:
            images: Tensor of shape (B, 3, H, W), normalized to [0, 1]

        Returns:
            z_v: Visual latent representation (B, latent_dim)
        """
        features = self.conv_net(images)
        features = features.view(features.size(0), -1)
        z_v = self.fc(features)

        return z_v

    def _initialize_weights(self):
        """
        Xavier initialization for stable causal representation learning.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

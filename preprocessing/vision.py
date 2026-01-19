"""
Vision preprocessing and encoding module for TC-MMWM.

This file implements:
- Visual preprocessing (resize, normalization)
- Optional data augmentation hooks
- Vision encoder wrapper producing Δz_t^(vision)

Aligned with:
Section 2.2.2.1 Vision-Induced State Change
Equation (2): Δz_t^(vision) = g_v(v_t, z_{t-1})
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Tuple, Optional


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

class VisionPreprocessor:
    """
    Preprocess raw RGB images before encoding.

    Input:
        - Raw image: uint8 tensor or PIL image
        - Shape before: (H, W, 3)

    Output:
        - Normalized tensor: (3, 224, 224)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ):
        self.augment = augment

        base_transforms = [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        if augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = T.Compose(base_transforms)

    def __call__(self, image):
        """
        Apply preprocessing to a single image.
        """
        return self.transform(image)


# ------------------------------------------------------------
# Vision Encoder
# ------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    Vision encoder producing vision-induced causal contribution Δz_t^(vision).

    Implements g_v(v_t, z_{t-1}) from Eq. (2).

    Supports:
        - CNN backbone (ResNet-style)
        - Transformer backbone (ViT-style)

    Output:
        - Δz_v ∈ R^{latent_dim}
    """

    def __init__(
        self,
        backbone: str = "cnn",
        latent_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_type = backbone
        self.latent_dim = latent_dim

        if backbone == "cnn":
            self.backbone = self._build_cnn_backbone()
            backbone_out_dim = 512

        elif backbone == "vit":
            self.backbone = self._build_vit_backbone()
            backbone_out_dim = 768

        else:
            raise ValueError(f"Unsupported vision backbone: {backbone}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Projection to latent causal space
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # Optional conditioning on previous latent state z_{t-1}
        self.state_condition = nn.Linear(latent_dim, latent_dim)

    def _build_cnn_backbone(self) -> nn.Module:
        """
        Lightweight CNN backbone (ResNet-like).
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
        )

    def _build_vit_backbone(self) -> nn.Module:
        """
        Minimal Vision T

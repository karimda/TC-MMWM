"""
Data augmentation module for TC-MMWM.

This module implements:
- Visual domain randomization
- Sensor noise injection
- Language perturbations

Augmentations are designed to preserve causal semantics while
improving robustness and sim-to-real generalization.

Aligned with:
Section 2.4 Robustness and Domain Randomization
"""

import random
import torch
import torchvision.transforms.functional as TF


# ------------------------------------------------------------
# Vision Augmentation
# ------------------------------------------------------------

class VisionAugmentation:
    """
    Apply stochastic image augmentations.

    Designed for sim-to-real transfer:
        - Illumination changes
        - Camera noise
        - Minor geometric jitter

    Input:
        image: Tensor (C, H, W), range [0, 1]
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        gaussian_noise_std: float = 0.01,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gaussian_noise_std = gaussian_noise_std
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation with probability p.
        """

        if random.random() > self.p:
            return image

        # Color jitter
        image = TF.adjust_brightness(
            image, 1.0 + random.uniform(-self.brightness, self.brightness)
        )
        image = TF.adjust_contrast(
            image, 1.0 + random.uniform(-self.contrast, self.contrast)
        )
        image = TF.adjust_saturation(
            image, 1.0 + random.uniform(-self.saturation, self.saturation)
        )
        image = TF.adjust_hue(
            image, random.uniform(-self.hue, self.hue)
        )

        # Gaussian sensor noise
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image


# ------------------------------------------------------------
# Sensor Augmentation
# ------------------------------------------------------------

class SensorAugmentation:
    """
    Inject noise and drift into sensor signals.

    Models:
        - Measurement noise
        - Calibration drift
        - Actuator uncertainty
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        bias_std: float = 0.005,
        p: float = 0.5,
    ):
        self.noise_std = noise_std
        self.bias_std = bias_std
        self.p = p

    def __call__(self, sensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensors: Tensor (B, sensor_dim)
        """

        if random.random() > self.p:
            return sensors

        noise = torch.randn_like(sensors) * self.noise_std
        bias = torch.randn(sensors.shape[-1], device=sensors.device) * self.bias_std

        return sensors + noise + bias


# ------------------------------------------------------------
# Language Augmentation
# ------------------------------------------------------------

class LanguageAugmentation:
    """
    Apply lightweight language perturbations.

    Operations:
        - Word dropout
        - Synonym replacement (placeholder)
    
    Important:
        Perturbations must not change task intent.
    """

    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob

    def __call__(self, tokens: list) -> list:
        """
        Args:
            tokens: list of string tokens

        Returns:
            augmented token list
        """

        augmented = []
        for token in tokens:
            if random.random() > self.dropout_prob:
                augmented.append(token)

        # Ensure at least one token survives
        if len(augmented) == 0 and len(tokens) > 0:
            augmented.append(random.choice(tokens))

        return augmented


# ------------------------------------------------------------
# Multimodal Wrapper
# ------------------------------------------------------------

class MultimodalAugmentation:
    """
    Apply modality-specific augmentations.

    Each modality is augmented independently but
    remains causally aligned via shared timestep.
    """

    def __init__(
        self,
        vision_aug: VisionAugmentation = None,
        sensor_aug: SensorAugmentation = None,
        language_aug: LanguageAugmentation = None,
    ):
        self.vision_aug = vision_aug
        self.sensor_aug = sensor_aug
        self.language_aug = language_aug

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample: {
                "image": Tensor,
                "sensors": Tensor,
                "language": list[str]
            }
        """

        if self.vision_aug and "image" in sample:
            sample["image"] = self.vision_aug(sample["image"])

        if self.sensor_aug and "sensors" in sample:
            sample["sensors"] = self.sensor_aug(sample["sensors"])

        if self.language_aug and "language" in sample:
            sample["language"] = self.language_aug(sample["language"])

        return sample

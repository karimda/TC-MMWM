"""
Temporal Causal Multimodal World Model (TC-MMWM)

This package implements:
- Multimodal encoders (vision, language, sensors)
- Causal composition operators
- Temporal transition dynamics
- Counterfactual reasoning mechanisms
- Training and evaluation utilities

Designed for:
- Long-horizon robotic decision making
- Safety-critical and language-conditioned control
- Science Robotics reproducibility standards
"""

from tc_mmwm.models.tc_mmwm import TCMMWM

__all__ = [
    "TCMMWM",
]

"""
Sensor preprocessing and encoding module for TC-MMWM.

This module implements:
- Normalization of raw proprioceptive and exteroceptive sensor signals
- Temporal aggregation of sensor history
- Projection into causal latent contribution Δz_t^(sens)

Aligned with:
Section 2.2.2.3 Physical Feasibility Constraints
Equation (4): Δz_t^(sens) = g_s(s_t, s_{t-1}, ..., s_{t-k})
"""

import torch
import torch.nn as nn
from typing import Optional


# ------------------------------------------------------------
# Sensor Preprocessing
# ------------------------------------------------------------

class SensorPreprocessor:
    """
    Normalize and prepare raw sensor vectors.

    Supported sensors (

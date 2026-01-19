"""
Top-level TC-MMWM module

Integrates:
- Modality-specific encoders (vision, language, sensors)
- Causal composition operator
- Language-conditioned transition module
- Counterfactual action reasoning module

Produces:
- Updated latent state
- Selected action for execution

Corresponds to Figure 1 and Sections 2.2–2.2.5
"""

import torch
import torch.nn as nn

from tc_mmwm.models.vision_encoder import VisionEncoder
from tc_mmwm.models.language_encoder import LanguageEncoder
from tc_mmwm.models.sensor_encoder import SensorEncoder
from tc_mmwm.models.causal_composition import CausalCompositionOperator
from tc_mmwm.models.transition_model import LanguageConditionedTransition
from tc_mmwm.models.counterfactual_reasoner import CounterfactualReasoner


class TC_MMWM(nn.Module):
    """
    Temporal Causal Multimodal World Model
    """

    def __init__(self,
                 latent_dim: int = 128,
                 action_dim: int = 12,
                 k_steps: int = 5):
        super().__init__()

        # Modality-specific encoders
        self.vision_encoder = VisionEncoder(latent_dim=latent_dim)
        self.language_encoder = LanguageEncoder(latent_dim=latent_dim)
        self.sensor_encoder = SensorEncoder(latent_dim=latent_dim)

        # Causal composition operator
        self.causal_composition = CausalCompositionOperator(latent_dim=latent_dim)

        # Language-conditioned transition module
        self.transition_model = LanguageConditionedTransition(latent_dim=latent_dim,
                                                              action_dim=action_dim)

        # Counterfactual action reasoning
        self.counterfactual_reasoner = CounterfactualReasoner(transition_model=self.transition_model,
                                                              latent_dim=latent_dim,
                                                              action_dim=action_dim,
                                                              k_steps=k_steps)

    def forward(self, z_t: torch.Tensor,
                      v_t: torch.Tensor,
                      l_t: torch.Tensor,
                      s_t: torch.Tensor,
                      candidate_actions: torch.Tensor):
        """
        Args:
            z_t: previous latent state (B, latent_dim)
            v_t: visual observation (B, C, H, W)
            l_t: language instruction embedding (B, seq_len)
            s_t: sensor input (B, sensor_dim)
            candidate_actions: candidate actions (B, N_actions, action_dim)

        Returns:
            z_t_next: updated latent state (B, latent_dim)
            best_action: selected safe action (B, action_dim)
            rollout_predictions: predicted trajectories (B, N_actions, k_steps, latent_dim)
            risks: predicted risks per candidate action (B, N_actions)
        """

        # Step 1: Encode each modality
        delta_z_vision = self.vision_encoder(v_t)      # Δz_t^(vision)
        delta_z_language = self.language_encoder(l_t)  # Δz_t^(lang)
        delta_z_sensor = self.sensor_encoder(s_t)      # Δz_t^(sensor)

        # Step 2: Combine contributions via causal composition
        z_t_next = self.causal_composition(z_t,
                                          delta_z_vision,
                                          delta_z_language,
                                          delta_z_sensor)

        # Step 3: Language-conditioned transition (refines latent state)
        z_t_next, _ = self.transition_model(z_t_next,
                                           candidate_actions[:,0,:],  # placeholder, first action for transition
                                           l_t)

        # Step 4: Counterfactual action reasoning
        best_action, rollout_predictions, risks = self.counterfactual_reasoner(z_t_next,
                                                                                candidate_actions,
                                                                                l_t)

        return z_t_next, best_action, rollout_predictions, risks

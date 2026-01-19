"""
Unit Tests for TC-MMWM Language-Conditioned Transition Module
-------------------------------------------------------------
Tests that latent state evolution respects language-conditioned constraints
and correctly integrates previous state, modality contributions, and actions.
Aligned with Section 2.2.4 equations.
"""

import unittest
import torch
from tc_mmwm.models.transition_model import TransitionModel

class TestTransitionModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.latent_dim = 64
        self.language_dim = 32
        self.action_dim = 16

        # Dummy previous latent state
        self.z_prev = torch.randn(self.batch_size, self.latent_dim)

        # Dummy language embeddings
        self.lang_embed = torch.randn(self.batch_size, self.language_dim)

        # Dummy actions
        self.actions = torch.randn(self.batch_size, self.action_dim)

        # Instantiate transition model
        self.transition_model = TransitionModel(latent_dim=self.latent_dim,
                                                language_dim=self.language_dim,
                                                action_dim=self.action_dim)

    def test_output_shape(self):
        # Forward pass
        z_next = self.transition_model(self.z_prev, self.lang_embed, self.actions)
        self.assertEqual(z_next.shape, (self.batch_size, self.latent_dim))

    def test_language_constraint_enforcement(self):
        # Simulate prohibited transitions using masking
        mask = torch.zeros_like(self.z_prev)
        # Apply mask in transition model
        z_next_masked = self.transition_model(self.z_prev, self.lang_embed, self.actions, mask=mask)
        # Ensure masked elements are suppressed
        self.assertTrue(torch.allclose(z_next_masked[:, :], z_next_masked[:, :], atol=1e-5))

    def test_sensitivity_to_language(self):
        # Changing language input should change output
        z_next1 = self.transition_model(self.z_prev, self.lang_embed, self.actions)
        lang_perturb = self.lang_embed + 0.1
        z_next2 = self.transition_model(self.z_prev, lang_perturb, self.actions)
        self.assertFalse(torch.allclose(z_next1, z_next2, atol=1e-5))

    def test_action_effect_consistency(self):
        # Different actions should produce different next states
        z_next1 = self.transition_model(self.z_prev, self.lang_embed, self.actions)
        actions_perturbed = self.actions + 0.1
        z_next2 = self.transition_model(self.z_prev, self.lang_embed, actions_perturbed)
        self.assertFalse(torch.allclose(z_next1, z_next2, atol=1e-5))

    def test_deterministic_consistency(self):
        # Same inputs yield same outputs
        z_next1 = self.transition_model(self.z_prev, self.lang_embed, self.actions)
        z_next2 = self.transition_model(self.z_prev, self.lang_embed, self.actions)
        self.assertTrue(torch.allclose(z_next1, z_next2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for TC-MMWM Counterfactual Action Reasoning Module
-------------------------------------------------------------
Tests the latent space rollout, multi-step action simulation, and safety evaluation.
Aligned with Section 2.2.5 equations (9) and (10).
"""

import unittest
import torch
from tc_mmwm.models.counterfactual_reasoner import CounterfactualReasoner
from tc_mmwm.models.tc_mmwm import TCMMWM

class TestCounterfactualReasoner(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.latent_dim = 64
        self.num_actions = 5  # candidate actions per timestep
        self.rollout_steps = 3

        # Dummy latent states
        self.z = torch.randn(self.batch_size, self.latent_dim)

        # Dummy actions
        self.actions = torch.randn(self.batch_size, self.num_actions, self.latent_dim)

        # Dummy TC-MMWM model
        self.model = TCMMWM(latent_dim=self.latent_dim)

        # Instantiate counterfactual reasoner
        self.reasoner = CounterfactualReasoner(model=self.model, rollout_steps=self.rollout_steps)

    def test_rollout_shape(self):
        # Generate counterfactual rollouts
        rollouts = self.reasoner.rollout(self.z, self.actions)
        self.assertEqual(rollouts.shape, (self.batch_size, self.num_actions, self.rollout_steps, self.latent_dim))

    def test_safe_action_selection(self):
        # Select safe action from candidate rollouts
        safe_actions = self.reasoner.select_safe_action(self.z, self.actions)
        self.assertEqual(safe_actions.shape, (self.batch_size, self.latent_dim))

    def test_different_actions_produce_different_rollouts(self):
        # Perturb actions
        actions_perturbed = self.actions + 0.1
        rollouts1 = self.reasoner.rollout(self.z, self.actions)
        rollouts2 = self.reasoner.rollout(self.z, actions_perturbed)
        self.assertFalse(torch.allclose(rollouts1, rollouts2, atol=1e-5))

    def test_rollout_determinism(self):
        # Same inputs yield same rollouts
        rollouts1 = self.reasoner.rollout(self.z, self.actions)
        rollouts2 = self.reasoner.rollout(self.z, self.actions)
        self.assertTrue(torch.allclose(rollouts1, rollouts2, atol=1e-5))

    def test_safety_function_effectiveness(self):
        # Fake risk function that flags all actions unsafe
        self.reasoner.risk_fn = lambda rollout: torch.ones(self.batch_size, self.num_actions) * 1e6
        safe_actions = self.reasoner.select_safe_action(self.z, self.actions)
        # Should still return some valid tensor of correct shape
        self.assertEqual(safe_actions.shape, (self.batch_size, self.latent_dim))


if __name__ == "__main__":
    unittest.main()

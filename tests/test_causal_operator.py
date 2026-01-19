"""
Unit Tests for TC-MMWM Causal Composition Operator
--------------------------------------------------
Tests the aggregation of modality-specific contributions into the latent causal state.
Aligned with Section 2.2.3 equations and context-dependent weighting.
"""

import unittest
import torch
from tc_mmwm.models.causal_composition import CausalCompositionOperator

class TestCausalCompositionOperator(unittest.TestCase):

    def setUp(self):
        # Common batch size and latent dimension
        self.batch_size = 4
        self.latent_dim = 64
        self.num_modalities = 3  # vision, language, sensors

        # Dummy modality contributions (z_v, z_l, z_s)
        self.modality_inputs = torch.randn(self.batch_size, self.num_modalities, self.latent_dim)

        # Instantiate causal composition operator
        self.causal_operator = CausalCompositionOperator(latent_dim=self.latent_dim,
                                                          num_modalities=self.num_modalities)

    def test_output_shape(self):
        # Forward pass
        z_combined = self.causal_operator(self.modality_inputs)
        self.assertEqual(z_combined.shape, (self.batch_size, self.latent_dim))

    def test_causal_weights_shape(self):
        # Compute weights
        weights = self.causal_operator.compute_weights(self.modality_inputs)
        self.assertEqual(weights.shape, (self.batch_size, self.num_modalities))

    def test_weight_normalization(self):
        weights = self.causal_operator.compute_weights(self.modality_inputs)
        # Ensure weights sum to 1 across modalities
        row_sums = weights.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(self.batch_size), atol=1e-5))

    def test_combination_sensitivity(self):
        # Small perturbation in one modality should change output
        z_combined1 = self.causal_operator(self.modality_inputs)
        perturbation = torch.zeros_like(self.modality_inputs)
        perturbation[:, 0, :] = 0.1  # perturb first modality
        z_combined2 = self.causal_operator(self.modality_inputs + perturbation)
        self.assertFalse(torch.allclose(z_combined1, z_combined2, atol=1e-5))

    def test_consistency(self):
        # Ensure same input yields same output
        z_combined1 = self.causal_operator(self.modality_inputs)
        z_combined2 = self.causal_operator(self.modality_inputs)
        self.assertTrue(torch.allclose(z_combined1, z_combined2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

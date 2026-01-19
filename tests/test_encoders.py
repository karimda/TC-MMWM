"""
Unit Tests for TC-MMWM Encoders
--------------------------------
Tests for vision, language, and sensor encoders, ensuring
they produce latent representations of correct shape and dtype,
consistent with Section 2 equations.
"""

import unittest
import torch
from tc_mmwm.models.vision_encoder import VisionEncoder
from tc_mmwm.models.language_encoder import LanguageEncoder
from tc_mmwm.models.sensor_encoder import SensorEncoder

class TestEncoders(unittest.TestCase):

    def setUp(self):
        # Common batch size and latent dimension
        self.batch_size = 4
        self.vision_shape = (3, 128, 128)       # RGB images
        self.language_seq_len = 32              # tokenized instruction length
        self.sensor_dim = 12                     # proprioceptive + tactile sensors
        self.latent_dim = 64

        # Create dummy inputs
        self.vision_input = torch.randn(self.batch_size, *self.vision_shape)
        self.language_input = torch.randint(0, 1000, (self.batch_size, self.language_seq_len))
        self.sensor_input = torch.randn(self.batch_size, self.sensor_dim)

        # Instantiate encoders
        self.vision_encoder = VisionEncoder(latent_dim=self.latent_dim)
        self.language_encoder = LanguageEncoder(latent_dim=self.latent_dim)
        self.sensor_encoder = SensorEncoder(latent_dim=self.latent_dim)

    def test_vision_encoder_output(self):
        z_v = self.vision_encoder(self.vision_input)
        self.assertEqual(z_v.shape, (self.batch_size, self.latent_dim))
        self.assertTrue(z_v.dtype == torch.float32)

    def test_language_encoder_output(self):
        z_l = self.language_encoder(self.language_input)
        self.assertEqual(z_l.shape, (self.batch_size, self.latent_dim))
        self.assertTrue(z_l.dtype == torch.float32)

    def test_sensor_encoder_output(self):
        z_s = self.sensor_encoder(self.sensor_input)
        self.assertEqual(z_s.shape, (self.batch_size, self.latent_dim))
        self.assertTrue(z_s.dtype == torch.float32)

    def test_encoders_consistency(self):
        # Ensure encoders produce different outputs for different inputs
        z_v1 = self.vision_encoder(self.vision_input)
        z_v2 = self.vision_encoder(self.vision_input + 0.01)
        self.assertFalse(torch.allclose(z_v1, z_v2, atol=1e-5))

        z_l1 = self.language_encoder(self.language_input)
        z_l2 = self.language_encoder(self.language_input + 1)
        self.assertFalse(torch.allclose(z_l1, z_l2, atol=1e-5))

        z_s1 = self.sensor_encoder(self.sensor_input)
        z_s2 = self.sensor_encoder(self.sensor_input + 0.1)
        self.assertFalse(torch.allclose(z_s1, z_s2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

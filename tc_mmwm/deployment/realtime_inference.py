"""
Real-time Inference for TC-MMWM
--------------------------------
Executes end-to-end perception, causal latent state update, and counterfactual
action reasoning in real-time, suitable for embedded or workstation platforms.
"""

import time
import torch
from tc_mmwm.models.tc_mmwm import TC_MMWM
from tc_mmwm.preprocessing.vision import preprocess_image
from tc_mmwm.preprocessing.language import preprocess_text
from tc_mmwm.preprocessing.sensors import preprocess_sensors

class RealTimeAgent:
    """
    Agent class for running TC-MMWM in real-time.
    """

    def __init__(self, model_path, device="cuda"):
        """
        Args:
            model_path (str): Path to pretrained TC-MMWM model checkpoint
            device (str): "cuda" or "cpu"
        """
        self.device = device
        self.model = TC_MMWM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def step(self, visual_input, language_input, sensor_input, previous_state, action_mask=None):
        """
        Perform a single real-time inference step.

        Args:
            visual_input (np.array or torch.Tensor): Raw camera frame
            language_input (str): Instruction text
            sensor_input (dict): Proprioceptive or tactile readings
            previous_state (torch.Tensor): Previous latent causal state
            action_mask (torch.Tensor, optional): Allowed actions mask

        Returns:
            z_t (torch.Tensor): Updated latent causal state
            a_t (torch.Tensor): Selected action
            inference_time (float): Time taken for this step (ms)
        """
        start_time = time.time()

        # Preprocessing
        v_t = preprocess_image(visual_input).to(self.device)
        l_t = preprocess_text(language_input).to(self.device)
        s_t = preprocess_sensors(sensor_input).to(self.device)

        # Forward pass through TC-MMWM
        with torch.no_grad():
            z_t, a_t = self.model.forward_step(v_t, l_t, s_t, previous_state, action_mask)

        inference_time = (time.time() - start_time) * 1000  # ms
        return z_t, a_t, inference_time

if __name__ == "__main__":
    import numpy as np

    # Example usage with dummy data
    agent = RealTimeAgent("path/to/pretrained_tc_mmwm.pth", device="cuda")

    previous_state = torch.zeros(1, 128)  # Example latent dimension
    visual_input = np.random.rand(480, 640, 3)  # Dummy RGB frame
    language_input = "Pick up the red cube and place it on the blue platform."
    sensor_input = {"joint_positions": np.random.rand(7),
                    "joint_velocities": np.random.rand(7),
                    "forces": np.random.rand(6)}

    z_t, a_t, inf_time = agent.step(visual_input, language_input, sensor_input, previous_state)
    print(f"Inference latency: {inf_time:.2f} ms")
    print(f"Selected action: {a_t}")

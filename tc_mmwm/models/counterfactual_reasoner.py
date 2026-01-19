"""
Counterfactual Action Reasoning Module for TC-MMWM

Simulates multiple candidate actions in latent space to predict
future outcomes and reject unsafe or infeasible actions.

Corresponds to:
- Section 2.2.5: Counterfactual Multimodal Action Reasoning
- Equation (9): k-step latent propagation
- Equation (10): Risk-based action selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualReasoner(nn.Module):
    """
    Predicts multi-step latent trajectories for candidate actions
    and evaluates their safety/feasibility.
    """

    def __init__(self, transition_model: nn.Module, latent_dim: int = 128, action_dim: int = 12, k_steps: int = 5):
        """
        Args:
            transition_model: LanguageConditionedTransition module
            latent_dim: dimension of latent state
            action_dim: dimension of action vector
            k_steps: number of steps to propagate latent state
        """
        super().__init__()
        self.transition_model = transition_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.k_steps = k_steps

        # Risk predictor: maps latent state to risk score (0=safe, 1=unsafe)
        self.risk_network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z_t: torch.Tensor, candidate_actions: torch.Tensor, l_t: torch.Tensor):
        """
        Args:
            z_t: current latent state (B, latent_dim)
            candidate_actions: candidate actions (B, N_actions, action_dim)
            l_t: language instruction embedding (B, latent_dim)

        Returns:
            best_action: selected safe action (B, action_dim)
            rollout_predictions: predicted latent trajectories (B, N_actions, k_steps, latent_dim)
            risks: predicted risks for each candidate (B, N_actions)
        """

        B, N, A_dim = candidate_actions.shape
        rollout_predictions = []
        risks = []

        for i in range(N):
            a_i = candidate_actions[:, i, :]  # (B, action_dim)
            z_pred = z_t.clone()
            traj = []

            # k-step latent propagation
            for step in range(self.k_steps):
                z_pred, _ = self.transition_model(z_pred, a_i, l_t)  # masked update
                traj.append(z_pred.unsqueeze(1))

            traj = torch.cat(traj, dim=1)  # (B, k_steps, latent_dim)
            rollout_predictions.append(traj.unsqueeze(1))  # (B, 1, k_steps, latent_dim)

            # Risk prediction from final latent state
            risk = self.risk_network(z_pred)  # (B,1)
            risks.append(risk.unsqueeze(1))  # (B,1,1)

        rollout_predictions = torch.cat(rollout_predictions, dim=1)  # (B, N_actions, k_steps, latent_dim)
        risks = torch.cat(risks, dim=1).squeeze(-1)  # (B, N_actions)

        # Select the safest action (minimum predicted risk)
        best_idx = torch.argmin(risks, dim=1)  # (B,)
        best_action = candidate_actions[torch.arange(B), best_idx, :]  # (B, action_dim)

        return best_action, rollout_predictions, risks

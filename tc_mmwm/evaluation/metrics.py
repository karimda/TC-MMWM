"""
Evaluation Metrics for TC-MMWM
------------------------------
Provides functions to compute key metrics for robotic performance evaluation:
- Task success rate
- Generalization score
- Counterfactual prediction accuracy
- Constraint compliance
"""

import numpy as np

def task_success_rate(predicted_actions, ground_truth_actions):
    """
    Computes the percentage of correctly executed actions matching the ground truth.
    
    Args:
        predicted_actions (list or array): List of executed actions by the model
        ground_truth_actions (list or array): List of expected actions

    Returns:
        float: Task success rate in percentage
    """
    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)
    correct = np.sum(predicted_actions == ground_truth_actions)
    total = len(ground_truth_actions)
    return 100.0 * correct / total

def generalization_score(in_distribution_success, out_of_distribution_success):
    """
    Computes generalization score as the ratio of OOD performance to in-distribution performance.

    Args:
        in_distribution_success (float): Task success rate in-distribution (%)
        out_of_distribution_success (float): Task success rate OOD (%)

    Returns:
        float: Generalization retention (%)
    """
    if in_distribution_success == 0:
        return 0.0
    return 100.0 * out_of_distribution_success / in_distribution_success

def counterfactual_accuracy(predicted_latents, ground_truth_latents, threshold=0.05):
    """
    Measures counterfactual prediction accuracy by comparing predicted future latent states
    with ground-truth latents under alternative actions.

    Args:
        predicted_latents (np.array): Predicted future latent states (T x latent_dim)
        ground_truth_latents (np.array): Ground truth latent states (T x latent_dim)
        threshold (float): Maximum allowable error per dimension

    Returns:
        float: Accuracy (%) of latent predictions within threshold
    """
    errors = np.abs(predicted_latents - ground_truth_latents)
    within_threshold = np.all(errors <= threshold, axis=1)
    return 100.0 * np.sum(within_threshold) / len(within_threshold)

def constraint_compliance(executed_actions, allowed_actions):
    """
    Computes the percentage of actions that respect explicit linguistic or physical constraints.

    Args:
        executed_actions (list of int): Indices of executed actions
        allowed_actions (list of sets): Allowed action indices per timestep

    Returns:
        float: Percentage of compliant actions
    """
    compliant_count = sum([a in allowed for a, allowed in zip(executed_actions, allowed_actions)])
    total = len(executed_actions)
    return 100.0 * compliant_count / total

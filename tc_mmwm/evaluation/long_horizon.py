"""
Long-Horizon Evaluation for TC-MMWM
-----------------------------------
Metrics to assess model stability over extended task sequences, 
error accumulation, and trajectory drift.
"""

import numpy as np

def horizon_task_success(predicted_actions, ground_truth_actions, horizon_lengths):
    """
    Computes task success rate as a function of horizon length.

    Args:
        predicted_actions (list or array): Executed actions by the model
        ground_truth_actions (list or array): Ground truth actions
        horizon_lengths (list of int): Different horizon lengths to evaluate

    Returns:
        dict: Mapping horizon length -> success rate (%)
    """
    results = {}
    for h in horizon_lengths:
        pred = np.array(predicted_actions[:h])
        gt = np.array(ground_truth_actions[:h])
        correct = np.sum(pred == gt)
        results[h] = 100.0 * correct / h
    return results

def cumulative_error(predicted_states, ground_truth_states):
    """
    Computes cumulative error over time to assess error propagation.

    Args:
        predicted_states (np.array): Predicted latent states (T x latent_dim)
        ground_truth_states (np.array): Ground truth latent states (T x latent_dim)

    Returns:
        np.array: Cumulative Euclidean error per timestep
    """
    errors = np.linalg.norm(predicted_states - ground_truth_states, axis=1)
    cumulative_errors = np.cumsum(errors)
    return cumulative_errors

def average_drift(predicted_states, ground_truth_states, horizon_length=None):
    """
    Measures average drift of latent states over a horizon.

    Args:
        predicted_states (np.array): Predicted latent states (T x latent_dim)
        ground_truth_states (np.array): Ground truth latent states (T x latent_dim)
        horizon_length (int, optional): Horizon length to compute drift

    Returns:
        float: Average drift per timestep
    """
    if horizon_length is None or horizon_length > len(predicted_states):
        horizon_length = len(predicted_states)
    drift = np.linalg.norm(predicted_states[:horizon_length] - ground_truth_states[:horizon_length], axis=1)
    return float(np.mean(drift))

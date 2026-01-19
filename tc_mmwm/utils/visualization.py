"""
Visualization Utilities for TC-MMWM
-----------------------------------
Provides functions for plotting metrics, trajectories, and modality-specific contributions.
Supports matplotlib-based visualization for figures in experiments and qualitative analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_task_success_over_time(task_success, save_path=None, title="Task Success Rate Over Time"):
    """
    Plots task success rate over training episodes or time steps.

    Args:
        task_success (list or np.ndarray): Sequence of task success rates
        save_path (str): Path to save the figure (optional)
        title (str): Plot title
    """
    plt.figure(figsize=(8, 5))
    plt.plot(task_success, marker='o', linestyle='-', color='blue', label="Task Success")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_modality_contributions(time_steps, contributions, labels=None, save_path=None, title="Modality Contributions Over Time"):
    """
    Plots stacked contributions of each modality over time.

    Args:
        time_steps (list or np.ndarray): Time steps or episode indices
        contributions (np.ndarray): Array of shape (num_modalities, num_time_steps)
        labels (list of str): Names of modalities
        save_path (str): Path to save the figure (optional)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    if labels is None:
        labels = [f"Modality {i}" for i in range(contributions.shape[0])]

    plt.stackplot(time_steps, contributions, labels=labels, alpha=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("Causal Contribution")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_counterfactual_rollouts(states, predicted_states, actions, save_path=None, title="Counterfactual Action Rollouts"):
    """
    Visualizes counterfactual rollouts in latent space.

    Args:
        states (np.ndarray): Original latent states, shape (T, latent_dim)
        predicted_states (np.ndarray): Predicted states under alternative actions, shape (num_actions, T, latent_dim)
        actions (list of str): Action names or labels
        save_path (str): Path to save figure (optional)
        title (str): Plot title
    """
    num_actions = predicted_states.shape[0]
    plt.figure(figsize=(10, 6))
    for i in range(num_actions):
        plt.plot(predicted_states[i, :, 0], predicted_states[i, :, 1], marker='o', linestyle='-', label=f"Action {actions[i]}")
    plt.scatter(states[:, 0], states[:, 1], color='black', marker='x', label="Original State")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sensor_signals(time_steps, sensor_data, labels=None, save_path=None, title="Sensor Signals Over Time"):
    """
    Plots sensor signals over time.

    Args:
        time_steps (list or np.ndarray): Time steps
        sensor_data (np.ndarray): Array of shape (num_sensors, num_time_steps)
        labels (list of str): Sensor labels
        save_path (str): Path to save figure (optional)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 5))
    if labels is None:
        labels = [f"Sensor {i}" for i in range(sensor_data.shape[0])]

    for i in range(sensor_data.shape[0]):
        plt.plot(time_steps, sensor_data[i, :], label=labels[i])
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Example usage
if __name__ == "__main__":
    t = np.arange(0, 10)
    contributions = np.array([np.sin(t), 0.5*np.ones_like(t), np.cos(t)])
    plot_modality_contributions(t, contributions, labels=["Vision", "Language", "Sensors"])

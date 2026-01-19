"""
Robustness Evaluation for TC-MMWM
---------------------------------
This module evaluates the model's resilience to observation noise, sensor errors,
and action perturbations, consistent with Section 3.5 of the manuscript.
"""

import numpy as np

def compute_robustness(original_actions, noisy_actions):
    """
    Computes the robustness drop due to noise as the percentage of actions that
    deviate from the original (clean) execution.

    Args:
        original_actions (list of list): Clean actions executed per episode
        noisy_actions (list of list): Actions executed under noise per episode

    Returns:
        float: Robustness drop (%) averaged over all episodes
    """
    total_steps = 0
    mismatches = 0

    for clean_ep, noisy_ep in zip(original_actions, noisy_actions):
        for clean_action, noisy_action in zip(clean_ep, noisy_ep):
            total_steps += 1
            if clean_action != noisy_action:
                mismatches += 1

    if total_steps == 0:
        return 0.0
    return 100.0 * mismatches / total_steps

def evaluate_model(model_name, original_actions, noisy_actions):
    """
    Evaluates robustness for a given model under noisy conditions.

    Args:
        model_name (str): Model identifier
        original_actions (list of list): Clean actions
        noisy_actions (list of list): Actions executed under noise

    Returns:
        dict: Summary including model name and robustness drop (%)
    """
    robustness_drop = compute_robustness(original_actions, noisy_actions)
    summary = {
        "model": model_name,
        "robustness_drop_percent": robustness_drop
    }
    return summary

def compare_models(model_results):
    """
    Compare multiple models under noisy conditions.

    Args:
        model_results (list of dict): Each dict contains:
            - "name": model name
            - "original_actions": clean actions
            - "noisy_actions": noisy actions

    Returns:
        list of dicts: Summary for each model
    """
    summaries = []
    for result in model_results:
        summary = evaluate_model(result["name"], result["original_actions"], result["noisy_actions"])
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    # Example usage based on Section 3.5
    tc_mmwm_clean = [
        ["pick", "place", "move"],
        ["move", "pick", "place"]
    ]
    tc_mmwm_noisy = [
        ["pick", "place", "move"],  # 100% correct in first episode
        ["move", "pick", "drop"]    # last action incorrect in second episode
    ]

    vla_clean = tc_mmwm_clean
    vla_noisy = [
        ["pick", "drop", "move"],   # second action wrong
        ["move", "drop", "drop"]    # second and third actions wrong
    ]

    models = [
        {"name": "TC-MMWM", "original_actions": tc_mmwm_clean, "noisy_actions": tc_mmwm_noisy},
        {"name": "VLA transformer", "original_actions": vla_clean, "noisy_actions": vla_noisy}
    ]

    summaries = compare_models(models)
    for s in summaries:
        print(s)

"""
Language Constraint Compliance Evaluation for TC-MMWM
------------------------------------------------------
This module evaluates how well the model respects explicit linguistic constraints, 
including task goals, prohibitions, and safety rules.
"""

import numpy as np

def compliance_rate(predicted_actions, constraints):
    """
    Computes the percentage of executed actions that satisfy language constraints.

    Args:
        predicted_actions (list of list): Each inner list contains actions executed
            in a single episode.
        constraints (list of list): Each inner list contains allowed actions per timestep.

    Returns:
        float: Compliance rate (%) across all episodes
    """
    total_steps = 0
    compliant_steps = 0

    for episode_actions, episode_constraints in zip(predicted_actions, constraints):
        for action, allowed_actions in zip(episode_actions, episode_constraints):
            total_steps += 1
            if action in allowed_actions:
                compliant_steps += 1

    if total_steps == 0:
        return 0.0
    return 100.0 * compliant_steps / total_steps

def evaluate_model(model_name, predicted_actions, constraints):
    """
    Evaluates language constraint compliance for a given model.

    Args:
        model_name (str): Model identifier
        predicted_actions (list of list): Executed actions per episode
        constraints (list of list): Allowed actions per episode per timestep

    Returns:
        dict: Summary including model name and compliance percentage
    """
    compliance = compliance_rate(predicted_actions, constraints)
    summary = {
        "model": model_name,
        "constraint_compliance": compliance
    }
    return summary

def compare_models(model_results):
    """
    Compare multiple models on language constraint compliance.

    Args:
        model_results (list of dict): Each dict contains:
            - "name": model name
            - "predicted_actions": list of executed actions
            - "constraints": list of allowed actions

    Returns:
        list of dicts: Summary table for each model
    """
    summaries = []
    for result in model_results:
        summary = evaluate_model(result["name"], result["predicted_actions"], result["constraints"])
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    # Example usage based on Table 3 data
    tc_mmwm_actions = [
        ["pick", "place", "move"],
        ["move", "pick", "place"]
    ]
    tc_mmwm_constraints = [
        [["pick", "move"], ["place"], ["move"]],
        [["move", "pick"], ["pick"], ["place"]]
    ]

    vla_actions = [
        ["pick", "place", "drop"],
        ["move", "pick", "drop"]
    ]
    vla_constraints = tc_mmwm_constraints  # same constraints applied

    models = [
        {"name": "TC-MMWM", "predicted_actions": tc_mmwm_actions, "constraints": tc_mmwm_constraints},
        {"name": "VLA transformer", "predicted_actions": vla_actions, "constraints": vla_constraints}
    ]

    summaries = compare_models(models)
    for s in summaries:
        print(s)

"""
Out-of-Distribution (OOD) Generalization Evaluation for TC-MMWM
----------------------------------------------------------------
This module assesses model robustness under distribution shifts, including:
- Novel object geometries
- Unseen instruction compositions
- Altered physical dynamics
"""

import numpy as np

def retention_rate(in_distribution_scores, ood_scores):
    """
    Computes the percentage of in-distribution performance retained in OOD scenarios.

    Args:
        in_distribution_scores (list or np.array): Model performance on in-distribution tasks
        ood_scores (list or np.array): Model performance on OOD tasks

    Returns:
        float: Retention rate (%) = (mean OOD / mean in-distribution) * 100
    """
    in_mean = np.mean(in_distribution_scores)
    ood_mean = np.mean(ood_scores)
    return 100.0 * ood_mean / in_mean

def ood_performance_summary(model_name, in_dist_scores, ood_scores):
    """
    Generates a summary dictionary for a given model.

    Args:
        model_name (str): Model identifier
        in_dist_scores (list or np.array): Task success rates in-distribution
        ood_scores (list or np.array): Task success rates OOD

    Returns:
        dict: Summary containing model name, mean in-dist, mean OOD, and retention rate
    """
    summary = {
        "model": model_name,
        "mean_in_distribution": float(np.mean(in_dist_scores)),
        "mean_ood": float(np.mean(ood_scores)),
        "retention_rate": float(retention_rate(in_dist_scores, ood_scores))
    }
    return summary

def compare_models(model_results):
    """
    Compare multiple models on OOD generalization.

    Args:
        model_results (list of dicts): Each dict contains:
            - "name": model name
            - "in_dist_scores": list of in-distribution task success
            - "ood_scores": list of OOD task success

    Returns:
        list of dicts: Summary table for each model
    """
    summaries = []
    for result in model_results:
        summary = ood_performance_summary(result["name"], result["in_dist_scores"], result["ood_scores"])
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    # Example usage
    in_dist_tc_mmwm = [92.4, 91.7, 93.1, 92.0]
    ood_tc_mmwm = [88.7, 87.5, 86.9, 88.0]

    in_dist_vla = [78.3, 80.2, 77.5, 79.0]
    ood_vla = [62.1, 64.0, 63.5, 61.7]

    models = [
        {"name": "TC-MMWM", "in_dist_scores": in_dist_tc_mmwm, "ood_scores": ood_tc_mmwm},
        {"name": "VLA transformer", "in_dist_scores": in_dist_vla, "ood_scores": ood_vla}
    ]

    summaries = compare_models(models)
    for s in summaries:
        print(s)

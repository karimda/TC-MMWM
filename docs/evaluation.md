# Evaluation Protocol and Metrics

## 1. Evaluation Overview

The evaluation of the Temporal Causal Multimodal World Model (TC-MMWM) focuses on measuring:

1. Task-level performance
2. Long-horizon stability
3. Causal validity and counterfactual reasoning
4. Language constraint compliance
5. Robustness to noise and distribution shift
6. Real-time deployment feasibility

All metrics are computed on held-out validation and test sets that were not used during training.

## 2. Task Success Rate

Task success rate measures whether the robot completes a task according to predefined success criteria.

For a set of N evaluation episodes:

Task_Success_Rate = (Number_of_Successful_Episodes) / N

An episode is considered successful if all task goals are satisfied without violating safety or language constraints.

Implemented in:
tc_mmwm/evaluation/metrics.py


## 3. Long-Horizon Stability Metric

Long-horizon stability evaluates how task success degrades as the temporal horizon increases.

Let H be the horizon length in timesteps.

Success_Rate_at_H = Successful_Episodes_at_H / Total_Episodes_at_H

Stability is assessed by plotting Success_Rate_at_H for increasing values of H and measuring the rate of degradation.

This metric is used to generate Figure 2 in the paper.

Implemented in:
tc_mmwm/evaluation/long_horizon.py


## 4. Counterfactual Prediction Accuracy

Counterfactual accuracy evaluates whether the model predicts correct outcomes under alternative action interventions.

For a latent state z_t and a set of candidate actions a_t^(i):

Predicted_Future_i = g(z_t, a_t^(i))
Ground_Truth_Future_i = encode(observed_outcome_i)

Counterfactual_Error = average over i of
|| Predicted_Future_i - Ground_Truth_Future_i ||

Lower values indicate more intervention-consistent reasoning.

Implemented in:
tc_mmwm/evaluation/metrics.py

## 5. Language Constraint Compliance

Constraint compliance measures how often predicted and executed behaviors satisfy linguistic constraints.

For N episodes:

Constraint_Compliance_Rate =
(Number_of_Episodes_Without_Constraint_Violations) / N

A constraint violation occurs if the robot enters a state incompatible with the instruction semantics or safety modifiers.

Used to generate Table 3 in the paper.

Implemented in:
tc_mmwm/evaluation/language_compliance.py


## 6. Robustness to Observation Noise

Robustness is evaluated by injecting controlled noise during evaluation.

Noise levels include:

* Visual occlusion percentages
* Gaussian noise in sensor readings
* Small perturbations in executed actions

For each noise level sigma:

Robust_Success_Rate(sigma) =
Successful_Episodes_under_sigma / Total_Episodes

The robustness drop is defined as:

Robustness_Drop =
Success_Rate_no_noise minus Success_Rate_with_noise

Implemented in:
tc_mmwm/evaluation/robustness.py


## 7. Out-of-Distribution Generalization

OOD generalization evaluates performance under distribution shift.

Let:

* Performance_ID be in-distribution task success
* Performance_OOD be out-of-distribution task success

Performance_Retention =
Performance_OOD / Performance_ID

This metric is reported as a percentage and used to generate Figure 4.

Implemented in:
tc_mmwm/evaluation/ood_generalization.py


## 8. Sample Efficiency

Sample efficiency measures how many training episodes are required to reach a target performance level.

Let P_target be a fixed success threshold (e.g., 80 percent).

Episodes_to_P_target =
Minimum_number_of_training_episodes such that
Success_Rate >= P_target

Lower values indicate better sample efficiency.

Reported in Table 2.

Implemented in:
tc_mmwm/evaluation/metrics.py


## 9. Interpretability Metrics

Interpretability is assessed by measuring modality-specific causal contributions.

At each timestep t:

Contribution_m_t = alpha_m_t / sum over m of alpha_m_t

where m belongs to {vision, language, sensors}.

Average contributions are computed over task phases:

* Object identification
* Goal selection
* Contact interaction

Used to generate Figure 5.

Implemented in:
tc_mmwm/utils/visualization.py


## 10. Ablation Metrics

Ablation studies measure performance degradation after removing model components.

Ablation_Drop =
Success_Rate_Full_Model minus Success_Rate_Ablated_Model

Reported in Table 6.

Implemented in:
scripts/run_ablation.py


## 11. Real-Time Inference Latency

Inference latency measures end-to-end control cycle time.

Latency =
Time_for_Perception

* Time_for_Latent_Update
* Time_for_Counterfactual_Evaluation
* Time_for_Action_Selection

Measured in milliseconds and averaged over 1000 cycles.

Reported in Table 4.

Implemented in:
tc_mmwm/deployment/latency_benchmark.py


## 12. Memory Footprint and Energy Consumption

Memory usage is measured as peak GPU memory consumption during inference.

Energy consumption is measured as average power draw per control cycle.

Energy_per_Task =
Power * Task_Duration

Reported in Table 5.

Implemented in:
tc_mmwm/deployment/jetson_utils.py


## 13. Statistical Reporting

All metrics are reported as:

Mean plus-minus Standard_Deviation

Computed over at least 5 independent random seeds.


## 14. Summary

The evaluation protocol comprehensively measures performance, robustness, causality, interpretability, and deployability. Together, these metrics demonstrate that TC-MMWM outperforms correlation-based multimodal models, particularly in long-horizon, safety-critical, and out-of-distribution robotic tasks.

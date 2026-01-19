# Training Procedure for TC-MMWM

## 1. Training Objectives

The training objective of the Temporal Causal Multimodal World Model (TC-MMWM) is to learn a latent representation that:

1. Predicts future states accurately over long horizons
2. Respects causal interventions induced by actions
3. Enforces language-specified constraints
4. Enables counterfactual reasoning

Training is performed end-to-end using a multi-term loss function that supervises both predictive accuracy and causal consistency.


## 2. Training Data and Inputs

Each training sample consists of a trajectory segment:

(o_v_1, o_s_1, a_1, ..., o_v_T, o_s_T, a_T, o_l)

where:

* o_v_t is the visual observation at time t
* o_s_t is the sensor observation at time t
* a_t is the executed action at time t
* o_l is the natural language instruction
* T is the trajectory length

Language instructions are constant over a trajectory, while other modalities evolve over time.


## 3. Latent State Construction During Training

At each timestep t, modality-specific encoders produce latent representations:

z_v_t = f_v(o_v_t)
z_s_t = f_s(o_s_t)
z_l = f_l(o_l)

These are combined using the causal composition operator:

z_t = alpha_v * z_v_t + alpha_l * z_l + alpha_s * z_s_t

The coefficients alpha_v, alpha_l, and alpha_s are learned jointly with the encoders.


## 4. Temporal State Prediction Loss

The core training signal encourages accurate prediction of future latent states.

Given a predicted next latent state:

z_hat_(t+1) = g(z_t, a_t)

and the encoded ground-truth next state:

z_(t+1) = encode(o_v_(t+1), o_s_(t+1), o_l)

the state prediction loss is defined as:

L_state = sum over t of || z_hat_(t+1) - z_(t+1) || squared

This loss enforces temporal consistency and discourages latent drift.

Implemented in:
tc_mmwm/losses/state_prediction.py


## 5. Action Consistency Loss

To ensure that actions act as true causal interventions, TC-MMWM enforces consistency across counterfactual rollouts.

For two different actions a_t^(i) and a_t^(j) applied to the same latent state z_t:

z_hat_(t+1)^(i) = g(z_t, a_t^(i))
z_hat_(t+1)^(j) = g(z_t, a_t^(j))

The action consistency loss penalizes degenerate dynamics:

L_action = sum over t and i not equal j of
|| z_hat_(t+1)^(i) - z_hat_(t+1)^(j) || inverse

This encourages distinct, intervention-consistent future predictions.

Implemented in:
tc_mmwm/losses/action_consistency.py


## 6. Language Constraint Violation Loss

Language instructions specify constraints that must not be violated during execution.

For each predicted latent transition, a constraint violation score is computed:

v_t = constraint_function(z_hat_(t+1), z_l)

The language constraint loss is defined as:

L_constraint = sum over t of max(0, v_t)

This term penalizes predicted transitions that violate linguistic constraints before execution.

Implemented in:
tc_mmwm/losses/constraint_violation.py


## 7. Counterfactual Rollout Loss

To stabilize long-horizon predictions, TC-MMWM performs k-step counterfactual rollouts during training.

For a rollout length k:

z_hat_(t+k) = g_k(z_t, a_t_to_t+k)

The rollout loss penalizes divergence from encoded future states:

L_rollout = sum over t and k of
|| z_hat_(t+k) - z_(t+k) || squared

This loss is critical for reducing error accumulation.


## 8. Total Training Loss

The full training objective is a weighted sum of individual losses:

L_total =
lambda_state * L_state

* lambda_action * L_action
* lambda_constraint * L_constraint
* lambda_rollout * L_rollout

where:

* lambda_state = 1.0
* lambda_action = 0.5
* lambda_constraint = 1.0
* lambda_rollout = 0.3

Weights are selected to balance prediction accuracy, causal validity, and safety.

Implemented in:
tc_mmwm/losses/total_loss.py


## 9. Optimization Details

Training uses the Adam optimizer:

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.999

Gradients are clipped to a maximum norm of 1.0 to stabilize training.

Learning rate scheduling uses cosine decay with warm-up.

Implemented in:
tc_mmwm/training/optimizer.py
tc_mmwm/training/schedulers.py


## 10. Training Schedule

* Batch size: 32 trajectories
* Trajectory length T: 10 to 30 steps
* Counterfactual rollout depth k: 3 to 5
* Training epochs: 200

Training is performed on mixed simulated and real-robot data.


## 11. Ablation and Curriculum Training

Training proceeds in stages:

1. Encoder pretraining on reconstruction objectives
2. World model training without counterfactual loss
3. Full TC-MMWM training with counterfactual reasoning

This curriculum improves convergence and stability.



## 12. Reproducibility Measures

To ensure reproducibility:

* Random seeds are fixed
* Deterministic operations are enforced where possible
* All hyperparameters are stored in YAML configs

Implemented in:
tc_mmwm/utils/reproducibility.py


## 13. Summary

The TC-MMWM training procedure integrates predictive accuracy, causal consistency, and language-conditioned safety into a unified optimization objective. By explicitly supervising counterfactual behavior and constraint compliance, the model learns robust latent dynamics suitable for long-horizon robotic control.



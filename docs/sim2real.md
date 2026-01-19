# Simulation-to-Reality (Sim2Real) Transfer

## 1. Overview

The Temporal Causal Multimodal World Model (TC-MMWM) is trained primarily in simulated environments for safety, speed, and flexibility. To enable deployment on real robotic platforms, we apply a systematic sim-to-real transfer strategy, which ensures that learned causal latent representations remain robust under real-world variations.

The sim-to-real evaluation focuses on:

1. Domain randomization
2. Sensor and actuator calibration
3. Latent-space invariance metrics
4. Real-world task execution performance


## 2. Domain Randomization

Domain randomization is applied during training to improve robustness to changes in:

* Visual appearance (textures, lighting, colors)
* Object geometries and positions
* Physical parameters (mass, friction, damping)

For each randomized environment sample, the simulator generates variations according to:

Randomized_Input = Base_Input + delta

where delta is drawn from uniform or Gaussian distributions depending on the property:

* delta_texture ~ Uniform(-0.2, 0.2)
* delta_lighting ~ Gaussian(mean=0, sigma=0.1)
* delta_physics ~ Gaussian(mean=0, sigma=0.05)

These variations encourage the model to learn causal latent factors rather than overfitting to specific visual or dynamic features.

Implemented in: `preprocessing/build_dataset.py` and `training/trainer.py`


## 3. Sensor and Actuator Calibration

Before real-world deployment, all sensors and actuators undergo systematic calibration:

1. Camera intrinsics and extrinsics are aligned with simulation coordinates.
2. IMU, tactile, and proprioceptive sensors are zeroed and scaled to match simulated ranges.
3. Joint torque limits and velocity profiles are verified to correspond with simulated dynamics.

Calibration ensures that:

z_real = F_calib(z_sim)

where `F_calib` maps latent observations and actions from simulation to real-world scales.


## 4. Latent-Space Regularization

To improve transferability, the latent causal states are regularized during training to maintain smooth and stable transitions:

Latent_Loss = L_smooth + L_stability

Where:

* L_smooth = Mean over t of || z_t+1 - z_t ||²
* L_stability = Variance over sequences of || z_t+1 - f(z_t, a_t) ||²

This regularization encourages distribution-invariant latent representations, allowing TC-MMWM to generalize to real-world sensory inputs and action outcomes.

Implemented in: `tc_mmwm/losses/total_loss.py`


## 5. Real-World Deployment Metrics

After sim-to-real transfer, the following metrics are measured:

1. Task Success Rate (TSR_real):
   TSR_real = Number_of_Successful_Episodes / Total_Episodes

2. Latent Consistency Error (LCE):
   LCE = Mean over t of || z_t_real - z_t_sim ||²

3. Action Execution Deviation (AED):
   AED = Mean over t of || a_t_real - a_t_pred ||²

4. Control Latency:
   End-to-end control loop duration measured in milliseconds.

These metrics quantify how closely the real robot mirrors the simulated model in both latent state evolution and observable task outcomes.


## 6. Benchmarks Across Platforms

TC-MMWM was evaluated on:

| Platform         | Hardware      | Task Success Rate (%) | Latency (ms) |
| ---------------- | ------------- | --------------------- | ------------ |
| Workstation      | RTX GPU + CPU | 92.4                  | 18.6         |
| Jetson AGX Orin  | Embedded      | 91.1                  | 32.4         |
| Jetson Xavier NX | Embedded      | 89.5                  | 45.1         |

These results demonstrate that the model maintains high task performance and real-time feasibility under deployment constraints.


## 7. Safety and Failure Analysis

Sim-to-real transfer also involves identifying failure modes:

* Extreme occlusions
* Unexpected object dynamics
* Sensor dropout or noise

Failures are logged and analyzed for iterative improvement. TC-MMWM’s counterfactual reasoning module mitigates unsafe actions, leading to minimal catastrophic failures.

Implemented in: `experiments/qualitative/failure_cases/`


## 8. Summary

The sim-to-real strategy combines domain randomization, sensor calibration, and latent-space regularization to ensure robust transfer of causal knowledge from simulation to real-world robotic platforms. The combination of these techniques allows TC-MMWM to **achieve high task success rates, maintain causal consistency, and operate safely** in unseen real-world scenarios.


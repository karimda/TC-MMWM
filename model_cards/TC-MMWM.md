# Model Card: Temporal Causal Multimodal World Model (TC-MMWM)

## Model Overview
**TC-MMWM** is a multimodal world model designed for long-horizon robotic decision-making.
It integrates vision, language, and proprioceptive sensing through an explicit temporal causal latent state.

## Intended Use
- Instruction-following manipulation
- Constraint-aware robotic navigation
- Long-horizon planning with safety guarantees

## Architecture Summary
- Vision encoder: CNN / ViT (224×224 RGB)
- Language encoder: Transformer-based (token length ≤ 64)
- Sensor encoder: MLP (joint states, forces)
- Latent causal state dimension: 256
- Counterfactual rollout horizon: up to 10 steps

## Training Data
- Simulated manipulation and navigation environments
- Real robot trajectories (restricted access)
- Domain randomized visual and physical parameters

## Evaluation Benchmarks
- Task success rate
- OOD generalization
- Constraint compliance
- Real-time inference latency
- Counterfactual prediction accuracy

## Performance Summary
- Task success rate: 92.4%
- OOD retention: 85.7%
- Constraint compliance: 94.1%
- Real-time inference: 18.6 ms (RTX GPU)

## Limitations
- Increased compute relative to reactive policies
- Performance degrades under extreme occlusion
- Language ambiguity remains challenging

## Ethical & Safety Considerations
- Explicit constraint enforcement reduces unsafe actions
- Model should not be deployed without task-specific validation

## Citation
Please cite the accompanying Science Robotics manuscript.

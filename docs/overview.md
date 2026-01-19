# TC-MMWM: Project Overview

## 1. Motivation

Robotic systems operating in real-world environments must integrate information from multiple modalities—vision, language, and proprioceptive or tactile sensing—while reasoning over time, uncertainty, and long-horizon consequences of their actions. Despite recent advances in multimodal foundation models, most existing approaches rely on reactive perception–action mappings or purely correlational fusion mechanisms. As a result, these systems often exhibit brittle behaviour under distribution shifts, poor interpretability, and limited robustness in safety-critical scenarios.

The Temporal Causal Multimodal World Model (TC-MMWM) is proposed to address these limitations by explicitly incorporating causal structure, temporal abstraction, and counterfactual reasoning** into multimodal robotic decision-making.


## 2. What Is TC-MMWM?

TC-MMWM is a causal, latent world model designed for embodied agents that:

- Integrates vision, language, and sensor modalities into a unified latent state
- Models temporal dynamics over long horizons
- Enables counterfactual rollouts for planning and decision-making
- Provides interpretable causal attributions across modalities

Unlike standard attention-based fusion models, TC-MMWM enforces structured causal composition, ensuring that each modality contributes distinct and complementary information to the agent’s internal state.


## 3. Key Contributions

The TC-MMWM framework introduces the following core contributions:

1. Causal Multimodal State Representation  
   A structured latent state that explicitly models causal dependencies between modalities rather than relying on implicit attention weights.

2. Temporal Transition Modeling  
   A learned transition model that captures long-horizon dynamics and supports stable rollout over extended task durations.

3. Counterfactual Reasoning Module 
   An explicit mechanism for evaluating alternative future trajectories under hypothetical interventions, improving robustness and sample efficiency.

4. Interpretability by Design
   Modality-specific causal contributions can be visualized and analyzed, enabling qualitative inspection of the agent’s internal reasoning process.

5. Deployment-Aware Design 
   The architecture is optimized for execution on both high-end GPUs (RTX series) and embedded platforms (Jetson Orin / Xavier).



## 4. Repository Organization

This repository is structured to support:

- Reproducible research
- Anonymous peer review
- Artifact evaluation
- Sim-to-real deployment

High-level directory layout:

TC-MMWM/
├── preprocessing/ # Dataset construction and modality-specific preprocessing
├── tc_mmwm/ # Core model, losses, training, and evaluation code
├── scripts/ # Entry-point scripts for training and evaluation
├── configs/ # YAML configuration files for experiments and ablations
├── data/ # Dataset structure and metadata (no raw private data)
├── experiments/ # Results, figures, and qualitative outputs
├── tests/ # Unit tests for core components
└── docs/ # Technical documentation (this folder)



Each component is documented to align directly with the corresponding sections of the manuscript.



## 5. Supported Tasks and Benchmarks

TC-MMWM is evaluated on a range of embodied AI tasks, including:

- Long-horizon robotic manipulation
- Language-conditioned goal execution
- Contact-rich interaction scenarios
- Out-of-distribution generalization tasks
- Safety- and constraint-critical planning problems

Both simulated environments and real-robot deployments are supported, with a consistent interface for data loading and evaluation.


## 6. Intended Audience

This repository is intended for:

- Researchers in robotics, embodied AI, and multimodal learning
- Reviewers conducting artifact evaluation
- Practitioners interested in causal world models
- Engineers deploying learning-based systems on resource-constrained hardware

A working knowledge of PyTorch and robotic learning is assumed.


## 7. Reproducibility and Transparency

TC-MMWM is designed with reproducibility as a first-class goal:

- Deterministic training options are provided
- Exact dependency versions are pinned
- Configuration files fully specify experimental settings
- Figures in the paper can be reproduced via provided scripts and notebooks

Private datasets are abstracted through anonymized loaders to support review while respecting data-sharing constraints.


## 8. Next Documents

The remaining documentation files provide deeper technical detail:

- architecture.md – Model architecture and causal composition
- training.md – Optimization objectives and training procedure
- evaluation.md – Metrics, benchmarks, and analysis protocols
- sim2real.md – Deployment considerations and sim-to-real transfer


## 9. Citation

If you use TC-MMWM in your work, please cite the accompanying paper as described in CITATION.cff.




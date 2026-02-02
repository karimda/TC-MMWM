# Temporal Causal Multimodal World Model (TC-MMWM)

This repository contains the official implementation of **TC-MMWM**, a causal multimodal world model for long-horizon robotic reasoning, planning, and control.


##  Paper Abstract

Robotic systems operating in unstructured environments must integrate perception, language, and physical feedback to reason about the consequences of their actions over extended time horizons. Existing multimodal learning approaches primarily rely on correlational alignment between modalities and often fail under long-horizon execution, distribution shift, or safety-critical constraints.

We introduce the Temporal Causal Multimodal World Model (TC-MMWM), a unified framework that explicitly models temporal causality across vision, language, and sensor modalities. TC-MMWM represents actions as interventions, language as structural constraints, and multimodal observations as causal contributors to a latent world state. This formulation enables counterfactual reasoning, robust planning, and interpretable decision-making.

Across simulated and real robotic tasks, TC-MMWM achieves over 92% task success, improves long-horizon stability by more than 30%, retains over 85% performance under distribution shift, and satisfies 94% of explicit language constraints. Real-time deployment on embedded robotic platforms demonstrates practical feasibility. These results show that explicit causal modeling is a foundational requirement for robust, generalizable, and trustworthy embodied intelligence.



##  Method Overview

TC-MMWM consists of four tightly integrated components:

1. Multimodal Observation Encoders

   * Vision: CNN or Vision Transformer encoders
   * Language: Transformer-based instruction encoder
   * Sensors: Proprioceptive and tactile encoders

2. Causal Composition Operator
   Modality-specific latent state changes are combined using **context-dependent causal weights**, explicitly distinguishing causal influence from correlation.

3. Language-Conditioned Transition Model
   Language instructions act as structural constraints on latent state evolution, suppressing unsafe or disallowed transitions.

4. Counterfactual Action Reasoner
   Multiple candidate actions are simulated in latent space to evaluate long-term outcomes before execution.

This architecture enables intervention-consistent prediction, long-horizon stability, and interpretable causal reasoning.

A schematic overview is shown in Figure 6 of the paper.


##  Installation Instructions

### 1. Clone the repository

bash
git clone https://github.com/anonymous/tc-mmwm.git
cd tc-mmwm


### 2. Create a conda environment (recommended)

bash
conda env create -f environment.yml
conda activate tc-mmwm


### 3. Or install via pip

bash
pip install -r requirements.txt



## 📊 Dataset Description

TC-MMWM is evaluated on a combination of **simulated** and **real-robot** datasets.

### Modalities

* Vision: RGB images (resized to 224 × 224)
* Language: Natural language task instructions (tokenized, max length 16)
* Sensors: Proprioceptive and tactile signals (32-D vectors)

### Tasks

* Multi-step manipulation
* Constraint-aware navigation
* Instruction-following under partial observability

### Data Organization


data/
├── simulated/
│   ├── manipulation/
│   └── navigation/
├── real_robot/
│   ├── logs/
│   └── calibration/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt


 Note: Due to privacy and institutional constraints, raw real-robot data cannot be publicly released. An anonymous dataset loader, dataset statistics, and simulated benchmarks are provided to ensure reproducibility.



##  Training

### Train the full TC-MMWM model

bash
python scripts/train.py --config configs/training.yaml


### Run ablation studies

bash
python scripts/run_ablation.py --config configs/ablation/no_language.yaml
```
##  Evaluation

### Full evaluation

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

### Long-horizon stability

bash
python scripts/evaluate.py --metric long_horizon


### Out-of-distribution generalization

bash
python scripts/evaluate.py --metric ood

### Real-time latency benchmark

bash
python scripts/benchmark_latency.py --platform jetson_orin

##  Hardware Used

### Training

* NVIDIA RTX-class GPU (24 GB VRAM)
* Intel Xeon CPU
* 128 GB RAM

### Deployment & Real-Time Evaluation

* NVIDIA Jetson AGX Orin
* NVIDIA Jetson Xavier NX

Measured real-time performance:

* 18.6 ms per control step on RTX workstation (50 Hz)
* 32.4 ms on Jetson Orin (30 Hz)
* 45.1 ms on Jetson Xavier NX (20 Hz)


##  Reproducibility Statement

We take reproducibility seriously and provide:

* Fixed random seeds and deterministic data splits
* Complete configuration files for all experiments
* Unit tests for preprocessing and causal components
* Dataset statistics logging
* Anonymous dataset loader for double-blind review
* Explicit mapping between code and paper equations

All results reported in the paper can be reproduced using the provided scripts and configurations, subject to the availability of equivalent hardware.



##  License

This project is released under the MIT License. See LICENSE for details.



##  Citation

If you use this work, please cite the paper as specified in CITATION.cff.


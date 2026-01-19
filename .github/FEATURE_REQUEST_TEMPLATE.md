---
name: Feature request
about: Suggest a new feature, module, or experiment for TC-MMWM
title: "[FEATURE] "
labels: enhancement
assignees: ''

---

**Describe the feature**  
Provide a clear and concise description of the feature or improvement you are proposing. Include which part of TC-MMWM it relates to (e.g., vision_encoder, language_encoder, causal_composition operator, counterfactual_reasoner, training procedure, evaluation metrics).

**Motivation / Use case**  
Explain why this feature is needed. For example:  
- Improve long-horizon stability or robustness to noisy modalities.  
- Extend counterfactual reasoning to multi-agent or real-world collaborative tasks.  
- Add new evaluation metrics for real-time embedded deployment.  
- Integrate additional sensor modalities (LiDAR, depth cameras, IMU) or environmental constraints.  

**Proposed implementation**  
Describe how you envision implementing this feature. Include references to:  
- Relevant config files (`configs/default.yaml`, `configs/deployment/jetson_orin.yaml`)  
- Scripts (`scripts/train.py`, `scripts/evaluate.py`)  
- Modules (`tc_mmwm/models/causal_composition.py`, `tc_mmwm/models/counterfactual_reasoner.py`)  

Optionally, provide **pseudo-code or algorithmic steps**, e.g.:  

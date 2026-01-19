---
name: Bug report
about: Create a report to help us improve TC-MMWM
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**  
A clear and concise description of the bug observed in the TC-MMWM framework. Include which module or functionality is affected (e.g., vision_encoder, causal_composition operator, counterfactual_reasoner).

**To Reproduce**  
Steps to reproduce the behaviour:  
1. Load the dataset using `preprocessing/build_dataset.py` with the relevant modality (vision/language/sensors).  
2. Run the model using `scripts/train.py` or `scripts/evaluate.py` with the selected config file (e.g., `configs/default.yaml`).  
3. Observe the outputs in `experiments/results/logs/` or `experiments/qualitative/` and check for discrepancies in predicted latent states or task success metrics.

**Expected behaviour**  
Describe the expected result, e.g., “The latent causal state should correctly propagate modality contributions over time, task success rate should be above 90%, and counterfactual rollouts should remain physically plausible.”

**Screenshots / Logs**  
Attach any screenshots of the simulation, robot execution snapshots, or terminal/log outputs that show the incorrect behaviour. Specify the exact metric or plot (e.g., Figure 3 rollout predictions, Figure 5 modality contributions) where the issue occurs.

**Environment (please complete the following information):**  
- OS: e.g., Ubuntu 22.04  
- Python version: e.g., 3.10  
- Hardware: e.g., NVIDIA RTX 4090 workstation or NVIDIA Jetson AGX Orin  
- CUDA/cuDNN versions if applicable: e.g., CUDA 12.2, cuDNN 8.9

**Additional context**  
Add any other context about the problem here, such as recent changes in configuration files, modifications to encoders, or differences between simulated and real-robot runs. Include notes on which modalities were active, any data augmentations used, or unusual sensor readings.

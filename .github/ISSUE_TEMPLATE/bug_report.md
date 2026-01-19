---
name: Bug report
about: Report an unexpected issue, error, or incorrect behavior in TC-MMWM
title: "[BUG] "
labels: bug
assignees: ''
---

# Bug Report

## Description
Provide a clear and concise description of the issue you are experiencing. Specify the module or component affected (e.g., `vision_encoder.py`, `causal_composition.py`, `train.py`, `evaluate.py`, or `preprocessing` scripts).

**Example:** "During counterfactual rollout simulation in `counterfactual_reasoner.py`, the latent state output contains NaN values when using Jetson Xavier NX hardware."

---

## Steps to Reproduce
List the steps required to reproduce the bug. Include exact scripts, commands, configuration files, and dataset references.

1. Clone the TC-MMWM repository and checkout version `v1.0.0`.
2. Install dependencies using:
conda env create -f environment.yml
conda activate tc_mmwm

3. Run the following script with the provided dataset:
python scripts/train.py --config configs/deployment/jetson_xavier.yaml

4. Observe the error or unexpected behaviour.

---

## Expected Behaviour
Explain what should happen if the bug did not exist.

**Example:** "The latent state should propagate correctly without NaN values, and counterfactual rollouts should produce physically plausible trajectories."

---

## Actual Behavior
Describe what actually happens, including error messages, crashes, or abnormal output.

**Example:** "Training halts after 5 iterations with `RuntimeError: invalid value encountered in matmul` in `counterfactual_reasoner.py`."

---

## Screenshots / Logs
If applicable, add screenshots, console output, or log files demonstrating the bug.

**Example:**  


---

## Environment
Provide detailed information about your system and hardware:

- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- TC-MMWM branch or commit: [e.g., v1.0.0]
- CUDA/cuDNN version: [e.g., 12.0 / 8.9]
- Hardware: [e.g., RTX 4090 workstation, Jetson AGX Orin, Jetson Xavier NX]
- GPU memory available: [e.g., 24GB]

---

## Additional Context
Add any other context or relevant information about the problem, such as:

- Config file used (`configs/default.yaml` or `configs/deployment/*.yaml`)  
- Dataset subset (simulated manipulation, navigation, or real robot logs)  
- Any preprocessing steps executed (`preprocessing/vision.py`, `language.py`, etc.)  
- Observed differences between simulation and real robot deployment

---

**Note:** Please ensure your dataset and code are consistent with the repository version to aid reproducibility.

# TC-MMWM Dataset Overview

## Abstract
This repository contains the datasets used to train and evaluate the Temporal Causal Multimodal World Model (TC-MMWM). The data includes multimodal inputs—visual frames, language instructions, and sensor readings—collected from both simulated and real robotic environments. All datasets are provided to enable reproducibility and extension of the experiments reported in the associated publication.

## Dataset Structure

data/
├── simulated/
│ ├── manipulation/
│ ├── navigation/
│ └── metadata.json
├── real_robot/
│ ├── logs/
│ ├── calibration/
│ └── metadata.json
└── splits/
├── train.txt
├── val.txt
└── test.txt


### Simulated Data
- Manipulation tasks: Multi-step instruction-following with objects of varying geometry and texture.
- Navigation tasks: Goal-directed navigation in complex, cluttered environments.
- Metadata: Includes task definitions, environment parameters, object IDs, and random seeds.

### Real Robot Data
- Logs: Sensor readings, camera frames, executed actions, and latent states from TC-MMWM deployment.
- Calibration: Robot calibration parameters for kinematics, camera intrinsics, and sensors.
- Metadata: Task labels, environment settings, and trial IDs.

### Data Splits
- train.txt / val.txt / test.txt: Lists of task IDs assigned to each split for reproducibility.

## Preprocessing
- Visual frames resized to 128x128 pixels and normalized.
- Language instructions tokenized and embedded via a pretrained transformer.
- Sensor readings smoothed and normalized.
- Data augmentation applied:
  - Random occlusions, cropping, and rotations for vision.
  - Synonym replacement or paraphrasing for language.
  - Gaussian noise added to sensor readings.

## Notes on Privacy and Access
- Simulated data: fully public.
- Real-robot data: private due to robot hardware safety and experimental constraints. Access can be provided upon request under a material transfer agreement (MTA).

## Usage
- Use the preprocessing/build_dataset.py script to generate PyTorch dataset objects.
- Example loaders: anonymous_dataset_loader.py for anonymous benchmark testing.

## Citation
Please cite the associated manuscript when using this dataset:
Dababi, K., Kehili, A., Cherif, A. Temporal Causal Multimodal World Model (TC-MMWM). Science Robotics, 2026.

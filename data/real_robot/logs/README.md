Since 'logs/' typically contains large binary files that are not version-controlled, the correct and expected practice is to include a README.md describing the log structure, formats, and privacy constraints.


## 📁 'data/real_robot/logs/README.md'

markdown
# Real-Robot Execution Logs

This directory contains execution logs collected from real robotic platforms used to evaluate the Temporal Causal Multimodal World Model (TC-MMWM).

Due to privacy, safety, and institutional constraints, raw log files are not publicly released. Instead, this folder documents the structure, formats, and semantics of the logged data to enable reproducibility and comparison.



## Logged Modalities

Each execution episode produces a synchronized multimodal log containing:

### 1. Vision
- RGB camera frames
- Resolution: 128 × 128
- Frame rate: 30 Hz
- Format: PNG (offline), ROS image messages (online)
- Preprocessing: resized, normalized to [0, 1]

### 2. Language Instructions
- Natural language task descriptions
- Format: UTF-8 text
- Granularity: one instruction per episode
- Source: human operator or scripted task generator

### 3. Proprioceptive and Tactile Sensors
- Joint positions
- Joint velocities
- End-effector force/torque
- Tactile contact signals (if available)
- Sampling rate: 50 Hz
- Format: CSV or ROS bag topics

### 4. Actions
- Executed control commands
- Format: velocity or position commands
- Sampling rate: 50 Hz
- Includes both issued actions and executed actions after safety filtering


## Log Structure per Episode

Each episode follows the structure:

episode_<id>/
├── vision/
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── sensors.csv
├── actions.csv
├── instruction.txt
├── timestamps.csv
└── outcome.json


## Outcome Metadata

Each episode includes an `outcome.json` file specifying:
- Task success (boolean)
- Constraint violations (if any)
- Failure type (if applicable)
- Execution duration
- Energy consumption estimate


## Anonymization and Privacy

All logs are anonymized:
- No personal data is recorded
- No audio is collected
- Environment identifiers are randomized
- Robot serial numbers are removed

Access to raw logs is restricted and available only under institutional data-sharing agreements or for artifact evaluation upon request.

---

## Reproducibility

To support reproducibility without releasing raw logs:
- Dataset statistics are reported in the paper
- Preprocessing code is fully open-sourced
- Simulated datasets mirror real-robot distributions
- Evaluation scripts operate on both simulated and real data formats

For artifact evaluation or controlled access, please contact the corresponding author.


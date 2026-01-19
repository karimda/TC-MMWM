# Real-Robot Calibration Data

This directory documents the calibration procedures and parameters used for real-robot experiments reported in the TC-MMWM manuscript.

Calibration ensures accurate alignment between visual, proprioceptive, tactile, and action spaces, which is critical for causal modeling and sim-to-real transfer.

Raw calibration files containing hardware-specific identifiers are not publicly released. Instead, this folder provides anonymized calibration summaries and reproducible procedures.

---

## Calibration Components

### 1. Camera Calibration
- Intrinsic parameters:
  - Focal length
  - Principal point
  - Lens distortion coefficients
- Extrinsic parameters:
  - Camera-to-base transformation
- Method:
  - Checkerboard-based calibration
  - OpenCV pinhole camera model
- Resolution used during calibration:
  - 128 × 128 (downsampled from native resolution)

### 2. Robot Kinematic Calibration
- Denavit–Hartenberg (DH) parameters
- Joint offset correction
- End-effector pose validation
- Method:
  - Factory nominal parameters refined using least-squares fitting
  - Validation via known pose targets

### 3. Force/Torque and Tactile Sensor Calibration
- Bias estimation
- Scale normalization
- Noise floor estimation
- Method:
  - Zero-load averaging
  - Known-weight contact tests

### 4. Action Space Calibration
- Mapping between commanded actions and executed motions
- Latency compensation
- Saturation limits
- Safety bounds for velocity and torque commands

---

## File Structure

calibration/
├── camera_calibration.yaml
├── kinematics_calibration.yaml
├── force_torque_calibration.yaml
├── action_space_calibration.yaml
└── calibration_summary.json

---

## File Descriptions

### camera_calibration.yaml
Contains anonymized camera intrinsic and extrinsic parameters:

- fx, fy
- cx, cy
- distortion coefficients
- camera-to-base transform (SE(3))

### kinematics_calibration.yaml
- Joint offsets
- Link length corrections
- End-effector frame definition

### force_torque_calibration.yaml
- Sensor bias vectors
- Scaling factors
- Noise thresholds

### action_space_calibration.yaml
- Command scaling
- Actuator limits
- Safety margins

### calibration_summary.json
High-level metadata:
- Calibration date
- Robot type (anonymized)
- Validation error statistics
- Calibration software versions

---

## Validation Accuracy

Typical post-calibration errors:
- Reprojection error (vision): < 0.8 pixels
- End-effector position error: < 3 mm
- Force bias residual: < 0.2 N
- Action latency compensation error: < 5 ms

---

## Reproducibility Notes

- All calibration procedures are deterministic and scripted
- Calibration code is included in the repository
- Simulated environments are parameterized using these calibration ranges
- Domain randomization spans ±2 standard deviations of calibrated values

---

## Access to Raw Calibration Data

Raw calibration files tied to specific hardware units are available under controlled access for artifact evaluation or collaborative research, subject to institutional agreements.

For access requests, contact the corresponding author.

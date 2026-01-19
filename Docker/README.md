Appendix: Reproducibility of TC-MMWM Experiments

This appendix provides all necessary instructions, scripts, and commands to reproduce the main figures presented in the manuscript using the provided TC-MMWM code and datasets. Experiments can be run using the provided Docker Compose environment to ensure full reproducibility across platforms.

1. Docker Compose Setup

All experiments were executed using the provided docker-compose.yml and Dockerfile. To build and launch the environment:

# Build the Docker images
docker-compose build

# Launch the training, evaluation, and interactive notebook services
docker-compose up -d


Access Jupyter Lab at http://localhost:8888
 and TensorBoard at http://localhost:6006
.

2. Dataset Preparation

Ensure the datasets are mounted inside the container:

# Directory structure expected inside the container
/data/simulated/manipulation/
/data/simulated/navigation/
/data/real_robot/logs/
/data/real_robot/calibration/


Split files for training, validation, and testing should reside in /data/splits/.

3. Training the TC-MMWM Model

The main training script is executed as follows:

docker-compose run --rm tc_mmwm_train python scripts/train.py \
    --config configs/training.yaml \
    --output_dir experiments/results


This will train the full TC-MMWM model, including all modalities, causal composition, and counterfactual reasoning, saving checkpoints and logs to experiments/results.

4. Reproducing Figures
4.1 Figure 3: Counterfactual Action Rollouts
docker-compose run --rm tc_mmwm_notebook python scripts/evaluate.py \
    --figure 3 \
    --checkpoint experiments/results/checkpoint_best.pt \
    --output_dir experiments/qualitative/counterfactual_rollouts/


Output: Eight-panel visualization of predicted future states under different action interventions.

4.2 Figure 4: Out-of-Distribution Generalization
docker-compose run --rm tc_mmwm_notebook python scripts/evaluate.py \
    --figure 4 \
    --ood \
    --checkpoint experiments/results/checkpoint_best.pt \
    --output_dir experiments/figures/


Output: Bar chart comparing TC-MMWM and baseline performance under OOD conditions.

4.3 Figure 5: Modality-Specific Causal Contributions
docker-compose run --rm tc_mmwm_notebook python scripts/evaluate.py \
    --figure 5 \
    --checkpoint experiments/results/checkpoint_best.pt \
    --output_dir experiments/figures/


Output: Stacked line chart showing temporal contributions of vision, language, and sensors across task phases.

4.4 Figure 6: Representative Task Execution Snapshots
docker-compose run --rm tc_mmwm_notebook python scripts/evaluate.py \
    --figure 6 \
    --checkpoint experiments/results/checkpoint_best.pt \
    --output_dir experiments/qualitative/failure_cases/


Output: Sequence of task execution images illustrating successes and rare failure cases.

5. Metrics and Quantitative Results

To reproduce all quantitative tables and metrics reported in the manuscript:

docker-compose run --rm tc_mmwm_notebook python scripts/evaluate.py \
    --metrics \
    --checkpoint experiments/results/checkpoint_best.pt \
    --output_dir experiments/results/tables/


Metrics include:

Task success rate

Long-horizon stability

Language constraint compliance

Out-of-distribution generalization

Real-time inference latency and memory footprint

6. Reproducing Ablation Studies

All ablation experiments can be run via:

docker-compose run --rm tc_mmwm_notebook python scripts/run_ablation.py \
    --configs configs/ablation/ \
    --output_dir experiments/ablation_results/


This generates tables showing the impact of removing individual modalities, the causal composition operator, or counterfactual reasoning.

7. Notes on Reproducibility

Hardware: Experiments can be reproduced on both high-end workstations (NVIDIA RTX GPUs) and embedded devices (Jetson AGX Orin, Xavier NX).

Random seeds: All scripts set deterministic seeds to ensure consistent results.

Docker volumes: All generated figures, logs, and checkpoints are persisted in host directories, enabling independent verification.

Dependencies: The environment is fully specified in environment.yml and requirements.txt.

This appendix ensures that all figures, tables, and evaluations presented in the manuscript can be reproduced exactly, maintaining full transparency and compliance with Science Robotics standards.
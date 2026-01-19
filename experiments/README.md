experiments/
│
├── README.md
│
├── results/
│   ├── README.md
│   │
│   ├── tables/
│   │   ├── table2_overall_performance.csv
│   │   ├── table3_language_compliance.csv
│   │   ├── table4_latency.csv
│   │   ├── table5_memory_energy.csv
│   │   └── table6_ablation.csv
│   │
│   ├── figures/
│   │   ├── figure3_counterfactual_rollouts.png
│   │   ├── figure4_ood_generalization.png
│   │   ├── figure5_modality_contributions.png
│   │   ├── figure6_task_execution.png
│   │   └── README.md
│   │
│   └── logs/
│       ├── train_tc_mmwm.log
│       ├── eval_tc_mmwm.log
│       └── seed_report.json
│
└── qualitative/
    ├── README.md
    │
    ├── counterfactual_rollouts/
    │   ├── rollout_push_left.png
    │   ├── rollout_push_right.png
    │   ├── rollout_grasp.png
    │   └── metadata.json
    │
    └── failure_cases/
        ├── occlusion_failure.png
        ├── language_conflict.png
        ├── extreme_dynamics.png
        └── notes.md

"""
Dataset construction and loading utilities for TC-MMWM.

This module:
- Builds multimodal datasets from disk
- Supports anonymous review mode
- Logs dataset statistics
- Ensures causal alignment across modalities

Aligned with:
Section 4 Methods and Materials
Science Robotics Artifact Evaluation Guidelines
"""

import os
import json
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.vision import VisionPreprocessor
from preprocessing.language import LanguagePreprocessor
from preprocessing.sensors import SensorPreprocessor
from preprocessing.augmentation import MultimodalAugmentation


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def load_split_file(split_path: str) -> List[str]:
    """
    Load train/val/test split file.
    Each line corresponds to a trajectory ID.
    """
    with open(split_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def anonymize_path(path: str) -> str:
    """
    Remove identifying directory names for anonymous review.
    """
    return path.replace("real_robot", "dataset").replace("simulated", "dataset")


# ------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------

class TCMMWMDataset(Dataset):
    """
    Multimodal dataset for TC-MMWM.

    Each item corresponds to a single timestep:
        - Image
        - Sensor vector
        - Language instruction
        - Action
        - Next-state metadata
    """

    def __init__(
        self,
        root_dir: str,
        split_file: str,
        augmentations: MultimodalAugmentation = None,
        anonymous: bool = False,
        max_timesteps: int = None,
    ):
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.anonymous = anonymous
        self.max_timesteps = max_timesteps

        self.vision = VisionPreprocessor()
        self.language = LanguagePreprocessor()
        self.sensors = SensorPreprocessor()

        self.trajectory_ids = load_split_file(split_file)
        self.samples = self._index_dataset()

        self._log_statistics()

    # --------------------------------------------------------

    def _index_dataset(self) -> List[Dict]:
        """
        Index all samples across trajectories.
        """
        samples = []

        for traj_id in self.trajectory_ids:
            traj_path = os.path.join(self.root_dir, traj_id)
            meta_path = os.path.join(traj_path, "metadata.json")

            if not os.path.exists(meta_path):
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            timesteps = meta["timesteps"]
            if self.max_timesteps:
                timesteps = timesteps[: self.max_timesteps]

            for t in timesteps:
                sample = {
                    "trajectory": traj_id,
                    "timestep": t["t"],
                    "image_path": os.path.join(traj_path, t["image"]),
                    "sensor_path": os.path.join(traj_path, t["sensors"]),
                    "language": meta["language"],
                    "action": t["action"],
                    "done": t.get("done", False),
                }

                if self.anonymous:
                    sample["image_path"] = anonymize_path(sample["image_path"])
                    sample["sensor_path"] = anonymize_path(sample["sensor_path"])

                samples.append(sample)

        return samples

    # --------------------------------------------------------

    def _log_statistics(self):
        """
        Compute and log dataset statistics.
        """
        num_samples = len(self.samples)
        num_trajectories = len(set(s["trajectory"] for s in self.samples))
        avg_len = num_samples / max(num_trajectories, 1)

        print("==== TC-MMWM Dataset Statistics ====")
        print(f"Number of trajectories: {num_trajectories}")
        print(f"Total samples: {num_samples}")
        print(f"Average trajectory length: {avg_len:.2f}")
        print("===================================")

    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = self.vision(sample["image_path"])
        sensors = self.sensors(sample["sensor_path"])
        language_tokens = self.language(sample["language"])

        data = {
            "image": image,
            "sensors": sensors,
            "language": language_tokens,
            "action": torch.tensor(sample["action"], dtype=torch.float32),
            "done": torch.tensor(sample["done"], dtype=torch.bool),
        }

        if self.augmentations:
            data = self.augmentations(data)

        return data


# ------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------

def build_dataloader(
    root_dir: str,
    split_file: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    augmentations: MultimodalAugmentation = None,
    anonymous: bool = False,
):
    """
    Construct PyTorch DataLoader for TC-MMWM.
    """

    dataset = TCMMWMDataset(
        root_dir=root_dir,
        split_file=split_file,
        augmentations=augmentations,
        anonymous=anonymous,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Running dataset build sanity check...")

    dummy_root = "data/simulated"
    dummy_split = "data/splits/train.txt"

    loader = build_dataloader(
        root_dir=dummy_root,
        split_file=dummy_split,
        batch_size=2,
        anonymous=True,
    )

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())

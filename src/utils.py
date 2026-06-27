from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


LABEL_TO_INDEX = {"cat": 0, "dog": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass
class TrainingConfig:
    train_dir: str
    test_dir: str
    output_dir: str
    artifact_dir: str
    model_name: str
    image_size: int
    batch_size: int
    epochs: int
    num_workers: int
    learning_rate_head: float
    learning_rate_backbone: float
    weight_decay: float
    scheduler: str
    min_learning_rate: float
    train_split: float
    random_seed: int
    early_stopping_patience: int
    label_smoothing: float
    freeze_epochs: int
    tta_passes: int
    use_mixed_precision: bool
    enterprise_threshold: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(**data)


def load_config(config_path: str | os.PathLike[str]) -> TrainingConfig:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return TrainingConfig.from_dict(data)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
    config: TrainingConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": asdict(config),
        "label_to_index": LABEL_TO_INDEX,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike[str], map_location: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils import LABEL_TO_INDEX


@dataclass
class Sample:
    image_path: Path
    label: int
    image_id: str


class DogCatDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "image_id": sample.image_id,
            "path": str(sample.image_path),
        }


class DogCatInferenceDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform=None) -> None:
        self.image_paths = sorted(image_paths, key=lambda path: int(path.stem))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": image_path.stem,
            "path": str(image_path),
        }


def build_train_valid_samples(
    train_dir: str,
    train_split: float,
    random_seed: int,
) -> tuple[list[Sample], list[Sample]]:
    image_paths = sorted(Path(train_dir).glob("*.jpg"))
    samples: list[Sample] = []
    labels: list[int] = []

    for image_path in image_paths:
        label_name = image_path.stem.split(".")[0]
        label = LABEL_TO_INDEX[label_name]
        samples.append(Sample(image_path=image_path, label=label, image_id=image_path.stem))
        labels.append(label)

    train_samples, valid_samples = train_test_split(
        samples,
        train_size=train_split,
        random_state=random_seed,
        stratify=labels,
    )
    return train_samples, valid_samples


def list_test_images(test_dir: str) -> list[Path]:
    return sorted(Path(test_dir).glob("*.jpg"), key=lambda path: int(path.stem))

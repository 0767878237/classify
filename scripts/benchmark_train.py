from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import DogCatDataset, build_train_valid_samples
from src.model import build_model, freeze_backbone
from src.transforms import build_eval_transforms, build_fast_train_transforms, build_train_transforms
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training bottlenecks.")
    parser.add_argument("--config", type=str, default="configs/resnet18_cpu_fast.yaml")
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--valid-batches", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    return parser.parse_args()


def build_train_transform(image_size: int, use_fast_transforms: bool):
    if use_fast_transforms:
        return build_fast_train_transforms(image_size)
    return build_train_transforms(image_size)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.cpu_threads is not None:
        config.cpu_num_threads = args.cpu_threads

    if config.cpu_num_threads > 0 and not torch.cuda.is_available():
        torch.set_num_threads(config.cpu_num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_start = time.perf_counter()
    train_samples, valid_samples = build_train_valid_samples(
        train_dir=config.train_dir,
        train_split=config.train_split,
        random_seed=config.random_seed,
    )
    split_sec = time.perf_counter() - split_start

    train_dataset = DogCatDataset(
        train_samples,
        transform=build_train_transform(config.image_size, config.use_fast_transforms),
    )
    valid_dataset = DogCatDataset(
        valid_samples,
        transform=build_eval_transforms(config.image_size),
    )

    loader_start = time.perf_counter()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
    )
    loader_sec = time.perf_counter() - loader_start

    first_batch_start = time.perf_counter()
    first_batch = next(iter(train_loader))
    first_batch_sec = time.perf_counter() - first_batch_start

    model_start = time.perf_counter()
    model = build_model(config.model_name).to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=config.learning_rate_head,
        weight_decay=config.weight_decay,
    )
    model_sec = time.perf_counter() - model_start

    train_start = time.perf_counter()
    train_batches_ran = 0
    for train_batches_ran, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if train_batches_ran >= args.train_batches:
            break
    train_sec = time.perf_counter() - train_start

    valid_start = time.perf_counter()
    valid_batches_ran = 0
    with torch.no_grad():
        for valid_batches_ran, batch in enumerate(valid_loader, start=1):
            images = batch["image"].to(device)
            _ = model(images)
            if valid_batches_ran >= args.valid_batches:
                break
    valid_sec = time.perf_counter() - valid_start

    result = {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "num_workers": config.num_workers,
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "sample_split_sec": round(split_sec, 3),
        "loader_build_sec": round(loader_sec, 3),
        "first_batch_sec": round(first_batch_sec, 3),
        "model_build_sec": round(model_sec, 3),
        "train_batches_ran": train_batches_ran,
        "train_total_sec": round(train_sec, 3),
        "train_sec_per_batch": round(train_sec / max(train_batches_ran, 1), 3),
        "valid_batches_ran": valid_batches_ran,
        "valid_total_sec": round(valid_sec, 3),
        "valid_sec_per_batch": round(valid_sec / max(valid_batches_ran, 1), 3),
        "rough_train_epoch_min": round((len(train_loader) * (train_sec / max(train_batches_ran, 1))) / 60, 2),
        "rough_valid_epoch_min": round((len(valid_loader) * (valid_sec / max(valid_batches_ran, 1))) / 60, 2),
        "first_batch_keys": sorted(first_batch.keys()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

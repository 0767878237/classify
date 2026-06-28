from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.datasets import DogCatDataset, build_train_valid_samples
from src.model import build_model, freeze_backbone, unfreeze_model
from src.transforms import build_eval_transforms, build_fast_train_transforms, build_train_transforms
from src.utils import (
    INDEX_TO_LABEL,
    TrainingConfig,
    ensure_dir,
    get_device,
    load_config,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dog vs Cat classifier.")
    parser.add_argument("--config", type=str, default="configs/resnet34_accuracy.yaml")
    return parser.parse_args()


def create_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    train_samples, valid_samples = build_train_valid_samples(
        train_dir=config.train_dir,
        train_split=config.train_split,
        random_seed=config.random_seed,
    )

    train_transform = (
        build_fast_train_transforms(config.image_size)
        if config.use_fast_transforms
        else build_train_transforms(config.image_size)
    )
    train_dataset = DogCatDataset(train_samples, transform=train_transform)
    valid_dataset = DogCatDataset(valid_samples, transform=build_eval_transforms(config.image_size))

    pin_memory = torch.cuda.is_available()
    persistent_workers = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, valid_loader


def build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return AdamW(
        [
            {"params": backbone_params, "lr": config.learning_rate_backbone},
            {"params": head_params, "lr": config.learning_rate_head},
        ],
        weight_decay=config.weight_decay,
    )


def create_scheduler(optimizer: AdamW, config: TrainingConfig) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_learning_rate)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    collect_metrics: bool = True,
) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    model.train(is_train)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        if collect_metrics:
            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    return average_loss, all_labels, all_predictions


def calculate_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.random_seed)
    if config.cpu_num_threads > 0 and not torch.cuda.is_available():
        torch.set_num_threads(config.cpu_num_threads)
    device = get_device()
    output_dir = ensure_dir(config.output_dir)
    artifact_dir = ensure_dir(config.artifact_dir)

    train_loader, valid_loader = create_dataloaders(config)
    model = build_model(config.model_name).to(device)
    freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and config.use_mixed_precision else None

    history: list[dict[str, float]] = []
    best_score = -math.inf
    patience_counter = 0
    best_checkpoint_path = artifact_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        if config.fine_tune_backbone and epoch == config.freeze_epochs + 1:
            unfreeze_model(model)
            optimizer = build_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config)

        train_loss, train_labels, train_predictions = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            collect_metrics=config.compute_train_metrics,
        )
        scheduler.step()

        if train_labels and train_predictions:
            train_metrics = calculate_metrics(train_labels, train_predictions)
        else:
            train_metrics = {"accuracy": 0.0, "f1": 0.0}

        should_validate = epoch == 1 or epoch % config.validate_every == 0 or epoch == config.epochs
        if should_validate:
            valid_loss, valid_labels, valid_predictions = run_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                collect_metrics=True,
            )
            valid_metrics = calculate_metrics(valid_labels, valid_predictions)
            valid_score = float(valid_metrics["f1"])
        else:
            valid_loss = float("nan")
            valid_metrics = {"accuracy": 0.0, "f1": 0.0}
            valid_score = best_score

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(valid_loss, 6) if not math.isnan(valid_loss) else "",
            "train_accuracy": round(float(train_metrics["accuracy"]), 6),
            "valid_accuracy": round(float(valid_metrics["accuracy"]), 6),
            "train_f1": round(float(train_metrics["f1"]), 6),
            "valid_f1": round(float(valid_metrics["f1"]), 6),
        }
        history.append(epoch_summary)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss if math.isnan(valid_loss) else f'{valid_loss:.4f}'} "
            f"train_f1={train_metrics['f1']:.4f} valid_f1={valid_metrics['f1']:.4f} "
            f"device={device.type}"
        )

        if should_validate and valid_score > best_score:
            best_score = valid_score
            patience_counter = 0
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                config=config,
            )
            save_json(
                {
                    "epoch": epoch,
                    "device": device.type,
                    "best_score": best_score,
                    "metrics": valid_metrics,
                    "labels": INDEX_TO_LABEL,
                },
                artifact_dir / "best_metrics.json",
            )
        elif should_validate:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    if history:
        write_history_csv(history, output_dir / "training_history.csv")

    print(f"Best checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    main()

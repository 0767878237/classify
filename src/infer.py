from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import DogCatInferenceDataset, list_test_images
from src.model import build_model
from src.transforms import build_eval_transforms
from src.utils import INDEX_TO_LABEL, ensure_dir, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Dog vs Cat classifier.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best.pt")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def predict_with_tta(model, images, device, tta_passes: int) -> torch.Tensor:
    if tta_passes <= 1:
        return torch.softmax(model(images.to(device)), dim=1)

    images = images.to(device)
    probabilities = [torch.softmax(model(images), dim=1)]

    # Lightweight TTA that works directly on tensors and stays friendly to CPU inference.
    flipped = torch.flip(images, dims=[3])
    probabilities.append(torch.softmax(model(flipped), dim=1))

    for _ in range(max(0, tta_passes - 2)):
        probabilities.append(torch.softmax(model(images), dim=1))

    return torch.mean(torch.stack(probabilities), dim=0)


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    output_dir = ensure_dir(args.output_dir)
    model = build_model(config["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = DogCatInferenceDataset(
        list_test_images(args.test_dir),
        transform=build_eval_transforms(config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    submission_path = Path(output_dir) / "submission.csv"
    enterprise_path = Path(output_dir) / "predictions.jsonl"

    with open(submission_path, "w", newline="", encoding="utf-8") as submission_file, open(
        enterprise_path, "w", encoding="utf-8"
    ) as enterprise_file:
        submission_writer = csv.writer(submission_file)
        submission_writer.writerow(["id", "label"])

        for batch in loader:
            images = batch["image"]
            image_ids = batch["image_id"]
            paths = batch["path"]

            with torch.no_grad():
                probabilities = predict_with_tta(model, images, device, config.get("tta_passes", 1))

            dog_scores = probabilities[:, 1].detach().cpu().tolist()

            for image_id, image_path, dog_score in zip(image_ids, paths, dog_scores):
                submission_writer.writerow([image_id, f"{dog_score:.6f}"])

                predicted_index = 1 if dog_score >= 0.5 else 0
                confidence = dog_score if predicted_index == 1 else 1.0 - dog_score
                record = {
                    "image_id": image_id,
                    "file_path": image_path,
                    "predicted_label": INDEX_TO_LABEL[predicted_index],
                    "confidence": round(confidence, 6),
                    "dog_probability": round(dog_score, 6),
                    "cat_probability": round(1.0 - dog_score, 6),
                    "review_recommended": confidence < config["enterprise_threshold"],
                }
                enterprise_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved competition output to: {submission_path}")
    print(f"Saved enterprise output to: {enterprise_path}")


if __name__ == "__main__":
    main()

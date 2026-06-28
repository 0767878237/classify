from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.model import build_model
from src.transforms import build_eval_transforms
from src.utils import INDEX_TO_LABEL, load_checkpoint


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    dog_probability: float
    cat_probability: float
    review_recommended: bool


class DogCatPredictor:
    def __init__(self, checkpoint_path: str | Path = "artifacts/best.pt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.config: dict[str, Any] = checkpoint["config"]

        self.model = build_model(self.config["model_name"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_eval_transforms(self.config["image_size"])
        self.enterprise_threshold = float(self.config.get("enterprise_threshold", 0.75))

    def predict_pil(self, image: Image.Image) -> PredictionResult:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)

        dog_probability = float(probabilities[0, 1].item())
        cat_probability = float(probabilities[0, 0].item())
        predicted_index = 1 if dog_probability >= 0.5 else 0
        confidence = dog_probability if predicted_index == 1 else cat_probability

        return PredictionResult(
            predicted_label=INDEX_TO_LABEL[predicted_index],
            confidence=round(confidence, 6),
            dog_probability=round(dog_probability, 6),
            cat_probability=round(cat_probability, 6),
            review_recommended=confidence < self.enterprise_threshold,
        )

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(image)

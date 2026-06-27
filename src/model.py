from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc")


def unfreeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

"""Construct the configured forecasting backbone."""
from __future__ import annotations

from torch import nn

from utils.config import ModelConfig

from .SimVPv2 import SimVPv2
from .UNetStandard import UNetStandard


def model_parameter_count(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "unet_standard":
        model = UNetStandard(config)
        count = model_parameter_count(model)["total"]
        if not config.parameter_min <= count <= config.parameter_max:
            raise RuntimeError(f"U-Net parameter count {count:,} outside [{config.parameter_min:,}, {config.parameter_max:,}]")
        return model
    if config.architecture == "simvpv2":
        return SimVPv2(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")

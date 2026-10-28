"""Optimizer and learning rate schedule."""
from __future__ import annotations

import math
import torch
from torch.optim import AdamW

from utils.config import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.peak_learning_rate,
        weight_decay=config.weight_decay,
    )


def compute_lr_factor(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    peak_lr: float,
    final_lr: float,
) -> float:
    if epoch <= warmup_epochs:
        return epoch / max(warmup_epochs, 1)
    decay_ratio = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    min_ratio = final_lr / peak_lr
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * decay_ratio))

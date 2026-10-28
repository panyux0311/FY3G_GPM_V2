"""PyTorch target normalization and decoding transforms for GPM surface precipitation."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
from torch import nn

from schema import FORECAST_LEADS, TARGET_LEVELS

EXCEEDANCE_THRESHOLDS_MM_H = (5.0, 10.0, 20.0)


def intensity_edges() -> np.ndarray:
    return np.asarray([0.0, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 300.0001], np.float32)


def threshold_positive_counts(statistics: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        statistics.get("threshold_positive_count", np.zeros((FORECAST_LEADS, TARGET_LEVELS, 3))),
        dtype=np.int64,
    )


class StrongRainTargetTransform(nn.Module):
    def __init__(
        self,
        statistics: dict[str, Any] | None = None,
        enabled: bool = True,
        level_scales: object = None,
    ) -> None:
        super().__init__()
        default_mean = np.zeros((FORECAST_LEADS, TARGET_LEVELS), np.float32)
        default_std = np.ones((FORECAST_LEADS, TARGET_LEVELS), np.float32)
        statistics = statistics or {}
        mean = torch.as_tensor(statistics.get("rain_log_mean", default_mean), dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(statistics.get("rain_log_std", default_std), dtype=torch.float32).reshape(-1).clamp_min(1e-3)
        q90 = torch.as_tensor(statistics.get("q90_mm_h", np.ones_like(default_mean)), dtype=torch.float32).reshape(-1).clamp_min(0.1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("q90", q90)
        self.enabled = bool(enabled)

    @property
    def targets(self) -> int:
        return int(self.mean.numel())

    def _flat(self, value: torch.Tensor) -> torch.Tensor:
        return value.float().reshape(value.shape[0], self.targets, *value.shape[-2:])

    def normalize_layer(self, amount: torch.Tensor) -> torch.Tensor:
        value = torch.log1p(amount.float().clamp(0.0, 300.0))
        flat = self._flat(value)
        if self.enabled:
            flat = (flat - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        return flat.reshape_as(value)

    def decode_layer(self, normalized: torch.Tensor) -> torch.Tensor:
        flat = self._flat(normalized)
        if self.enabled:
            flat = flat * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)
        return flat.clamp(0.0, math.log1p(300.0)).reshape_as(normalized)

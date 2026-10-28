"""The same final precipitation gate for models, grid metrics, and plots."""
from __future__ import annotations

import numpy as np


def gate_rain_numpy(probability, amount, threshold, mode: str = "hard"):
    if mode == "expected_log":
        amount = np.expm1(probability * np.log1p(np.maximum(amount, 0)))
    elif mode != "hard":
        raise ValueError(f"Unknown rain gate mode: {mode}")
    return np.where(probability >= threshold, amount, 0.0)


def gate_rain_torch(probability, amount, threshold, mode: str = "hard"):
    import torch
    if mode == "expected_log":
        amount = torch.expm1(probability * torch.log1p(amount.clamp_min(0)))
    elif mode != "hard":
        raise ValueError(f"Unknown rain gate mode: {mode}")
    return torch.where(probability >= threshold, amount, torch.zeros_like(amount))

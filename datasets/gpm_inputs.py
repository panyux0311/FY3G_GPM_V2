"""Assemble 43-channel model input from batch tensors."""
from __future__ import annotations

import torch


def assemble_model_input(
    batch: dict[str, torch.Tensor],
    reflectivity_mean: torch.Tensor | None = None,
    reflectivity_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize reflectivity and assemble 43-channel input:
    - 20 normalized reflectivity channels (2 bands × 10 levels)
    - 20 binary valid mask channels
    - 2 normalized lat/lon coordinate channels
    - 1 orbit direction channel
    """
    reflectivity = batch["x_reflectivity"].float()
    valid = batch["x_valid"].bool()
    if reflectivity_mean is not None and reflectivity_std is not None:
        mean = reflectivity_mean.to(reflectivity.device).view(1, 1, 2, 10, 1, 1)
        std = reflectivity_std.to(reflectivity.device).view(1, 1, 2, 10, 1, 1).clamp_min(1e-6)
        reflectivity = torch.where(valid, (reflectivity - mean) / std, torch.zeros_like(reflectivity))

    geo = batch["x_geo"].float()
    direction = batch["x_direction"].float().unsqueeze(2)
    return torch.cat((
        reflectivity.reshape(reflectivity.shape[0], 3, 20, 64, 64),
        valid.float().reshape(valid.shape[0], 3, 20, 64, 64),
        geo,
        direction,
    ), dim=2)

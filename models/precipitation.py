"""Shared precipitation heads and output contract for forecasting backbones."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from utils.config import ModelConfig
from utils.target_transform import EXCEEDANCE_THRESHOLDS_MM_H, StrongRainTargetTransform, threshold_positive_counts
from utils.rain_gate import gate_rain_torch


class PrecipitationModel(nn.Module):
    def __init__(self, config: ModelConfig, decoded_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.input_channels = int(config.input_channels)
        self.input_steps = int(config.input_steps)
        self.forecast_leads = int(config.forecast_leads)
        self.target_levels = int(config.target_levels)
        self.output_targets = int(config.output_targets)
        self.multi_lead = config.simvpv2_multi_lead
        self.rain_gate_mode = config.rain_gate_mode
        self.patch_blend_mode = config.patch_blend_mode
        self.patch_blend_min_weight = config.patch_blend_min_weight
        if self.output_targets != self.forecast_leads * self.target_levels:
            raise ValueError("output_targets must equal forecast_leads × target_levels")

        self.distillation_projection = nn.Conv2d(feature_channels, config.distillation_channels, 1)
        self.occurrence_head = nn.Conv2d(decoded_channels, self.output_targets, 1)
        self.amount_head = nn.Conv2d(decoded_channels, self.output_targets, 1)
        self.exceedance_head = (
            nn.Conv2d(decoded_channels, self.output_targets * 3, 1)
            if config.exceedance_head_enabled else None
        )
        nn.init.zeros_(self.occurrence_head.weight)
        nn.init.constant_(self.occurrence_head.bias, -2.0)
        nn.init.zeros_(self.amount_head.weight)
        nn.init.zeros_(self.amount_head.bias)

        self.register_buffer("occurrence_thresholds", torch.full((self.output_targets,), 0.5))
        self.register_buffer("occurrence_thresholds_by_zone", torch.full((self.forecast_leads, self.target_levels, 3), 0.5))
        self.target_transform = StrongRainTargetTransform(enabled=False)

    def configure_target_transform(
        self,
        statistics: dict[str, Any] | None,
        enabled: bool = True,
        level_scales: object = None,
    ) -> None:
        transform = StrongRainTargetTransform(statistics, enabled=enabled).to(self.amount_head.bias.device)
        if transform.targets != self.output_targets:
            raise ValueError(f"target statistics have {transform.targets} channels, expected {self.output_targets}")
        self.target_transform = transform
        if statistics:
            with torch.no_grad():
                self.amount_head.bias.copy_((-transform.mean / transform.std).to(self.amount_head.bias))
                if self.exceedance_head is not None:
                    positive = torch.as_tensor(threshold_positive_counts(statistics), dtype=torch.float32)
                    valid = torch.as_tensor(statistics["valid_count"], dtype=torch.float32).unsqueeze(-1).clamp_min(1)
                    prior = (positive / valid).clamp(1e-5, 1 - 1e-5)
                    self.exceedance_head.bias.copy_(torch.logit(prior).reshape(-1).to(self.exceedance_head.bias))

    def initialize_occurrence_bias(
        self,
        probabilities: torch.Tensor | list[float],
        minimum_logit: float = -8.0,
        maximum_logit: float = -2.0,
    ) -> None:
        value = torch.as_tensor(probabilities, dtype=self.occurrence_head.bias.dtype)
        if value.numel() != self.output_targets:
            raise ValueError("incorrect occurrence-bias length")
        with torch.no_grad():
            self.occurrence_head.bias.copy_(
                torch.logit(value.reshape(-1).clamp(1e-7, 1 - 1e-7)).clamp(minimum_logit, maximum_logit)
            )

    def set_occurrence_thresholds(self, thresholds: torch.Tensor | list[float]) -> None:
        value = torch.as_tensor(thresholds, dtype=self.occurrence_thresholds.dtype, device=self.occurrence_thresholds.device).reshape(-1)
        if value.numel() != self.output_targets:
            raise ValueError("incorrect occurrence threshold length")
        self.occurrence_thresholds.copy_(value)
        self.occurrence_thresholds_by_zone.copy_(value.reshape(self.forecast_leads, self.target_levels, 1).expand(-1, -1, 3))

    def set_occurrence_thresholds_by_zone(self, thresholds: torch.Tensor | list[list[list[float]]]) -> None:
        value = torch.as_tensor(thresholds, dtype=self.occurrence_thresholds_by_zone.dtype, device=self.occurrence_thresholds_by_zone.device)
        expected = (self.forecast_leads, self.target_levels, 3)
        if tuple(value.shape) != expected:
            raise ValueError(f"expected zone thresholds {expected}, got {tuple(value.shape)}")
        self.occurrence_thresholds_by_zone.copy_(value)
        self.occurrence_thresholds.copy_(value.mean(dim=2).reshape(-1))

    def _forecast_outputs(
        self, x: torch.Tensor, decoded: torch.Tensor, feature: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        direct = x[:, :, 20:40].bool().any(dim=(1, 2))
        near = F.max_pool2d(direct.float().unsqueeze(1), 9, stride=1, padding=4).squeeze(1).bool()
        coverage_zone = torch.where(
            direct,
            torch.zeros_like(direct, dtype=torch.long),
            torch.where(near, torch.ones_like(direct, dtype=torch.long), torch.full_like(direct, 2, dtype=torch.long)),
        )
        def project(head: nn.Conv2d) -> torch.Tensor:
            if decoded.ndim == 4:
                return head(decoded)
            batch, leads, channels, height, width = decoded.shape
            if leads != self.forecast_leads:
                raise ValueError("Decoded features must have one plane per forecast lead")
            return F.conv2d(decoded.reshape(batch, leads * channels, height, width),
                            head.weight, head.bias, groups=leads)

        logits, normalized = project(self.occurrence_head), project(self.amount_head)
        base_log = self.target_transform.decode_layer(normalized)
        amount = torch.expm1(base_log.clamp(0.0, math.log1p(300.0)))
        lookup = self.occurrence_thresholds_by_zone.reshape(self.output_targets, 3)
        threshold = lookup[:, coverage_zone].permute(1, 0, 2, 3).contiguous()
        probability = logits.sigmoid()
        prediction = gate_rain_torch(probability, amount, threshold, self.rain_gate_mode)
        shape = (x.shape[0], self.forecast_leads, self.target_levels, *x.shape[-2:])
        exceedance = (
            project(self.exceedance_head).reshape(*shape[:3], 3, *shape[-2:])
            if self.exceedance_head is not None else
            4.0 * (base_log.reshape(shape).unsqueeze(3)
                   - torch.log1p(torch.as_tensor(EXCEEDANCE_THRESHOLDS_MM_H, dtype=base_log.dtype, device=base_log.device)).view(1, 1, 1, 3, 1, 1))
        )
        if feature.ndim == 5:
            feature = feature.mean(dim=1)
        return {
            "occurrence_logits": logits.reshape(shape),
            "occurrence_probability": probability.reshape(shape),
            "amount_normalized": normalized.reshape(shape),
            "amount_log": base_log.reshape(shape),
            "prediction": prediction.reshape(shape),
            "ungated_amount": amount.reshape(shape),
            "intensity_exceedance_logits": exceedance,
            "coverage_zone": coverage_zone,
            "occurrence_threshold_map": threshold.reshape(shape),
            "feature": self.distillation_projection(feature),
        }

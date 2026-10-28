"""Masked V6 composite loss for 3-hour GPM surface precipitation."""
from __future__ import annotations

import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from utils.target_transform import EXCEEDANCE_THRESHOLDS_MM_H, StrongRainTargetTransform, threshold_positive_counts


class ThreeHourStrongRainLoss(nn.Module):
    def __init__(
        self,
        statistics: dict[str, Any],
        rain_threshold: float = 0.1,
        focal_gamma: float = 2.0,
        positive_weight_cap: float = 20.0,
        focal_weight: float = 1.0,
        soft_csi_weight: float = 0.25,
        rain_amount_weight: float = 1.0,
        dry_amount_weight: float = 0.05,
        neighborhood_fss_weight: float = 0.10,
        calibration_bce_weight: float = 0.02,
        exceedance_weight: float = 0.20,
        exceedance_threshold_weights: tuple[float, ...] = (0.4, 0.35, 0.25),
        strong_fss_weight: float = 0.0,
        power_cap: float = 2.0,
        lead_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        hard_negative_fraction: float = 0.02,
        hard_negative_weight: float = 0.02,
        zone_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        expected_amount_weight: float = 0.0,
        dry_probability_weighted: bool = False,
        exceedance_positive_weighting: bool = False,
        exceedance_positive_weight_cap: float = 25.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.transform = StrongRainTargetTransform(statistics, enabled=True)
        self.targets = self.transform.targets
        valid_count = torch.as_tensor(statistics["valid_count"], dtype=torch.float32).reshape(-1)
        rain_count = torch.as_tensor(statistics["rain_count"], dtype=torch.float32).reshape(-1)
        if valid_count.numel() != self.targets or len(lead_weights) == 0 or self.targets % len(lead_weights):
            raise ValueError("statistics / lead weights do not match output targets")

        self.register_buffer(
            "positive_weight",
            torch.sqrt((valid_count - rain_count).clamp_min(1) / rain_count.clamp_min(1)).clamp(1, positive_weight_cap),
        )
        self.register_buffer("supported", rain_count >= 100)
        self.register_buffer("edges", torch.as_tensor(statistics["intensity_edges_mm_h"], dtype=torch.float32))
        self.register_buffer("group_weight", torch.as_tensor(statistics["group_weight"], dtype=torch.float32).reshape(self.targets, -1))
        self.register_buffer("strong_supported", torch.as_tensor(threshold_positive_counts(statistics), dtype=torch.bool).reshape(self.targets, 3))
        positive = torch.as_tensor(threshold_positive_counts(statistics), dtype=torch.float32).reshape(self.targets, 3)
        exceed_weight = torch.sqrt((valid_count[:, None] - positive).clamp_min(1) / positive.clamp_min(1))
        self.register_buffer("exceed_pos_weight", exceed_weight.clamp(1, exceedance_positive_weight_cap)
                             if exceedance_positive_weighting else torch.ones_like(exceed_weight))
        self.register_buffer("zone_loss_weights", torch.as_tensor(zone_loss_weights, dtype=torch.float32))
        self.expected_amount_weight = float(expected_amount_weight)
        self.dry_probability_weighted = bool(dry_probability_weighted)
        self.register_buffer("exceedance_thresholds", torch.as_tensor(EXCEEDANCE_THRESHOLDS_MM_H, dtype=torch.float32))
        weights = torch.as_tensor(lead_weights, dtype=torch.float32)
        self.register_buffer("channel_weight", (weights / weights.mean()).repeat_interleave(self.targets // len(weights)))
        threshold_weights = torch.as_tensor(exceedance_threshold_weights, dtype=torch.float32)
        self.register_buffer("exceedance_weight_by_threshold", threshold_weights / threshold_weights.sum())

        self.rain_threshold = float(rain_threshold)
        self.focal_gamma = float(focal_gamma)
        self.focal_weight = float(focal_weight)
        self.soft_csi_weight = float(soft_csi_weight)
        self.rain_amount_weight = float(rain_amount_weight)
        self.dry_amount_weight = float(dry_amount_weight)
        self.neighborhood_fss_weight = float(neighborhood_fss_weight)
        self.calibration_bce_weight = float(calibration_bce_weight)
        self.exceedance_weight = float(exceedance_weight)
        self.strong_fss_weight = float(strong_fss_weight)
        self.power_cap = float(power_cap)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hard_negative_weight = float(hard_negative_weight)

    def _flat(self, value: torch.Tensor) -> torch.Tensor:
        return value.float().reshape(value.shape[0], self.targets, *value.shape[-2:])

    def _mean(self, value: torch.Tensor, mask: torch.Tensor, spatial_weight: torch.Tensor | None = None) -> torch.Tensor:
        weight = mask.float() * self.channel_weight.view(1, -1, 1, 1)
        total = weight.sum((0, 2, 3))
        # Normalize by valid counts, not zone-weight sums: an all-uncovered
        # batch must still have its gradients attenuated by the configured weight.
        if spatial_weight is not None:
            weight = weight * spatial_weight
        terms = (value * weight).sum((0, 2, 3)) / total.clamp_min(1)
        active = total > 0
        channel = self.channel_weight * active
        return (terms * channel).sum() / channel.sum().clamp_min(1)

    def _channel_mean(self, value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        weight = self.channel_weight * active.float()
        return (value * weight).sum() / weight.sum().clamp_min(1)

    @staticmethod
    def _pool(value: torch.Tensor, valid: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
        if scale == 1:
            return value, valid
        fraction = F.avg_pool2d(valid.float(), scale, stride=scale)
        pooled = F.avg_pool2d(value * valid.float(), scale, stride=scale) / fraction.clamp_min(1e-6)
        return pooled, fraction >= 0.5

    def _soft_csi(self, probability: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
                  spatial_weight: torch.Tensor | None = None) -> torch.Tensor:
        terms = []
        for scale, weight in ((1, 0.60), (2, 0.25), (4, 0.15)):
            prediction, mask = self._pool(probability, valid, scale)
            observed, _ = self._pool(truth.float(), valid, scale)
            maskf = mask.float()
            attenuation = 1.0
            if spatial_weight is not None:
                pooled_weight, _ = self._pool(spatial_weight.expand_as(probability), valid, scale)
                weighted_mask = maskf * pooled_weight
                attenuation = weighted_mask.sum((0, 2, 3)) / maskf.sum((0, 2, 3)).clamp_min(1)
                maskf = weighted_mask
            intersection = (prediction * observed * maskf).sum((0, 2, 3))
            union = ((prediction + observed - prediction * observed) * maskf).sum((0, 2, 3))
            active = mask.any((0, 2, 3)) & self.supported
            terms.append(weight * self._channel_mean((1.0 - (intersection + 1) / (union + 1)) * attenuation, active))
        return torch.stack(terms).sum()

    def _fss(self, probability: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
             spatial_weight: torch.Tensor | None = None) -> torch.Tensor:
        terms = []
        for window in (3, 5):
            fraction = F.avg_pool2d(valid.float(), window, 1, window // 2)
            mask = fraction > 0
            prediction = F.avg_pool2d(probability * valid.float(), window, 1, window // 2) / fraction.clamp_min(1e-6)
            observed = F.avg_pool2d(truth.float() * valid.float(), window, 1, window // 2) / fraction.clamp_min(1e-6)
            error = (prediction - observed).square()
            reference = prediction.square() + observed.square()
            terms.append(self._mean(error / reference.clamp_min(1e-6), mask & self.supported.view(1, -1, 1, 1), spatial_weight))
        return torch.stack(terms).mean()

    def _regression_weights(self, target: torch.Tensor, rainy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = torch.bucketize(target.contiguous(), self.edges[1:-1], right=True).clamp(0, self.group_weight.shape[1] - 1)
        lookup = self.group_weight.view(1, self.targets, -1, 1, 1).expand(target.shape[0], -1, -1, *target.shape[-2:])
        weight = torch.gather(lookup, 2, groups.unsqueeze(2)).squeeze(2)
        active = rainy & self.supported.view(1, -1, 1, 1)
        weight = weight * (1 + (target / self.transform.q90.view(1, -1, 1, 1)).square()).clamp_max(self.power_cap)
        # Adjusted clamp: allow computed weights down to 0.1, up to 10.0
        return torch.where(active, weight.clamp(0.1, 10.0), torch.zeros_like(weight)), active

    def _hard_negative(self, logits: torch.Tensor, rainy: torch.Tensor, valid: torch.Tensor,
                       spatial_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dry = valid & ~rainy
        bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
        if spatial_weight is not None:
            bce = bce * spatial_weight
        selected = torch.zeros(self.targets, device=logits.device, dtype=torch.int64)
        available = dry.sum((0, 2, 3))
        terms = []
        for channel in range(self.targets):
            count = int(available[channel].item())
            if count:
                k = max(1, math.ceil(self.hard_negative_fraction * count))
                selected[channel] = k
                terms.append(bce[:, channel][dry[:, channel]].topk(k).values.mean())
        return (torch.stack(terms).mean() if terms else logits.sum() * 0), selected, available

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self._flat(outputs["occurrence_logits"])
        normalized = self._flat(outputs["amount_normalized"])
        amount_log = self._flat(outputs["amount_log"])
        target = self._flat(target).clamp(0.0, 300.0)
        valid = self._flat(valid).bool()
        rainy = valid & (target >= self.rain_threshold)
        probability = logits.sigmoid()
        spatial_weight = None
        if not torch.all(self.zone_loss_weights == 1):
            if "coverage_zone" not in outputs:
                raise ValueError("Zone-aware loss requires outputs['coverage_zone']")
            zone = outputs["coverage_zone"].long()
            spatial_weight = self.zone_loss_weights[zone].unsqueeze(1)

        bce = F.binary_cross_entropy_with_logits(logits, rainy.float(), pos_weight=self.positive_weight.view(1, -1, 1, 1), reduction="none")
        focal = (1 - torch.where(rainy, probability, 1 - probability)).pow(self.focal_gamma) * bce
        focal_loss = self._mean(focal, valid, spatial_weight)
        csi_loss = self._soft_csi(probability, rainy, valid, spatial_weight)
        fss_loss = self._fss(probability, rainy, valid, spatial_weight)
        calibration = self._mean(F.binary_cross_entropy_with_logits(logits, rainy.float(), reduction="none"), valid, spatial_weight)

        if self.hard_negative_weight:
            hard_loss, hard_selected, hard_available = self._hard_negative(logits, rainy, valid, spatial_weight)
        else:
            hard_loss = logits.sum() * 0
            hard_selected = torch.zeros(self.targets, device=logits.device, dtype=torch.int64)
            hard_available = (valid & ~rainy).sum((0, 2, 3))

        occurrence = (
            self.focal_weight * focal_loss
            + self.soft_csi_weight * csi_loss
            + self.neighborhood_fss_weight * fss_loss
            + self.calibration_bce_weight * calibration
            + self.hard_negative_weight * hard_loss
        )

        target_normalized = self._flat(self.transform.normalize_layer(target.reshape_as(outputs["amount_log"])))
        regression = F.smooth_l1_loss(normalized, target_normalized, reduction="none")
        weight, rain_mask = self._regression_weights(target, rainy)
        rain_amount = self._mean(regression * weight, rain_mask, spatial_weight)
        dry_penalty = amount_log.clamp_min(0).square()
        if self.dry_probability_weighted:
            dry_penalty = probability.detach() * dry_penalty
        dry = self._mean(dry_penalty, valid & ~rainy, spatial_weight)
        amount = self.rain_amount_weight * rain_amount + self.dry_amount_weight * dry
        expected = self._mean(F.smooth_l1_loss(probability * amount_log, torch.log1p(target), reduction="none"),
                              valid, spatial_weight) if self.expected_amount_weight else logits.sum() * 0

        exceed_logits = outputs["intensity_exceedance_logits"].float().reshape(target.shape[0], self.targets, 3, *target.shape[-2:])
        exceed_truth = target.unsqueeze(2) >= self.exceedance_thresholds.view(1, 1, 3, 1, 1)
        exceed_mask = valid.unsqueeze(2) & self.strong_supported.view(1, self.targets, 3, 1, 1)
        exceed_bce = F.binary_cross_entropy_with_logits(
            exceed_logits, exceed_truth.float(), reduction="none",
            pos_weight=self.exceed_pos_weight.view(1, self.targets, 3, 1, 1))
        exceed = sum(self.exceedance_weight_by_threshold[i] * self._mean(exceed_bce[:, :, i], exceed_mask[:, :, i], spatial_weight) for i in range(3))

        total = occurrence + amount + self.exceedance_weight * exceed + self.expected_amount_weight * expected
        return {
            "loss": total,
            "occurrence_loss": occurrence,
            "amount_loss": amount,
            "focal_loss": focal_loss,
            "soft_csi_loss": csi_loss,
            "neighborhood_fss_loss": fss_loss,
            "calibration_bce_loss": calibration,
            "rain_amount_loss": rain_amount,
            "dry_amount_loss": dry,
            "exceedance_loss": exceed,
            "expected_amount_loss": expected,
            "hard_negative_loss": hard_loss,
            "hard_negative_selected": hard_selected.sum().detach(),
            "hard_negative_available": hard_available.sum().detach(),
            "valid_voxels": valid.sum().detach(),
            "rainy_voxels": rainy.sum().detach(),
        }


def build_loss(statistics: dict[str, Any], config: Any) -> ThreeHourStrongRainLoss:
    """Use identical configured losses in training and CPU preflight."""
    return ThreeHourStrongRainLoss(
        statistics, rain_threshold=config.rain_threshold_mm_h,
        focal_gamma=config.focal_gamma, positive_weight_cap=config.positive_weight_cap,
        focal_weight=config.focal_weight, soft_csi_weight=config.soft_csi_weight,
        rain_amount_weight=config.rain_amount_weight, dry_amount_weight=config.dry_amount_weight,
        neighborhood_fss_weight=config.neighborhood_fss_weight,
        calibration_bce_weight=config.calibration_bce_weight,
        exceedance_weight=config.exceedance_weight,
        exceedance_threshold_weights=tuple(config.exceedance_threshold_weights),
        lead_weights=tuple(config.lead_weights), hard_negative_fraction=config.hard_negative_fraction,
        hard_negative_weight=0.0, zone_loss_weights=tuple(config.zone_loss_weights),
        expected_amount_weight=config.expected_amount_weight,
        dry_probability_weighted=config.dry_probability_weighted,
        exceedance_positive_weighting=config.exceedance_positive_weighting,
        exceedance_positive_weight_cap=config.exceedance_positive_weight_cap,
    )

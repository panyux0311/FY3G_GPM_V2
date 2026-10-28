from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import yaml

from schema import DATASET_VERSION, FORECAST_LEADS, GRID_DEGREES, INPUT_CHANNELS, INPUT_LEVELS_KM, INPUT_STEPS, PATCH_SIZE, TARGET_LEVELS


@dataclass
class DataConsumerConfig:
    dataset_root: str = "/lixj/pyx/dataset/FY3G_after"
    dataset_version: str = DATASET_VERSION
    grid_degrees: float = GRID_DEGREES
    input_levels_km: list[int] = field(default_factory=lambda: list(INPUT_LEVELS_KM))
    patch_size: int = PATCH_SIZE


@dataclass
class ModelConfig:
    architecture: str = "simvpv2"
    input_channels: int = INPUT_CHANNELS
    input_steps: int = INPUT_STEPS
    forecast_leads: int = FORECAST_LEADS
    target_levels: int = TARGET_LEVELS
    output_targets: int = FORECAST_LEADS * TARGET_LEVELS
    distillation_channels: int = 48
    base_width: int = 64
    strong_branch_enabled: bool = False
    strong_branch_width: int = 32
    strong_correction_log_cap: float = 1.3862943611198906
    parameter_min: int = 30_000_000
    parameter_max: int = 40_000_000
    simvpv2_hidden_spatial: int = 32
    simvpv2_hidden_temporal: int = 128
    simvpv2_spatial_layers: int = 4
    simvpv2_temporal_layers: int = 4
    simvpv2_mlp_ratio: float = 4.0
    simvpv2_drop_path: float = 0.0
    simvpv2_multi_lead: bool = False
    exceedance_head_enabled: bool = False
    rain_gate_mode: str = "hard"
    patch_blend_mode: str = "uniform"
    patch_blend_min_weight: float = 0.05


@dataclass
class TrainConfig:
    experiment_group: str = "gpm_surface_v1"
    seed: int = 14
    epochs: int = 100
    effective_batch_size: int = 64
    microbatch_size: int = 64
    num_workers: int = 2
    prefetch_factor: int = 1
    disable_tqdm: bool = False
    peak_learning_rate: float = 2.0e-4
    final_learning_rate: float = 5.0e-6
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    rain_threshold_mm_h: float = 0.1
    focal_gamma: float = 2.0
    positive_weight_cap: float = 20.0
    focal_weight: float = 1.0
    soft_csi_weight: float = 0.25
    rain_amount_weight: float = 1.0
    dry_amount_weight: float = 0.05
    neighborhood_fss_weight: float = 0.10
    calibration_bce_weight: float = 0.02
    exceedance_weight: float = 0.20
    exceedance_threshold_weights: list[float] = field(default_factory=lambda: [0.40, 0.35, 0.25])
    strong_fss_weight: float = 0.0
    lead_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    hard_negative_fraction: float = 0.02
    hard_negative_weight: float = 0.02
    hard_negative_ramp_start_epoch: int = 21
    hard_negative_ramp_end_epoch: int = 60
    ema_decay: float = 0.999
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    output_dir: str = "outputs/gpm_surface_v1"
    min_metric_positive_voxels: int = 100
    validation_far_max: float = 0.90
    threshold_min: float = 0.05
    threshold_max: float = 0.99
    threshold_step: float = 0.01
    threshold_pod_retention: float = 0.0
    quality_gate_enabled: bool = False
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    diagnostics_enabled: bool = False
    diagnostics_train_forecast_times: int = 8
    diagnostics_every_epochs: int = 1
    diagnostics_probability_threshold: float = 0.5
    diagnostics_compare_raw_ema: bool = False
    intensity_weight_strategy: str = "dataset"
    save_best_by_metrics: list[str] = field(default_factory=list)
    zone_loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    expected_amount_weight: float = 0.0
    dry_probability_weighted: bool = False
    exceedance_positive_weighting: bool = False
    exceedance_positive_weight_cap: float = 25.0
    validation_far_max_by_zone: list[float] | None = None
    primary_selection_domain: str = "macro_all"
    uncovered_patch_keep_ratio: float = 1.0


@dataclass
class ProjectConfig:
    data: DataConsumerConfig = field(default_factory=DataConsumerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "gpm_surface_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.data.dataset_version != DATASET_VERSION:
            raise ValueError(f"dataset_version must be {DATASET_VERSION}")
        if self.data.grid_degrees != GRID_DEGREES or self.data.patch_size != PATCH_SIZE:
            raise ValueError(f"dataset grid_degrees and patch_size must be {GRID_DEGREES} and {PATCH_SIZE}")
        if self.data.input_levels_km != list(INPUT_LEVELS_KM):
            raise ValueError("input_levels_km must be exactly [2,3,4,5,6,7,8,9,10,11]")
        if self.model.architecture not in ("unet_standard", "simvpv2"):
            raise ValueError("model.architecture must be unet_standard or simvpv2")
        if self.model.architecture == "simvpv2":
            if self.model.simvpv2_hidden_spatial <= 0 or self.model.simvpv2_hidden_temporal <= 0:
                raise ValueError("SimVPv2 hidden widths must be positive")
            if self.model.simvpv2_spatial_layers < 2 or self.model.simvpv2_spatial_layers % 2:
                raise ValueError("simvpv2_spatial_layers must be an even number >= 2")
            if self.model.simvpv2_temporal_layers < 2 or self.model.simvpv2_mlp_ratio <= 0:
                raise ValueError("SimVPv2 temporal layers must be >= 2 and mlp ratio positive")
            if not 0.0 <= self.model.simvpv2_drop_path < 1.0:
                raise ValueError("simvpv2_drop_path must lie in [0, 1)")
        contract = (
            self.model.input_channels,
            self.model.input_steps,
            self.model.forecast_leads,
            self.model.target_levels,
            self.model.output_targets,
        )
        if contract != (43, 3, 3, 1, 3):
            raise ValueError("model contract must be [3,43] -> [3,1,64,64]")
        if self.model.strong_branch_enabled:
            raise ValueError("strong-rain auxiliary branch remains disabled")
        if self.model.simvpv2_multi_lead and self.model.architecture != "simvpv2":
            raise ValueError("simvpv2_multi_lead requires architecture=simvpv2")
        if self.model.rain_gate_mode not in ("hard", "expected_log"):
            raise ValueError("rain_gate_mode must be hard or expected_log")
        if self.model.patch_blend_mode not in ("uniform", "hann"):
            raise ValueError("patch_blend_mode must be uniform or hann")
        if not 0 < self.model.patch_blend_min_weight <= 1:
            raise ValueError("patch_blend_min_weight must be in (0,1]")
        if len(self.train.zone_loss_weights) != 3 or any(not isfinite(v) or v < 0 for v in self.train.zone_loss_weights) or not any(self.train.zone_loss_weights):
            raise ValueError("zone_loss_weights must contain three finite nonnegative weights, with at least one positive")
        if not isfinite(self.train.expected_amount_weight) or self.train.expected_amount_weight < 0:
            raise ValueError("expected_amount_weight must be finite and nonnegative")
        if not isfinite(self.train.exceedance_positive_weight_cap) or self.train.exceedance_positive_weight_cap < 1:
            raise ValueError("exceedance_positive_weight_cap must be at least 1")
        caps = self.train.validation_far_max_by_zone
        if caps is not None and (len(caps) != 3 or any(not isfinite(v) or not 0 <= v <= 1 for v in caps)):
            raise ValueError("validation_far_max_by_zone must contain three FAR caps in [0,1]")
        if self.train.primary_selection_domain not in ("macro_all", "observable", "direct_only"):
            raise ValueError("primary_selection_domain must be macro_all, observable or direct_only")
        if not isfinite(self.train.uncovered_patch_keep_ratio) or not 0 <= self.train.uncovered_patch_keep_ratio <= 1:
            raise ValueError("uncovered_patch_keep_ratio must lie in [0,1]")
        if self.train.effective_batch_size % self.train.microbatch_size:
            raise ValueError("effective_batch_size must be divisible by microbatch_size")
        if len(self.train.lead_weights) != FORECAST_LEADS or len(self.train.exceedance_threshold_weights) != 3:
            raise ValueError("lead and exceedance weights have invalid length")
        if not (0.0 < self.train.threshold_min < self.train.threshold_max < 1.0):
            raise ValueError("threshold range must lie inside (0,1)")
        if self.train.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not isfinite(self.train.early_stopping_min_delta) or self.train.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be finite and nonnegative")
        if self.train.diagnostics_enabled and self.train.diagnostics_train_forecast_times <= 0:
            raise ValueError("diagnostics_train_forecast_times must be positive when diagnostics are enabled")
        if self.train.diagnostics_every_epochs <= 0:
            raise ValueError("diagnostics_every_epochs must be positive")
        if not 0.0 < self.train.diagnostics_probability_threshold < 1.0:
            raise ValueError("diagnostics_probability_threshold must lie inside (0,1)")
        if self.train.intensity_weight_strategy not in ("dataset", "sqrt_inverse_frequency_v1"):
            raise ValueError("Unsupported intensity_weight_strategy")
        if any(name not in ("csi_0.1", "csi_2", "csi_5") for name in self.train.save_best_by_metrics):
            raise ValueError("Unsupported save_best_by_metrics entry")
        if len(set(self.train.save_best_by_metrics)) != len(self.train.save_best_by_metrics):
            raise ValueError("Duplicate save_best_by_metrics entry")


def set_nested(config: dict[str, Any], expression: str) -> None:
    """Apply a ``section.key=value`` override to a loaded configuration dictionary."""
    if "=" not in expression:
        raise ValueError(f"--set expects key=value, got {expression!r}")
    key, raw_value = expression.split("=", 1)
    parts = [part.strip() for part in key.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid override key in {expression!r}")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    for override in overrides:
        set_nested(payload, override)
    model = ModelConfig(**payload.get("model", {}))
    train = TrainConfig(**payload.get("train", {}))
    config_name = payload.get("config", {}).get("name", "gpm_surface_v1")
    config = ProjectConfig(
        data=DataConsumerConfig(**payload.get("data", {})),
        model=model,
        train=train,
        name=config_name,
    )
    config.validate()
    return config

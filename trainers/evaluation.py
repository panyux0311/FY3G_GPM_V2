"""Deduplicated-grid evaluation and threshold optimization."""
from __future__ import annotations

from typing import Any, Callable
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.gpm_dataset import FY3G3HDataset
from datasets.gpm_inputs import assemble_model_input
from metrics.gpm_metrics import GridMetrics
from metrics.thresholds import ZoneThresholdSearch
from schema import FORECAST_LEADS, GRID_SHAPE, PATCH_SIZE, TARGET_LEVELS
from utils.config import ProjectConfig
from utils.progress import should_disable_tqdm


@lru_cache(maxsize=16)
def _patch_blend_weights(patch_size: int = 64, min_weight: float = 0.05) -> np.ndarray:
    if patch_size < 2 or not 0 < min_weight <= 1:
        raise ValueError("Blending requires patch_size >= 2 and min_weight in (0,1]")
    one = np.hanning(patch_size)
    window = np.outer(one, one)
    window = min_weight + (1 - min_weight) * window / max(float(window.max()), 1e-12)
    window.setflags(write=False)
    return window


def for_each_deduplicated_grid(
    model: torch.nn.Module,
    dataset: FY3G3HDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    callback: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None],
    *,
    progress_desc: str | None = None,
    disable_tqdm: bool = False,
    criterion: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Average overlapping patch predictions before emitting one complete time-grid."""
    model.eval()
    current = None
    state: dict[str, np.ndarray] | None = None
    loss_totals: dict[str, float] = {}
    loss_counts: dict[str, int] = {}
    blend_mode = getattr(model, "patch_blend_mode", "uniform")
    blend = (_patch_blend_weights(PATCH_SIZE, getattr(model, "patch_blend_min_weight", 0.05))
             if blend_mode == "hann" else np.ones((PATCH_SIZE, PATCH_SIZE), np.float64))

    def start() -> dict[str, np.ndarray]:
        surface_shape = (FORECAST_LEADS, TARGET_LEVELS, *GRID_SHAPE)
        return {
            "prob": np.zeros(surface_shape, np.float64),
            "amount": np.zeros(surface_shape, np.float64),
            "pred_weight": np.zeros(surface_shape, np.float64),
            "target": np.zeros(surface_shape, np.float64),
            "target_count": np.zeros(surface_shape, np.uint16),
            "zone": np.full(GRID_SHAPE, 2, np.int8),
        }

    def finish(value: dict[str, np.ndarray]) -> None:
        valid = value["target_count"] > 0
        prob = np.divide(value["prob"], value["pred_weight"], out=np.zeros_like(value["prob"]), where=value["pred_weight"] > 0)
        amount = np.divide(value["amount"], value["pred_weight"], out=np.zeros_like(value["amount"]), where=value["pred_weight"] > 0)
        target = np.divide(value["target"], value["target_count"], out=np.zeros_like(value["target"]), where=valid)
        callback(prob.astype(np.float32), amount.astype(np.float32), target.astype(np.float32), valid, value["zone"])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cursor = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=progress_desc, leave=False, disable=disable_tqdm or progress_desc is None):
            moved = {key: value.to(device) for key, value in batch.items() if key != "_sample_index"}
            outputs = model(assemble_model_input(moved, mean, std))
            if criterion is not None:
                loss_values = criterion(outputs, moved["y_amount"], moved["y_valid"])
                batch_size_actual = len(batch["y_amount"])
                for key, value in loss_values.items():
                    if (key == "loss" or key.endswith("_loss")) and isinstance(value, torch.Tensor) and value.numel() == 1:
                        loss_totals[key] = loss_totals.get(key, 0.0) + float(value.item()) * batch_size_actual
                        loss_counts[key] = loss_counts.get(key, 0) + batch_size_actual
            prob = outputs["occurrence_probability"].cpu().numpy()
            amount = outputs["ungated_amount"].cpu().numpy()
            target = batch["y_amount"].numpy()
            valid = batch["y_valid"].numpy().astype(bool)
            zone = outputs["coverage_zone"].cpu().numpy()
            for index in range(len(prob)):
                sample_index = int(batch["_sample_index"][index]) if "_sample_index" in batch else cursor
                meta = dataset.metadata(sample_index)
                cursor += 1
                time = meta["forecast_time"]
                if current is not None and time != current:
                    finish(state)
                    state = start()
                if state is None:
                    state = start()
                current = time
                row, col = int(meta["grid_row"]), int(meta["grid_column"])
                area = (slice(row, row + PATCH_SIZE), slice(col, col + PATCH_SIZE))
                mask = valid[index]
                state["prob"][(..., *area)] += prob[index] * blend
                state["amount"][(..., *area)] += amount[index] * blend
                state["pred_weight"][(..., *area)] += blend
                state["target"][(..., *area)] += target[index] * mask
                state["target_count"][(..., *area)] += mask
                state["zone"][area] = np.minimum(state["zone"][area], zone[index])
    if state is not None:
        finish(state)
    return {key: total / loss_counts[key] for key, total in loss_totals.items()}


def tune_zone_thresholds(
    model: torch.nn.Module,
    dataset: FY3G3HDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    config: ProjectConfig,
    *,
    criterion: torch.nn.Module | None = None,
    diagnostics_probability_threshold: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Search for optimal zone thresholds on a validation dataset and evaluate with those thresholds."""
    search = ZoneThresholdSearch(
        minimum_positive=config.train.min_metric_positive_voxels,
        threshold_min=config.train.threshold_min,
        threshold_max=config.train.threshold_max,
        threshold_step=config.train.threshold_step,
        far_max=config.train.validation_far_max,
        far_max_by_zone=config.train.validation_far_max_by_zone,
    )
    tqdm_disabled = should_disable_tqdm(config.train.disable_tqdm)
    for_each_deduplicated_grid(
        model, dataset, mean, std, device, config.train.microbatch_size,
        lambda p, a, t, v, z: search.update(p, t, v, z),
        progress_desc="Validation thresholds", disable_tqdm=tqdm_disabled,
    )
    selection = search.best()
    thresholds = np.asarray(selection["thresholds"], dtype=np.float32)
    gate_mode = getattr(model, "rain_gate_mode", "hard")
    metrics = GridMetrics(thresholds, rain_gate_mode=gate_mode)
    fixed_thresholds = (
        np.full_like(thresholds, diagnostics_probability_threshold)
        if diagnostics_probability_threshold is not None else None
    )
    fixed_metrics = GridMetrics(fixed_thresholds, rain_gate_mode=gate_mode) if fixed_thresholds is not None else None

    def update_metrics(
        probability: np.ndarray, amount: np.ndarray, target: np.ndarray,
        valid: np.ndarray, zone: np.ndarray,
    ) -> None:
        metrics.update(probability, amount, target, valid, zone)
        if fixed_metrics is not None:
            fixed_metrics.update(probability, amount, target, valid, zone)

    losses = for_each_deduplicated_grid(
        model, dataset, mean, std, device, config.train.microbatch_size,
        update_metrics,
        progress_desc="Validation metrics", disable_tqdm=tqdm_disabled, criterion=criterion,
    )
    result = metrics.finish()
    result["threshold_selection"] = selection
    result["losses"] = losses
    if fixed_metrics is not None:
        fixed_result = fixed_metrics.finish()
        fixed_result["threshold_selection"] = {
            "status": "fixed",
            "thresholds": fixed_thresholds.tolist(),
        }
        result["fixed_threshold_metrics"] = fixed_result
    return result, thresholds


def evaluate_with_fixed_thresholds(
    model: torch.nn.Module,
    dataset: FY3G3HDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    thresholds: np.ndarray,
    microbatch_size: int = 64,
    *,
    criterion: torch.nn.Module | None = None,
    progress_desc: str | None = None,
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Evaluate dataset using strictly fixed zone thresholds without searching or data leakage."""
    thresholds = np.asarray(thresholds, dtype=np.float32)
    metrics = GridMetrics(thresholds, rain_gate_mode=getattr(model, "rain_gate_mode", "hard"))
    losses = for_each_deduplicated_grid(
        model, dataset, mean, std, device, microbatch_size,
        lambda p, a, t, v, z: metrics.update(p, a, t, v, z),
        progress_desc=progress_desc, disable_tqdm=disable_tqdm, criterion=criterion,
    )
    result = metrics.finish()
    result["losses"] = losses
    result["threshold_selection"] = {
        "status": "fixed",
        "thresholds": thresholds.tolist(),
    }
    return result


def evaluate_deduplicated(
    model: torch.nn.Module,
    dataset: FY3G3HDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    config: ProjectConfig,
    *,
    criterion: torch.nn.Module | None = None,
    diagnostics_probability_threshold: float | None = None,
) -> dict[str, Any]:
    """Validation evaluation helper that searches for optimal thresholds and evaluates."""
    result, _ = tune_zone_thresholds(
        model, dataset, mean, std, device, config,
        criterion=criterion,
        diagnostics_probability_threshold=diagnostics_probability_threshold,
    )
    return result

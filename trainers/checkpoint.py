"""Checkpoint save and load utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch

from schema import DATASET_VERSION, FORECAST_LEADS, PATCH_SIZE, TARGET_LEVELS

CHECKPOINT_VERSION = DATASET_VERSION
OUTPUT_CONTRACT = [FORECAST_LEADS, TARGET_LEVELS, PATCH_SIZE, PATCH_SIZE]


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
    inputs: dict[str, Any],
    targets: dict[str, Any],
    metrics: dict[str, Any] | None,
    best_score: float | None = None,
    history: list[dict[str, Any]] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = (
        np.asarray(metrics["threshold_selection"]["thresholds"], np.float32)
        if metrics and "threshold_selection" in metrics
        else model.occurrence_thresholds_by_zone.detach().cpu().numpy()
    )
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": "inference",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "input_statistics": inputs,
        "target_statistics": targets,
        "occurrence_thresholds_by_zone": thresholds,
        "output_contract": OUTPUT_CONTRACT,
        "metrics": metrics,
        "best_score": best_score,
        "history": history,
        "training_state": training_state,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION or payload.get("output_contract") != OUTPUT_CONTRACT:
        raise RuntimeError(f"Checkpoint must use dataset contract {DATASET_VERSION} and output shape {OUTPUT_CONTRACT}")
    return payload


def validate_checkpoint_dataset(payload: dict[str, Any], manifest_sha256: str) -> None:
    """Reject checkpoints trained with statistics from another dataset build."""
    for key in ("input_statistics", "target_statistics"):
        stored = payload.get(key, {}).get("manifest_sha256")
        if not manifest_sha256 or stored != manifest_sha256:
            raise ValueError(f"Checkpoint {key} manifest_sha256 does not match the selected dataset")


def validate_checkpoint_architecture(payload: dict[str, Any], architecture: str, model_config: Any = None) -> None:
    saved = payload.get("config", {}).get("model", {}).get("architecture")
    if saved != architecture:
        raise ValueError(f"Checkpoint architecture {saved!r} does not match configured {architecture!r}")
    if model_config is not None:
        stored = payload.get("config", {}).get("model", {})
        for name, default in (("simvpv2_multi_lead", False), ("exceedance_head_enabled", False),
                              ("rain_gate_mode", "hard"), ("patch_blend_mode", "uniform"),
                              ("patch_blend_min_weight", 0.05)):
            if stored.get(name, default) != getattr(model_config, name):
                raise ValueError(f"Checkpoint model.{name} differs from configuration; use its original config or train a new model")


def validate_checkpoint_selection(payload: dict[str, Any], domain: str) -> None:
    saved = payload.get("config", {}).get("train", {}).get("primary_selection_domain", "macro_all")
    state = payload.get("training_state") or {}
    if saved != domain or state.get("selection_domain", saved) != domain:
        raise ValueError("Checkpoint primary_selection_domain differs; start a new experiment instead of mixing selection scores")

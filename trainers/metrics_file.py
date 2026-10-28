"""Compact, atomically updated metrics for a training run."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def epoch_metrics(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["validation"]
    result = {
        "epoch": record["epoch"],
        "train_loss": record.get("train_loss"),
        "validation_loss": record.get("validation_loss"),
        "learning_rate": record.get("learning_rate"),
        "validation": {
            key: summary.get(key) for key in ("csi", "pod", "far", "mae_mm_h", "rmse_mm_h", "cc")
        } | {
            "csi_by_threshold": {
                key: summary.get("csi_by_threshold", {}).get(key)
                for key in ("0.1", "2", "5")
            }
        },
    }
    for name in ("observable_summary", "pooled_summary", "direct_summary", "selection_summary", "selection_domain"):
        if name in record:
            result[name] = record[name]
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_metrics(path: Path, history: list[dict[str, Any]], *, status: str,
                  best_epoch: int, best_score: float, best_by_metric: dict[str, dict[str, Any]],
                  manifest_sha256: str | None, strategy: Any, selection_domain: str = "macro_all") -> None:
    payload = {
        "status": status,
        "completed_epochs": history[-1]["epoch"] if history else 0,
        "manifest_sha256": manifest_sha256,
        "weight_strategy": strategy or {"name": "dataset"},
        "selection_domain": selection_domain,
        "best": {"epoch": best_epoch, "score": best_score if math.isfinite(best_score) else None},
        "best_by_objective": {
            name: {"epoch": value["epoch"], "score": value["score"],
                   "checkpoint": f"checkpoints/best_{name}_inference.pt"}
            for name, value in best_by_metric.items() if value["epoch"] > 0
        },
        "epochs": [epoch_metrics(row) if "train_loss" in row else row for row in history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

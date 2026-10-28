"""Case selection, track cropping, and results packaging logic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from schema import FORECAST_LEADS, GRID_SHAPE, INPUT_STEPS, PATCH_SIZE, TARGET_LEVELS

SURFACE_SHAPE = (FORECAST_LEADS, TARGET_LEVELS, *GRID_SHAPE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track_crop(valid_mask: np.ndarray, padding: int = 4) -> tuple[slice, slice]:
    """Find bounding slice covering valid cells with padding, clamped to grid."""
    active = np.asarray(valid_mask, dtype=bool)
    if active.ndim > 2:
        active = active.any(axis=tuple(range(active.ndim - 2)))
    rows = np.flatnonzero(active.any(axis=1))
    cols = np.flatnonzero(active.any(axis=0))
    if not rows.size or not cols.size:
        return slice(0, GRID_SHAPE[0]), slice(0, GRID_SHAPE[1])
    r_start = max(0, int(rows.min()) - padding)
    r_stop = min(GRID_SHAPE[0], int(rows.max()) + padding + 1)
    c_start = max(0, int(cols.min()) - padding)
    c_stop = min(GRID_SHAPE[1], int(cols.max()) + padding + 1)
    return slice(r_start, r_stop), slice(c_start, c_stop)


def select_single_input_track(valid_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the largest continuous history strip among the 3 input hours."""
    from scipy.ndimage import label
    # valid_mask: [3, 2, 10, 600, 900]
    best_size = -1
    best_slice = np.zeros(GRID_SHAPE, dtype=bool)
    best_hour_idx = 0
    best_hour_before = 3

    for h_idx in range(INPUT_STEPS):
        # 3 - h_idx hours before forecast
        h_before = INPUT_STEPS - h_idx
        grid_h = valid_mask[h_idx].any(axis=(0, 1))
        labeled, num_features = label(grid_h)
        for f_idx in range(1, num_features + 1):
            comp = (labeled == f_idx)
            sz = int(comp.sum())
            if sz > best_size:
                best_size = sz
                best_slice = comp
                best_hour_idx = h_idx
                best_hour_before = h_before

    info = {
        "selected_input_hour_index": best_hour_idx,
        "selected_input_hour_before_forecast": best_hour_before,
        "selected_track_cells": best_size if best_size > 0 else 0,
    }
    return best_slice, info


def strong_case_score(dataset: Any, indices: list[int]) -> dict[str, Any]:
    target_sum = np.zeros(SURFACE_SHAPE, dtype=np.float64)
    target_count = np.zeros(SURFACE_SHAPE, dtype=np.uint16)
    direct = np.zeros(GRID_SHAPE, dtype=bool)
    for index in indices:
        shard_idx, local_idx = dataset.index[index]
        values = dataset._load(shard_idx)
        metadata = dataset.metadata(index)
        row, column = int(metadata["grid_row"]), int(metadata["grid_column"])
        area = (slice(row, row + PATCH_SIZE), slice(column, column + PATCH_SIZE))
        valid = np.asarray(values["y_valid"][local_idx], dtype=bool)
        amount = np.asarray(values["y_amount"][local_idx], dtype=np.float32)
        target_sum[(..., *area)] += amount * valid
        target_count[(..., *area)] += valid
        direct[area] |= np.asarray(values["x_valid"][local_idx], dtype=bool).any(axis=(0, 1, 2))

    valid = target_count > 0
    target = np.divide(target_sum, target_count, out=np.zeros_like(target_sum), where=valid)
    observed = valid & direct[None, None]
    values = target[observed]
    strong = target >= 5.0
    return {
        "coverage_aware_excess_over_5_mm_h": float(np.maximum(target[observed] - 5.0, 0.0).sum()),
        "coverage_aware_strong_voxels": int(np.count_nonzero(observed & strong)),
        "coverage_aware_max_mm_h": float(values.max()) if values.size else 0.0,
        "direct_observed_voxels": int(np.count_nonzero(observed)),
    }


def select_strong_cases(dataset: Any, requested: int) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        value = dataset.metadata(index)
        forecast_time = str(value["forecast_time"])
        groups.setdefault(forecast_time, []).append(index)
        metadata.setdefault(forecast_time, value)

    candidates = []
    for forecast_time, indices in groups.items():
        score = strong_case_score(dataset, indices)
        candidates.append({
            "forecast_time": forecast_time,
            "indices": indices,
            "metadata": metadata[forecast_time],
            **score,
        })
    candidates.sort(
        key=lambda v: (-v["coverage_aware_excess_over_5_mm_h"], -v["coverage_aware_max_mm_h"], v["forecast_time"])
    )
    return candidates[:max(1, min(requested, len(candidates)))]


def create_result_archive(project_root: Path, output_dir: Path, config_path: Path, archive_path: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_files: list[dict[str, Any]] = []
    archive_members: list[tuple[Path, str]] = []

    # Include config
    if config_path.exists():
        rel = f"config/{config_path.name}"
        archive_members.append((config_path, rel))
        manifest_files.append({"path": rel, "bytes": config_path.stat().st_size, "sha256": sha256_file(config_path)})

    # Include output files except checkpoints
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if "checkpoints" in path.parts:
            continue
        rel = f"outputs/{path.relative_to(output_dir).as_posix()}"
        archive_members.append((path, rel))
        manifest_files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest_payload = {
        "status": "complete",
        "archive_name": archive_path.name,
        "checkpoint_included": False,
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    manifest_tmp = output_dir / "results_package_manifest.json"
    manifest_tmp.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_members.append((manifest_tmp, "results_package_manifest.json"))

    with tarfile.open(archive_path, "w:gz") as tar:
        for fpath, arcname in archive_members:
            tar.add(fpath, arcname=arcname, recursive=False)

    sha_file = archive_path.with_name(archive_path.name + ".sha256")
    sha_file.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="ascii")

    return {
        "status": "created",
        "archive_path": str(archive_path),
        "sha256_file": str(sha_file),
        "checkpoint_included": False,
        "files_archived": len(archive_members),
    }

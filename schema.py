"""Data contract and schema validation for FY3G_GPM_V2 training project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

DATASET_VERSION = "fy3ggpm_surface_3h_l1only_0p1_v3"

# Expected dimensions and types
ARRAY_SPECS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "x_reflectivity": (np.float16, (3, 2, 10, 64, 64)),
    "x_valid": (np.uint8, (3, 2, 10, 64, 64)),
    "x_geo": (np.float16, (3, 2, 64, 64)),
    "x_direction": (np.int8, (3, 64, 64)),
    "y_amount": (np.float16, (3, 1, 64, 64)),
    "y_valid": (np.uint8, (3, 1, 64, 64)),
}

ARRAY_NAMES = tuple(ARRAY_SPECS.keys())

INPUT_LEVELS_KM = tuple(range(2, 12))
INPUT_CHANNELS = 43
INPUT_STEPS = 3
FORECAST_LEADS = 3
TARGET_LEVELS = 1
GRID_DEGREES = 0.1
GRID_SHAPE = (600, 900)
PATCH_SIZE = 64
PATCH_STRIDE = 32
SURFACE_GEO_INDEX = 0


def validate_dataset_contract(dataset_root: str | Path) -> dict[str, Any]:
    """Validate that the dataset directory conforms strictly to the required data contract."""
    root = Path(dataset_root)
    manifest_path = root / "source_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("dataset_version")
    if version != DATASET_VERSION:
        raise ValueError(f"Dataset version mismatch: found {version}, expected {DATASET_VERSION}")

    data_config = manifest.get("data_config", {})
    expected_config = {
        "grid_degrees": GRID_DEGREES,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "input_levels_km": list(INPUT_LEVELS_KM),
    }
    for key, expected in expected_config.items():
        if data_config.get(key) != expected:
            raise ValueError(f"Dataset {key} mismatch: found {data_config.get(key)}, expected {expected}")
    manifest_specs = manifest.get("array_specs", {})
    for name, (dtype, shape) in ARRAY_SPECS.items():
        expected = [str(dtype), list(shape)]
        if manifest_specs.get(name) != expected:
            raise ValueError(f"Dataset array_specs[{name}] mismatch: found {manifest_specs.get(name)}, expected {expected}")

    manifest_sha256 = manifest.get("manifest_sha256", "")

    inputs_path = root / "statistics/input_statistics.json"
    targets_path = root / "statistics/target_statistics.json"
    if not inputs_path.exists():
        raise FileNotFoundError(
            f"Missing input statistics: {inputs_path}. Please run compute_statistics.py in fy3g_gpm_process."
        )
    if not targets_path.exists():
        raise FileNotFoundError(
            f"Missing target statistics: {targets_path}. Please run compute_statistics.py in fy3g_gpm_process."
        )

    # Validate input statistics structure, shape, and values
    input_stats = json.loads(inputs_path.read_text(encoding="utf-8"))
    if "mean" not in input_stats or "std" not in input_stats:
        raise ValueError(f"Input statistics missing 'mean' or 'std': {inputs_path}")
    mean_arr = np.asarray(input_stats["mean"], dtype=np.float32)
    std_arr = np.asarray(input_stats["std"], dtype=np.float32)
    if mean_arr.shape != (2, 10) or std_arr.shape != (2, 10):
        raise ValueError(
            f"Input statistics shape mismatch: expected (2, 10), got mean={mean_arr.shape}, std={std_arr.shape}"
        )
    if not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(std_arr)):
        raise ValueError("Input statistics contain non-finite values (NaN or Inf)")
    if np.any(std_arr <= 0):
        raise ValueError("Input statistics standard deviation must be strictly positive")
    if input_stats.get("manifest_sha256") and manifest_sha256:
        if input_stats["manifest_sha256"] != manifest_sha256:
            raise ValueError(
                f"Input statistics manifest_sha256 mismatch: {input_stats['manifest_sha256']} != {manifest_sha256}"
            )

    # Validate target statistics structure, shape, and values
    target_stats = json.loads(targets_path.read_text(encoding="utf-8"))
    if "rain_log_mean" not in target_stats or "rain_log_std" not in target_stats:
        raise ValueError(f"Target statistics missing 'rain_log_mean' or 'rain_log_std': {targets_path}")
    for field_name in ("intensity_edges_mm_h", "group_weight", "threshold_positive_count"):
        if field_name not in target_stats:
            raise ValueError(f"Target statistics missing required field '{field_name}': {targets_path}")
    t_mean = np.asarray(target_stats["rain_log_mean"], dtype=np.float32)
    t_std = np.asarray(target_stats["rain_log_std"], dtype=np.float32)
    if t_mean.shape != (3, 1) or t_std.shape != (3, 1):
        raise ValueError(
            f"Target statistics shape mismatch: expected (3, 1), got mean={t_mean.shape}, std={t_std.shape}"
        )
    if not np.all(np.isfinite(t_mean)) or not np.all(np.isfinite(t_std)):
        raise ValueError("Target statistics contain non-finite values (NaN or Inf)")
    if target_stats.get("computed_from_split") and target_stats["computed_from_split"] != "train":
        raise ValueError(
            f"Target statistics must be computed from 'train' split, got {target_stats['computed_from_split']}"
        )
    if target_stats.get("manifest_sha256") and manifest_sha256:
        if target_stats["manifest_sha256"] != manifest_sha256:
            raise ValueError(
                f"Target statistics manifest_sha256 mismatch: {target_stats['manifest_sha256']} != {manifest_sha256}"
            )

    return {
        "dataset_version": version,
        "sample_counts": manifest.get("sample_counts", {}),
        "manifest_sha256": manifest_sha256,
        "input_statistics_present": True,
        "target_statistics_present": True,
        "input_shape": list(mean_arr.shape),
        "target_shape": list(t_mean.shape),
    }

"""Derive training-only rain intensity weights from existing target counts."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

STRATEGY = "sqrt_inverse_frequency_v1"
SMOOTHING = 10.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 10.0


def derive_intensity_weights(statistics: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with effective weights; never mutate the dataset statistics."""
    counts = np.asarray(statistics["group_count"], dtype=np.float64)
    rainy = np.asarray(statistics["rain_count"], dtype=np.float64)
    edges = np.asarray(statistics["intensity_edges_mm_h"], dtype=np.float64)
    valid = np.asarray(statistics["valid_count"], dtype=np.float64)
    if counts.ndim != 3 or counts.shape[:2] != rainy.shape or counts.shape[:2] != valid.shape or counts.shape[2] != len(edges) - 1:
        raise ValueError("group_count dimensions do not match rain_count, valid_count and intensity edges")
    if (not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or not np.all(np.isfinite(rainy)) or np.any(rainy < 0)
            or not np.all(np.isfinite(valid)) or np.any(valid < 0)):
        raise ValueError("Rain intensity counts must be finite and nonnegative")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("Rain intensity edges must be finite and increasing")
    if np.any(counts.sum(axis=-1) > valid):
        raise ValueError("Rain intensity counts exceed valid counts")
    rainy_bins = edges[:-1] >= 0.1
    if not np.array_equal(counts[..., rainy_bins].sum(axis=-1).astype(np.int64), rainy.astype(np.int64)):
        raise ValueError("Rain count does not match populated rainy intensity bins")

    weights = np.zeros_like(counts)
    for index in np.ndindex(rainy.shape):
        active = (counts[index] > 0) & rainy_bins
        if rainy[index] and not active.any():
            raise ValueError("Rain count has no populated intensity bins")
        if active.any():
            raw = np.sqrt(rainy[index] / (counts[index][active] + SMOOTHING))
            # Normalize by observed pixels, not by the number of intensity bins.
            raw /= np.average(raw, weights=counts[index][active])
            weights[index][active] = np.clip(raw, MIN_WEIGHT, MAX_WEIGHT)
    result = deepcopy(statistics)
    result["group_weight"] = weights.tolist()
    result["weight_strategy"] = {
        "name": STRATEGY, "smoothing": SMOOTHING,
        "minimum": MIN_WEIGHT, "maximum": MAX_WEIGHT,
        "source_statistics_sha256": statistics.get("sha256"),
    }
    return result

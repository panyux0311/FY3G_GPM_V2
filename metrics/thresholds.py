"""Zone-based probability threshold search on validation predictions."""
from __future__ import annotations

from typing import Any
import numpy as np

ZONE_NAMES = ("direct", "near", "uncovered")


def _rat(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray | float:
    return np.divide(num, den, out=np.full_like(np.asarray(num, dtype=float), np.nan), where=np.asarray(den) > 0)


class ZoneThresholdSearch:
    def __init__(
        self,
        leads: int = 3,
        targets: int = 1,
        minimum_positive: int = 100,
        threshold_min: float = 0.05,
        threshold_max: float = 0.99,
        threshold_step: float = 0.01,
        far_max: float = 0.90,
        far_max_by_zone: list[float] | None = None,
    ) -> None:
        self.leads, self.targets, self.minimum_positive = leads, targets, minimum_positive
        self.thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 8)
        shape = (leads, targets, 3, len(self.thresholds))
        self.tp = np.zeros(shape, np.int64)
        self.fp = np.zeros(shape, np.int64)
        self.positive = np.zeros((leads, targets, 3), np.int64)
        self.valid = np.zeros((leads, targets, 3), np.int64)
        self.far_max = far_max
        self.far_max_by_zone = np.asarray(far_max_by_zone if far_max_by_zone is not None else [far_max] * 3, dtype=float)
        if self.far_max_by_zone.shape != (3,) or not np.isfinite(self.far_max_by_zone).all() or np.any((self.far_max_by_zone < 0) | (self.far_max_by_zone > 1)):
            raise ValueError("FAR caps must contain three finite values in [0,1]")
        self.strict_far = far_max_by_zone is not None

    def update(self, probability: np.ndarray, target: np.ndarray, valid: np.ndarray, zone: np.ndarray) -> None:
        for lead in range(self.leads):
            for target_index in range(self.targets):
                for zone_index in range(3):
                    mask = np.asarray(valid[lead, target_index], bool) & (zone == zone_index)
                    p = np.asarray(probability[lead, target_index][mask], float)
                    truth = np.asarray(target[lead, target_index][mask] >= 0.1, bool)
                    self.valid[lead, target_index, zone_index] += len(p)
                    self.positive[lead, target_index, zone_index] += truth.sum()
                    for values, sink in ((p[truth], self.tp[lead, target_index, zone_index]), (p[~truth], self.fp[lead, target_index, zone_index])):
                        bins = np.searchsorted(self.thresholds, values, side="right")
                        hist = np.bincount(bins, minlength=len(self.thresholds) + 1)
                        sink += np.cumsum(hist[1:][::-1])[::-1]

    def best(self) -> dict[str, Any]:
        output = np.full((self.leads, self.targets, 3), 0.5, np.float32)
        supported = np.zeros_like(output, dtype=bool)
        details = []
        curves = []
        for lead in range(self.leads):
            for target_index in range(self.targets):
                for zone in range(3):
                    tp = self.tp[lead, target_index, zone]
                    fp = self.fp[lead, target_index, zone]
                    pos = self.positive[lead, target_index, zone]
                    fn = pos - tp
                    csi = _rat(tp, tp + fp + fn)
                    pod = _rat(tp, tp + fn)
                    far = _rat(fp, tp + fp)
                    feasible = np.isfinite(csi) & (far <= self.far_max_by_zone[zone])
                    ok = (pos >= self.minimum_positive) & feasible
                    candidates = feasible if self.strict_far else np.isfinite(csi)
                    selected = int(np.argmax(np.where(ok if ok.any() else candidates, csi, -np.inf)))
                    abstain = self.strict_far and not candidates.any()
                    if not np.isfinite(csi[selected]) and not abstain:
                        selected = len(self.thresholds) // 2
                    output[lead, target_index, zone] = np.nextafter(np.float32(1), np.float32(np.inf)) if abstain else self.thresholds[selected]
                    supported[lead, target_index, zone] = bool(ok.any())
                    details.append({
                        "lead_hour": lead + 1,
                        "target": "surface",
                        "zone": ZONE_NAMES[zone],
                        "threshold": float(output[lead, target_index, zone]),
                        "supported": bool(ok.any()),
                        "positive": int(pos),
                        "valid": int(self.valid[lead, target_index, zone]),
                        "far_max": float(self.far_max_by_zone[zone]),
                        "status": "no_feasible_threshold_abstain" if abstain else ("supported" if ok.any() else "insufficient_positive_support"),
                        "csi": float(_rat(0, pos)) if abstain else float(csi[selected]),
                        "pod": float(_rat(0, pos)) if abstain else float(pod[selected]),
                        "far": float("nan") if abstain else float(far[selected]),
                    })
                    curves.extend({
                        "lead_hour": lead + 1,
                        "target": "surface",
                        "zone": ZONE_NAMES[zone],
                        "threshold": float(threshold),
                        "tp": int(tp[index]),
                        "fp": int(fp[index]),
                        "fn": int(fn[index]),
                        "csi": float(csi[index]),
                        "pod": float(pod[index]),
                        "far": float(far[index]),
                    } for index, threshold in enumerate(self.thresholds))
        return {"thresholds": output.tolist(), "supported": supported.tolist(), "details": details, "curves": curves}

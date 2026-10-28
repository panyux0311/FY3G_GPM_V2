"""Deduplicated-grid GPM precipitation metrics (CSI, POD, FAR, MAE, RMSE, CC)."""
from __future__ import annotations

from typing import Any
import numpy as np

from .thresholds import ZONE_NAMES, _rat
from utils.rain_gate import gate_rain_numpy


CSI_RAIN_THRESHOLDS_MM_H = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def selection_summary(metrics: dict[str, Any], domain: str) -> dict[str, Any]:
    keys = {"macro_all": "summary", "observable": "observable_summary", "direct_only": "direct_summary"}
    return metrics[keys[domain]]


def _macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in ("csi", "pod", "far", "mae_mm_h", "rmse_mm_h", "cc"):
        values = [row[key] for row in rows if np.isfinite(row[key])]
        result[key] = float(np.mean(values)) if values else float("nan")
    result["csi_by_threshold"] = {}
    for threshold in CSI_RAIN_THRESHOLDS_MM_H:
        key = format(threshold, "g")
        values = [row["csi_by_threshold"][key] for row in rows if np.isfinite(row["csi_by_threshold"][key])]
        result["csi_by_threshold"][key] = float(np.mean(values)) if values else float("nan")
    result["balanced_score"] = result["csi"]
    return result


class GridMetrics:
    def __init__(self, thresholds: np.ndarray, leads: int = 3, targets: int = 1, *, rain_gate_mode: str = "hard") -> None:
        self.thresholds = np.asarray(thresholds, float).reshape(leads, targets, 3)
        self.rain_gate_mode = rain_gate_mode
        self.leads, self.targets = leads, targets
        self.tp = np.zeros((leads, targets, 3), np.int64)
        self.fp = np.zeros_like(self.tp)
        self.fn = np.zeros_like(self.tp)
        self.valid = np.zeros_like(self.tp)
        self.amount_abs = np.zeros_like(self.tp, dtype=float)
        self.amount_sq = np.zeros_like(self.tp, dtype=float)
        self.cc_mean_rate = np.zeros_like(self.tp, dtype=float)
        self.cc_mean_target = np.zeros_like(self.tp, dtype=float)
        self.cc_m2_rate = np.zeros_like(self.tp, dtype=float)
        self.cc_m2_target = np.zeros_like(self.tp, dtype=float)
        self.cc_covariance = np.zeros_like(self.tp, dtype=float)
        threshold_shape = (*self.tp.shape, len(CSI_RAIN_THRESHOLDS_MM_H))
        self.threshold_tp = np.zeros(threshold_shape, np.int64)
        self.threshold_fp = np.zeros(threshold_shape, np.int64)
        self.threshold_fn = np.zeros(threshold_shape, np.int64)

    def update(
        self,
        probability: np.ndarray,
        amount: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
        zone: np.ndarray,
    ) -> None:
        for lead in range(self.leads):
            for target_index in range(self.targets):
                threshold_map = self.thresholds[lead, target_index][zone]
                rate = gate_rain_numpy(probability[lead, target_index], amount[lead, target_index],
                                       threshold_map, self.rain_gate_mode)
                for zone_index in range(3):
                    mask = np.asarray(valid[lead, target_index], bool) & (zone == zone_index)
                    truth = np.asarray(target[lead, target_index] >= 0.1, bool)
                    pred = np.asarray(probability[lead, target_index] >= self.thresholds[lead, target_index, zone_index], bool)
                    self.tp[lead, target_index, zone_index] += np.count_nonzero(mask & truth & pred)
                    self.fp[lead, target_index, zone_index] += np.count_nonzero(mask & ~truth & pred)
                    self.fn[lead, target_index, zone_index] += np.count_nonzero(mask & truth & ~pred)
                    error = np.asarray(rate - target[lead, target_index], float)[mask]
                    self.amount_abs[lead, target_index, zone_index] += np.abs(error).sum()
                    self.amount_sq[lead, target_index, zone_index] += np.square(error).sum()

                    # Merge this grid's centered moments with earlier grids (Chan's formula).
                    x = np.asarray(rate[mask], dtype=np.float64)
                    y = np.asarray(target[lead, target_index][mask], dtype=np.float64)
                    if x.size:
                        key = (lead, target_index, zone_index)
                        for threshold_index, threshold in enumerate(CSI_RAIN_THRESHOLDS_MM_H):
                            predicted = x >= threshold
                            observed = y >= threshold
                            self.threshold_tp[(*key, threshold_index)] += np.count_nonzero(predicted & observed)
                            self.threshold_fp[(*key, threshold_index)] += np.count_nonzero(predicted & ~observed)
                            self.threshold_fn[(*key, threshold_index)] += np.count_nonzero(~predicted & observed)

                        previous = int(self.valid[key])
                        count = previous + x.size
                        mean_x, mean_y = x.mean(), y.mean()
                        dx = mean_x - self.cc_mean_rate[key]
                        dy = mean_y - self.cc_mean_target[key]
                        factor = float(previous) * x.size / count
                        self.cc_m2_rate[key] += np.square(x - mean_x).sum() + dx * dx * factor
                        self.cc_m2_target[key] += np.square(y - mean_y).sum() + dy * dy * factor
                        self.cc_covariance[key] += np.sum((x - mean_x) * (y - mean_y)) + dx * dy * factor
                        self.cc_mean_rate[key] += dx * x.size / count
                        self.cc_mean_target[key] += dy * x.size / count
                        self.valid[key] = count

    def finish(self) -> dict[str, Any]:
        rows = []
        for lead in range(self.leads):
            for target_index in range(self.targets):
                for zone in range(3):
                    tp, fp, fn = self.tp[lead, target_index, zone], self.fp[lead, target_index, zone], self.fn[lead, target_index, zone]
                    key = (lead, target_index, zone)
                    variance_product = self.cc_m2_rate[key] * self.cc_m2_target[key]
                    cc = (
                        float(np.clip(self.cc_covariance[key] / np.sqrt(variance_product), -1.0, 1.0))
                        if self.valid[key] >= 2 and variance_product > 0 else float("nan")
                    )
                    counts_by_threshold = {}
                    csi_by_threshold = {}
                    for threshold_index, threshold in enumerate(CSI_RAIN_THRESHOLDS_MM_H):
                        threshold_key = format(threshold, "g")
                        threshold_counts = (*key, threshold_index)
                        rain_tp = int(self.threshold_tp[threshold_counts])
                        rain_fp = int(self.threshold_fp[threshold_counts])
                        rain_fn = int(self.threshold_fn[threshold_counts])
                        counts_by_threshold[threshold_key] = {
                            "tp": rain_tp, "fp": rain_fp, "fn": rain_fn,
                            "positive": rain_tp + rain_fn,
                        }
                        csi_by_threshold[threshold_key] = float(_rat(rain_tp, rain_tp + rain_fp + rain_fn))
                    rows.append({
                        "lead_hour": lead + 1,
                        "target": "surface",
                        "zone": ZONE_NAMES[zone],
                        "threshold": float(self.thresholds[lead, target_index, zone]),
                        "valid": int(self.valid[lead, target_index, zone]),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "positive": int(tp + fn),
                        "rain_fraction": float(_rat(tp + fn, self.valid[key])),
                        "csi": float(_rat(tp, tp + fp + fn)),
                        "pod": float(_rat(tp, tp + fn)),
                        "far": float(_rat(fp, tp + fp)),
                        "mae_mm_h": float(_rat(self.amount_abs[lead, target_index, zone], self.valid[lead, target_index, zone])),
                        "rmse_mm_h": float(np.sqrt(_rat(self.amount_sq[lead, target_index, zone], self.valid[lead, target_index, zone]))),
                        "cc": cc,
                        "csi_by_threshold": csi_by_threshold,
                        "counts_by_threshold": counts_by_threshold,
                    })
        valid_rows = [row for row in rows if np.isfinite(row["csi"])]
        summary = {}
        for key in ("csi", "pod", "far", "mae_mm_h", "rmse_mm_h"):
            values = [row[key] for row in valid_rows if not np.isnan(row[key])]
            summary[key] = float(np.mean(values)) if values else float("nan")
        valid_cc = [row["cc"] for row in rows if np.isfinite(row["cc"])]
        summary["cc"] = float(np.mean(valid_cc)) if valid_cc else float("nan")
        summary["csi_by_threshold"] = {}
        for threshold in CSI_RAIN_THRESHOLDS_MM_H:
            threshold_key = format(threshold, "g")
            values = [row["csi_by_threshold"][threshold_key] for row in rows
                      if np.isfinite(row["csi_by_threshold"][threshold_key])]
            summary["csi_by_threshold"][threshold_key] = float(np.mean(values)) if values else float("nan")
        summary["balanced_score"] = summary["csi"]
        count = int(self.valid.sum())
        mean_x = float(np.sum(self.valid * self.cc_mean_rate) / count) if count else 0.0
        mean_y = float(np.sum(self.valid * self.cc_mean_target) / count) if count else 0.0
        m2_x = np.sum(self.cc_m2_rate + self.valid * (self.cc_mean_rate - mean_x) ** 2)
        m2_y = np.sum(self.cc_m2_target + self.valid * (self.cc_mean_target - mean_y) ** 2)
        covariance = np.sum(self.cc_covariance + self.valid * (self.cc_mean_rate - mean_x) * (self.cc_mean_target - mean_y))
        tp, fp, fn = int(self.tp.sum()), int(self.fp.sum()), int(self.fn.sum())
        pooled = {
            "valid": count, "tp": tp, "fp": fp, "fn": fn, "positive": tp + fn,
            "csi": float(_rat(tp, tp + fp + fn)), "pod": float(_rat(tp, tp + fn)),
            "far": float(_rat(fp, tp + fp)),
            "mae_mm_h": float(_rat(self.amount_abs.sum(), count)),
            "rmse_mm_h": float(np.sqrt(_rat(self.amount_sq.sum(), count))),
            "cc": float(np.clip(covariance / np.sqrt(m2_x * m2_y), -1, 1)) if count >= 2 and m2_x > 0 and m2_y > 0 else float("nan"),
            "csi_by_threshold": {}, "counts_by_threshold": {},
        }
        for i, threshold in enumerate(CSI_RAIN_THRESHOLDS_MM_H):
            key = format(threshold, "g")
            a, b, c = int(self.threshold_tp[..., i].sum()), int(self.threshold_fp[..., i].sum()), int(self.threshold_fn[..., i].sum())
            pooled["csi_by_threshold"][key] = float(_rat(a, a + b + c))
            pooled["counts_by_threshold"][key] = {"tp": a, "fp": b, "fn": c, "positive": a + c}
        return {
            "summary": summary,
            "observable_summary": _macro_summary([row for row in rows if row["zone"] in ("direct", "near")]),
            "direct_summary": _macro_summary([row for row in rows if row["zone"] == "direct"]),
            "pooled_summary": pooled,
            "by_lead_target_zone": rows,
            "rain_thresholds_mm_h": list(CSI_RAIN_THRESHOLDS_MM_H),
            "rain_gate_mode": self.rain_gate_mode,
        }

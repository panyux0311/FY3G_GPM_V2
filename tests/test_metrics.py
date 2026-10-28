from __future__ import annotations

import math
import unittest
import numpy as np

from metrics.gpm_metrics import CSI_RAIN_THRESHOLDS_MM_H, GridMetrics
from metrics.thresholds import ZoneThresholdSearch


class TestMetrics(unittest.TestCase):
    def test_zone_threshold_search_and_grid_metrics(self) -> None:
        probability = np.full((3, 1, 2, 2), 0.1, np.float32)
        probability[:, 0, 0, 0] = 0.9
        target = np.zeros_like(probability)
        target[:, 0, 0, 0] = 1.0
        valid = np.ones_like(target, bool)
        zone = np.array([[0, 1], [2, 0]], np.int8)

        search = ZoneThresholdSearch(minimum_positive=1, threshold_min=0.1, threshold_max=0.9, threshold_step=0.4)
        search.update(probability, target, valid, zone)
        selected = search.best()
        thresholds = np.asarray(selected["thresholds"])
        self.assertEqual(thresholds.shape, (3, 1, 3))

        metrics = GridMetrics(thresholds)
        metrics.update(probability, np.ones_like(probability), target, valid, zone)
        result = metrics.finish()
        self.assertIn("balanced_score", result["summary"])
        self.assertIn("csi", result["summary"])

    def test_fixed_thresholds_evaluation_structure(self) -> None:
        thresholds = np.full((3, 1, 3), 0.5, dtype=np.float32)
        metrics = GridMetrics(thresholds)
        prob = np.full((3, 1, 4, 4), 0.6, dtype=np.float32)
        amt = np.full((3, 1, 4, 4), 2.0, dtype=np.float32)
        tgt = np.full((3, 1, 4, 4), 2.0, dtype=np.float32)
        vld = np.ones((3, 1, 4, 4), dtype=bool)
        zne = np.zeros((4, 4), dtype=np.int8)
        metrics.update(prob, amt, tgt, vld, zne)
        res = metrics.finish()
        self.assertEqual(res["summary"]["csi"], 1.0)

    def test_cc_positive_and_negative_by_zone(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        amount = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float).reshape(1, 1, 1, 8)
        target = np.array([2, 4, 6, 8, 8, 6, 4, 2], dtype=float).reshape(1, 1, 1, 8)
        probability = np.ones_like(amount)
        valid = np.ones_like(amount, dtype=bool)
        zone = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])

        metrics.update(probability, amount, target, valid, zone)
        result = metrics.finish()
        rows = result["by_lead_target_zone"]
        self.assertAlmostEqual(rows[0]["cc"], 1.0)
        self.assertAlmostEqual(rows[1]["cc"], -1.0)
        self.assertTrue(np.isnan(rows[2]["cc"]))
        self.assertAlmostEqual(result["summary"]["cc"], 0.0)
        self.assertEqual(result["summary"]["balanced_score"], result["summary"]["csi"])

    def test_cc_streaming_matches_batch_reference_with_large_offsets(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        amount = 1e9 + np.array([1, 3, 2, 6, 4, 5], dtype=float)
        target = 1e9 + np.array([3, 1, 4, 2, 7, 5], dtype=float)
        for start, stop in ((0, 2), (2, 6)):
            x = amount[start:stop].reshape(1, 1, 1, stop - start)
            y = target[start:stop].reshape(1, 1, 1, stop - start)
            metrics.update(np.ones_like(x), x, y, np.ones_like(x, dtype=bool), np.zeros((1, stop - start), dtype=int))

        mean_x = math.fsum(amount) / len(amount)
        mean_y = math.fsum(target) / len(target)
        covariance = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(amount, target))
        variance_x = math.fsum((x - mean_x) ** 2 for x in amount)
        variance_y = math.fsum((y - mean_y) ** 2 for y in target)
        expected = covariance / math.sqrt(variance_x * variance_y)
        result = metrics.finish()
        self.assertAlmostEqual(result["by_lead_target_zone"][0]["cc"], expected, places=12)
        self.assertAlmostEqual(result["summary"]["cc"], expected, places=12)

    def test_cc_uses_gated_rate_and_only_valid_pixels(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        probability = np.array([0.9, 0.4, 0.9, 0.9]).reshape(1, 1, 1, 4)
        amount = np.array([1, 2, 3, 100], dtype=float).reshape(1, 1, 1, 4)
        target = np.array([2, 1, 6, -1000], dtype=float).reshape(1, 1, 1, 4)
        valid = np.array([True, True, True, False]).reshape(1, 1, 1, 4)

        metrics.update(probability, amount, target, valid, np.zeros((1, 4), dtype=int))
        row = metrics.finish()["by_lead_target_zone"][0]
        self.assertEqual(row["valid"], 3)
        self.assertAlmostEqual(row["cc"], 4 * np.sqrt(3) / 7)
        self.assertAlmostEqual(row["mae_mm_h"], 5 / 3)

    def test_cc_undefined_for_one_sample_or_constant_values(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        amount = np.array([1, 2, 2, 2, 1, 2, 3], dtype=float).reshape(1, 1, 1, 7)
        target = np.array([1, 1, 2, 3, 2, 2, 2], dtype=float).reshape(1, 1, 1, 7)
        zone = np.array([[0, 1, 1, 1, 2, 2, 2]])
        metrics.update(np.ones_like(amount), amount, target, np.ones_like(amount, dtype=bool), zone)

        result = metrics.finish()
        self.assertTrue(all(np.isnan(row["cc"]) for row in result["by_lead_target_zone"]))
        self.assertTrue(np.isnan(result["summary"]["cc"]))
        self.assertEqual(result["summary"]["csi"], 1.0)

    def test_cc_summary_filters_its_own_valid_groups(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        probability = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=float).reshape(1, 1, 1, 8)
        amount = np.array([0, 0, 1, 2, 3, 2, 2, 2], dtype=float).reshape(1, 1, 1, 8)
        target = np.array([0, 0.01, 1, 3, 2, 1, 2, 3], dtype=float).reshape(1, 1, 1, 8)
        zone = np.array([[0, 0, 1, 1, 1, 2, 2, 2]])
        metrics.update(probability, amount, target, np.ones_like(amount, dtype=bool), zone)

        result = metrics.finish()
        rows = result["by_lead_target_zone"]
        self.assertTrue(np.isnan(rows[0]["csi"]))
        self.assertTrue(np.isnan(rows[0]["cc"]))
        self.assertTrue(np.isfinite(rows[1]["cc"]))
        self.assertTrue(np.isnan(rows[2]["cc"]))
        self.assertAlmostEqual(result["summary"]["cc"], rows[1]["cc"])
        self.assertEqual(result["summary"]["csi"], 1.0)

    def test_six_rain_threshold_counts_use_gated_amount_and_inclusive_bounds(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        probability = np.array([.9] * 7 + [.1, .9, .1, .9]).reshape(1, 1, 1, 11)
        amount = np.array([.1, .5, 1, 2, 5, 10, 10, 20, 0, 20, 10]).reshape(1, 1, 1, 11)
        target = np.array([.1, .5, 1, 2, 5, 10, 0, 10, 10, 0, 0]).reshape(1, 1, 1, 11)
        valid = np.array([True] * 10 + [False]).reshape(1, 1, 1, 11)
        metrics.update(probability, amount, target, valid, np.zeros((1, 11), dtype=int))

        result = metrics.finish()
        self.assertEqual(result["rain_thresholds_mm_h"], list(CSI_RAIN_THRESHOLDS_MM_H))
        row = result["by_lead_target_zone"][0]
        self.assertEqual(row["valid"], 10)
        self.assertEqual(row["tp"], 7)  # Original occurrence-only CSI at 0.1 mm/h.
        self.assertEqual(row["fp"], 1)
        self.assertEqual(row["fn"], 1)
        self.assertAlmostEqual(row["csi"], 7 / 9)
        self.assertEqual(result["summary"]["balanced_score"], result["summary"]["csi"])
        expected_tp = {"0.1": 6, "0.5": 5, "1": 4, "2": 3, "5": 2, "10": 1}
        self.assertEqual(set(row["csi_by_threshold"]), set(expected_tp))
        for threshold, tp in expected_tp.items():
            self.assertEqual(row["counts_by_threshold"][threshold], {
                "tp": tp, "fp": 1, "fn": 2, "positive": tp + 2,
            })
            self.assertAlmostEqual(row["csi_by_threshold"][threshold], tp / (tp + 3))
            self.assertAlmostEqual(result["summary"]["csi_by_threshold"][threshold], tp / (tp + 3))
        self.assertNotEqual(result["summary"]["csi"], result["summary"]["csi_by_threshold"]["0.1"])

    def test_rain_threshold_csi_is_macro_average_across_leads_and_zones(self) -> None:
        metrics = GridMetrics(np.full((2, 1, 3), 0.5), leads=2)
        amount = np.array([[1, 0, 1, 1], [0, 0, 1, 1]], dtype=float).reshape(2, 1, 1, 4)
        target = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=float).reshape(2, 1, 1, 4)
        zone = np.array([[0, 0, 1, 1]])
        metrics.update(np.ones_like(amount), amount, target, np.ones_like(amount, dtype=bool), zone)

        result = metrics.finish()
        rows = result["by_lead_target_zone"]
        self.assertEqual([row["lead_hour"] for row in rows], [1, 1, 1, 2, 2, 2])
        self.assertEqual(rows[0]["csi_by_threshold"]["1"], 1.0)
        self.assertEqual(rows[1]["csi_by_threshold"]["1"], 0.0)
        self.assertTrue(np.isnan(rows[3]["csi_by_threshold"]["1"]))
        self.assertEqual(rows[4]["csi_by_threshold"]["1"], 1.0)
        self.assertAlmostEqual(result["summary"]["csi_by_threshold"]["1"], 2 / 3)

    def test_rain_threshold_counts_accumulate_across_grids(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        for amount, target in (([1, 2], [1, 0]), ([0, 2], [2, 2])):
            x = np.asarray(amount, dtype=float).reshape(1, 1, 1, 2)
            y = np.asarray(target, dtype=float).reshape(1, 1, 1, 2)
            metrics.update(np.ones_like(x), x, y, np.ones_like(x, dtype=bool), np.zeros((1, 2), dtype=int))

        result = metrics.finish()
        row = result["by_lead_target_zone"][0]
        self.assertEqual(row["counts_by_threshold"]["1"], {
            "tp": 2, "fp": 1, "fn": 1, "positive": 3,
        })
        self.assertEqual(row["csi_by_threshold"]["1"], 0.5)
        self.assertEqual(result["summary"]["csi_by_threshold"]["1"], 0.5)

    def test_rain_threshold_csi_undefined_without_events_and_zero_for_false_alarms(self) -> None:
        metrics = GridMetrics(np.full((1, 1, 3), 0.5), leads=1)
        probability = np.array([0, 1], dtype=float).reshape(1, 1, 1, 2)
        amount = np.array([20, 20], dtype=float).reshape(1, 1, 1, 2)
        target = np.zeros_like(amount)
        metrics.update(probability, amount, target, np.ones_like(amount, dtype=bool), np.array([[0, 1]]))

        result = metrics.finish()
        rows = result["by_lead_target_zone"]
        for threshold in ("0.1", "0.5", "1", "2", "5", "10"):
            self.assertTrue(np.isnan(rows[0]["csi_by_threshold"][threshold]))
            self.assertEqual(rows[1]["counts_by_threshold"][threshold], {
                "tp": 0, "fp": 1, "fn": 0, "positive": 0,
            })
            self.assertEqual(rows[1]["csi_by_threshold"][threshold], 0.0)
            self.assertEqual(result["summary"]["csi_by_threshold"][threshold], 0.0)

        empty = GridMetrics(np.full((1, 1, 3), 0.5), leads=1).finish()
        self.assertTrue(all(np.isnan(value) for value in empty["summary"]["csi_by_threshold"].values()))

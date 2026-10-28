from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from datasets.diagnostics import build_diagnostic_subset
from trainers.evaluation import (
    evaluate_with_fixed_thresholds,
    for_each_deduplicated_grid,
    tune_zone_thresholds,
)
from utils.config import ProjectConfig


RAIN_THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
RAIN_KEYS = {format(threshold, "g") for threshold in RAIN_THRESHOLDS}


class _PatchDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]], order: list[int] | None = None) -> None:
        self.records = records
        self.order = order or list(range(len(records)))

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        original = self.order[index]
        record = self.records[original]
        return {
            "x": torch.tensor([record["probability"]], dtype=torch.float32),
            "y_amount": torch.full((3, 1, 64, 64), float(record["target"])),
            "y_valid": torch.ones((3, 1, 64, 64), dtype=torch.bool),
            "_sample_index": torch.tensor(original, dtype=torch.int64),
        }

    def metadata(self, index: int) -> dict[str, object]:
        return self.records[index]


class _PatchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        probability = x[:, 0].reshape(-1, 1, 1, 1, 1).expand(-1, 3, 1, 64, 64)
        return {
            "occurrence_probability": probability,
            "ungated_amount": probability * 10,
            "coverage_zone": torch.zeros((len(x), 64, 64), dtype=torch.int8),
        }


class _ScalarLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, outputs, target, valid):
        self.calls += 1
        value = outputs["occurrence_probability"][:, 0, 0, 0, 0].mean()
        return {
            "loss": value,
            "focal_loss": value * 2,
            "hard_negative_selected": torch.tensor(99),
        }


def _evaluation_dataset() -> _PatchDataset:
    records = [
        {"forecast_time": "t2", "grid_row": 0, "grid_column": 0, "probability": 0.9, "target": 1.0},
        {"forecast_time": "t1", "grid_row": 32, "grid_column": 32, "probability": 0.8, "target": 6.0},
        {"forecast_time": "t1", "grid_row": 0, "grid_column": 0, "probability": 0.2, "target": 2.0},
    ]
    return _PatchDataset(records, order=[2, 1, 0])


class TestDiagnosticSubset(unittest.TestCase):
    def test_evenly_spaced_complete_times_and_local_metadata(self) -> None:
        records = [
            {"forecast_time": f"t{time}", "grid_row": row, "grid_column": 0,
             "probability": 0.5, "target": 1.0}
            for time in range(5)
            for row in (32, 0)
        ]
        dataset = _PatchDataset(records)
        groups = {f"t{time}": [2 * time, 2 * time + 1] for time in range(5)}

        subset, audit = build_diagnostic_subset(dataset, groups, 3)

        self.assertEqual(audit["selected_forecast_times"], ["t0", "t2", "t4"])
        self.assertEqual(audit["samples"], 6)
        self.assertEqual(subset.indices, [1, 0, 5, 4, 9, 8])
        self.assertEqual(subset.metadata(2)["forecast_time"], "t2")
        self.assertEqual(subset.metadata(2)["grid_row"], 0)
        self.assertEqual(subset[2]["_sample_index"].item(), 2)
        disabled, disabled_audit = build_diagnostic_subset(dataset, groups, 0)
        self.assertIsNone(disabled)
        self.assertFalse(disabled_audit["enabled"])


class TestEvaluationDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _evaluation_dataset()
        self.device = torch.device("cpu")

    def test_overlap_metadata_and_sample_weighted_loss(self) -> None:
        model = _PatchModel()
        criterion = _ScalarLoss()
        grids = []
        with patch("trainers.evaluation.GRID_SHAPE", (96, 96)), patch(
            "trainers.evaluation.assemble_model_input", side_effect=lambda batch, *_: batch["x"]
        ):
            losses = for_each_deduplicated_grid(
                model, self.dataset, None, None, self.device, 2,
                lambda *arrays: grids.append(arrays), criterion=criterion,
            )

        self.assertEqual(model.calls, 2)
        self.assertEqual(criterion.calls, 2)
        self.assertEqual(len(grids), 2)
        probability, amount, target, valid, zone = grids[0]
        self.assertAlmostEqual(float(probability[0, 0, 40, 40]), 0.5)
        self.assertAlmostEqual(float(amount[0, 0, 40, 40]), 5.0)
        self.assertAlmostEqual(float(target[0, 0, 40, 40]), 4.0)
        self.assertTrue(valid[0, 0, 40, 40])
        self.assertEqual(zone[40, 40], 0)
        self.assertAlmostEqual(losses["loss"], (0.2 + 0.8 + 0.9) / 3, places=6)
        self.assertAlmostEqual(losses["focal_loss"], 2 * (0.2 + 0.8 + 0.9) / 3, places=6)
        self.assertNotIn("hard_negative_selected", losses)

    def test_fixed_threshold_evaluation_does_not_search(self) -> None:
        model = _PatchModel()
        criterion = _ScalarLoss()
        thresholds = np.full((3, 1, 3), 0.5, dtype=np.float32)
        with patch("trainers.evaluation.GRID_SHAPE", (96, 96)), patch(
            "trainers.evaluation.assemble_model_input", side_effect=lambda batch, *_: batch["x"]
        ), patch("trainers.evaluation.ZoneThresholdSearch", side_effect=AssertionError("threshold search called")):
            result = evaluate_with_fixed_thresholds(
                model, self.dataset, None, None, self.device, thresholds, 2,
                criterion=criterion, disable_tqdm=True,
            )

        self.assertEqual(model.calls, 2)
        self.assertEqual(result["threshold_selection"]["status"], "fixed")
        self.assertEqual(result["threshold_selection"]["thresholds"], thresholds.tolist())
        self.assertAlmostEqual(result["losses"]["loss"], (0.2 + 0.8 + 0.9) / 3, places=6)
        self.assertEqual(result["rain_thresholds_mm_h"], RAIN_THRESHOLDS)
        self.assertEqual(set(result["summary"]["csi_by_threshold"]), RAIN_KEYS)

        # The two t1 patches overlap over 32x32 pixels; classify physical rain
        # after averaging both their probabilities and amounts on the grid.
        direct = result["by_lead_target_zone"][0]
        self.assertEqual(set(direct["csi_by_threshold"]), RAIN_KEYS)
        self.assertEqual(set(direct["counts_by_threshold"]), RAIN_KEYS)
        self.assertEqual(
            direct["counts_by_threshold"]["2"],
            {"tp": 4096, "fp": 4096, "fn": 3072, "positive": 7168},
        )
        self.assertEqual(
            direct["counts_by_threshold"]["5"],
            {"tp": 3072, "fp": 5120, "fn": 0, "positive": 3072},
        )
        self.assertEqual(
            direct["counts_by_threshold"]["10"],
            {"tp": 0, "fp": 0, "fn": 0, "positive": 0},
        )
        self.assertAlmostEqual(result["summary"]["csi_by_threshold"]["1"], 8192 / 11264)
        self.assertAlmostEqual(result["summary"]["csi_by_threshold"]["2"], 4096 / 11264)
        self.assertAlmostEqual(result["summary"]["csi_by_threshold"]["5"], 3072 / 8192)
        self.assertTrue(np.isnan(result["summary"]["csi_by_threshold"]["10"]))

    def test_diagnostic_fixed_score_uses_existing_second_pass(self) -> None:
        config = ProjectConfig()
        config.train.microbatch_size = 2
        config.train.disable_tqdm = True
        model = _PatchModel()
        criterion = _ScalarLoss()
        with patch("trainers.evaluation.GRID_SHAPE", (96, 96)), patch(
            "trainers.evaluation.assemble_model_input", side_effect=lambda batch, *_: batch["x"]
        ):
            result, selected = tune_zone_thresholds(
                model, self.dataset, None, None, self.device, config,
                criterion=criterion, diagnostics_probability_threshold=0.5,
            )

        self.assertEqual(model.calls, 4)  # Two batches in each of the existing two passes.
        self.assertEqual(criterion.calls, 2)  # Loss is evaluated in the metrics pass only.
        self.assertEqual(selected.shape, (3, 1, 3))
        self.assertEqual(result["fixed_threshold_metrics"]["threshold_selection"]["status"], "fixed")
        self.assertAlmostEqual(result["fixed_threshold_metrics"]["summary"]["csi"], 8192 / 11264)
        self.assertGreater(result["summary"]["csi"], result["fixed_threshold_metrics"]["summary"]["csi"])
        self.assertEqual(result["rain_thresholds_mm_h"], RAIN_THRESHOLDS)
        self.assertEqual(result["fixed_threshold_metrics"]["rain_thresholds_mm_h"], RAIN_THRESHOLDS)
        self.assertEqual(set(result["summary"]["csi_by_threshold"]), RAIN_KEYS)
        self.assertEqual(set(result["fixed_threshold_metrics"]["summary"]["csi_by_threshold"]), RAIN_KEYS)
        self.assertAlmostEqual(result["summary"]["csi_by_threshold"]["2"], 7168 / 11264)
        self.assertAlmostEqual(result["fixed_threshold_metrics"]["summary"]["csi_by_threshold"]["2"], 4096 / 11264)
        self.assertAlmostEqual(result["losses"]["loss"], (0.2 + 0.8 + 0.9) / 3, places=6)


if __name__ == "__main__":
    unittest.main()

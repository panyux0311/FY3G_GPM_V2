from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

from utils.config import TrainConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/default.yaml"
EXPERIMENT = ROOT / "configs/simvpv2_lr1e4_earlystop.yaml"


class TestTrainingConfig(unittest.TestCase):
    def test_experiment_changes_only_requested_training_options(self) -> None:
        baseline = load_config(DEFAULT)
        experiment = load_config(EXPERIMENT)

        self.assertEqual(baseline.name, "gpm_surface_v1")
        self.assertEqual(experiment.name, "gpm_surface_lr1e4_earlystop")
        self.assertEqual(baseline.data, experiment.data)
        self.assertEqual(baseline.model, experiment.model)
        self.assertEqual(experiment.model.architecture, "simvpv2")
        self.assertEqual(baseline.train.peak_learning_rate, 0.0002)
        self.assertEqual(experiment.train.peak_learning_rate, 0.0001)
        self.assertEqual(experiment.train.experiment_group, "simvpv2_lr1e4_earlystop")
        self.assertEqual(experiment.train.output_dir, "outputs/gpm_surface_lr1e4_earlystop")

        allowed_changes = {
            "experiment_group", "output_dir", "peak_learning_rate",
            "early_stopping_enabled", "diagnostics_enabled", "diagnostics_compare_raw_ema",
        }
        for field in fields(TrainConfig):
            if field.name not in allowed_changes:
                self.assertEqual(
                    getattr(baseline.train, field.name), getattr(experiment.train, field.name), field.name,
                )
        self.assertFalse(baseline.train.early_stopping_enabled)
        self.assertFalse(baseline.train.diagnostics_enabled)
        self.assertFalse(baseline.train.diagnostics_compare_raw_ema)
        self.assertTrue(experiment.train.early_stopping_enabled)
        self.assertTrue(experiment.train.diagnostics_enabled)
        self.assertTrue(experiment.train.diagnostics_compare_raw_ema)
        self.assertEqual(experiment.train.early_stopping_patience, 10)
        self.assertEqual(experiment.train.early_stopping_min_delta, 0.001)
        self.assertEqual(experiment.train.diagnostics_train_forecast_times, 8)
        self.assertEqual(experiment.train.diagnostics_every_epochs, 1)
        self.assertEqual(experiment.train.diagnostics_probability_threshold, 0.5)

    def test_invalid_early_stopping_and_diagnostics_options(self) -> None:
        bad_options = [
            ("train.early_stopping_patience=0", "early_stopping_patience"),
            ("train.early_stopping_min_delta=-0.001", "early_stopping_min_delta"),
            ("train.early_stopping_min_delta=.nan", "early_stopping_min_delta"),
            ("train.early_stopping_min_delta=.inf", "early_stopping_min_delta"),
            ("train.diagnostics_every_epochs=0", "diagnostics_every_epochs"),
            ("train.diagnostics_probability_threshold=0", "diagnostics_probability_threshold"),
            ("train.diagnostics_probability_threshold=1", "diagnostics_probability_threshold"),
        ]
        for override, field_name in bad_options:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, field_name):
                load_config(DEFAULT, overrides=[override])

        with self.assertRaisesRegex(ValueError, "diagnostics_train_forecast_times"):
            load_config(DEFAULT, overrides=[
                "train.diagnostics_enabled=true", "train.diagnostics_train_forecast_times=0",
            ])
        self.assertEqual(
            load_config(DEFAULT, overrides=["train.diagnostics_train_forecast_times=0"])
            .train.diagnostics_train_forecast_times,
            0,
        )

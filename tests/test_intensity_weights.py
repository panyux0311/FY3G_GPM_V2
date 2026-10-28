from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from utils.config import load_config
from utils.intensity_weights import derive_intensity_weights


class TestIntensityWeights(unittest.TestCase):
    def statistics(self):
        return {
            "group_count": [[[0, 10000, 100, 0]]] * 3,
            "rain_count": [[10100]] * 3,
            "valid_count": [[20000]] * 3,
            "intensity_edges_mm_h": [0, 0.1, 2, 5, 10],
            "sha256": "original", "manifest_sha256": "manifest",
        }

    def test_derivation_preserves_source_and_uses_counts(self):
        stats = self.statistics()
        original = deepcopy(stats)
        result = derive_intensity_weights(stats)
        self.assertEqual(stats, original)
        weights = np.asarray(result["group_weight"])
        self.assertEqual(weights.shape, (3, 1, 4))
        self.assertTrue(np.all(weights[..., 0] == 0))
        self.assertTrue(np.all(weights[..., 3] == 0))
        self.assertTrue(np.all(weights[..., 2] > weights[..., 1]))
        self.assertTrue(np.all(weights <= 10))
        self.assertEqual(result["weight_strategy"]["source_statistics_sha256"], "original")

    def test_no_rain_and_invalid_counts(self):
        stats = self.statistics()
        stats["group_count"] = [[[0, 0, 0, 0]]] * 3
        stats["rain_count"] = [[0]] * 3
        self.assertTrue(np.all(np.asarray(derive_intensity_weights(stats)["group_weight"]) == 0))
        stats["rain_count"] = [[1]] * 3
        with self.assertRaises(ValueError):
            derive_intensity_weights(stats)

    def test_new_config_loads(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v3.yaml")
        self.assertEqual(config.train.intensity_weight_strategy, "sqrt_inverse_frequency_v1")
        self.assertEqual(config.train.save_best_by_metrics, ["csi_0.1", "csi_2", "csi_5"])
        self.assertEqual(config.train.epochs, 25)

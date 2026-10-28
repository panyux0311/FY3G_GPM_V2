from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from schema import ARRAY_SPECS, DATASET_VERSION, GRID_DEGREES, GRID_SHAPE, INPUT_CHANNELS, INPUT_LEVELS_KM, PATCH_SIZE, PATCH_STRIDE, validate_dataset_contract


class TestDataContract(unittest.TestCase):
    def test_schema_contract(self) -> None:
        self.assertEqual(DATASET_VERSION, "fy3ggpm_surface_3h_l1only_0p1_v3")
        self.assertEqual((GRID_DEGREES, GRID_SHAPE), (0.1, (600, 900)))
        self.assertEqual(INPUT_CHANNELS, 43)
        self.assertEqual(ARRAY_SPECS["x_reflectivity"], (np.float16, (3, 2, 10, 64, 64)))
        self.assertEqual(ARRAY_SPECS["y_amount"], (np.float16, (3, 1, 64, 64)))

    def test_validate_dataset_contract_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                validate_dataset_contract(Path(tmpdir))

    def test_validate_dataset_contract_valid_and_invalid_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "source_manifest.json").write_text(
                json.dumps({
                    "dataset_version": DATASET_VERSION,
                    "manifest_sha256": "h123",
                    "data_config": {
                        "grid_degrees": GRID_DEGREES,
                        "patch_size": PATCH_SIZE,
                        "patch_stride": PATCH_STRIDE,
                        "input_levels_km": list(INPUT_LEVELS_KM),
                    },
                    "array_specs": {name: [str(dtype), list(shape)] for name, (dtype, shape) in ARRAY_SPECS.items()},
                }),
                encoding="utf-8",
            )
            stats = root / "statistics"
            stats.mkdir(parents=True)
            (stats / "input_statistics.json").write_text(
                json.dumps({"mean": [[0.0] * 10, [0.0] * 10], "std": [[1.0] * 10, [1.0] * 10], "manifest_sha256": "h123"}),
                encoding="utf-8",
            )
            (stats / "target_statistics.json").write_text(
                json.dumps({
                    "computed_from_split": "train",
                    "rain_log_mean": [[0.0], [0.0], [0.0]],
                    "rain_log_std": [[1.0], [1.0], [1.0]],
                    "intensity_edges_mm_h": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0],
                    "group_weight": [[[1.0] * 11], [[1.0] * 11], [[1.0] * 11]],
                    "threshold_positive_count": [[[10, 5, 2]], [[10, 5, 2]], [[10, 5, 2]]],
                    "manifest_sha256": "h123",
                }),
                encoding="utf-8",
            )
            # Valid case
            info = validate_dataset_contract(root)
            self.assertEqual(info["dataset_version"], DATASET_VERSION)

            manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
            manifest["data_config"]["grid_degrees"] = 0.25
            (root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "grid_degrees"):
                validate_dataset_contract(root)
            manifest["data_config"]["grid_degrees"] = GRID_DEGREES
            (root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            # Invalid input shape
            (stats / "input_statistics.json").write_text(
                json.dumps({"mean": [[0.0] * 5], "std": [[1.0] * 5]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_dataset_contract(root)

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from datasets.gpm_dataset import FY3G3HDataset
from datasets.gpm_inputs import assemble_model_input
from schema import ARRAY_SPECS, DATASET_VERSION


def _create_mock_dataset(root: Path) -> None:
    shard_dir = root / "shards/train/shard_00000"
    shard_dir.mkdir(parents=True)
    n_samples = 4
    for name, (dtype, shape) in ARRAY_SPECS.items():
        data = np.zeros((n_samples, *shape), dtype=dtype)
        np.save(shard_dir / f"{name}.npy", data)

    metadata_lines = [
        json.dumps({"sample_id": f"s_{i}", "forecast_time": "2025-07-01T03:00:00+00:00", "grid_row": 0, "grid_column": 0})
        for i in range(n_samples)
    ]
    (shard_dir / "metadata.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    (shard_dir / "shard_manifest.json").write_text(
        json.dumps({"status": "complete", "sample_count": n_samples}), encoding="utf-8"
    )
    (root / "source_manifest.json").write_text(
        json.dumps({
            "dataset_version": DATASET_VERSION,
            "sample_counts": {"train": n_samples, "val": 0, "test": 0},
            "manifest_sha256": "mock_hash",
        }),
        encoding="utf-8",
    )
    stats_dir = root / "statistics"
    stats_dir.mkdir(parents=True)
    (stats_dir / "input_statistics.json").write_text(
        json.dumps({
            "shape": [2, 10],
            "mean": [[0.0] * 10, [0.0] * 10],
            "std": [[1.0] * 10, [1.0] * 10],
            "manifest_sha256": "mock_hash",
        }),
        encoding="utf-8",
    )
    (stats_dir / "target_statistics.json").write_text(
        json.dumps({
            "shape": [3, 1],
            "computed_from_split": "train",
            "rain_log_mean": [[0.0], [0.0], [0.0]],
            "rain_log_std": [[1.0], [1.0], [1.0]],
            "rain_count": [[10], [10], [10]],
            "valid_count": [[100], [100], [100]],
            "q90_mm_h": [[5.0], [5.0], [5.0]],
            "intensity_edges_mm_h": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0],
            "group_weight": [[[1.0] * 11], [[1.0] * 11], [[1.0] * 11]],
            "threshold_positive_count": [[[10, 5, 2]], [[10, 5, 2]], [[10, 5, 2]]],
            "manifest_sha256": "mock_hash",
        }),
        encoding="utf-8",
    )


class TestDataset(unittest.TestCase):
    def test_dataset_loading_and_input_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_mock_dataset(root)
            dataset = FY3G3HDataset(root, "train")
            self.assertEqual(len(dataset), 4)

            sample = dataset[0]
            self.assertEqual(sample["x_reflectivity"].shape, (3, 2, 10, 64, 64))
            self.assertEqual(sample["y_amount"].shape, (3, 1, 64, 64))

            batch = torch.utils.data.default_collate([sample])
            inputs = assemble_model_input(batch)
            self.assertEqual(inputs.shape, (1, 3, 43, 64, 64))
            dataset.close()

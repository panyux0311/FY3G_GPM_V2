from __future__ import annotations

import unittest

import numpy as np
import torch

from schema import ARRAY_SPECS, GRID_SHAPE, PATCH_SIZE
from trainers.evaluation import for_each_deduplicated_grid


class _EdgeDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            name: torch.from_numpy(np.zeros(shape, dtype=dtype))
            for name, (dtype, shape) in ARRAY_SPECS.items()
        }
        sample["y_amount"].fill_(5)
        sample["y_valid"].fill_(1)
        return sample

    def metadata(self, index: int) -> dict[str, object]:
        return {"forecast_time": "2025-07-09T18:00:00+00:00", "grid_row": 512, "grid_column": 832}


class _ConstantModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (x.shape[0], 3, 1, PATCH_SIZE, PATCH_SIZE)
        return {
            "occurrence_probability": torch.full(shape, 0.8),
            "ungated_amount": torch.full(shape, 4.0),
            "coverage_zone": torch.zeros((x.shape[0], PATCH_SIZE, PATCH_SIZE), dtype=torch.int8),
        }


class TestEvaluationGrid(unittest.TestCase):
    def test_patch_beyond_old_grid_is_preserved(self) -> None:
        results = []
        for_each_deduplicated_grid(
            _ConstantModel(), _EdgeDataset(), torch.zeros((2, 10)), torch.ones((2, 10)),
            torch.device("cpu"), 1, lambda *arrays: results.append(arrays),
        )
        self.assertEqual(len(results), 1)
        probability, amount, target, valid, zone = results[0]
        self.assertEqual(probability.shape, (3, 1, *GRID_SHAPE))
        self.assertTrue(np.allclose(probability[:, :, 512:576, 832:896], 0.8))
        self.assertTrue(np.allclose(amount[:, :, 512:576, 832:896], 4.0))
        self.assertTrue(np.allclose(target[:, :, 512:576, 832:896], 5.0))
        self.assertTrue(valid[:, :, 512:576, 832:896].all())
        self.assertTrue((zone[512:576, 832:896] == 0).all())

from __future__ import annotations

import unittest
import torch

from losses.gpm_loss import ThreeHourStrongRainLoss
from models import build_model
from utils.config import ModelConfig


def _mock_stats() -> dict[str, object]:
    return {
        "valid_count": [[1000], [1000], [1000]],
        "rain_count": [[100], [100], [100]],
        "rain_log_mean": [[1.0], [1.0], [1.0]],
        "rain_log_std": [[0.5], [0.5], [0.5]],
        "q90_mm_h": [[5.0], [5.0], [5.0]],
        "intensity_edges_mm_h": [0.0, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 300.0001],
        "group_weight": [[[1.0] * 11]] * 3,
        "threshold_positive_count": [[[10, 5, 1]]] * 3,
    }


class TestLoss(unittest.TestCase):
    def test_loss_computation_and_backward(self) -> None:
        stats = _mock_stats()
        config = ModelConfig(architecture="unet_standard")
        model = build_model(config)
        model.configure_target_transform(stats, True)

        criterion = ThreeHourStrongRainLoss(stats)

        x = torch.zeros(1, 3, 43, 64, 64)
        x[:, :, 20:40] = 1.0
        out = model(x)

        target = torch.ones(1, 3, 1, 64, 64) * 2.0
        valid = torch.ones(1, 3, 1, 64, 64, dtype=torch.bool)

        loss_dict = criterion(out, target, valid)
        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() > 0.0)

        loss.backward()
        self.assertIsNotNone(model.enc1[0].weight.grad)
        self.assertTrue(torch.isfinite(model.enc1[0].weight.grad).all())

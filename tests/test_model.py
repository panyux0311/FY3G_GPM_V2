from __future__ import annotations

import unittest
from pathlib import Path
import torch

from losses.gpm_loss import ThreeHourStrongRainLoss
from models import SimVPv2, UNetStandard, build_model, model_parameter_count
from utils.config import ModelConfig, load_config

from tests.test_loss import _mock_stats


class TestModel(unittest.TestCase):
    def test_unet_standard_architecture(self) -> None:
        config = ModelConfig(architecture="unet_standard")
        model = build_model(config)
        self.assertIsInstance(model, UNetStandard)

        count = model_parameter_count(model)
        self.assertTrue(30_000_000 <= count["total"] <= 40_000_000)
        self.assertEqual(count["trainable"], count["total"])

        # Test forward pass with 43 input channels (B=2, steps=3, channels=43, 64, 64)
        x = torch.zeros(2, 3, 43, 64, 64)
        x[:, :, 20:40] = 1.0

        out = model(x)
        self.assertEqual(out["prediction"].shape, (2, 3, 1, 64, 64))
        self.assertEqual(out["occurrence_probability"].shape, (2, 3, 1, 64, 64))
        self.assertEqual(out["ungated_amount"].shape, (2, 3, 1, 64, 64))
        self.assertEqual(out["intensity_exceedance_logits"].shape, (2, 3, 1, 3, 64, 64))
        self.assertEqual(out["coverage_zone"].shape, (2, 64, 64))

    def test_occurrence_thresholds_setting(self) -> None:
        config = ModelConfig(architecture="unet_standard")
        model = build_model(config)
        model.set_occurrence_thresholds([0.3, 0.4, 0.5])
        self.assertTrue(torch.allclose(model.occurrence_thresholds, torch.tensor([0.3, 0.4, 0.5])))
        self.assertEqual(tuple(model.occurrence_thresholds_by_zone.shape), (3, 1, 3))

    def test_simvpv2_config_and_loss_backward(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v1.yaml")
        self.assertEqual(config.model.architecture, "simvpv2")
        self.assertEqual(config.train.output_dir, "outputs/gpm_surface_v1")
        self.assertEqual(config.name, "gpm_surface_v1")
        config.model.simvpv2_hidden_spatial = 8
        config.model.simvpv2_hidden_temporal = 16
        config.model.simvpv2_spatial_layers = 2
        config.model.simvpv2_temporal_layers = 2
        model = build_model(config.model)
        self.assertIsInstance(model, SimVPv2)
        stats = _mock_stats()
        model.configure_target_transform(stats, True)
        with torch.no_grad():
            model.occurrence_head.weight.fill_(0.01)
            model.amount_head.weight.fill_(0.01)
        x = torch.zeros(1, 3, 43, 64, 64)
        x[:, :, 20:40] = 1.0
        output = model(x)
        self.assertEqual(output["prediction"].shape, (1, 3, 1, 64, 64))
        self.assertEqual(output["intensity_exceedance_logits"].shape, (1, 3, 1, 3, 64, 64))
        self.assertEqual(output["coverage_zone"].shape, (1, 64, 64))
        self.assertEqual(output["feature"].shape[1], 48)
        criterion = ThreeHourStrongRainLoss(stats)
        target = torch.full((1, 3, 1, 64, 64), 2.0)
        valid = torch.ones_like(target, dtype=torch.bool)
        loss = criterion(output, target, valid)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model.encoder[0][0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_simvpv2_rejects_invalid_spatial_layers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "spatial_layers"):
            load_config(root / "configs/gpm_surface_v1.yaml", overrides=[
                "model.architecture=simvpv2", "model.simvpv2_spatial_layers=3",
            ])

    def test_default_config_matches_server_run(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v1.yaml")
        self.assertEqual(config.name, "gpm_surface_v1")
        self.assertEqual(config.data.dataset_root, "/lixj/pyx/dataset/FY3G_after")
        self.assertEqual(config.model.architecture, "simvpv2")
        self.assertEqual(config.train.epochs, 100)
        self.assertEqual(config.train.device, "cuda")
        self.assertEqual(config.train.output_dir, "outputs/gpm_surface_v1")

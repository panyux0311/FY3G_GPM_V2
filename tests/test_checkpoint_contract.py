from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from models import build_model
from schema import DATASET_VERSION
from trainers.checkpoint import (
    CHECKPOINT_VERSION, load_checkpoint, save_checkpoint,
    validate_checkpoint_architecture, validate_checkpoint_dataset,
)
from utils.config import ModelConfig, ProjectConfig


class TestCheckpointContract(unittest.TestCase):
    def test_dataset_version_and_manifest_match(self) -> None:
        self.assertEqual(CHECKPOINT_VERSION, DATASET_VERSION)
        payload = {
            "input_statistics": {"manifest_sha256": "current"},
            "target_statistics": {"manifest_sha256": "current"},
        }
        validate_checkpoint_dataset(payload, "current")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            validate_checkpoint_dataset(payload, "other")

    def test_model_choice_must_match_checkpoint(self) -> None:
        payload = {"config": {"model": {"architecture": "simvpv2"}}}
        validate_checkpoint_architecture(payload, "simvpv2")
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint_architecture(payload, "unet_standard")

    def test_simvpv2_checkpoint_round_trip(self) -> None:
        config = ProjectConfig(model=ModelConfig(
            architecture="simvpv2", simvpv2_hidden_spatial=8,
            simvpv2_hidden_temporal=16, simvpv2_spatial_layers=2,
            simvpv2_temporal_layers=2,
        ))
        model = build_model(config.model)
        optimizer = torch.optim.AdamW(model.parameters())
        stats = {"manifest_sha256": "current"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(path, model, model, optimizer, 1, config, stats, stats, None)
            payload = load_checkpoint(path, torch.device("cpu"))
            validate_checkpoint_dataset(payload, "current")
            validate_checkpoint_architecture(payload, "simvpv2")
            restored = build_model(config.model)
            restored.load_state_dict(payload["model_state"])

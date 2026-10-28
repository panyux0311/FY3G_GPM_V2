from __future__ import annotations

from copy import deepcopy
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trainers.gpm_trainer import GPMTrainer
from utils.config import load_config
from utils.paths import check_output_dir_safety


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"prediction": inputs * self.weight}

    def set_occurrence_thresholds_by_zone(self, thresholds: object) -> None:
        self.thresholds = thresholds


class _TinyLoss:
    hard_negative_weight = 0.0

    def __call__(self, outputs: dict[str, torch.Tensor], target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": ((outputs["prediction"] - target) ** 2).mean()}


class _OneBatchLoader:
    batch_sampler = object()

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield {
            "x": torch.ones((1, 1)),
            "y_amount": torch.zeros((1, 1)),
            "y_valid": torch.ones((1, 1), dtype=torch.bool),
        }


class TestTrainingEpochReporting(unittest.TestCase):
    def test_existing_metrics_cannot_be_overwritten_implicitly(self) -> None:
        with TemporaryDirectory() as directory:
            (Path(directory) / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                check_output_dir_safety(directory)

    def test_every_epoch_validates_logs_and_records_metrics(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v1.yaml", overrides=[
            "train.epochs=2", "train.effective_batch_size=1", "train.microbatch_size=1",
            "train.disable_tqdm=true", "train.device=cpu",
        ])
        model = _TinyModel()
        log_stream = io.StringIO()
        logger = logging.getLogger("test_training_epoch_reporting")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(log_stream))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        context = SimpleNamespace(
            config=config, output_dir=Path("unused"), resume_checkpoint=None,
            start_epoch=1, train_loader=_OneBatchLoader(), model=model,
            ema=deepcopy(model), optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            criterion=_TinyLoss(), device=torch.device("cpu"), mean=None, std=None,
            val_dataset=None, inputs={}, targets={}, logger=logger,
        )
        validation_epochs = []
        checkpoint_paths = []
        written = {}

        def evaluate(*args, **kwargs):
            epoch = len(validation_epochs) + 1
            validation_epochs.append(epoch)
            return {
                "summary": {
                    "csi": epoch / 10, "pod": 0.2, "far": 0.3,
                    "mae_mm_h": 1.0, "rmse_mm_h": 2.0,
                    "balanced_score": epoch / 10,
                },
                "threshold_selection": {"thresholds": [[[0.5]]]},
            }

        def write_text(path, content, **kwargs):
            written[path.name] = content
            return len(content)

        with TemporaryDirectory() as directory, \
             patch("trainers.gpm_trainer.assemble_model_input", side_effect=lambda batch, *_: batch["x"]), \
             patch("trainers.gpm_trainer.evaluate_deduplicated", side_effect=evaluate), \
             patch("trainers.gpm_trainer.save_checkpoint", side_effect=lambda **kwargs: checkpoint_paths.append(kwargs["path"].name)):
            context.output_dir = Path(directory)
            result = GPMTrainer(context).train()
            document = json.loads((context.output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(p.name for p in context.output_dir.glob("*.json")), ["metrics.json"])

        self.assertEqual(validation_epochs, [1, 2])
        history = document["epochs"]
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        self.assertEqual([row["validation"]["csi"] for row in history], [0.1, 0.2])
        self.assertIn("Epoch 1/2", log_stream.getvalue())
        self.assertIn("Epoch 2/2", log_stream.getvalue())
        self.assertIn("val_CSI=0.200000", log_stream.getvalue())
        self.assertEqual(checkpoint_paths.count("last_resume.pt"), 2)
        self.assertEqual(checkpoint_paths.count("best_balanced_inference.pt"), 2)
        self.assertEqual(checkpoint_paths.count("epoch2_inference.pt"), 1)
        self.assertEqual(result["status"], "complete")

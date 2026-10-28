from __future__ import annotations

from copy import deepcopy
from tempfile import TemporaryDirectory
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.diagnostics import build_diagnostic_subset
from losses.gpm_loss import ThreeHourStrongRainLoss
from models import build_model
from schema import ARRAY_SPECS
from tests.test_loss import _mock_stats
from tests.test_training_epoch_reporting import _OneBatchLoader, _TinyLoss, _TinyModel
from trainers.gpm_trainer import GPMTrainer
from utils.config import load_config


RAIN_THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
RAIN_KEYS = {format(threshold, "g") for threshold in RAIN_THRESHOLDS}


class _MemoryDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def metadata(self, index):
        return {"forecast_time": f"2025-07-01T0{index}:00:00", "grid_row": 0, "grid_column": 0}

    def __getitem__(self, index):
        sample = {name: torch.from_numpy(np.zeros(shape, dtype=dtype))
                  for name, (dtype, shape) in ARRAY_SPECS.items()}
        sample["x_reflectivity"].fill_(index + 1)
        sample["x_valid"].fill_(1)
        sample["y_amount"][:, :, 16:48, 16:48] = index + 2
        sample["y_valid"].fill_(1)
        sample["_sample_index"] = torch.tensor(index)
        return sample


class TestTrainingDiagnostics(unittest.TestCase):
    def test_frequency_controls_fixed_metrics_and_extra_evaluations(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v2.yaml", overrides=[
            "train.epochs=4", "train.effective_batch_size=1", "train.microbatch_size=1",
            "train.disable_tqdm=true", "train.device=cpu", "train.diagnostics_every_epochs=3",
        ])
        model = _TinyModel()
        logger = logging.getLogger("test_diagnostic_frequency")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        context = SimpleNamespace(
            config=config, output_dir=Path("unused"), resume_checkpoint=None, resume_payload=None,
            start_epoch=1, train_loader=_OneBatchLoader(), model=model, ema=deepcopy(model),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01), criterion=_TinyLoss(),
            device=torch.device("cpu"), mean=None, std=None, val_dataset=None,
            train_diagnostic_dataset=[None], inputs={}, targets={}, logger=logger,
        )
        written, thresholds_requested = {}, []
        metrics = {"summary": dict(csi=0.3, balanced_score=0.3, pod=0.5, far=0.2,
                                   mae_mm_h=1.0, rmse_mm_h=2.0, cc=0.7,
                                   csi_by_threshold={key: 0.3 for key in RAIN_KEYS}),
                   "rain_thresholds_mm_h": RAIN_THRESHOLDS,
                   "losses": {"loss": 0.5}, "by_lead_target_zone": [],
                   "threshold_selection": {"thresholds": [[[0.5] * 3]] * 3}}

        def evaluate(*args, **kwargs):
            thresholds_requested.append(kwargs["diagnostics_probability_threshold"])
            value = deepcopy(metrics)
            if kwargs["diagnostics_probability_threshold"] is not None:
                value["fixed_threshold_metrics"] = deepcopy(metrics)
            return value

        def write_text(path, content, **kwargs):
            written[path.name] = content
            return len(content)

        with TemporaryDirectory() as directory, \
             patch("trainers.gpm_trainer.assemble_model_input", side_effect=lambda batch, *_: batch["x"]), \
             patch("trainers.gpm_trainer.evaluate_deduplicated", side_effect=evaluate), \
             patch("trainers.gpm_trainer.evaluate_with_fixed_thresholds", side_effect=lambda *a, **kw: deepcopy(metrics)) as extra, \
             patch("trainers.gpm_trainer.save_checkpoint"):
            context.output_dir = Path(directory)
            GPMTrainer(context).train()
            document = json.loads((context.output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(thresholds_requested, [0.5, None, 0.5, 0.5])
        self.assertEqual(extra.call_count, 6)  # Train subset + raw validation in each diagnostic epoch.
        self.assertEqual([r["epoch"] for r in document["epochs"]], [1, 2, 3, 4])
        for call in extra.call_args_list:
            self.assertTrue(np.all(call.kwargs["thresholds"] == 0.5))
            self.assertIs(call.kwargs["criterion"], context.criterion)

    def test_real_model_loss_evaluation_and_checkpoint_flow(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v2.yaml", overrides=[
            "train.epochs=2", "train.effective_batch_size=2", "train.microbatch_size=2",
            "train.device=cpu", "train.disable_tqdm=true",
            "model.simvpv2_hidden_spatial=8", "model.simvpv2_hidden_temporal=16",
            "model.simvpv2_spatial_layers=2", "model.simvpv2_temporal_layers=2",
        ])
        torch.manual_seed(14)
        dataset = _MemoryDataset()
        subset, audit = build_diagnostic_subset(dataset, {dataset.metadata(i)["forecast_time"]: [i] for i in range(2)}, 8)
        model = build_model(config.model)
        stats = _mock_stats()
        stats["manifest_sha256"] = "fixture"
        model.configure_target_transform(stats, True)
        before = model.encoder[0][0].weight.detach().clone()
        logger = logging.getLogger("test_real_training_diagnostics")
        logger.handlers.clear()
        log = io.StringIO()
        logger.addHandler(logging.StreamHandler(log))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        context = SimpleNamespace(
            config=config, output_dir=Path("unused"), resume_checkpoint=None,
            resume_payload=None, start_epoch=1, train_loader=DataLoader(dataset, batch_size=2),
            model=model, ema=deepcopy(model), optimizer=torch.optim.AdamW(model.parameters()),
            criterion=ThreeHourStrongRainLoss(stats), device=torch.device("cpu"),
            mean=torch.zeros(2, 10), std=torch.ones(2, 10), val_dataset=dataset,
            train_diagnostic_dataset=subset, inputs={"manifest_sha256": "fixture"},
            targets=stats, logger=logger,
        )
        written, checkpoints = {}, {}
        original_save = torch.save

        def save_to_memory(payload, path):
            buffer = io.BytesIO()
            original_save(payload, buffer)
            checkpoints[path.name] = buffer.getvalue()

        def write_text(path, content, **kwargs):
            written[path.name] = content
            return len(content)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with TemporaryDirectory() as directory, \
                 patch("trainers.evaluation.GRID_SHAPE", (64, 64)), \
                 patch("trainers.checkpoint.torch.save", side_effect=save_to_memory):
                context.output_dir = Path(directory)
                result = GPMTrainer(context).train()
                document = json.loads((context.output_dir / "metrics.json").read_text(encoding="utf-8"))
        finally:
            torch.set_num_threads(previous_threads)

        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(torch.equal(before, model.encoder[0][0].weight))
        self.assertEqual(audit["samples"], 2)
        history = document["epochs"]
        for record in history:
            self.assertTrue(np.isfinite(record["validation_loss"]))
            self.assertIn("cc", record["validation"])
            self.assertEqual(set(record["validation"]["csi_by_threshold"]), {"0.1", "2", "5"})
        for record in torch.load(io.BytesIO(checkpoints["last_resume.pt"]), weights_only=False)["history"]:
            self.assertIn("validation", record)
        payload = torch.load(io.BytesIO(checkpoints["last_resume.pt"]), weights_only=False)
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["training_state"]["early_stopping"]["patience"], 10)
        self.assertEqual(len(payload["history"]), 2)
        self.assertIn("best_metrics", payload["training_state"])
        self.assertIn("val_CC=", log.getvalue())
        self.assertIn("Diagnostics val_raw", log.getvalue())
        self.assertIn("Validation lead=+1h zone=direct", log.getvalue())
        for threshold in RAIN_KEYS:
            self.assertIn(f"val_CSI@{threshold}=", log.getvalue())
            self.assertRegex(log.getvalue(), rf"Diagnostics val_raw [^\n]*CSI@{threshold}=")
            self.assertRegex(log.getvalue(), rf"Validation lead=\+1h zone=direct [^\n]*CSI@{threshold}=")

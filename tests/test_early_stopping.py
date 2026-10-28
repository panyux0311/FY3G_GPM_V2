from __future__ import annotations

import io
import unittest
from copy import deepcopy
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trainers.early_stopping import EarlyStopping
from trainers.gpm_trainer import GPMTrainer
from tests.test_training_epoch_reporting import _OneBatchLoader, _TinyLoss, _TinyModel
from utils.config import load_config


class TestEarlyStopping(unittest.TestCase):
    def test_small_gains_do_not_reset_patience_but_accumulated_gain_does(self) -> None:
        stop = EarlyStopping(True, patience=2, min_delta=0.001)
        self.assertFalse(stop.update(0.30, 1))
        self.assertFalse(stop.update(0.3005, 2))
        self.assertFalse(stop.update(0.3015, 3))
        self.assertEqual(stop.bad_epochs, 0)
        self.assertFalse(stop.update(0.3016, 4))
        self.assertTrue(stop.update(0.3017, 5))
        self.assertEqual(stop.best_epoch, 3)

    def test_disabled_stopping_and_undefined_metric(self) -> None:
        stop = EarlyStopping(False, patience=1)
        self.assertFalse(stop.update(float("nan"), 1))
        self.assertFalse(stop.update(0.2, 2))
        self.assertFalse(stop.update(0.1, 3))
        self.assertEqual(stop.best_epoch, 2)

    def test_restored_counter_stops_at_same_epoch(self) -> None:
        original = EarlyStopping(True, patience=2)
        original.update(0.3, 1)
        original.update(0.29, 2)
        restored = EarlyStopping(True, patience=2)
        restored.load_state_dict(original.state_dict())
        self.assertTrue(original.update(0.28, 3))
        self.assertTrue(restored.update(0.28, 3))
        self.assertEqual(original.state_dict(), restored.state_dict())

    def _context(self, start_epoch: int = 1, payload: dict | None = None):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/gpm_surface_v1.yaml", overrides=[
            "train.epochs=8", "train.effective_batch_size=1", "train.microbatch_size=1",
            "train.disable_tqdm=true", "train.device=cpu", "train.early_stopping_enabled=true",
            "train.early_stopping_patience=2", "train.early_stopping_min_delta=0.001",
        ])
        model = _TinyModel()
        stream = io.StringIO()
        logger = logging.getLogger(f"test_early_stopping_{start_epoch}")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(stream))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return SimpleNamespace(
            config=config, output_dir=Path("unused"),
            resume_checkpoint="resume.pt" if payload is not None else None,
            resume_payload=payload, start_epoch=start_epoch,
            train_loader=_OneBatchLoader(), model=model, ema=deepcopy(model),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01), criterion=_TinyLoss(),
            device=torch.device("cpu"), mean=None, std=None, val_dataset=None,
            inputs={}, targets={}, logger=logger, stream=stream,
        )

    def _run(self, context, scores, exists=None):
        checkpoints, writes, evaluated = [], {}, []

        def evaluate(*args, **kwargs):
            score = scores[len(evaluated)]
            evaluated.append(score)
            return {
                "summary": dict(csi=score, balanced_score=score, pod=0.5, far=0.3,
                                mae_mm_h=1.0, rmse_mm_h=2.0, cc=0.7),
                "losses": {"loss": 0.6, "focal_loss": 0.2},
                "threshold_selection": {"thresholds": [[[0.5]]]},
            }

        def write_text(path, content, **kwargs):
            writes[path.name] = content
            return len(content)

        with TemporaryDirectory() as directory, \
             patch("trainers.gpm_trainer.assemble_model_input", side_effect=lambda batch, *_: batch["x"]), \
             patch("trainers.gpm_trainer.evaluate_deduplicated", side_effect=evaluate), \
             patch("trainers.gpm_trainer.save_checkpoint", side_effect=lambda **kwargs: checkpoints.append(deepcopy(kwargs))), \
             patch.object(Path, "exists", new=exists or (lambda path: False)):
            context.output_dir = Path(directory)
            result = GPMTrainer(context).train()
            if (context.output_dir / "metrics.json").is_file():
                writes["metrics.json"] = (context.output_dir / "metrics.json").read_text(encoding="utf-8")
        return result, checkpoints, writes, evaluated

    def test_stopping_epoch_saved_and_best_preserved(self) -> None:
        result, saves, writes, evaluated = self._run(self._context(), [0.3, 0.3005, 0.299])
        self.assertEqual(evaluated, [0.3, 0.3005, 0.299])
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual((result["epochs"], result["planned_epochs"], result["best_epoch"]), (3, 8, 2))
        best = [s for s in saves if s["path"].name == "best_balanced_inference.pt"]
        self.assertEqual([s["epoch"] for s in best], [1, 2])
        self.assertEqual(saves[-1]["path"].name, "last_resume.pt")
        self.assertEqual(saves[-1]["training_state"]["early_stopping"]["bad_epochs"], 2)
        self.assertTrue(any(s["path"].name == "epoch3_inference.pt" for s in saves))
        history = json.loads(writes["metrics.json"])["epochs"]
        self.assertEqual(history[-1]["validation_loss"], 0.6)
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3])

    def test_resume_preserves_patience_without_local_history(self) -> None:
        stop = EarlyStopping(True, patience=2)
        stop.update(0.3, 1)
        stop.update(0.299, 2)
        payload = {
            "best_score": 0.3, "metrics": None,
            "history": [{"epoch": 1, "validation": {"csi": 0.3}},
                        {"epoch": 2, "validation": {"csi": 0.299}}],
            "training_state": {"early_stopping": stop.state_dict(), "best_epoch": 1},
        }
        result, saves, writes, evaluated = self._run(self._context(3, payload), [0.298])
        self.assertEqual(evaluated, [0.298])
        self.assertEqual(result["epochs"], 3)
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(len(json.loads(writes["metrics.json"])["epochs"]), 3)
        self.assertEqual(saves[-1]["training_state"]["early_stopping"]["bad_epochs"], 2)

    def test_legacy_checkpoint_replays_history_before_resuming(self) -> None:
        payload = {
            "best_score": 0.3, "metrics": None,
            "history": [{"epoch": 1, "validation": {"csi": 0.3}},
                        {"epoch": 2, "validation": {"csi": 0.299}}],
        }
        result, saves, _, evaluated = self._run(self._context(3, payload), [0.298])
        self.assertEqual(evaluated, [0.298])
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(saves[-1]["training_state"]["early_stopping"]["bad_epochs"], 2)

    def test_enabling_stopping_on_exhausted_history_runs_no_extra_epoch(self) -> None:
        previous = EarlyStopping(False, patience=2)
        for epoch, score in enumerate((0.3, 0.29, 0.28), 1):
            previous.update(score, epoch)
        payload = {"best_score": 0.3, "metrics": None,
                   "training_state": {"early_stopping": previous.state_dict(), "best_epoch": 1}}
        result, saves, _, evaluated = self._run(self._context(4, payload), [])
        self.assertEqual(evaluated, [])
        self.assertEqual(saves, [])
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(result["epochs"], 3)

    def test_resume_rejects_best_checkpoint_from_other_data_or_future_epoch(self) -> None:
        context = self._context(3, {"best_score": 0.3, "metrics": None})
        context.inputs = {"manifest_sha256": "current"}
        for epoch, manifest, error in ((3, "current", "newer"), (1, "other", "manifest_sha256")):
            best = {"epoch": epoch, "config": {"model": {"architecture": "simvpv2"}},
                    "input_statistics": {"manifest_sha256": manifest},
                    "target_statistics": {"manifest_sha256": manifest}}
            with self.subTest(epoch=epoch, manifest=manifest), \
                 patch.object(Path, "exists", new=lambda path: path.name == "best_balanced_inference.pt"), \
                 patch("trainers.gpm_trainer.load_checkpoint", return_value=best):
                with self.assertRaisesRegex(ValueError, error):
                    GPMTrainer(context).train()

    def test_checkpoint_history_takes_priority_over_local_history(self) -> None:
        payload = {"best_score": 0.3, "metrics": None,
                   "history": [{"epoch": 1, "validation": {"csi": 0.3}},
                               {"epoch": 2, "validation": {"csi": 0.29}}]}
        with patch("trainers.gpm_trainer.json.loads", side_effect=AssertionError("stale local history parsed")):
            result, _, _, evaluated = self._run(
                self._context(3, payload), [0.28], exists=lambda path: path.name == "metrics.json",
            )
        self.assertEqual(evaluated, [0.28])
        self.assertEqual(result["status"], "early_stopped")

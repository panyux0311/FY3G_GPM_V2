"""Regression coverage for the opt-in zone-coupled experiment."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.samplers import ForecastTimeRoundRobinBatchSampler, find_dry_uncovered_patches
from losses.gpm_loss import ThreeHourStrongRainLoss, build_loss
from metrics.gpm_metrics import GridMetrics, selection_summary
from metrics.thresholds import ZoneThresholdSearch
from models import build_model
from tests.test_evaluation_diagnostics import _PatchDataset, _PatchModel
from tests.test_loss import _mock_stats
from tests.test_training_diagnostics import _MemoryDataset
from tests.test_training_epoch_reporting import _OneBatchLoader, _TinyLoss, _TinyModel
from trainers.checkpoint import load_checkpoint, validate_checkpoint_architecture, validate_checkpoint_selection
from trainers.evaluation import _patch_blend_weights, for_each_deduplicated_grid
from trainers.gpm_trainer import GPMTrainer
from utils.config import ModelConfig, load_config
from utils.rain_gate import gate_rain_numpy, gate_rain_torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/simvpv2_zone_coupled_v2.yaml"


def tiny_config():
    return load_config(CONFIG, overrides=[
        "train.device=cpu", "train.disable_tqdm=true", "train.epochs=2",
        "train.effective_batch_size=2", "train.microbatch_size=2",
        "train.num_workers=0", "train.diagnostics_enabled=false",
        "model.simvpv2_hidden_spatial=4", "model.simvpv2_hidden_temporal=8",
        "model.simvpv2_spatial_layers=2", "model.simvpv2_temporal_layers=2",
    ])


def loss_outputs(zone=0):
    shape = (1, 3, 1, 8, 8)
    return {
        "occurrence_logits": torch.zeros(shape, requires_grad=True),
        "amount_normalized": torch.zeros(shape, requires_grad=True),
        "amount_log": torch.ones(shape, requires_grad=True),
        "intensity_exceedance_logits": torch.zeros((1, 3, 1, 3, 8, 8), requires_grad=True),
        "coverage_zone": torch.full((1, 8, 8), zone),
    }


class TestZoneConfig(unittest.TestCase):
    def test_new_experiment_and_legacy_defaults(self):
        config = load_config(CONFIG)
        self.assertEqual(config.train.zone_loss_weights, [1, .7, .05])
        self.assertEqual(config.train.primary_selection_domain, "observable")
        self.assertEqual(config.train.uncovered_patch_keep_ratio, .25)
        self.assertEqual(config.train.epochs, 25)
        self.assertEqual(config.train.seed, 14)
        self.assertTrue(config.model.simvpv2_multi_lead)
        old = load_config(ROOT / "configs/simvpv2_fixed_weights_short_schedule.yaml")
        self.assertFalse(old.model.simvpv2_multi_lead)
        self.assertFalse(old.model.exceedance_head_enabled)
        self.assertEqual(old.train.zone_loss_weights, [1, 1, 1])
        self.assertEqual(old.train.primary_selection_domain, "macro_all")
        self.assertEqual(old.model.rain_gate_mode, "hard")

    def test_invalid_options_rejected(self):
        for override in (
            "train.zone_loss_weights=[1,0]", "train.zone_loss_weights=[1,-1,0]",
            "train.zone_loss_weights=[0,0,0]", "train.expected_amount_weight=.nan",
            "train.uncovered_patch_keep_ratio=1.1", "train.validation_far_max_by_zone=[0.1,0.2,2]",
            "train.primary_selection_domain=all", "model.rain_gate_mode=unknown",
            "model.patch_blend_min_weight=0", "model.architecture=unet_standard",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(CONFIG, overrides=[override])


class TestZoneLoss(unittest.TestCase):
    def test_uncovered_penalty_attenuates_even_without_covered_pixels(self):
        loss = build_loss(_mock_stats(), tiny_config().train)
        target = torch.full((1, 3, 1, 8, 8), 10.)
        target[..., :4, :] = 0
        valid = torch.ones_like(target, dtype=torch.bool)
        direct = loss(loss_outputs(0), target, valid)
        uncovered = loss(loss_outputs(2), target, valid)
        for name in ("focal_loss", "calibration_bce_loss", "rain_amount_loss",
                     "dry_amount_loss", "exceedance_loss", "expected_amount_loss"):
            with self.subTest(name=name):
                self.assertGreater(float(direct[name].detach()), 0)
                torch.testing.assert_close(uncovered[name], direct[name] * .05)

    def test_expected_loss_couples_heads_and_dry_probability_is_detached(self):
        criterion = build_loss(_mock_stats(), tiny_config().train)
        outputs = loss_outputs()
        target = torch.full_like(outputs["amount_log"], 2)
        result = criterion(outputs, target, torch.ones_like(target, dtype=torch.bool))
        expected = .5 * (.5 - math.log1p(2)) ** 2
        self.assertAlmostEqual(float(result["expected_amount_loss"].detach()), expected, places=6)
        gradients = torch.autograd.grad(result["expected_amount_loss"],
                                        (outputs["occurrence_logits"], outputs["amount_log"]))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
        outputs = loss_outputs()
        target.zero_()
        dry = criterion(outputs, target, torch.ones_like(target, dtype=torch.bool))["dry_amount_loss"]
        gradients = torch.autograd.grad(dry, (outputs["occurrence_logits"], outputs["amount_log"]), allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertGreater(float(gradients[1].abs().sum()), 0)

    def test_positive_balance_finite_backward_and_empty_mask(self):
        criterion = build_loss(_mock_stats(), tiny_config().train)
        torch.testing.assert_close(criterion.exceed_pos_weight[0], torch.tensor([math.sqrt(99), math.sqrt(199), 25]))
        outputs = loss_outputs()
        target = torch.full_like(outputs["amount_log"], 25)
        result = criterion(outputs, target, torch.ones_like(target, dtype=torch.bool))
        self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
        result["loss"].backward()
        for name in ("occurrence_logits", "amount_normalized", "amount_log", "intensity_exceedance_logits"):
            self.assertTrue(torch.isfinite(outputs[name].grad).all())
            self.assertGreater(float(outputs[name].grad.abs().sum()), 0)
        outputs = loss_outputs(2)
        result = criterion(outputs, target, torch.zeros_like(target, dtype=torch.bool))
        self.assertEqual(float(result["loss"].detach()), 0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(outputs["occurrence_logits"].grad).all())


class TestMultiLeadModel(unittest.TestCase):
    def test_forecast_planes_and_masks_reach_decoder(self):
        torch.manual_seed(14)
        model = build_model(tiny_config().model).eval()
        model.configure_target_transform(_mock_stats())
        x = torch.randn(1, 3, 43, 64, 64)
        x[:, :, 20:40] = 0
        x[:, 0, 20:40, :32] = 1
        x[:, 1, 20:40, 32:] = 1
        captured = {}
        def capture_skip(module, args):
            captured["skip"] = args[0].detach().clone()
        def capture_decoder(module, args):
            captured["decoder"] = args[0].detach().clone()
        handles = [model.skip_fusion.register_forward_pre_hook(capture_skip),
                   model.decoder[-1].register_forward_pre_hook(capture_decoder)]
        try:
            out = model(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(out["prediction"].shape), (1, 3, 1, 64, 64))
        self.assertEqual(tuple(out["intensity_exceedance_logits"].shape), (1, 3, 1, 3, 64, 64))
        skip = captured["skip"].reshape(1, 3, 4, 64, 64)
        self.assertGreater(float(skip[:, 0, :, :32].abs().sum()), 0)
        self.assertGreater(float(skip[:, 1, :, 32:].abs().sum()), 0)
        self.assertEqual(float(skip[:, 0, :, 32:].abs().sum()), 0)
        self.assertEqual(float(skip[:, 2].abs().sum()), 0)
        self.assertEqual(captured["decoder"].shape[0], 3)
        self.assertFalse(torch.equal(captured["decoder"][0], captured["decoder"][1]))

    def test_grouped_heads_do_not_mix_forecast_planes(self):
        model = build_model(tiny_config().model)
        with torch.no_grad():
            model.occurrence_head.weight.fill_(1)
            model.occurrence_head.bias.zero_()
            model.exceedance_head.weight.fill_(1)
            model.exceedance_head.bias.zero_()
        decoded = torch.stack([torch.full((4, 8, 8), n) for n in (1., 2., 3.)]).unsqueeze(0).requires_grad_()
        out = model._forecast_outputs(torch.zeros(1, 3, 43, 8, 8), decoded, decoded)
        torch.testing.assert_close(out["occurrence_logits"][0, :, 0, 0, 0], torch.tensor([4., 8., 12.]))
        torch.testing.assert_close(out["intensity_exceedance_logits"][0, :, 0, 0, 0, 0], torch.tensor([4., 8., 12.]))
        out["occurrence_logits"][:, 0].sum().backward()
        self.assertGreater(float(decoded.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(decoded.grad[:, 1:].abs().sum()), 0)

    def test_old_model_state_loads_strictly_and_new_config_is_rejected(self):
        config = ModelConfig(simvpv2_hidden_spatial=4, simvpv2_hidden_temporal=8,
                             simvpv2_spatial_layers=2, simvpv2_temporal_layers=2)
        old = build_model(config)
        restored = build_model(config)
        restored.load_state_dict(old.state_dict(), strict=True)
        self.assertFalse(any("skip_fusion" in key or "exceedance_head" in key for key in old.state_dict()))
        payload = {"config": {"model": {"architecture": "simvpv2"}}}
        validate_checkpoint_architecture(payload, "simvpv2", config)
        with self.assertRaisesRegex(ValueError, "simvpv2_multi_lead"):
            validate_checkpoint_architecture(payload, "simvpv2", tiny_config().model)


class TestBlendAndMetrics(unittest.TestCase):
    def test_hann_overlap_matches_hand_calculation_and_preserves_truth(self):
        window = _patch_blend_weights()
        self.assertAlmostEqual(float(window.min()), .05)
        self.assertEqual(float(window.max()), 1)
        self.assertTrue(np.all(window > 0))
        records = [dict(forecast_time="t", grid_row=0, grid_column=0, probability=.2, target=2.),
                   dict(forecast_time="t", grid_row=32, grid_column=32, probability=.8, target=6.)]
        results = []
        model = _PatchModel()
        model.patch_blend_mode = "hann"
        for order in ([0, 1], [1, 0]):
            with patch("trainers.evaluation.GRID_SHAPE", (96, 96)), patch(
                "trainers.evaluation.assemble_model_input", side_effect=lambda batch, *_: batch["x"]
            ):
                for_each_deduplicated_grid(model, _PatchDataset(records, order), None, None,
                                          torch.device("cpu"), 2, lambda *a: results.append(a))
        probability, amount, target, valid, _ = results[0]
        a, b = window[40, 40], window[8, 8]
        self.assertAlmostEqual(float(probability[0, 0, 40, 40]), (.2*a+.8*b)/(a+b), places=6)
        self.assertAlmostEqual(float(amount[0, 0, 40, 40]), (2*a+8*b)/(a+b), places=6)
        self.assertEqual(float(target[0, 0, 40, 40]), 4)
        self.assertAlmostEqual(float(probability[0, 0, 0, 0]), .2)
        self.assertTrue(valid[0, 0, 0, 0])
        for before, after in zip(results[0], results[1]):
            np.testing.assert_allclose(before, after)

    def test_observable_macro_and_true_pooled_metrics(self):
        probability = np.array([.9, .9, .9, .1, .9, .1]).reshape(1, 1, 1, 6)
        amount = np.array([1, 2, 3, 5, 1, 4]).reshape(probability.shape)
        target = np.array([1, 0, 3, 0, 0, 2]).reshape(probability.shape)
        zone = np.array([[0, 1, 2, 2, 2, 2]])
        metrics = GridMetrics(np.full((1, 1, 3), .5), leads=1)
        metrics.update(probability, amount, target, np.ones_like(target, bool), zone)
        value = metrics.finish()
        self.assertAlmostEqual(value["observable_summary"]["csi"], .5)
        self.assertEqual(value["direct_summary"]["csi"], 1)
        pooled = value["pooled_summary"]
        self.assertEqual(pooled["valid"], 6)
        self.assertEqual((pooled["tp"], pooled["fp"], pooled["fn"]), (2, 2, 1))
        self.assertAlmostEqual(pooled["csi"], .4)
        self.assertAlmostEqual(pooled["mae_mm_h"], 5/6)
        self.assertAlmostEqual(pooled["rmse_mm_h"], math.sqrt(9/6))
        x = np.array([1, 2, 3, 0, 1, 0], dtype=float)
        y = target.ravel()
        dx, dy = x-x.mean(), y-y.mean()
        cc = (dx*dy).sum() / np.sqrt((dx*dx).sum() * (dy*dy).sum())
        self.assertAlmostEqual(pooled["cc"], cc)
        self.assertIs(selection_summary(value, "observable"), value["observable_summary"])
        self.assertTrue(math.isnan(pooled["csi_by_threshold"]["10"]))
        empty = GridMetrics(np.full((1, 1, 3), .5), leads=1)
        empty.update(probability, amount, target, np.ones_like(target, bool), np.full_like(zone, 2))
        self.assertTrue(math.isnan(empty.finish()["observable_summary"]["csi"]))

    def test_expected_log_gate_matches_loss_semantics(self):
        p = np.array([.5, .1], dtype=np.float32)
        amount = np.array([8., 8.], dtype=np.float32)
        expected = gate_rain_numpy(p, amount, .2, "expected_log")
        np.testing.assert_allclose(expected, [2, 0], atol=1e-6)
        torch.testing.assert_close(gate_rain_torch(torch.from_numpy(p), torch.from_numpy(amount), .2, "expected_log"),
                                   torch.from_numpy(expected))
        np.testing.assert_array_equal(gate_rain_numpy(p, amount, .2), [8, 0])
        metrics = GridMetrics(np.full((1, 1, 3), .2), leads=1, rain_gate_mode="expected_log")
        metrics.update(p[:1].reshape(1, 1, 1, 1), amount[:1].reshape(1, 1, 1, 1),
                       np.full((1, 1, 1, 1), 2.), np.ones((1, 1, 1, 1), bool), np.zeros((1, 1), int))
        self.assertAlmostEqual(metrics.finish()["pooled_summary"]["mae_mm_h"], 0, places=6)

    def test_zone_far_caps_and_no_feasible_threshold_abstention(self):
        probability = np.ones((1, 1, 1, 6), np.float32)
        target = np.array([1, 0, 1, 0, 1, 0]).reshape(probability.shape)
        zone = np.array([[0, 0, 1, 1, 2, 2]])
        kwargs = dict(leads=1, minimum_positive=1, threshold_min=.1, threshold_max=.9, threshold_step=.1)
        search = ZoneThresholdSearch(**kwargs, far_max_by_zone=[.75, .7, .4])
        search.update(probability, target, np.ones_like(target, bool), zone)
        result = search.best()
        self.assertLessEqual(result["thresholds"][0][0][0], 1)
        self.assertGreater(result["thresholds"][0][0][2], 1)
        self.assertEqual(result["details"][2]["status"], "no_feasible_threshold_abstain")
        self.assertEqual(result["details"][2]["csi"], 0)
        legacy = ZoneThresholdSearch(**kwargs, far_max=.4)
        legacy.update(probability, target, np.ones_like(target, bool), zone)
        self.assertLess(legacy.best()["thresholds"][0][0][2], 1)


class TestCoverageSampler(unittest.TestCase):
    def test_downsampling_preserves_all_protected_patches_and_is_repeatable(self):
        groups = {"a": list(range(11)), "b": list(range(11, 22))}
        sampler = ForecastTimeRoundRobinBatchSampler(groups, 4, 14,
                    dry_uncovered_indices=set(range(1, 21)), uncovered_patch_keep_ratio=.25)
        def emitted():
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            return [index for batch in batches for index in batch]
        first = emitted()
        self.assertEqual(first, emitted())
        self.assertEqual(len(first), 7)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue({0, 21}.issubset(first))
        sampler.set_epoch(1)
        second = emitted()
        self.assertNotEqual(set(first), set(second))
        self.assertTrue({0, 21}.issubset(second))
        sampler.set_epoch(0)
        self.assertEqual(first, emitted())

    def test_background_detection_uses_processed_targets_conservatively(self):
        with TemporaryDirectory() as directory:
            shard = Path(directory)
            amount = np.zeros((7, 3, 1, 2, 2), np.float32)
            valid = np.ones_like(amount, bool)
            amount[1, 2, 0, 0, 0] = .1  # Rain in any lead must be retained.
            valid[2] = False
            amount[4, 0, 0, 0, 0] = np.nan
            amount[5] = .05
            np.save(shard / "y_amount.npy", amount)
            np.save(shard / "y_valid.npy", valid)
            metadata = [{"input_valid_fraction": [1, 0, 0]}] + [
                {"input_valid_fraction": [0, 0, 0]} for _ in range(5)] + [{}]
            class Dataset:
                split = "train"
                shards = [shard]
                index = [(0, i) for i in range(7)]
                def __len__(self):
                    return 7
                def metadata(self, i):
                    return metadata[i]
            dataset = Dataset()
            self.assertEqual(find_dry_uncovered_patches(dataset, disable_tqdm=True), {3, 5})
            dataset.split = "val"
            with self.assertRaises(ValueError):
                find_dry_uncovered_patches(dataset)


class TestZoneTrainingFlow(unittest.TestCase):
    def test_selection_and_early_stop_follow_observable_not_macro(self):
        config = tiny_config()
        config.train.epochs = 3
        config.train.early_stopping_patience = 1
        config.train.effective_batch_size = config.train.microbatch_size = 1
        model = _TinyModel()
        context = SimpleNamespace(config=config, output_dir=Path("unused"), resume_checkpoint=None,
            start_epoch=1, train_loader=_OneBatchLoader(), model=model, ema=deepcopy(model),
            optimizer=torch.optim.SGD(model.parameters(), lr=.01), criterion=_TinyLoss(),
            device=torch.device("cpu"), mean=None, std=None, val_dataset=None, inputs={}, targets={},
            logger=logging.getLogger("test_zone_selection"))
        calls = []
        def evaluate(*args, **kwargs):
            epoch = len(calls) + 1
            calls.append(epoch)
            def summary(score):
                return dict(csi=score, pod=.5, far=.2, mae_mm_h=.1, rmse_mm_h=.2, cc=.3,
                            balanced_score=score, csi_by_threshold={key: score for key in ("0.1", "2", "5")})
            return {"summary": summary(.1*epoch), "observable_summary": summary(.8-.1*epoch),
                    "threshold_selection": {"thresholds": [[[.5]*3]]*3}}
        with TemporaryDirectory() as directory, patch(
            "trainers.gpm_trainer.assemble_model_input", side_effect=lambda b, *_: b["x"]
        ), patch("trainers.gpm_trainer.evaluate_deduplicated", side_effect=evaluate), patch(
            "trainers.gpm_trainer.save_checkpoint"
        ) as save:
            context.output_dir = Path(directory)
            result = GPMTrainer(context).train()
            metrics = json.loads((context.output_dir / "metrics.json").read_text())
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(metrics["best_by_objective"]), 3)
        self.assertTrue(all(v["epoch"] == 1 for v in metrics["best_by_objective"].values()))
        self.assertEqual(metrics["best"]["epoch"], 1)
        self.assertEqual(metrics["selection_domain"], "observable")
        self.assertGreater(metrics["epochs"][0]["selection_summary"]["csi"], metrics["epochs"][1]["selection_summary"]["csi"])
        self.assertEqual(sum(c.kwargs["path"].name == "best_balanced_inference.pt" for c in save.call_args_list), 1)

    def test_resume_rejects_mixed_selection_domains(self):
        validate_checkpoint_selection({}, "macro_all")
        with self.assertRaisesRegex(ValueError, "primary_selection_domain"):
            validate_checkpoint_selection({}, "observable")
        payload = {"config": {"train": {"primary_selection_domain": "observable"}},
                   "training_state": {"selection_domain": "observable"}}
        validate_checkpoint_selection(payload, "observable")
        payload["training_state"]["selection_domain"] = "direct_only"
        with self.assertRaises(ValueError):
            validate_checkpoint_selection(payload, "observable")

    def test_two_epoch_real_model_evaluation_and_checkpoint_reload(self):
        config = tiny_config()
        stats = _mock_stats() | {"manifest_sha256": "fixture"}
        model = build_model(config.model)
        model.configure_target_transform(stats)
        dataset = _MemoryDataset()
        context = SimpleNamespace(config=config, output_dir=Path("unused"), resume_checkpoint=None,
            resume_payload=None, start_epoch=1, train_loader=DataLoader(dataset, batch_size=2),
            model=model, ema=deepcopy(model), optimizer=torch.optim.AdamW(model.parameters()),
            criterion=build_loss(stats, config.train), device=torch.device("cpu"),
            mean=torch.zeros(2, 10), std=torch.ones(2, 10), val_dataset=dataset,
            inputs={"manifest_sha256": "fixture"}, targets=stats, logger=logging.getLogger("test_zone_smoke"))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with TemporaryDirectory() as directory, patch("trainers.evaluation.GRID_SHAPE", (64, 64)):
                context.output_dir = Path(directory)
                result = GPMTrainer(context).train()
                metrics = json.loads((context.output_dir / "metrics.json").read_text())
                payload = load_checkpoint(context.output_dir / "checkpoints/last_resume.pt", torch.device("cpu"))
                restored = build_model(config.model)
                restored.configure_target_transform(stats)
                restored.load_state_dict(payload["ema_state"], strict=True)
                validate_checkpoint_architecture(payload, "simvpv2", config.model)
                self.assertEqual(payload["training_state"]["selection_domain"], "observable")
                self.assertEqual(len(payload["history"]), 2)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(metrics["selection_domain"], "observable")
        for row in metrics["epochs"]:
            self.assertTrue(math.isfinite(row["train_loss"]))
            self.assertTrue(math.isfinite(row["validation_loss"]))
            self.assertIn("observable_summary", row)
            self.assertIn("pooled_summary", row)
            self.assertEqual(len(row["observable_summary"]["csi_by_threshold"]), 6)


if __name__ == "__main__":
    unittest.main()

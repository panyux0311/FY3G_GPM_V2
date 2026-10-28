"""Pipeline context assembly for FY-3G GPM training."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.gpm_dataset import FY3G3HDataset
from datasets.diagnostics import build_diagnostic_subset
from datasets.samplers import ForecastTimeRoundRobinBatchSampler, build_forecast_time_index, find_dry_uncovered_patches
from losses.gpm_loss import build_loss
from models import build_model
from schema import validate_dataset_contract
from utils.config import ProjectConfig
from utils.intensity_weights import derive_intensity_weights
from utils.logging import setup_logger
from utils.paths import check_output_dir_safety, resolve_output_dir
from utils.progress import should_disable_tqdm
from .checkpoint import load_checkpoint, validate_checkpoint_architecture, validate_checkpoint_dataset, validate_checkpoint_selection
from .optimization import build_optimizer


def seed_everything(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_dataloader(dataset: FY3G3HDataset, config: ProjectConfig, train: bool) -> DataLoader:
    if not train:
        return DataLoader(
            dataset,
            batch_size=config.train.microbatch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=config.train.prefetch_factor if config.train.num_workers else None,
        )
    groups, _ = build_forecast_time_index(dataset, disable_tqdm=should_disable_tqdm(config.train.disable_tqdm))
    background = (find_dry_uncovered_patches(dataset, config.train.rain_threshold_mm_h,
                  disable_tqdm=should_disable_tqdm(config.train.disable_tqdm))
                  if config.train.uncovered_patch_keep_ratio < 1 else set())
    sampler = ForecastTimeRoundRobinBatchSampler(
        groups, config.train.microbatch_size, config.train.seed,
        dry_uncovered_indices=background,
        uncovered_patch_keep_ratio=config.train.uncovered_patch_keep_ratio,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config.train.prefetch_factor if config.train.num_workers else None,
    )


class PipelineContext:
    def __init__(
        self,
        config: ProjectConfig,
        data_root: str | Path | None = None,
        resume_checkpoint: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        device: torch.device | None = None,
    ) -> None:
        config.validate()
        self.config = config
        seed_everything(config.train.seed)
        self.root = Path(data_root or config.data.dataset_root)

        # Output directory resolution and safety check
        self.output_dir = resolve_output_dir(output_dir or config.train.output_dir, config.name)
        check_output_dir_safety(self.output_dir, allow_resume=bool(resume_checkpoint), overwrite=overwrite)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unified logging
        self.logger = setup_logger(name="FY3G_GPM", log_file=self.output_dir / "training.log")

        # Device determination
        if device is not None:
            self.device = device
        else:
            cfg_device = getattr(config.train, "device", "auto").lower()
            if cfg_device == "cuda":
                self.device = torch.device("cuda")
            elif cfg_device == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate dataset contract
        dataset_contract = validate_dataset_contract(self.root)

        # Load statistics
        inputs_path = self.root / "statistics/input_statistics.json"
        targets_path = self.root / "statistics/target_statistics.json"
        self.inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        original_targets = json.loads(targets_path.read_text(encoding="utf-8"))
        self.targets = (
            derive_intensity_weights(original_targets)
            if config.train.intensity_weight_strategy == "sqrt_inverse_frequency_v1"
            else original_targets
        )
        self.logger.info("Intensity weight strategy: %s", config.train.intensity_weight_strategy)

        self.mean = torch.tensor(self.inputs["mean"], device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.inputs["std"], device=self.device, dtype=torch.float32).clamp_min(1e-6)

        # Build model and EMA
        self.model = build_model(config.model).to(self.device)
        self.model.configure_target_transform(self.targets, True)
        self.model.initialize_occurrence_bias(
            (torch.tensor(self.targets["rain_count"]) / torch.tensor(self.targets["valid_count"]).clamp_min(1)).reshape(-1)
        )
        self.ema = deepcopy(self.model).eval()

        # Build optimizer and loss
        self.optimizer = build_optimizer(self.model, config.train)
        self.criterion = build_loss(self.targets, config.train).to(self.device)

        # Datasets and Loaders
        self.train_dataset = FY3G3HDataset(self.root, "train")
        self.val_dataset = FY3G3HDataset(self.root, "val")
        self.train_loader = build_dataloader(self.train_dataset, config, True)
        sampler = self.train_loader.batch_sampler
        self.logger.info("Training sampler: total=%d dry_uncovered=%d kept_background=%d samples_per_epoch=%d keep_ratio=%.3f",
                         len(self.train_dataset), len(sampler.background), sampler.background_kept,
                         sampler.sample_count, sampler.keep_ratio)
        self.logger.info("Primary selection domain: %s", config.train.primary_selection_domain)
        self.train_diagnostic_dataset = None
        self.diagnostic_audit = None
        if config.train.diagnostics_enabled:
            self.train_diagnostic_dataset, audit = build_diagnostic_subset(
                self.train_dataset,
                self.train_loader.batch_sampler.patches_by_time,
                config.train.diagnostics_train_forecast_times,
            )
            audit["manifest_sha256"] = dataset_contract["manifest_sha256"]
            self.diagnostic_audit = audit
            self.logger.info(
                "Fixed train diagnostic subset: %d forecast times, %d patches",
                len(audit["selected_forecast_times"]), audit["samples"],
            )

        self.start_epoch = 1
        self.resume_checkpoint = resume_checkpoint
        self.resume_payload: dict[str, Any] | None = None
        if resume_checkpoint:
            payload = load_checkpoint(resume_checkpoint, self.device)
            validate_checkpoint_dataset(payload, dataset_contract["manifest_sha256"])
            validate_checkpoint_architecture(payload, config.model.architecture, config.model)
            validate_checkpoint_selection(payload, config.train.primary_selection_domain)
            previous_train = payload.get("config", {}).get("train", {})
            if previous_train.get("uncovered_patch_keep_ratio", 1.0) != config.train.uncovered_patch_keep_ratio:
                raise ValueError("Checkpoint uncovered_patch_keep_ratio differs from current configuration; start a new experiment")
            stored_strategy = payload.get("target_statistics", {}).get("weight_strategy")
            if stored_strategy != self.targets.get("weight_strategy"):
                raise ValueError("Checkpoint intensity weight strategy differs from current training configuration")
            previous_audit = (payload.get("training_state") or {}).get("diagnostic_audit")
            if previous_audit is not None and previous_audit != self.diagnostic_audit:
                raise ValueError("Diagnostic subset changed on resume; keep the original subset setting")
            self.resume_payload = payload
            self.model.load_state_dict(payload["model_state"])
            self.ema.load_state_dict(payload.get("ema_state", payload["model_state"]))
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.start_epoch = int(payload["epoch"]) + 1

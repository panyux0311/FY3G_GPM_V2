#!/usr/bin/env python3
"""Evaluate a trained model checkpoint on test or val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.gpm_dataset import FY3G3HDataset
from models import build_model
from metrics.gpm_metrics import selection_summary
from schema import validate_dataset_contract
from trainers.checkpoint import load_checkpoint, validate_checkpoint_architecture, validate_checkpoint_dataset
from trainers.evaluation import evaluate_with_fixed_thresholds, tune_zone_thresholds
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--config", default=ROOT / "configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Search for optimal thresholds (only allowed on val split to prevent test leakage)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        help="Override config values, e.g. --set train.microbatch_size=32",
    )
    args = parser.parse_args()

    if args.split == "test" and args.tune_thresholds:
        raise ValueError("Searching thresholds on the test split is forbidden (data leakage). Use val split to tune.")

    config = load_config(args.config, overrides=args.overrides)
    data_root = Path(args.data_root or config.data.dataset_root)
    dataset_contract = validate_dataset_contract(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = load_checkpoint(args.checkpoint, device)
    validate_checkpoint_dataset(payload, dataset_contract["manifest_sha256"])
    validate_checkpoint_architecture(payload, config.model.architecture, config.model)
    model = build_model(config.model).to(device)
    model.configure_target_transform(payload["target_statistics"], True)
    model.load_state_dict(payload.get("ema_state", payload["model_state"]))
    frozen_thresholds = np.asarray(payload["occurrence_thresholds_by_zone"], dtype=np.float32)
    model.set_occurrence_thresholds_by_zone(frozen_thresholds)
    model.eval()

    mean = torch.tensor(payload["input_statistics"]["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(payload["input_statistics"]["std"], device=device, dtype=torch.float32).clamp_min(1e-6)

    dataset = FY3G3HDataset(data_root, args.split)

    if args.tune_thresholds:
        metrics, selected_thresholds = tune_zone_thresholds(
            model, dataset, mean, std, device, config
        )
    else:
        metrics = evaluate_with_fixed_thresholds(
            model=model,
            dataset=dataset,
            mean=mean,
            std=std,
            device=device,
            thresholds=frozen_thresholds,
            microbatch_size=config.train.microbatch_size,
        )

    metrics["selection_domain"] = config.train.primary_selection_domain
    metrics["selection_summary"] = selection_summary(metrics, config.train.primary_selection_domain)
    print(json.dumps({key: metrics[key] for key in (
        "summary", "observable_summary", "pooled_summary", "direct_summary", "selection_domain", "selection_summary",
    )}, ensure_ascii=False, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()

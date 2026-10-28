#!/usr/bin/env python3
"""Freeze completed GPM run, evaluate test split with fixed thresholds, render full cases, and package results."""
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
from datasets.diagnostics import ForecastTimeSubset
from metrics.gpm_metrics import selection_summary
from models import build_model
from schema import validate_dataset_contract
from trainers.checkpoint import load_checkpoint, validate_checkpoint_architecture, validate_checkpoint_dataset
from trainers.evaluation import evaluate_with_fixed_thresholds, for_each_deduplicated_grid
from utils.config import load_config
from utils.reporting import create_result_archive, select_strong_cases
from visualization.gpm import plot_surface_leads


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize GPM training run")
    parser.add_argument("--config", default=ROOT / "configs/gpm_surface_v1.yaml")
    parser.add_argument("--checkpoint", help="Path to checkpoint (default: best_balanced_inference.pt)")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive-path", help="Destination path for packaged tar.gz")
    parser.add_argument("--cases", type=int, default=3, help="Number of strong cases to visualize")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        help="Override config values, e.g. --set train.microbatch_size=32",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    output_dir = Path(args.output_dir or config.train.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints/best_balanced_inference.pt"
    data_root = Path(args.data_root or config.data.dataset_root)
    dataset_contract = validate_dataset_contract(data_root)
    archive_path = Path(args.archive_path) if args.archive_path else output_dir.parent / f"{config.name}_results.tar.gz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(checkpoint_path, device)
    validate_checkpoint_dataset(payload, dataset_contract["manifest_sha256"])
    validate_checkpoint_architecture(payload, config.model.architecture, config.model)

    model = build_model(config.model).to(device)
    model.configure_target_transform(payload["target_statistics"], True)
    model.load_state_dict(payload.get("ema_state", payload["model_state"]))
    thresholds = np.asarray(payload["occurrence_thresholds_by_zone"], dtype=np.float32)
    model.set_occurrence_thresholds_by_zone(thresholds)
    model.eval()

    mean = torch.tensor(payload["input_statistics"]["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(payload["input_statistics"]["std"], device=device, dtype=torch.float32).clamp_min(1e-6)

    # 1. Test split evaluation with FIXED thresholds (no search, no test data leakage)
    test_dataset = FY3G3HDataset(data_root, "test")
    test_metrics = evaluate_with_fixed_thresholds(
        model=model,
        dataset=test_dataset,
        mean=mean,
        std=std,
        device=device,
        thresholds=thresholds,
        microbatch_size=config.train.microbatch_size,
    )
    test_metrics["selection_domain"] = config.train.primary_selection_domain
    test_metrics["selection_summary"] = selection_summary(test_metrics, config.train.primary_selection_domain)
    metrics_file = output_dir / "test_evaluation.json"
    metrics_file.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Test evaluation saved: {metrics_file}")

    # 2. Case visualization: stitch full 0.1-degree grid for selected strong precipitation cases
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    cases = select_strong_cases(test_dataset, args.cases)

    for idx, case in enumerate(cases, 1):
        time_str = case["forecast_time"]
        sub_dir = vis_dir / f"case_{idx}_{time_str.replace(':', '').replace('-', '')}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        def render(probability, amount, target, valid, zone):
            plot_surface_leads(
                target=target, probability=probability, ungated_amount=amount,
                valid=valid, zone=zone, thresholds=thresholds, output=sub_dir,
                title=f"Case {idx} ({time_str})", rain_gate_mode=model.rain_gate_mode,
            )

        # Use the same weighted stitching as validation and test metrics.
        for_each_deduplicated_grid(
            model, ForecastTimeSubset(test_dataset, case["indices"]), mean, std,
            device, config.train.microbatch_size, render,
            progress_desc=f"Case {idx}/{len(cases)}",
        )
    test_dataset.close()

    # 3. Create results archive
    archive_info = create_result_archive(
        project_root=ROOT,
        output_dir=output_dir,
        config_path=Path(args.config),
        archive_path=archive_path,
    )
    print(json.dumps(archive_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

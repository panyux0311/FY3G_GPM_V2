#!/usr/bin/env python3
"""Main training script for FY-3G GPM precipitation forecasting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainers.gpm_pipeline import PipelineContext
from trainers.gpm_trainer import GPMTrainer
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FY-3G GPM Model")
    parser.add_argument("--config", default=ROOT / "configs/gpm_surface_v1.yaml", help="Path to YAML config")
    parser.add_argument("--data-root", help="Override dataset root path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing to existing non-empty output directory")
    parser.add_argument("--dry-run", action="store_true", help="Assemble pipeline, print summary and exit without training")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        help="Override config values, e.g. --set train.epochs=100 train.microbatch_size=32",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    if args.data_root:
        config.data.dataset_root = args.data_root
    if args.output_dir:
        config.train.output_dir = args.output_dir

    context = PipelineContext(
        config=config,
        data_root=args.data_root,
        resume_checkpoint=args.resume,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        param_count = sum(p.numel() for p in context.model.parameters())
        summary = {
            "status": "dry_run_passed",
            "config_name": config.name,
            "device": str(context.device),
            "output_dir": str(context.output_dir),
            "model_parameters": param_count,
            "train_samples": len(context.train_dataset),
            "val_samples": len(context.val_dataset),
            "epochs": config.train.epochs,
            "effective_batch_size": config.train.effective_batch_size,
            "microbatch_size": config.train.microbatch_size,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    trainer = GPMTrainer(context)
    result = trainer.train()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

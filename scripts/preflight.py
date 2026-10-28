#!/usr/bin/env python3
"""CPU-only preflight check for model architecture, inputs, and loss computation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.gpm_dataset import FY3G3HDataset
from datasets.gpm_inputs import assemble_model_input
from losses.gpm_loss import build_loss
from utils.intensity_weights import derive_intensity_weights
from models import build_model
from schema import validate_dataset_contract
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check")
    parser.add_argument("--config", default=ROOT / "configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--set", dest="overrides", action="extend", nargs="+", default=[])
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    root = Path(args.data_root or config.data.dataset_root)
    validate_dataset_contract(root)

    stats_path = root / "statistics/target_statistics.json"
    inputs_path = root / "statistics/input_statistics.json"
    if not stats_path.exists() or not inputs_path.exists():
        raise FileNotFoundError(f"Statistics files missing in {root}/statistics")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    inputs_stat = json.loads(inputs_path.read_text(encoding="utf-8"))
    if config.train.intensity_weight_strategy == "sqrt_inverse_frequency_v1":
        stats = derive_intensity_weights(stats)

    dataset = FY3G3HDataset(root, "train")
    batch = torch.utils.data.default_collate([dataset[0]])

    model = build_model(config.model).cpu()
    model.configure_target_transform(stats, True)

    mean = torch.tensor(inputs_stat["mean"], dtype=torch.float32)
    std = torch.tensor(inputs_stat["std"], dtype=torch.float32)

    model_input = assemble_model_input(batch, mean, std)
    out = model(model_input)
    criterion = build_loss(stats, config.train)
    loss = criterion(out, batch["y_amount"], batch["y_valid"])["loss"]

    result = {
        "status": "passed",
        "device": "cpu",
        "train_samples": len(dataset),
        "model_output_shape": list(out["prediction"].shape),
        "loss_finite": bool(torch.isfinite(loss)),
        "loss_value": float(loss.item()),
    }
    dataset.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["loss_finite"]:
        raise RuntimeError("CPU preflight loss is not finite")


if __name__ == "__main__":
    main()

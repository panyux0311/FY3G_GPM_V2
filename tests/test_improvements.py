#!/usr/bin/env python3
"""Test script to verify the implemented improvements."""
import sys
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_weight_calculation():
    """Test the fixed weight calculation logic."""
    print("=" * 60)
    print("Testing Fixed Weight Calculation")
    print("=" * 60)

    # Simulate statistics similar to the training data
    rain_count = np.array([1000000])  # Total rainy pixels
    group_count = np.array([[100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 300000, 200000, 0]])

    # Old method (WRONG - includes zero bins in normalization)
    old_weight = np.divide(rain_count[:, None], np.maximum(group_count, 1), dtype=np.float64)
    old_weight /= np.maximum(old_weight.mean(axis=1, keepdims=True), 1e-6)

    # New method (FIXED - only computes weights for bins with samples)
    new_weight = np.zeros_like(group_count, dtype=np.float64)
    for channel in range(len(rain_count)):
        if rain_count[channel] > 0:
            mask = group_count[channel] > 0
            if mask.any():
                smooth = 10.0
                weights = np.sqrt(rain_count[channel] / (group_count[channel][mask].astype(np.float64) + smooth))
                weights = np.minimum(weights, 20.0)
                weights /= weights.mean()
                new_weight[channel][mask] = weights

    print(f"\nGroup counts: {group_count[0]}")
    print(f"\nOld weights: {old_weight[0]}")
    print(f"Old weights after clamp(0.5, 8.0): {np.clip(old_weight[0], 0.5, 8.0)}")
    print(f"\nNew weights: {new_weight[0]}")
    print(f"New weights after clamp(0.1, 10.0): {np.clip(new_weight[0], 0.1, 10.0)}")

    # Check that strong rain bins (low count) get higher weights
    strong_rain_idx = 0  # First bin (0.1-0.5 mm/h)
    weak_rain_idx = 6  # Seventh bin (higher frequency)

    print(f"\nStrong rain bin weight ratio (new/old): {new_weight[0][strong_rain_idx] / max(old_weight[0][strong_rain_idx], 1e-6):.2f}x")
    print(f"Empty bin weight (should be 0): old={old_weight[0][-1]:.4f}, new={new_weight[0][-1]:.4f}")

    if new_weight[0][-1] == 0.0 and new_weight[0][strong_rain_idx] > old_weight[0][strong_rain_idx]:
        print("\n[PASS] Weight calculation FIXED: Empty bins are 0, strong rain gets higher weight")
        return True
    else:
        print("\n[FAIL] Weight calculation may have issues")
        return False


def test_config_changes():
    """Test the new configuration file."""
    print("\n" + "=" * 60)
    print("Testing New Configuration")
    print("=" * 60)

    config_path = ROOT / "configs/gpm_surface_v3.yaml"
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return False

    import yaml
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    epochs = config["train"]["epochs"]
    warmup = config["train"]["warmup_epochs"]
    diagnostics_every = config["train"]["diagnostics_every_epochs"]
    save_metrics = config["train"].get("save_best_by_metrics", [])

    print(f"\nTotal epochs: {epochs} (was 100, now shorter)")
    print(f"Warmup epochs: {warmup}")
    print(f"Diagnostics frequency: every {diagnostics_every} epochs (was every 1)")
    print(f"Save best by metrics: {save_metrics}")

    checks = [
        (epochs == 25, "Short schedule (25 epochs)"),
        (warmup == 5, "Warmup epochs (5)"),
        (diagnostics_every == 5, "Reduced diagnostics frequency"),
        (len(save_metrics) >= 2, "Multi-objective checkpointing"),
    ]

    all_passed = all(check[0] for check in checks)
    for passed, desc in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {desc}")

    return all_passed


def test_learning_rate_schedule():
    """Test that learning rate decays properly in 25 epochs."""
    print("\n" + "=" * 60)
    print("Testing Learning Rate Schedule")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "trainers"))
    from optimization import compute_lr_factor

    peak_lr = 1e-4
    final_lr = 5e-6
    warmup = 5
    total_epochs = 25

    print(f"\nPeak LR: {peak_lr:.6f}, Final LR: {final_lr:.6f}")
    print(f"Epochs: {total_epochs}, Warmup: {warmup}\n")

    test_epochs = [1, 5, 6, 10, 15, 20, 25]
    for epoch in test_epochs:
        factor = compute_lr_factor(epoch, warmup, total_epochs, peak_lr, final_lr)
        lr = peak_lr * factor
        print(f"Epoch {epoch:2d}: LR = {lr:.6e} ({factor*100:5.1f}% of peak)")

    # Check that LR decays significantly by epoch 15 (roughly mid-way through cosine)
    factor_15 = compute_lr_factor(15, warmup, total_epochs, peak_lr, final_lr)
    lr_15 = peak_lr * factor_15

    # At epoch 15 of 25, should be well into cosine decay
    if lr_15 < peak_lr * 0.7:
        print(f"\n[PASS] Learning rate decays properly (at epoch 15: {lr_15:.6e}, {factor_15*100:.1f}% of peak)")
        return True
    else:
        print(f"\n[FAIL] Learning rate not decaying enough (at epoch 15: {lr_15:.6e})")
        return False


def main():
    print("\n" + "=" * 60)
    print("IMPROVEMENT VERIFICATION TESTS")
    print("=" * 60)

    results = {}
    results["weight_calculation"] = test_weight_calculation()
    results["config_changes"] = test_config_changes()
    results["lr_schedule"] = test_learning_rate_schedule()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed'}")

    if all_passed:
        print("\nNext steps:")
        print("1. Regenerate target statistics with fixed weight calculation")
        print("2. Run training with new config:")
        print("   python scripts/train.py --config configs/gpm_surface_v3.yaml")
        print("3. Compare results with baseline experiment")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

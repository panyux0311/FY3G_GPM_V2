"""Training history and validation curves visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_training_curves(history: list[dict[str, Any]], output_path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", float("nan")) for h in history]
    val_epochs = [h["epoch"] for h in history if "validation" in h]
    val_csi = [h["validation"]["csi"] for h in history if "validation" in h]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(epochs, train_loss, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    if val_csi:
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Val CSI", color=color)
        ax2.plot(val_epochs, val_csi, color=color, linestyle="--", marker="o", label="Val CSI")
        ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

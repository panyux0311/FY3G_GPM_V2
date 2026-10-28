"""Diagnostic figures: truth/prediction/absolute-error maps per lead."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from utils.rain_gate import gate_rain_numpy


def plot_surface_leads(
    target: np.ndarray,
    probability: np.ndarray,
    ungated_amount: np.ndarray,
    valid: np.ndarray,
    zone: np.ndarray,
    thresholds: np.ndarray,
    output: str | Path,
    title: str = "GPM surface precipitation",
    rain_gate_mode: str = "hard",
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = np.asarray(thresholds).reshape(3, 1, 3)
    paths = []

    for lead in range(3):
        threshold = thresholds[lead, 0][zone]
        prediction = gate_rain_numpy(probability[lead, 0], ungated_amount[lead, 0], threshold, rain_gate_mode)
        mask = ~np.asarray(valid[lead, 0], bool)
        truth = np.ma.masked_where(mask, target[lead, 0])
        pred = np.ma.masked_where(mask, prediction)
        error = np.ma.masked_where(mask, np.abs(prediction - target[lead, 0]))
        positive = truth.compressed()
        vmax = float(np.clip(np.percentile(positive, 99) if positive.size else 1, 1, 100))
        errmax = float(np.clip(np.percentile(error.compressed(), 99) if error.count() else 1, 1, 100))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for axis, value, label, cmap, limit in zip(
            axes,
            (truth, pred, error),
            ("GPM truth (mm/h)", "prediction (mm/h)", "absolute error (mm/h)"),
            ("turbo", "turbo", "magma"),
            (vmax, vmax, errmax),
        ):
            image = axis.imshow(value, origin="lower", extent=(60, 150, 0, 60), cmap=cmap, vmin=0, vmax=limit)
            axis.set(title=label, xlabel="longitude (°E)", ylabel="latitude (°N)")
            fig.colorbar(image, ax=axis, shrink=0.82)
        fig.suptitle(f"{title}: +{lead+1} h")
        path = output / f"surface_lead_{lead+1}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths

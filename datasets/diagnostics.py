"""Deterministic, complete forecast-time subsets for training diagnostics."""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .gpm_dataset import FY3G3HDataset


class ForecastTimeSubset(Dataset):
    """Expose selected patches with local indices and matching metadata."""

    def __init__(self, dataset: FY3G3HDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.dataset[self.indices[index]])
        sample["_sample_index"] = torch.tensor(index, dtype=torch.int64)
        return sample

    def metadata(self, index: int) -> dict[str, Any]:
        return self.dataset.metadata(self.indices[index])


def build_diagnostic_subset(
    dataset: FY3G3HDataset,
    groups: dict[str, list[int]],
    max_forecast_times: int,
) -> tuple[ForecastTimeSubset | None, dict[str, Any]]:
    """Choose evenly spaced sorted time groups, keeping every patch in each group."""
    if max_forecast_times < 0:
        raise ValueError("max_forecast_times must be nonnegative")

    times = sorted(groups)
    selected_times: list[str] = []
    if max_forecast_times > 0 and times:
        count = min(max_forecast_times, len(times))
        positions = (
            [len(times) // 2] if count == 1 else
            [index * (len(times) - 1) // (count - 1) for index in range(count)]
        )
        selected_times = [times[position] for position in positions]

    indices = [
        index
        for time in selected_times
        for index in sorted(
            groups[time],
            key=lambda value: (
                int(dataset.metadata(value)["grid_row"]),
                int(dataset.metadata(value)["grid_column"]),
                value,
            ),
        )
    ]
    audit = {
        "enabled": bool(selected_times),
        "available_forecast_times": len(times),
        "selected_forecast_times": selected_times,
        "samples": len(indices),
    }
    return (ForecastTimeSubset(dataset, indices) if indices else None), audit

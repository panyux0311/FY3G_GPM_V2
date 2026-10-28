"""Forecast-time round robin batch sampler."""
from __future__ import annotations

from collections import defaultdict, deque
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm

from .gpm_dataset import FY3G3HDataset


def build_forecast_time_index(
    dataset: FY3G3HDataset,
    *,
    disable_tqdm: bool = False,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Index every patch by forecast time without reading large NPY arrays."""
    patches_by_time: dict[str, list[int]] = defaultdict(list)
    for index in tqdm(range(len(dataset)), desc="Index train", leave=False, disable=disable_tqdm):
        patches_by_time[str(dataset.metadata(index)["forecast_time"])].append(index)
    counts = [len(indices) for indices in patches_by_time.values()]
    audit = {
        "definition": "all patches exactly once per epoch; forecast-time round robin",
        "forecast_times_total": len(patches_by_time),
        "patches_total": sum(counts),
        "patches_per_forecast_time": {
            "minimum": min(counts, default=0),
            "maximum": max(counts, default=0),
            "mean": sum(counts) / max(len(counts), 1),
        },
    }
    return dict(patches_by_time), audit


def find_dry_uncovered_patches(dataset: FY3G3HDataset, rain_threshold: float = 0.1,
                              *, disable_tqdm: bool = False) -> set[int]:
    """Scan processed targets only for patches whose three radar frames are empty.

    Missing coverage metadata or missing target observations are kept conservatively.
    No validation/test samples may be filtered by this training-only helper.
    """
    if dataset.split != "train":
        raise ValueError("Background downsampling is allowed only on the training split")
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index in range(len(dataset)):
        fraction = np.asarray(dataset.metadata(index).get("input_valid_fraction", [np.nan]), dtype=float)
        if fraction.size == 3 and np.isfinite(fraction).all() and np.all(fraction == 0):
            shard, local = dataset.index[index]
            candidates[shard].append((index, local))
    background = set()
    for shard, rows in tqdm(candidates.items(), desc="Index dry background", leave=False, disable=disable_tqdm):
        amount = np.load(dataset.shards[shard] / "y_amount.npy", mmap_mode="r")
        valid = np.load(dataset.shards[shard] / "y_valid.npy", mmap_mode="r")
        try:
            for index, local in rows:
                mask = np.asarray(valid[local], bool)
                observed = np.asarray(amount[local])[mask]
                if observed.size and np.isfinite(observed).all() and not np.any(observed >= rain_threshold):
                    background.add(index)
        finally:
            amount._mmap.close()
            valid._mmap.close()
    return background


class ForecastTimeRoundRobinBatchSampler(Sampler[list[int]]):
    """Visit selected patches once, preserving covered/rainy patches in every epoch."""

    def __init__(
        self,
        patches_by_time: dict[str, list[int]],
        batch_size: int,
        seed: int,
        *,
        dry_uncovered_indices: set[int] | None = None,
        uncovered_patch_keep_ratio: float = 1.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not patches_by_time:
            raise ValueError("At least one forecast time is required")
        flat = [index for indices in patches_by_time.values() for index in indices]
        if len(flat) != len(set(flat)):
            raise ValueError("A patch appears under more than one forecast time")
        self.patches_by_time = {key: list(value) for key, value in patches_by_time.items()}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        if not math.isfinite(uncovered_patch_keep_ratio) or not 0 <= uncovered_patch_keep_ratio <= 1:
            raise ValueError("uncovered_patch_keep_ratio must lie in [0,1]")
        self.background = set(dry_uncovered_indices or ())
        if not self.background.issubset(flat):
            raise ValueError("Background patch indices must belong to the sampler")
        self.keep_ratio = float(uncovered_patch_keep_ratio)
        self.background_kept = math.ceil(len(self.background) * self.keep_ratio)
        self.sample_count = len(flat) - len(self.background) + self.background_kept
        if self.sample_count == 0:
            raise ValueError("Background downsampling removed every training sample")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.sample_count / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        kept_background = (set(rng.sample(sorted(self.background), self.background_kept))
                           if self.keep_ratio < 1 else self.background)
        queues: dict[str, deque[int]] = {}
        for forecast_time, indices in self.patches_by_time.items():
            shuffled = [i for i in indices if i not in self.background or i in kept_background]
            if not shuffled:
                continue
            rng.shuffle(shuffled)
            queues[forecast_time] = deque(shuffled)

        active = list(queues)
        rng.shuffle(active)
        emitted: set[int] = set()
        while active:
            batch: list[int] = []
            batch_times: set[str] = set()
            deferred: list[str] = []
            while active and len(batch) < self.batch_size:
                forecast_time = active.pop()
                if forecast_time in batch_times:
                    deferred.append(forecast_time)
                    continue
                index = queues[forecast_time].popleft()
                if index in emitted:
                    raise RuntimeError(f"Patch {index} was sampled twice in Epoch {self.epoch}")
                emitted.add(index)
                batch.append(index)
                batch_times.add(forecast_time)
                if queues[forecast_time]:
                    deferred.append(forecast_time)
            rng.shuffle(deferred)
            active = deferred + active

            while active and len(batch) < self.batch_size:
                forecast_time = active.pop()
                index = queues[forecast_time].popleft()
                if index in emitted:
                    raise RuntimeError(f"Patch {index} was sampled twice in Epoch {self.epoch}")
                emitted.add(index)
                batch.append(index)
                if queues[forecast_time]:
                    active.insert(0, forecast_time)
            if batch:
                yield batch

        if len(emitted) != self.sample_count:
            raise RuntimeError(
                f"Epoch {self.epoch} emitted {len(emitted)} of {self.sample_count} patches"
            )


def write_sampler_audit(
    path: str | Path,
    sampler: ForecastTimeRoundRobinBatchSampler,
    patches_by_time: dict[str, list[int]],
    metadata: dict[str, Any],
) -> None:
    first_epoch = list(iter(sampler))
    flat = [index for batch in first_epoch for index in batch]
    duplicate_patches = len(flat) - len(set(flat))
    index_to_time = {
        index: forecast_time
        for forecast_time, indices in patches_by_time.items()
        for index in indices
    }
    duplicate_times_by_batch = []
    for batch_index, batch in enumerate(first_epoch):
        times = [index_to_time[index] for index in batch]
        duplicate_times_by_batch.append({
            "batch": batch_index,
            "samples": len(batch),
            "duplicate_forecast_times": len(times) - len(set(times)),
        })
    payload = {
        **metadata,
        "batch_size": sampler.batch_size,
        "batches_per_epoch": len(sampler),
        "dry_uncovered_patches": len(sampler.background),
        "background_kept_per_epoch": sampler.background_kept,
        "uncovered_patch_keep_ratio": sampler.keep_ratio,
        "first_epoch_unique_patch_check": {
            "emitted": len(flat),
            "expected": sampler.sample_count,
            "duplicate_patches": duplicate_patches,
            "passed": len(flat) == sampler.sample_count and duplicate_patches == 0,
        },
        "duplicate_forecast_times_by_batch": duplicate_times_by_batch,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

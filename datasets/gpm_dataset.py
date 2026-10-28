"""PyTorch Dataset for FY-3G GPM surface precipitation shards."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from schema import ARRAY_NAMES
from .storage import shard_selections


class FY3G3HDataset(Dataset):
    """Loads NPY shards with LRU caching for array slices and metadata."""

    def __init__(self, root: str | Path, split: str, cache_size: int = 2) -> None:
        self.split = split
        self.selections = shard_selections(root, split)
        self.shards = [selection.path for selection in self.selections]
        self.index: list[tuple[int, int]] = []
        for shard_index, selection in enumerate(self.selections):
            self.index.extend((shard_index, int(local)) for local in selection.rows)
        self.cache_size = cache_size
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.metadata_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, index: int) -> dict[str, np.ndarray]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        shard = self.shards[index]
        value = {name: np.load(shard / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
        self.cache[index] = value
        return value

    def metadata(self, index: int) -> dict[str, Any]:
        shard_index, local = self.index[index]
        if shard_index not in self.metadata_cache:
            if len(self.metadata_cache) >= self.cache_size:
                self.metadata_cache.popitem(last=False)
            path = self.shards[shard_index] / "metadata.jsonl"
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.metadata_cache[shard_index] = values
        else:
            self.metadata_cache.move_to_end(shard_index)
        value = dict(self.metadata_cache[shard_index][local])
        value["source_shard"] = self.selections[shard_index].source_shard
        value["split"] = self.split
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, local = self.index[index]
        values = self._load(shard_index)
        sample = {name: torch.from_numpy(np.asarray(array[local]).copy()) for name, array in values.items()}
        sample["_sample_index"] = torch.tensor(index, dtype=torch.int64)
        return sample

    def close(self) -> None:
        for val in self.cache.values():
            for arr in val.values():
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    try:
                        arr._mmap.close()
                    except Exception:
                        pass
        self.cache.clear()
        self.metadata_cache.clear()

    def __enter__(self) -> "FY3G3HDataset":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# Alias
GPMDataset = FY3G3HDataset

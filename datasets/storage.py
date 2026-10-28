"""Discover completed shards in the current dataset layout."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShardSelection:
    path: Path
    rows: np.ndarray
    source_shard: str


def list_shards(root: str | Path, split: str) -> list[Path]:
    result = []
    for path in sorted((Path(root) / "shards" / split).glob("shard_*")):
        manifest = path / "shard_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete":
            result.append(path)
    return result


def shard_selections(root: str | Path, split: str) -> list[ShardSelection]:
    root = Path(root)
    selections = []
    for shard in list_shards(root, split):
        count = int(json.loads((shard / "shard_manifest.json").read_text(encoding="utf-8"))["sample_count"])
        selections.append(ShardSelection(shard, np.arange(count, dtype=np.int64), f"{split}/{shard.name}"))
    return selections

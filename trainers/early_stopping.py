"""Patience-based stopping on validation CSI, with resumable state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass
class EarlyStopping:
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.001
    best_score: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, score: float, epoch: int) -> bool:
        if math.isfinite(score) and score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return self.enabled and self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_score = float(state["best_score"])
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])

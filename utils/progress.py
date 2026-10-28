"""Shared progress-bar settings for training and validation."""

import os


def should_disable_tqdm(disable_tqdm: bool) -> bool:
    if disable_tqdm:
        return True
    return os.getenv("TQDM_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}

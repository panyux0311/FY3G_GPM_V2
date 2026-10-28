from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "FY3G_GPM",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    if isinstance(name, Path) or (isinstance(name, str) and ("/" in name or "\\" in name)):
        log_file = name
        name = "FY3G_GPM"
    logger = logging.getLogger(str(name))
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

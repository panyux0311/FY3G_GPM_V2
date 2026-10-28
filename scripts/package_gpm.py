#!/usr/bin/env python3
"""Create an in-place source archive for server deployment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "fy3ggpm_training_0p1_v3.tar.gz"

EXCLUDED = {"__pycache__", ".pytest_cache", ".tmp-tests", "outputs", "output", "checkpoints", ".git", ".vscode", ".venv", "venv"}
ACTIVE_SCRIPTS = {
    "train.py",
    "evaluate.py",
    "preflight.py",
    "finalize.py",
    "package_gpm.py",
    "run_gpm_training.sh",
    "start_gpm_tmux.sh",
    "run_gpm_finalize.sh",
    "start_gpm_finalize_tmux.sh",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in EXCLUDED for part in p.relative_to(ROOT).parts) or p.suffix == ".pyc":
            continue
        relative = p.relative_to(ROOT)
        if relative.parts[0] == "scripts" and p.name not in ACTIVE_SCRIPTS:
            continue
        files.append(p)

    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        names = tar.getnames()
        required = {
            "scripts/train.py", "configs/default.yaml",
            "models/UNetStandard.py", "models/SimVPv2.py", "models/precipitation.py", "models/factory.py",
        }
        if not required.issubset(names):
            raise RuntimeError(f"Archive is missing required files: {required - set(names)}")

    with ARCHIVE.with_name(ARCHIVE.name + ".sha256").open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{sha(ARCHIVE)}  {ARCHIVE.name}\n")

    print(f"Training archive created: {ARCHIVE}")


if __name__ == "__main__":
    main()

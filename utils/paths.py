from __future__ import annotations

from pathlib import Path


def resolve_output_dir(
    output_dir: str | Path | None = None,
    config_name: str = "gpm_surface_v1",
    base_dir: str | Path = "outputs",
) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(base_dir) / config_name


def check_output_dir_safety(
    output_dir: str | Path,
    allow_resume: bool = False,
    overwrite: bool = False,
) -> None:
    p = Path(output_dir)
    if not p.exists() or allow_resume or overwrite:
        return
    existing_artifacts = [
        f for f in ["metrics.json", "history.json", "run_status.json", "checkpoints/last_resume.pt", "checkpoints/best_balanced_inference.pt"]
        if (p / f).exists()
    ]
    if existing_artifacts:
        raise FileExistsError(
            f"Output directory '{p}' already contains existing experiment artifacts ({', '.join(existing_artifacts)}). "
            f"Use --resume to continue training or --overwrite to overwrite."
        )


def get_experiment_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_checkpoint_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_visualization_dir(output_dir: str | Path, name: str = "visualizations") -> Path:
    p = Path(output_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

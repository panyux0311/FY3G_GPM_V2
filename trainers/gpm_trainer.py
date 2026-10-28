"""Main training loop with EMA, cosine LR decay, and deduplicated grid evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from datasets.gpm_inputs import assemble_model_input
from metrics.gpm_metrics import selection_summary
from utils.config import ProjectConfig
from utils.progress import should_disable_tqdm
from .checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint_architecture, validate_checkpoint_dataset, validate_checkpoint_selection
from .evaluation import evaluate_deduplicated, evaluate_with_fixed_thresholds
from .early_stopping import EarlyStopping
from .gpm_pipeline import PipelineContext
from .optimization import compute_lr_factor
from .metrics_file import epoch_metrics, write_metrics


def copy_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def _format_rain_csi(value: dict[str, Any], prefix: str = "CSI") -> str:
    return " ".join(
        f"{prefix}@{threshold}={score:.6f}"
        for threshold, score in value.get("csi_by_threshold", {}).items()
    )


class GPMTrainer:
    def __init__(self, context: PipelineContext) -> None:
        self.ctx = context
        self.config = context.config
        self.output = context.output_dir

    def _selection(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return selection_summary(metrics, self.config.train.primary_selection_domain)

    def _history_selection(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.config.train.primary_selection_domain == "macro_all":
            return record["validation"]
        return record["selection_summary"]

    def _run_diagnostics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        threshold = self.config.train.diagnostics_probability_threshold
        thresholds = np.full_like(
            np.asarray(metrics["threshold_selection"]["thresholds"], dtype=np.float32), threshold,
        )
        common = dict(
            mean=self.ctx.mean, std=self.ctx.std, device=self.ctx.device,
            thresholds=thresholds, microbatch_size=self.config.train.microbatch_size,
            criterion=self.ctx.criterion,
            disable_tqdm=should_disable_tqdm(self.config.train.disable_tqdm),
        )
        val_ema = metrics["fixed_threshold_metrics"]
        val_ema["losses"] = metrics["losses"]
        result = {
            "probability_threshold": threshold,
            "train_samples": len(self.ctx.train_diagnostic_dataset),
            "train_ema": evaluate_with_fixed_thresholds(
                self.ctx.ema, self.ctx.train_diagnostic_dataset,
                progress_desc="Train subset diagnostics", **common,
            ),
            "val_ema": val_ema,
        }
        if self.config.train.diagnostics_compare_raw_ema:
            result["val_raw"] = evaluate_with_fixed_thresholds(
                self.ctx.model, self.ctx.val_dataset,
                progress_desc="Validation raw model", **common,
            )
        for name in ("train_ema", "val_ema", "val_raw"):
            if name not in result:
                continue
            value = result[name]
            summary = value["summary"]
            rows = value["by_lead_target_zone"]
            valid = sum(row["valid"] for row in rows)
            value["data_profile"] = {
                "valid_voxels": valid,
                "rain_fraction": sum(row["tp"] + row["fn"] for row in rows) / valid if valid else math.nan,
                "coverage_fraction_by_zone": {
                    zone: sum(row["valid"] for row in rows if row["zone"] == zone) / valid if valid else math.nan
                    for zone in ("direct", "near", "uncovered")
                },
            }
            self.ctx.logger.info(
                "Diagnostics %s probability_threshold=%.2f loss=%.6f "
                "CSI=%.6f POD=%.6f FAR=%.6f MAE=%.6f RMSE=%.6f CC=%.6f",
                name, threshold, value["losses"].get("loss", math.nan),
                summary["csi"], summary["pod"], summary["far"],
                summary["mae_mm_h"], summary["rmse_mm_h"], summary["cc"],
            )
            if summary.get("csi_by_threshold"):
                self.ctx.logger.info("Diagnostics %s rain-rate: %s", name, _format_rain_csi(summary))
            self.ctx.logger.info("Diagnostics %s data_profile: %s", name, json.dumps(value["data_profile"]))
        return result

    def train(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        best_score = -math.inf
        best: dict[str, Any] | None = None
        best_epoch = 0
        best_by_metric: dict[str, dict[str, Any]] = {}
        save_metrics = self.config.train.save_best_by_metrics
        for metric_name in save_metrics:
            best_by_metric[metric_name] = {"score": -math.inf, "epoch": 0}

        stopping = EarlyStopping(
            self.config.train.early_stopping_enabled,
            self.config.train.early_stopping_patience,
            self.config.train.early_stopping_min_delta,
        )
        resume_payload = getattr(self.ctx, "resume_payload", None) or {}

        # Restore history and best_score if resuming
        if self.ctx.resume_checkpoint is not None:
            validate_checkpoint_selection(resume_payload, self.config.train.primary_selection_domain)
            history = [h for h in (resume_payload.get("history") or []) if h["epoch"] < self.ctx.start_epoch]
            history_file = self.output / "metrics.json"
            if not history and history_file.exists():
                loaded = json.loads(history_file.read_text(encoding="utf-8"))
                history = [h for h in loaded.get("epochs", []) if h.get("epoch", 0) < self.ctx.start_epoch]

            # 2. Restore best validation score and checkpoint info
            best_ckpt_path = self.output / "checkpoints/best_balanced_inference.pt"
            if best_ckpt_path.exists():
                best_payload = load_checkpoint(best_ckpt_path, torch.device("cpu"))
                validate_checkpoint_architecture(best_payload, self.config.model.architecture, self.config.model)
                validate_checkpoint_selection(best_payload, self.config.train.primary_selection_domain)
                validate_checkpoint_dataset(best_payload, self.ctx.inputs["manifest_sha256"])
                if best_payload.get("target_statistics", {}).get("weight_strategy") != self.ctx.targets.get("weight_strategy"):
                    raise ValueError("Best checkpoint intensity weight strategy differs from resume checkpoint")
                if int(best_payload["epoch"]) >= self.ctx.start_epoch:
                    raise ValueError("Best checkpoint is newer than the resume checkpoint; resume the latest checkpoint or use a new output directory")
                if best_payload.get("metrics") is not None:
                    best = best_payload["metrics"]
                    best_score = float(self._selection(best).get("balanced_score", -math.inf))
                    best_epoch = int(best_payload["epoch"])

            if best is None and self.ctx.resume_payload is not None:
                if self.ctx.resume_payload.get("best_score") is not None:
                    best_score = float(self.ctx.resume_payload["best_score"])
                if self.ctx.resume_payload.get("metrics") is not None:
                    best = self.ctx.resume_payload["metrics"]

            training_state = resume_payload.get("training_state") or {}
            saved_objectives = training_state.get("best_by_metric", {})
            for name in save_metrics:
                saved = saved_objectives.get(name)
                if saved is None:
                    continue
                checkpoint = self.output / "checkpoints" / f"best_{name}_inference.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Resume requires the existing best checkpoint: {checkpoint}")
                objective_payload = load_checkpoint(checkpoint, torch.device("cpu"))
                validate_checkpoint_dataset(objective_payload, self.ctx.inputs["manifest_sha256"])
                validate_checkpoint_architecture(objective_payload, self.config.model.architecture, self.config.model)
                validate_checkpoint_selection(objective_payload, self.config.train.primary_selection_domain)
                if objective_payload.get("target_statistics", {}).get("weight_strategy") != self.ctx.targets.get("weight_strategy"):
                    raise ValueError(f"Best {name} checkpoint uses a different intensity weight strategy")
                if int(objective_payload["epoch"]) != int(saved["epoch"]) or int(saved["epoch"]) >= self.ctx.start_epoch:
                    raise ValueError(f"Best {name} checkpoint does not match the resume state")
                best_by_metric[name] = {"epoch": int(saved["epoch"]), "score": float(saved["score"])}
            if best is None:
                best = training_state.get("best_metrics")
            if not best_epoch:
                best_epoch = int(training_state.get("best_epoch", 0))
            state = training_state.get("early_stopping")
            if state and state.get("patience") == stopping.patience and state.get("min_delta") == stopping.min_delta:
                stopping.load_state_dict(state)
            else:
                for row in history:
                    if "validation" in row:
                        stopping.update(self._history_selection(row)["csi"], row["epoch"])
                if not history and math.isfinite(best_score):
                    stopping.best_score = best_score
                    stopping.best_epoch = best_epoch
            if not best_epoch and history:
                best_epoch = max(
                    (h for h in history if "validation" in h),
                    key=lambda h: self._history_selection(h)["csi"], default={"epoch": 0},
                )["epoch"]

        accumulate = max(1, self.config.train.effective_batch_size // self.config.train.microbatch_size)
        tqdm_disabled = should_disable_tqdm(self.config.train.disable_tqdm)

        # Determine autocast configuration
        amp_dtype_str = getattr(self.config.train, "amp_dtype", "bfloat16").lower()
        if self.ctx.device.type == "cuda":
            if amp_dtype_str == "bfloat16":
                amp_dtype = torch.bfloat16
                amp_enabled = True
            elif amp_dtype_str == "float16":
                amp_dtype = torch.float16
                amp_enabled = True
            else:
                amp_dtype = torch.float32
                amp_enabled = False
        else:
            amp_dtype = torch.float32
            amp_enabled = False

        epoch_progress = tqdm(
            () if stopping.should_stop else range(self.ctx.start_epoch, self.config.train.epochs + 1),
            desc="Epochs",
            leave=True,
            disable=tqdm_disabled,
        )
        completed_epoch = self.ctx.start_epoch - 1
        if stopping.should_stop:
            self.ctx.logger.info("Resume: early-stopping patience already exhausted; no further epochs run")
            return {
                "status": "early_stopped", "epochs": completed_epoch,
                "planned_epochs": self.config.train.epochs, "best_epoch": best_epoch,
                "early_stopping": stopping.state_dict(), "best_validation": best,
                "checkpoint": str(self.output / "checkpoints/best_balanced_inference.pt"),
                "best_by_metric": best_by_metric,
            }
        for epoch in epoch_progress:
            if hasattr(self.ctx.train_loader.batch_sampler, "set_epoch"):
                self.ctx.train_loader.batch_sampler.set_epoch(epoch)

            self.ctx.model.train()
            factor = compute_lr_factor(
                epoch,
                self.config.train.warmup_epochs,
                self.config.train.epochs,
                self.config.train.peak_learning_rate,
                self.config.train.final_learning_rate,
            )
            for group in self.ctx.optimizer.param_groups:
                group["lr"] = self.config.train.peak_learning_rate * factor

            ramp = (
                0.0
                if epoch < self.config.train.hard_negative_ramp_start_epoch
                else min(
                    1.0,
                    (epoch - self.config.train.hard_negative_ramp_start_epoch + 1)
                    / max(1, self.config.train.hard_negative_ramp_end_epoch - self.config.train.hard_negative_ramp_start_epoch + 1),
                )
            )
            self.ctx.criterion.hard_negative_weight = self.config.train.hard_negative_weight * ramp

            total = 0.0
            steps = 0
            loss_totals: dict[str, float] = {}
            self.ctx.optimizer.zero_grad(set_to_none=True)

            train_progress = tqdm(
                self.ctx.train_loader,
                desc=f"Train {epoch}/{self.config.train.epochs}",
                leave=False,
                disable=tqdm_disabled,
            )
            for step, batch in enumerate(train_progress, 1):
                moved = {key: value.to(self.ctx.device) for key, value in batch.items() if key != "_sample_index"}
                autocast = torch.autocast(device_type=self.ctx.device.type, dtype=amp_dtype, enabled=amp_enabled)
                with autocast:
                    inputs = assemble_model_input(moved, self.ctx.mean, self.ctx.std)
                    outputs = self.ctx.model(inputs)
                    losses = self.ctx.criterion(outputs, moved["y_amount"], moved["y_valid"])
                    loss = losses["loss"] / accumulate

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")

                loss.backward()
                total += float(losses["loss"].detach())
                for key, value in losses.items():
                    if (key == "loss" or key.endswith("_loss")) and value.numel() == 1:
                        loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach())
                steps += 1
                if step % 10 == 0 or step == len(self.ctx.train_loader):
                    train_progress.set_postfix(loss=f"{total / steps:.4f}", refresh=False)

                if step % accumulate == 0 or step == len(self.ctx.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.ctx.model.parameters(), self.config.train.grad_clip, error_if_nonfinite=True)
                    self.ctx.optimizer.step()
                    self.ctx.optimizer.zero_grad(set_to_none=True)
                    copy_ema(self.ctx.ema, self.ctx.model, self.config.train.ema_decay)

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": total / max(steps, 1),
                "learning_rate": self.ctx.optimizer.param_groups[0]["lr"],
                "hard_negative_weight": self.ctx.criterion.hard_negative_weight,
                "train_losses": {key: value / max(steps, 1) for key, value in loss_totals.items()},
            }

            diagnostic_epoch = self.config.train.diagnostics_enabled and (
                epoch == self.ctx.start_epoch
                or epoch % self.config.train.diagnostics_every_epochs == 0
                or epoch == self.config.train.epochs
            )

            metrics = evaluate_deduplicated(
                self.ctx.ema,
                self.ctx.val_dataset,
                self.ctx.mean,
                self.ctx.std,
                self.ctx.device,
                self.config,
                criterion=self.ctx.criterion,
                diagnostics_probability_threshold=(
                    self.config.train.diagnostics_probability_threshold
                    if diagnostic_epoch else None
                ),
            )
            summary = metrics["summary"]
            selected = self._selection(metrics)
            metrics["selection_domain"] = self.config.train.primary_selection_domain
            metrics["selection_summary"] = selected
            record["validation"] = summary
            for name in ("observable_summary", "pooled_summary", "direct_summary"):
                if name in metrics:
                    record[name] = metrics[name]
            record["selection_domain"] = self.config.train.primary_selection_domain
            record["selection_summary"] = selected
            record["validation_losses"] = metrics.get("losses", {})
            record["validation_loss"] = record["validation_losses"].get("loss", math.nan)
            record["validation_by_lead_target_zone"] = metrics.get("by_lead_target_zone", [])
            self.ctx.ema.set_occurrence_thresholds_by_zone(metrics["threshold_selection"]["thresholds"])
            improved = selected["balanced_score"] > best_score
            if improved:
                best_score = selected["balanced_score"]
                best = metrics
                best_epoch = epoch

            objective_improvements = set()
            for metric_name in save_metrics:
                current_score = selected.get("csi_by_threshold", {}).get(metric_name.removeprefix("csi_"), math.nan)
                if current_score is not None and math.isfinite(float(current_score)) and current_score > best_by_metric[metric_name]["score"]:
                    best_by_metric[metric_name] = {"score": float(current_score), "epoch": epoch}
                    objective_improvements.add(metric_name)

            stop = stopping.update(selected["csi"], epoch)
            record["early_stopping"] = stopping.state_dict()
            if diagnostic_epoch:
                record["diagnostics"] = self._run_diagnostics(metrics)

            history.append(record)
            epoch_progress.set_postfix(
                loss=f"{record['train_loss']:.4f}", csi=f"{selected['csi']:.4f}", refresh=False,
            )
            self.ctx.logger.info(
                "Epoch %d/%d train_loss=%.6f lr=%.6g hard_negative_weight=%.6f "
                "val_CSI=%.6f val_POD=%.6f val_FAR=%.6f val_MAE=%.6f "
                "val_RMSE=%.6f val_CC=%.6f val_loss=%.6f balanced_score=%.6f",
                epoch, self.config.train.epochs, record["train_loss"], record["learning_rate"],
                record["hard_negative_weight"], summary["csi"], summary["pod"],
                summary["far"], summary["mae_mm_h"], summary["rmse_mm_h"],
                summary.get("cc", math.nan), record["validation_loss"], summary["balanced_score"],
            )
            if summary.get("csi_by_threshold"):
                self.ctx.logger.info(
                    "Epoch %d/%d rain-rate validation: %s",
                    epoch, self.config.train.epochs, _format_rain_csi(summary, "val_CSI"),
                )
            self.ctx.logger.info("Selection domain=%s CSI=%.6f MAE=%.6f %s",
                                 self.config.train.primary_selection_domain, selected["csi"],
                                 selected["mae_mm_h"], _format_rain_csi(selected))
            for name in ("observable_summary", "pooled_summary"):
                if name in metrics:
                    value = metrics[name]
                    self.ctx.logger.info("%s CSI=%.6f POD=%.6f FAR=%.6f MAE=%.6f RMSE=%.6f CC=%.6f %s",
                                         name, value["csi"], value["pod"], value["far"], value["mae_mm_h"],
                                         value["rmse_mm_h"], value["cc"], _format_rain_csi(value))
            for name in ("train_losses", "validation_losses"):
                self.ctx.logger.info("Epoch %d %s: %s", epoch, name, " ".join(
                    f"{key}={value:.6f}" for key, value in record[name].items()
                ))
            for row in record["validation_by_lead_target_zone"]:
                self.ctx.logger.info(
                    "Validation lead=+%dh zone=%s CSI=%.6f POD=%.6f FAR=%.6f "
                    "MAE=%.6f RMSE=%.6f CC=%.6f probability_threshold=%.2f %s",
                    row["lead_hour"], row["zone"], row["csi"], row["pod"], row["far"],
                    row["mae_mm_h"], row["rmse_mm_h"], row["cc"], row["threshold"],
                    _format_rain_csi(row),
                )
            if stopping.enabled:
                self.ctx.logger.info(
                    "Early stopping: CSI reference=%.6f epoch=%d stale_epochs=%d/%d min_delta=%.6f",
                    stopping.best_score, stopping.best_epoch, stopping.bad_epochs,
                    stopping.patience, stopping.min_delta,
                )

            training_state = {
                "early_stopping": stopping.state_dict(), "best_epoch": best_epoch,
                "best_metrics": best, "best_by_metric": best_by_metric,
                "diagnostic_audit": getattr(self.ctx, "diagnostic_audit", None),
                "selection_domain": self.config.train.primary_selection_domain,
            }
            compact_history = [epoch_metrics(row) for row in history]
            if epoch % 5 == 0 or epoch == self.config.train.epochs or stop:
                save_checkpoint(
                    path=self.output / "checkpoints" / f"epoch{epoch}_inference.pt",
                    model=self.ctx.ema,
                    ema=self.ctx.ema,
                    optimizer=self.ctx.optimizer,
                    epoch=epoch,
                    config=self.config,
                    inputs=self.ctx.inputs,
                    targets=self.ctx.targets,
                    metrics=metrics,
                    best_score=best_score,
                    history=compact_history,
                    training_state=training_state,
                )
            if improved:
                save_checkpoint(
                    path=self.output / "checkpoints" / "best_balanced_inference.pt",
                    model=self.ctx.ema,
                    ema=self.ctx.ema,
                    optimizer=self.ctx.optimizer,
                    epoch=epoch,
                    config=self.config,
                    inputs=self.ctx.inputs,
                    targets=self.ctx.targets,
                    metrics=metrics,
                    best_score=best_score,
                    history=compact_history,
                    training_state=training_state,
                )

            # Save best checkpoint for each tracked metric
            for metric_name in objective_improvements:
                metric_path = self.output / "checkpoints" / f"best_{metric_name}_inference.pt"
                save_checkpoint(
                    path=metric_path,
                    model=self.ctx.ema,
                    ema=self.ctx.ema,
                    optimizer=self.ctx.optimizer,
                    epoch=epoch,
                    config=self.config,
                    inputs=self.ctx.inputs,
                    targets=self.ctx.targets,
                    metrics=metrics,
                    best_score=best_by_metric[metric_name]["score"],
                    history=compact_history,
                    training_state=training_state,
                )
                self.ctx.logger.info(
                    "Saved best %s checkpoint: epoch=%d score=%.6f",
                    metric_name, epoch, best_by_metric[metric_name]["score"],
                )

            save_checkpoint(
                path=self.output / "checkpoints" / "last_resume.pt",
                model=self.ctx.model,
                ema=self.ctx.ema,
                optimizer=self.ctx.optimizer,
                epoch=epoch,
                config=self.config,
                inputs=self.ctx.inputs,
                targets=self.ctx.targets,
                metrics=None,
                best_score=best_score,
                history=compact_history,
                training_state=training_state,
            )
            write_metrics(
                self.output / "metrics.json", compact_history,
                status="early_stopped" if stop else ("complete" if epoch == self.config.train.epochs else "running"),
                best_epoch=best_epoch, best_score=best_score,
                best_by_metric=best_by_metric,
                manifest_sha256=self.ctx.targets.get("manifest_sha256"),
                strategy=self.ctx.targets.get("weight_strategy"),
                selection_domain=self.config.train.primary_selection_domain,
            )
            completed_epoch = epoch
            if stop:
                self.ctx.logger.info(
                    "Early stopping at epoch %d; best CSI=%.6f at epoch %d. Last resume checkpoint saved.",
                    epoch, best_score, best_epoch,
                )
                break
        epoch_progress.close()

        best_cp = str(self.output / "checkpoints" / "best_balanced_inference.pt")
        if not Path(best_cp).exists():
            self.ctx.logger.warning("Best checkpoint is not present in this output directory; retain the original best checkpoint when resuming")
            best_cp = None

        # Build multi-objective summary
        best_checkpoints = {}
        for metric_name in save_metrics:
            ckpt_path = self.output / "checkpoints" / f"best_{metric_name}_inference.pt"
            if ckpt_path.exists() and best_by_metric[metric_name]["epoch"] > 0:
                best_checkpoints[metric_name] = {
                    "path": str(ckpt_path),
                    "epoch": best_by_metric[metric_name]["epoch"],
                    "score": best_by_metric[metric_name]["score"],
                }

        result = {
            "status": "early_stopped" if stopping.should_stop else "complete",
            "epochs": completed_epoch,
            "planned_epochs": self.config.train.epochs,
            "best_epoch": best_epoch,
            "early_stopping": stopping.state_dict(),
            "best_validation": best,
            "checkpoint": best_cp,
            "best_by_metric": best_checkpoints,
        }

        if completed_epoch < self.ctx.start_epoch:
            write_metrics(
                self.output / "metrics.json", history,
                status=result["status"], best_epoch=best_epoch, best_score=best_score,
                best_by_metric=best_by_metric,
                manifest_sha256=self.ctx.targets.get("manifest_sha256"),
                strategy=self.ctx.targets.get("weight_strategy"),
                selection_domain=self.config.train.primary_selection_domain,
            )
        return result


def train_project(
    config: ProjectConfig,
    data_root: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    context = PipelineContext(
        config=config,
        data_root=data_root,
        resume_checkpoint=resume_checkpoint,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    trainer = GPMTrainer(context)
    return trainer.train()

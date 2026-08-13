"""Strict, content-addressed configuration loading for training stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from kcorrdiff.data.condition_augmentation import ConditionAugmentationPolicy
from kcorrdiff.data.condition_schema import parse_condition_signature

from .losses import RegressionLossConfig
from .runtime import RuntimeContract, contract_from_mapping


STAGE2_NAME = "deterministic_regression_crossfit_oof"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    name: str,
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise ValueError(f"{name} schema mismatch: missing={missing}, extra={extra}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _int_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a two-element sequence")
    result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain two positive integers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Stage2DataConfig:
    target_coordinates: Path
    context_coordinates: Path
    normalization: Path
    condition_signature: str
    radar_timezone: str
    context_detail_mode: str
    context_wet_threshold_mm_per_hour: float
    condition_augmentation: ConditionAugmentationPolicy | None = None

    @property
    def condition_signatures(self) -> tuple[str, ...]:
        if self.condition_augmentation is None:
            return (self.condition_signature,)
        return tuple(
            str(entry.condition_signature)
            for entry in self.condition_augmentation.entries
        )

    @property
    def condition_augmentation_sha256(self) -> str | None:
        if self.condition_augmentation is None:
            return None
        return self.condition_augmentation.semantic_sha256


@dataclass(frozen=True, slots=True)
class Stage2OptimizationConfig:
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    gradient_clip_norm: float
    epochs: int
    scheduler_minimum_ratio: float
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    global_effective_batch_size: int
    checkpoint_every_optimizer_steps: int
    log_every_optimizer_steps: int


@dataclass(frozen=True, slots=True)
class Stage2OOFRemoteSpillConfig:
    endpoint: str
    remote_prefix: str
    local_reserve_bytes: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Stage2CrossfitConfig:
    folds: int
    train_split: str
    oof_shard_items: int
    maximum_oof_bytes: int
    maximum_oof_compressed_bytes: int
    maximum_oof_compression_ratio: float
    oof_compression_probe_bytes: int
    epsilon_residual_scale: float
    remote_spill: Stage2OOFRemoteSpillConfig


@dataclass(frozen=True, slots=True)
class Stage2Config:
    raw: Mapping[str, object]
    sha256: str
    protocol_version: str
    research_track: str
    seed: int
    runtime: RuntimeContract
    data: Stage2DataConfig
    loss: RegressionLossConfig
    optimization: Stage2OptimizationConfig
    crossfit: Stage2CrossfitConfig

    def validate_topology(self, *, world_size: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        expected = (
            world_size
            * self.optimization.per_rank_microbatch_size
            * self.optimization.gradient_accumulation_steps
        )
        if expected != self.optimization.global_effective_batch_size:
            raise ValueError(
                "global effective batch mismatch: "
                f"config={self.optimization.global_effective_batch_size}, "
                f"topology={expected}"
            )


def _canonical_hash(raw: Mapping[str, object]) -> str:
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_stage2_config(path: Path) -> Stage2Config:
    """Load the full-width Stage 2 config without accepting hidden defaults."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(loaded, "stage2 config")
    required_top = {
        "protocol_version",
        "stage",
        "research_track",
        "seed",
        "runtime",
        "data",
        "model",
        "loss",
        "optimization",
        "crossfit",
        "loader_tuning",
        "tracking",
        "publication",
    }
    _exact_keys(raw, required=required_top, name="stage2 config")
    if raw["protocol_version"] != "v1.1.3b" or raw["stage"] != STAGE2_NAME:
        raise ValueError("Stage 2 protocol/stage identifier mismatch")
    seed = _positive_int(raw["seed"], "seed")
    runtime = contract_from_mapping(_mapping(raw["runtime"], "runtime"))

    data = _mapping(raw["data"], "data")
    _exact_keys(
        data,
        required={
            "radar_timezone",
            "history_frames",
            "future_target_frames",
            "target_shape",
            "context_shape",
            "era_shape",
            "target_coordinates",
            "context_coordinates",
            "normalization",
            "condition_signature",
            "context_detail_mode",
            "context_wet_threshold_mm_per_hour",
            "mmap_lazy_per_worker",
            "pin_memory",
            "persistent_workers",
        },
        optional={"condition_augmentation"},
        name="data",
    )
    if data["history_frames"] != 12 or data["future_target_frames"] != 7:
        raise ValueError("radar temporal fallback is forbidden")
    if _int_pair(data["target_shape"], "target_shape") != (256, 256):
        raise ValueError("target spatial fallback is forbidden")
    if _int_pair(data["context_shape"], "context_shape") != (256, 256):
        raise ValueError("context spatial fallback is forbidden")
    if _int_pair(data["era_shape"], "era_shape") != (33, 33):
        raise ValueError("ERA spatial fallback is forbidden")
    for key in ("mmap_lazy_per_worker", "pin_memory", "persistent_workers"):
        if not _strict_bool(data[key], key):
            raise ValueError(f"{key} must remain enabled in production")
    detail_mode = str(data["context_detail_mode"])
    if detail_mode not in {"local_max", "nearest"}:
        raise ValueError("unsupported context detail mode")
    condition_signature = parse_condition_signature(
        str(data["condition_signature"])
    ).key
    condition_augmentation = (
        ConditionAugmentationPolicy.from_mapping(
            _mapping(data["condition_augmentation"], "condition_augmentation")
        )
        if "condition_augmentation" in data
        else None
    )
    if (
        condition_augmentation is not None
        and condition_augmentation.source_signature.key != condition_signature
    ):
        raise ValueError(
            "condition augmentation source signature disagrees with data config"
        )
    data_config = Stage2DataConfig(
        target_coordinates=Path(str(data["target_coordinates"])),
        context_coordinates=Path(str(data["context_coordinates"])),
        normalization=Path(str(data["normalization"])),
        condition_signature=condition_signature,
        radar_timezone=str(data["radar_timezone"]),
        context_detail_mode=detail_mode,
        context_wet_threshold_mm_per_hour=_positive_float(
            data["context_wet_threshold_mm_per_hour"],
            "context_wet_threshold_mm_per_hour",
            allow_zero=True,
        ),
        condition_augmentation=condition_augmentation,
    )
    if not data_config.condition_signature:
        raise ValueError("condition_signature cannot be empty")

    loss = _mapping(raw["loss"], "loss")
    _exact_keys(
        loss,
        required={
            "lambda_occurrence",
            "lambda_positive",
            "lambda_mean",
            "mean_warmup_steps",
            "detach_probability_in_mean",
            "importance_weight_clipping",
        },
        name="loss",
    )
    if loss["importance_weight_clipping"] is not None:
        raise ValueError("importance-weight clipping is not declared for v1.1.3b")
    loss_config = RegressionLossConfig(
        lambda_occurrence=_positive_float(
            loss["lambda_occurrence"], "lambda_occurrence", allow_zero=True
        ),
        lambda_positive=_positive_float(
            loss["lambda_positive"], "lambda_positive", allow_zero=True
        ),
        lambda_mean=_positive_float(
            loss["lambda_mean"], "lambda_mean", allow_zero=True
        ),
        mean_warmup_steps=int(loss["mean_warmup_steps"]),
        detach_probability_in_mean=_strict_bool(
            loss["detach_probability_in_mean"], "detach_probability_in_mean"
        ),
    )
    loss_config.validate()

    optimization = _mapping(raw["optimization"], "optimization")
    _exact_keys(
        optimization,
        required={
            "optimizer",
            "learning_rate",
            "weight_decay",
            "betas",
            "gradient_clip_norm",
            "epochs",
            "scheduler",
            "scheduler_minimum_ratio",
            "per_rank_microbatch_size",
            "gradient_accumulation_steps",
            "global_effective_batch_size",
            "checkpoint_every_optimizer_steps",
            "log_every_optimizer_steps",
        },
        name="optimization",
    )
    if optimization["optimizer"] != "AdamW" or optimization["scheduler"] != "cosine":
        raise ValueError("only the declared AdamW/cosine Stage 2 path is supported")
    betas_value = optimization["betas"]
    if not isinstance(betas_value, Sequence) or isinstance(betas_value, (str, bytes)):
        raise TypeError("betas must be a two-element sequence")
    betas = tuple(float(value) for value in betas_value)
    if len(betas) != 2 or not 0.0 < betas[0] < 1.0 or not 0.0 < betas[1] < 1.0:
        raise ValueError("AdamW betas must lie in (0,1)")
    optimization_config = Stage2OptimizationConfig(
        learning_rate=_positive_float(optimization["learning_rate"], "learning_rate"),
        weight_decay=_positive_float(
            optimization["weight_decay"], "weight_decay", allow_zero=True
        ),
        betas=betas,  # type: ignore[arg-type]
        gradient_clip_norm=_positive_float(
            optimization["gradient_clip_norm"], "gradient_clip_norm"
        ),
        epochs=_positive_int(optimization["epochs"], "epochs"),
        scheduler_minimum_ratio=_positive_float(
            optimization["scheduler_minimum_ratio"], "scheduler_minimum_ratio"
        ),
        per_rank_microbatch_size=_positive_int(
            optimization["per_rank_microbatch_size"], "per_rank_microbatch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            optimization["gradient_accumulation_steps"],
            "gradient_accumulation_steps",
        ),
        global_effective_batch_size=_positive_int(
            optimization["global_effective_batch_size"],
            "global_effective_batch_size",
        ),
        checkpoint_every_optimizer_steps=_positive_int(
            optimization["checkpoint_every_optimizer_steps"],
            "checkpoint_every_optimizer_steps",
        ),
        log_every_optimizer_steps=_positive_int(
            optimization["log_every_optimizer_steps"],
            "log_every_optimizer_steps",
        ),
    )
    if not 0.0 < optimization_config.scheduler_minimum_ratio <= 1.0:
        raise ValueError("scheduler_minimum_ratio must lie in (0,1]")

    crossfit = _mapping(raw["crossfit"], "crossfit")
    _exact_keys(
        crossfit,
        required={
            "folds",
            "fold_group",
            "train_split",
            "train_fold_models",
            "train_deployment_model",
            "held_out_inference_distributed",
            "oof_dtype",
            "oof_fields",
            "oof_shard_items",
            "maximum_oof_bytes",
            "oof_storage_encoding",
            "maximum_oof_compressed_bytes",
            "maximum_oof_compression_ratio",
            "oof_compression_probe_bytes",
            "epsilon_residual_scale",
            "oof_remote_spill",
        },
        name="crossfit",
    )
    if crossfit["fold_group"] != "manifest_block":
        raise ValueError("cross-fitting must be grouped by manifest block")
    if crossfit["train_split"] != "outer_train":
        raise ValueError("cross-fitting must use outer_train")
    for key in (
        "train_fold_models",
        "train_deployment_model",
        "held_out_inference_distributed",
    ):
        if not _strict_bool(crossfit[key], key):
            raise ValueError(f"{key} must remain enabled")
    if crossfit["oof_dtype"] != "float32" or tuple(crossfit["oof_fields"]) != (
        "occurrence_probability",
        "mu_z",
    ):
        raise ValueError("OOF field schema/precision fallback is forbidden")
    if (
        crossfit["oof_storage_encoding"]
        != "npz-deflate-byte-shuffle-lossless-float32"
    ):
        raise ValueError("OOF storage must use lossless byte-shuffled FP32 NPZ")
    remote_spill = _mapping(
        crossfit["oof_remote_spill"], "crossfit oof_remote_spill"
    )
    _exact_keys(
        remote_spill,
        required={
            "endpoint",
            "remote_prefix",
            "local_reserve_bytes",
            "timeout_seconds",
        },
        name="crossfit oof_remote_spill",
    )
    endpoint = str(remote_spill["endpoint"])
    remote_prefix = str(remote_spill["remote_prefix"])
    if not endpoint.startswith("https://") or not remote_prefix.startswith("/"):
        raise ValueError("OOF remote spill requires HTTPS endpoint/absolute prefix")
    remote_spill_config = Stage2OOFRemoteSpillConfig(
        endpoint=endpoint,
        remote_prefix=remote_prefix,
        local_reserve_bytes=_positive_int(
            remote_spill["local_reserve_bytes"], "OOF local reserve bytes"
        ),
        timeout_seconds=_positive_float(
            remote_spill["timeout_seconds"], "OOF remote timeout seconds"
        ),
    )
    if remote_spill_config.local_reserve_bytes < 10 * 1024**3:
        raise ValueError("OOF remote spill must preserve at least 10 GiB locally")
    if remote_spill_config.timeout_seconds > 3600.0:
        raise ValueError("OOF remote timeout cannot exceed one hour")
    crossfit_config = Stage2CrossfitConfig(
        folds=_positive_int(crossfit["folds"], "folds"),
        train_split=str(crossfit["train_split"]),
        oof_shard_items=_positive_int(crossfit["oof_shard_items"], "oof_shard_items"),
        maximum_oof_bytes=_positive_int(
            crossfit["maximum_oof_bytes"], "maximum_oof_bytes"
        ),
        maximum_oof_compressed_bytes=_positive_int(
            crossfit["maximum_oof_compressed_bytes"],
            "maximum_oof_compressed_bytes",
        ),
        maximum_oof_compression_ratio=_positive_float(
            crossfit["maximum_oof_compression_ratio"],
            "maximum_oof_compression_ratio",
        ),
        oof_compression_probe_bytes=_positive_int(
            crossfit["oof_compression_probe_bytes"],
            "oof_compression_probe_bytes",
        ),
        epsilon_residual_scale=_positive_float(
            crossfit["epsilon_residual_scale"], "epsilon_residual_scale"
        ),
        remote_spill=remote_spill_config,
    )
    if crossfit_config.folds not in {3, 5}:
        raise ValueError("the preregistered cross-fit choices are 3 or 5 folds")
    if crossfit_config.maximum_oof_compressed_bytes >= crossfit_config.maximum_oof_bytes:
        raise ValueError("compressed OOF cap must be smaller than logical FP32 bytes")

    return Stage2Config(
        raw=raw,
        sha256=_canonical_hash(raw),
        protocol_version=str(raw["protocol_version"]),
        research_track=str(raw["research_track"]),
        seed=seed,
        runtime=runtime,
        data=data_config,
        loss=loss_config,
        optimization=optimization_config,
        crossfit=crossfit_config,
    )


__all__ = [
    "STAGE2_NAME",
    "Stage2Config",
    "Stage2CrossfitConfig",
    "Stage2DataConfig",
    "Stage2OptimizationConfig",
    "Stage2OOFRemoteSpillConfig",
    "load_stage2_config",
]

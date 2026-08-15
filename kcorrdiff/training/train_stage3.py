"""Production Stage 3 residual-EDM training and selection governance.

This module deliberately stops at the boundary between training and
independent evaluation.  It can train the two pre-registered screening arms,
continue externally nominated finalists on the three frozen training seeds,
and bind an externally produced final model-selection decision.  It never
reads model-selection labels and contains no implementation that can choose a
candidate from training loss.

Production execution is exactly two ``torchrun`` ranks with NCCL, CUDA,
float32 parameters/activations, TF32 disabled, and no architecture or
precision fallback.  Unequal valid mass and explicit tail padding are handled
with a global numerator/denominator and manual gradient SUM, avoiding both the
mathematical error of averaging rank-local means and unequal-forward DDP
hangs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import yaml

from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.models.common import (
    CONDITION_DIM,
    CONTEXT_WIDTHS,
    PRODUCTION_INPUT_SIZE,
    TARGET_WIDTHS,
)
from kcorrdiff.models.physical_attention import ATTENTION_DIM, ATTENTION_HEADS
from kcorrdiff.models.regression_system import RegressionSystem
from kcorrdiff.models.residual_edm import (
    EDM_A_PRODUCTION_PARAMETER_COUNT,
    EDM_B_PRODUCTION_PARAMETER_COUNT,
    ResidualEDM,
    ResidualEDMConfig,
)
from kcorrdiff.training.batch import TrainingBatchCollator
from kcorrdiff.training.calibration import POOLING_ORDER
from kcorrdiff.training.calibration_artifact import (
    MONOTONE_LOGIT_LINEAR_FAMILY,
    FrozenModelSelectionDecision,
)
from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    TrainingCursor,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
)
from kcorrdiff.training.config import Stage2Config, load_stage2_config
from kcorrdiff.training.data_factory import (
    FactoryInputs,
    KCorrDiffDataFactory,
    build_data_factory,
    dataloader_options,
)
from kcorrdiff.training.edm_loss import (
    build_noisy_residual_training_state,
    masked_edm_loss_terms,
)
from kcorrdiff.training.stage3_data import (
    PROTOCOL_VERSION,
    STAGE2_TARGET_BUILDER_VERSION,
    Stage2FactoryTargetLoaderFactory,
    Stage3ArtifactBundle,
    Stage3ArtifactInputs,
    Stage3BatchCollator,
    Stage3DistributedPlan,
    Stage3RankDataset,
    Stage3TrainingBatch,
    build_stage3_distributed_plan,
    load_stage3_artifacts,
)
from kcorrdiff.training.oof_remote import AuthenticatedHTTPSShardStore
from kcorrdiff.training.runtime import assert_float32_tree
from kcorrdiff.training.release_index import (
    Stage2ReleaseIndex,
    load_stage2_release_index,
)
from kcorrdiff.training.tracking import TrackingRun, initialize_tracking
from kcorrdiff.training.train_stage2 import (
    DistributedRuntime,
    initialize_distributed_runtime,
    reduce_gradients_sum,
    regression_system_config_from_stage2,
)


STAGE3_NAME = "residual_edm_independent_calibration_inference"
STAGE3_CHECKPOINT_FORMAT = "kcorrdiff.stage3-training-checkpoint.v1"
STAGE3_PARTIAL_MANIFEST_FORMAT = "kcorrdiff.stage3-training-partial.v1"
STAGE3_SCREENING_MANIFEST_FORMAT = "kcorrdiff.stage3-screening-training.v1"
STAGE3_FINALIST_MANIFEST_FORMAT = "kcorrdiff.stage3-finalist-training.v1"
STAGE3_TRAINING_MANIFEST_FORMAT = "kcorrdiff.stage3-training.v1"
SCREENING_EVALUATION_FORMAT = "kcorrdiff.stage3-screening-evaluation.v1"
FINAL_SELECTION_FORMAT = "kcorrdiff.stage3-model-selection-decision.v1"

SCREEN_TRAINING_SEED = 11103
COMMON_ENSEMBLE_SEED = 11104
REPEAT_TRAINING_SEEDS = (11103, 11105, 11106)
DEPLOYMENT_TRAINING_SEED = 11103
EDM_VARIANTS = ("edm_a", "edm_b")
SELECTION_MEMBERS = 16
SELECTION_STEPS = 8
TRAINING_NOISE_PURPOSE = "edm-training-noise-v1"
CURSOR_SEMANTICS = "rank_local_plan_slots_consumed_including_padding"
PRODUCTION_CONDITION_AUGMENTATION_SHA256 = (
    "813d7395ac59897a7a258ed07205fff289c262b7e8df93cd2cef4deeace9c9f8"
)
PRODUCTION_CONDITION_SIGNATURES = (
    "era5_oracle:era=1:tp=0:full_trajectory",
    "era5_oracle:era=1:tp=1:full_trajectory",
    "null_provider:era=0:tp=0:no_era_access",
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FALLBACK_ENVIRONMENT = (
    "KCORRDIFF_ALLOW_CPU_FALLBACK",
    "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK",
    "KCORRDIFF_ALLOW_PRECISION_FALLBACK",
    "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _deep_freeze_config(value: object) -> object:
    """Return an immutable, detached copy of a validated config tree."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze_config(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_config(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze_config(item) for item in value)
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, name: str
) -> None:
    actual = {str(key) for key in value}
    if actual != set(expected):
        raise ValueError(
            f"{name} schema mismatch; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_power_of_two(value: object, *, name: str) -> int:
    result = _positive_int(value, name=name)
    if result & (result - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    result = float(value)
    accepted = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not accepted:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fixed_number(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == float(expected)
    )


def _integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain only integers")
    return tuple(value)


def _number_sequence(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    return tuple(
        _finite_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact_semantic_sha256(value: Mapping[str, object]) -> str:
    return _semantic_sha256(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    )


def source_tree_sha256(root: Path) -> str:
    """Hash the immutable Stage 3 Python/config/docs source tree."""

    selected = Path(root).resolve()
    if not selected.is_dir():
        raise NotADirectoryError(selected)
    files: list[Path] = []
    for relative_root in ("kcorrdiff", "configs", "docs"):
        base = selected / relative_root
        if base.is_dir():
            files.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    for name in ("pyproject.toml",):
        path = selected / name
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("Stage 3 source-tree identity has no files")
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda value: value.relative_to(selected).as_posix()):
        relative = path.relative_to(selected).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_container_image_identity(
    value: str, *, expected_sha256: str
) -> str:
    """Require an immutable image reference whose digest matches the CLI hash."""

    expected = _sha256(expected_sha256, name="expected container image SHA-256")
    if not isinstance(value, str) or "@sha256:" not in value:
        raise ValueError("KCORRDIFF_CONTAINER_IMAGE must be an immutable @sha256 reference")
    digest = value.rsplit("@sha256:", 1)[1]
    if digest != expected:
        raise ValueError("container image reference digest disagrees with CLI identity")
    return value


def verify_stage3_launch_identity(
    *,
    source_root: Path,
    expected_source_tree_sha256: str,
    expected_container_image_sha256: str,
    expected_runtime_report_sha256: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Detect source/runtime/image mutation at every durable publication."""

    environment = os.environ if environ is None else environ
    if source_tree_sha256(source_root) != _sha256(
        expected_source_tree_sha256, name="expected Stage 3 source tree"
    ):
        raise ValueError("Stage 3 source tree changed after launch identity capture")
    if _runtime_report_sha256() != _sha256(
        expected_runtime_report_sha256, name="expected Stage 3 runtime report"
    ):
        raise ValueError("Stage 3 runtime changed after launch identity capture")
    image = environment.get("KCORRDIFF_CONTAINER_IMAGE")
    if image is None:
        raise ValueError("KCORRDIFF_CONTAINER_IMAGE is required")
    validate_container_image_identity(
        image, expected_sha256=expected_container_image_sha256
    )


def _runtime_report_payload() -> dict[str, object]:
    distributions = sorted(
        (
            str(distribution.metadata.get("Name") or distribution.name).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    return {
        "format_version": "kcorrdiff.stage3-runtime-report.v1",
        "protocol_version": PROTOCOL_VERSION,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "torch_version": str(torch.__version__),
        "python_version": sys.version,
        "platform": platform.platform(),
        "installed_distributions": [list(value) for value in distributions],
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "cuda_device_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def _runtime_report_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(_runtime_report_payload())).hexdigest()


def write_runtime_report(path: Path) -> str:
    """Atomically record the exact CUDA/PyTorch runtime and return file hash."""

    return _atomic_json(Path(path), _runtime_report_payload(), replace=False)


def _atomic_json(path: Path, value: Mapping[str, object], *, replace: bool) -> str:
    """Durably publish canonical JSON, optionally replacing only a partial path."""

    selected = Path(path).resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, selected)
        else:
            try:
                os.link(temporary, selected)
            except FileExistsError:
                if selected.read_bytes() != payload:
                    raise FileExistsError(
                        f"immutable artifact already exists: {selected}"
                    ) from None
        directory = os.open(selected.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    if selected.read_bytes() != payload:
        raise RuntimeError(f"atomic JSON verification failed: {selected}")
    return sha256_file(selected)


@dataclass(frozen=True, slots=True)
class Stage3NoiseConfig:
    log_sigma_mean: float
    log_sigma_standard_deviation: float
    minimum_sigma: float
    maximum_sigma: float
    purpose_id: str


@dataclass(frozen=True, slots=True)
class Stage3OptimizationConfig:
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
class Stage3TrackingConfig:
    project: str
    job_type: str
    mode: str


@dataclass(frozen=True, slots=True)
class Stage3Config:
    raw: Mapping[str, object]
    sha256: str
    protocol_version: str
    research_track: str
    seed: int
    target_widths: tuple[int, ...]
    context_widths: tuple[int, ...]
    input_size: int
    activation_checkpoint: bool
    query_chunk_size: int
    sigma_data: float
    noise: Stage3NoiseConfig
    optimization: Stage3OptimizationConfig
    selected_num_workers: int
    selected_prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    condition_augmentation_sha256: str
    condition_signatures: tuple[str, ...]
    tracking: Stage3TrackingConfig
    selection_decision_file: Path
    probability_mapping_family: str
    pooling_order: tuple[str, ...]

    def validate_topology(self, *, world_size: int) -> None:
        if world_size != 2:
            raise ValueError("Stage 3 production topology requires exactly two ranks")
        expected = (
            world_size
            * self.optimization.per_rank_microbatch_size
            * self.optimization.gradient_accumulation_steps
        )
        if expected != self.optimization.global_effective_batch_size:
            raise ValueError(
                "global effective batch mismatch: "
                f"config={self.optimization.global_effective_batch_size}, topology={expected}"
            )


def load_stage3_config(path: Path) -> Stage3Config:
    """Load and fail-closed validate the frozen full-width Stage 3 config."""

    loaded = yaml.load(
        Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
    )
    raw = _mapping(loaded, name="Stage 3 config")
    _exact_keys(
        raw,
        {
            "protocol_version",
            "stage",
            "research_track",
            "seed",
            "runtime",
            "data",
            "stage2_contract",
            "model",
            "noise",
            "loss",
            "optimization",
            "loader_tuning",
            "sampling_profiles",
            "model_selection",
            "calibration",
            "inference",
            "tracking",
            "publication",
        },
        name="Stage 3 config",
    )
    if raw["protocol_version"] != PROTOCOL_VERSION or raw["stage"] != STAGE3_NAME:
        raise ValueError("Stage 3 protocol/stage identifier mismatch")
    if raw["research_track"] != "cprecnet_event_conditioned_era5_full_trajectory_oracle":
        raise ValueError("Stage 3 research-track fallback is forbidden")
    seed = _positive_int(raw["seed"], name="seed")
    if seed != SCREEN_TRAINING_SEED:
        raise ValueError("screening training seed must remain 11103")

    runtime = _mapping(raw["runtime"], name="runtime")
    _exact_keys(
        runtime,
        {
            "target_widths",
            "context_widths",
            "era_latent_channels",
            "era_grid_size",
            "target_grid_size",
            "target_spacing_km",
            "precision",
            "tf32",
            "allow_cpu_fallback",
            "allow_model_width_fallback",
            "allow_precision_fallback",
            "allow_era_grid_fallback",
        },
        name="runtime",
    )
    target_widths = _integer_sequence(
        runtime["target_widths"], name="runtime target_widths"
    )
    context_widths = _integer_sequence(
        runtime["context_widths"], name="runtime context_widths"
    )
    if target_widths != TARGET_WIDTHS or context_widths != CONTEXT_WIDTHS:
        raise ValueError("original full model widths are mandatory")
    if (
        type(runtime["era_latent_channels"]) is not int
        or runtime["era_latent_channels"] != 128
        or type(runtime["era_grid_size"]) is not int
        or runtime["era_grid_size"] != 33
        or type(runtime["target_grid_size"]) is not int
        or runtime["target_grid_size"] != 256
        or not _fixed_number(runtime["target_spacing_km"], 0.5)
        or runtime["precision"] != "float32"
        or runtime["tf32"] is not False
    ):
        raise ValueError("Stage 3 runtime geometry/precision fallback is forbidden")
    for key in (
        "allow_cpu_fallback",
        "allow_model_width_fallback",
        "allow_precision_fallback",
        "allow_era_grid_fallback",
    ):
        if _strict_bool(runtime[key], name=key):
            raise ValueError(f"runtime fallback must remain disabled: {key}")

    data = _mapping(raw["data"], name="data")
    _exact_keys(
        data,
        {
            "radar_timezone",
            "target_coordinates",
            "context_coordinates",
            "target_static",
            "normalization",
            "draw_manifest",
            "candidate_manifest",
            "bundle_metadata",
            "condition_augmentation_policy_sha256",
            "condition_signatures",
            "oof_dtype",
            "oof_fields",
            "target_builder_version",
            "mmap_lazy_per_worker",
            "pin_memory",
            "persistent_workers",
        },
        name="data",
    )
    if (
        data.get("radar_timezone") != "Asia/Seoul"
        or data.get("target_builder_version") != STAGE2_TARGET_BUILDER_VERSION
        or data.get("oof_dtype") != "float32"
        or tuple(data.get("oof_fields", ())) != ("occurrence_probability", "mu_z")
        or data.get("mmap_lazy_per_worker") is not True
        or data.get("pin_memory") is not True
        or data.get("persistent_workers") is not True
    ):
        raise ValueError("Stage 3 OOF/loader data contract fallback is forbidden")
    for key in (
        "target_coordinates",
        "context_coordinates",
        "target_static",
        "normalization",
        "draw_manifest",
        "candidate_manifest",
        "bundle_metadata",
    ):
        configured_path = data.get(key)
        if (
            not isinstance(configured_path, str)
            or not configured_path.strip()
            or not Path(configured_path).is_absolute()
        ):
            raise ValueError(f"Stage 3 data.{key} must be a non-empty absolute path")
    signatures_raw = data.get("condition_signatures")
    if not isinstance(signatures_raw, Sequence) or isinstance(
        signatures_raw, (str, bytes)
    ):
        raise TypeError("Stage 3 condition_signatures must be a sequence")
    condition_signatures = tuple(str(value) for value in signatures_raw)
    if condition_signatures != PRODUCTION_CONDITION_SIGNATURES:
        raise ValueError("Stage 3 production condition-signature support changed")
    condition_augmentation_sha256 = _sha256(
        data.get("condition_augmentation_policy_sha256"),
        name="Stage 3 condition augmentation policy",
    )
    if (
        condition_augmentation_sha256
        != PRODUCTION_CONDITION_AUGMENTATION_SHA256
    ):
        raise ValueError("Stage 3 production condition policy changed")

    stage2_contract = _mapping(raw["stage2_contract"], name="stage2_contract")
    _exact_keys(
        stage2_contract,
        {
            "require_complete_stage_manifest",
            "require_three_fold_checkpoints",
            "require_deployment_checkpoint",
            "require_complete_oof",
            "require_residual_scales",
            "require_float32_oof",
            "freeze_deployment_condition_encoder",
        },
        name="stage2_contract",
    )
    if any(value is not True for value in stage2_contract.values()):
        raise ValueError("every Stage 2 lineage/freeze requirement must remain enabled")

    model = _mapping(raw["model"], name="model")
    _exact_keys(
        model,
        {
            "condition_dimension",
            "target_widths",
            "activation_checkpoint",
            "sigma_data",
            "state_channels",
            "regression_condition_channels",
            "raw_positive_amount_condition",
            "target_grid_residual_state",
            "diffusion_specific_condition_adapters",
            "diffusion_specific_qkv",
            "physical_attention_levels",
            "physical_attention_heads",
            "physical_attention_dimension",
            "physical_attention_query_chunk_size",
            "cache_source_kv_per_lead_signature",
            "source_kv_depends_on_sigma",
            "source_gate_depends_on_sigma",
            "candidates",
            "selection_decision_file",
        },
        name="model",
    )
    model_target_widths = _integer_sequence(
        model.get("target_widths"), name="model target_widths"
    )
    if (
        model_target_widths != TARGET_WIDTHS
        or type(model.get("condition_dimension")) is not int
        or model.get("condition_dimension") != CONDITION_DIM
        or not _fixed_number(model.get("sigma_data"), 1.0)
        or type(model.get("state_channels")) is not int
        or model.get("state_channels") != 1
        or tuple(model.get("regression_condition_channels", ()))
        != ("mu_z", "probability_wet")
        or model.get("raw_positive_amount_condition") is not False
        or model.get("target_grid_residual_state") is not True
        or model.get("activation_checkpoint") is not True
        or model.get("diffusion_specific_condition_adapters") is not True
        or model.get("diffusion_specific_qkv") is not True
        or _integer_sequence(
            model.get("physical_attention_levels"),
            name="physical_attention_levels",
        )
        != (3, 4)
        or type(model.get("physical_attention_heads")) is not int
        or model.get("physical_attention_heads") != ATTENTION_HEADS
        or type(model.get("physical_attention_dimension")) is not int
        or model.get("physical_attention_dimension") != ATTENTION_DIM
        or model.get("cache_source_kv_per_lead_signature") is not True
        or model.get("source_kv_depends_on_sigma") is not False
        or model.get("source_gate_depends_on_sigma") is not True
    ):
        raise ValueError("Stage 3 model contract differs from the frozen architecture")
    candidates = model.get("candidates")
    expected_candidates = (
        {
            "name": "edm_a",
            "deployment_encoder_pyramid": False,
            "static_and_advection_conditions": True,
        },
        {
            "name": "edm_b",
            "deployment_encoder_pyramid": True,
            "static_and_advection_conditions": True,
        },
    )
    if (
        not isinstance(candidates, list)
        or len(candidates) != len(expected_candidates)
        or any(not isinstance(item, Mapping) for item in candidates)
    ):
        raise ValueError("Stage 3 must retain exactly the preregistered EDM-A/B arms")
    for index, (candidate, expected_candidate) in enumerate(
        zip(candidates, expected_candidates, strict=True)
    ):
        candidate_mapping = _mapping(candidate, name=f"model candidate {index}")
        _exact_keys(
            candidate_mapping,
            set(expected_candidate),
            name=f"model candidate {index}",
        )
        if dict(candidate_mapping) != expected_candidate:
            raise ValueError(
                "Stage 3 candidate architecture differs from the preregistered arms"
            )
    query_chunk_size = _positive_int(
        model.get("physical_attention_query_chunk_size"),
        name="physical_attention_query_chunk_size",
    )
    selection_decision_file_raw = model.get("selection_decision_file")
    if (
        not isinstance(selection_decision_file_raw, str)
        or not selection_decision_file_raw.strip()
        or not Path(selection_decision_file_raw).is_absolute()
    ):
        raise ValueError("selection_decision_file must be a non-empty absolute path")

    noise = _mapping(raw["noise"], name="noise")
    _exact_keys(
        noise,
        {
            "distribution",
            "log_sigma_mean",
            "log_sigma_standard_deviation",
            "minimum_sigma",
            "maximum_sigma",
            "purpose_id",
            "invalid_clean_residual_fill",
            "gaussian_noise_full_grid",
        },
        name="noise",
    )
    if (
        noise.get("distribution") != "lognormal"
        or noise.get("purpose_id") != TRAINING_NOISE_PURPOSE
        or not _fixed_number(noise.get("invalid_clean_residual_fill"), 0.0)
        or noise.get("gaussian_noise_full_grid") is not True
    ):
        raise ValueError("EDM training-noise contract mismatch")
    noise_config = Stage3NoiseConfig(
        log_sigma_mean=_finite_float(noise["log_sigma_mean"], name="log sigma mean"),
        log_sigma_standard_deviation=_positive_float(
            noise["log_sigma_standard_deviation"], name="log sigma standard deviation"
        ),
        minimum_sigma=_positive_float(noise["minimum_sigma"], name="minimum sigma"),
        maximum_sigma=_positive_float(noise["maximum_sigma"], name="maximum sigma"),
        purpose_id=str(noise["purpose_id"]),
    )
    if (
        not math.isfinite(noise_config.log_sigma_mean)
        or noise_config.minimum_sigma != 0.002
        or noise_config.maximum_sigma != 80.0
        or noise_config.minimum_sigma >= noise_config.maximum_sigma
    ):
        raise ValueError("EDM lognormal sigma bounds changed")

    loss = _mapping(raw["loss"], name="loss")
    if set(loss) != {
        "masked_edm",
        "sigma_data",
        "full_item_importance_weight",
        "target_validity_loss_only",
        "distributed_global_numerator_denominator",
    } or not _fixed_number(loss.get("sigma_data"), 1.0) or any(
        loss[key] is not True for key in loss if key != "sigma_data"
    ):
        raise ValueError("masked global EDM objective must remain fully enabled")

    optimization = _mapping(raw["optimization"], name="optimization")
    _exact_keys(
        optimization,
        {
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
    if optimization.get("optimizer") != "AdamW" or optimization.get("scheduler") != "cosine":
        raise ValueError("only the declared AdamW/cosine path is supported")
    betas_raw = optimization["betas"]
    if not isinstance(betas_raw, Sequence) or isinstance(betas_raw, (str, bytes)):
        raise TypeError("AdamW betas must be a two-element numeric sequence")
    betas = tuple(
        _positive_float(value, name=f"AdamW beta {index}")
        for index, value in enumerate(betas_raw)
    )
    if len(betas) != 2 or not all(value < 1.0 for value in betas):
        raise ValueError("AdamW betas must lie in (0,1)")
    optimization_config = Stage3OptimizationConfig(
        learning_rate=_positive_float(optimization["learning_rate"], name="learning rate"),
        weight_decay=_positive_float(
            optimization["weight_decay"], name="weight decay", allow_zero=True
        ),
        betas=betas,  # type: ignore[arg-type]
        gradient_clip_norm=_positive_float(
            optimization["gradient_clip_norm"], name="gradient clip norm"
        ),
        epochs=_positive_int(optimization["epochs"], name="epochs"),
        scheduler_minimum_ratio=_positive_float(
            optimization["scheduler_minimum_ratio"], name="scheduler minimum ratio"
        ),
        per_rank_microbatch_size=_positive_power_of_two(
            optimization["per_rank_microbatch_size"], name="per-rank microbatch size"
        ),
        gradient_accumulation_steps=_positive_int(
            optimization["gradient_accumulation_steps"], name="gradient accumulation steps"
        ),
        global_effective_batch_size=_positive_int(
            optimization["global_effective_batch_size"], name="global effective batch size"
        ),
        checkpoint_every_optimizer_steps=_positive_int(
            optimization["checkpoint_every_optimizer_steps"], name="checkpoint interval"
        ),
        log_every_optimizer_steps=_positive_int(
            optimization["log_every_optimizer_steps"], name="log interval"
        ),
    )
    if optimization_config.epochs != 1 or not (
        0.0 < optimization_config.scheduler_minimum_ratio <= 1.0
    ):
        raise ValueError("Stage 3 uses one immutable draw epoch and a cosine floor in (0,1]")

    loader = _mapping(raw["loader_tuning"], name="loader_tuning")
    _exact_keys(
        loader,
        {
            "selected_batch_size_per_rank",
            "selected_num_workers",
            "selected_prefetch_factor",
            "require_production_pipeline_benchmark",
        },
        name="loader_tuning",
    )
    selected_loader_batch = _positive_power_of_two(
        loader.get("selected_batch_size_per_rank"), name="selected loader batch"
    )
    if (
        selected_loader_batch != optimization_config.per_rank_microbatch_size
        or loader.get("require_production_pipeline_benchmark") is not True
    ):
        raise ValueError("Stage 3 loader selection is not production-bound")
    workers = _nonnegative_int(loader.get("selected_num_workers"), name="selected workers")
    prefetch = _positive_int(loader.get("selected_prefetch_factor"), name="selected prefetch")

    profiles = _mapping(raw["sampling_profiles"], name="sampling_profiles")
    _exact_keys(
        profiles,
        {
            "development_smoke",
            "selection_signature",
            "final_primary_signature",
            "operational_transition_signature",
            "solver",
            "sigma_schedule",
            "minimum_sigma",
            "maximum_sigma",
            "rho",
        },
        name="sampling_profiles",
    )
    expected_profile_mappings: Mapping[str, Mapping[str, object]] = {
        "development_smoke": {"members": 4, "edm_steps": 6},
        "selection_signature": {
            "members": SELECTION_MEMBERS,
            "edm_steps": SELECTION_STEPS,
        },
        "final_primary_signature": {"members": 32, "edm_steps": 12},
        "operational_transition_signature": {
            "members": 8,
            "edm_steps": 4,
            "distilled_only": True,
        },
    }
    for profile_name, expected_profile in expected_profile_mappings.items():
        profile = _mapping(profiles.get(profile_name), name=profile_name)
        _exact_keys(profile, set(expected_profile), name=profile_name)
        if (
            _positive_int(profile.get("members"), name=f"{profile_name} members")
            != expected_profile["members"]
            or _positive_int(
                profile.get("edm_steps"), name=f"{profile_name} edm_steps"
            )
            != expected_profile["edm_steps"]
            or (
                "distilled_only" in expected_profile
                and profile.get("distilled_only") is not True
            )
        ):
            raise ValueError(f"frozen sampling profile changed: {profile_name}")
    if (
        profiles.get("solver") != "heun"
        or profiles.get("sigma_schedule") != "karras_rho7"
        or not _fixed_number(profiles.get("minimum_sigma"), 0.002)
        or not _fixed_number(profiles.get("maximum_sigma"), 80.0)
        or not _fixed_number(profiles.get("rho"), 7.0)
    ):
        raise ValueError("frozen Stage 3 sampler contract mismatch")
    model_selection = _mapping(raw["model_selection"], name="model_selection")
    _exact_keys(
        model_selection,
        {
            "split",
            "labels_never_fit_calibration_parameters",
            "candidates",
            "sampler_bias_d_candidates",
            "profile",
            "decision_file_required_before_calibration",
        },
        name="model_selection",
    )
    if (
        model_selection.get("split") != "model_selection"
        or model_selection.get("labels_never_fit_calibration_parameters") is not True
        or tuple(model_selection.get("candidates", ())) != EDM_VARIANTS
        or tuple(model_selection.get("sampler_bias_d_candidates", ()))
        != ("disabled", "enabled")
        or model_selection.get("profile") != "selection_signature"
        or model_selection.get("decision_file_required_before_calibration") is not True
    ):
        raise ValueError("model-selection governance contract mismatch")
    calibration = _mapping(raw["calibration"], name="calibration")
    _exact_keys(
        calibration,
        {
            "split",
            "require_frozen_model_selection_decision",
            "order",
            "fold_mixture_weights",
            "variance",
            "d_default_when_disabled",
            "gamma_preserve_member_mean",
            "probability_family",
            "probability_clip",
            "p_thresholds_mm",
            "q_thresholds_mm",
            "pooling_order",
            "minimum_independent_blocks",
            "minimum_block_ess",
            "minimum_positive_support_blocks",
            "minimum_negative_support_blocks",
        },
        name="calibration",
    )
    if (
        calibration.get("split") != "calibration"
        or calibration.get("require_frozen_model_selection_decision") is not True
        or tuple(calibration.get("order", ()))
        != (
            "oof_scale_restore",
            "location_b",
            "total_scale_c",
            "optional_sampler_d",
            "spread_gamma",
            "probability_maps",
        )
        or calibration.get("fold_mixture_weights") != "oof_weighted_valid_mass"
        or calibration.get("variance") != "weighted_population"
        or not _fixed_number(calibration.get("d_default_when_disabled"), 0.0)
        or calibration.get("gamma_preserve_member_mean") is not True
        or calibration.get("probability_family") != MONOTONE_LOGIT_LINEAR_FAMILY
        or not _fixed_number(calibration.get("probability_clip"), 0.000001)
        or _number_sequence(
            calibration.get("p_thresholds_mm"), name="calibration p_thresholds_mm"
        )
        != (0.1,)
        or _number_sequence(
            calibration.get("q_thresholds_mm"), name="calibration q_thresholds_mm"
        )
        != (0.1, 1.0, 5.0)
        or tuple(calibration.get("pooling_order", ())) != POOLING_ORDER
        or type(calibration.get("minimum_independent_blocks")) is not int
        or calibration.get("minimum_independent_blocks") != 30
        or not _fixed_number(calibration.get("minimum_block_ess"), 20.0)
        or type(calibration.get("minimum_positive_support_blocks")) is not int
        or calibration.get("minimum_positive_support_blocks") != 20
        or type(calibration.get("minimum_negative_support_blocks")) is not int
        or calibration.get("minimum_negative_support_blocks") != 20
    ):
        raise ValueError("independent calibration family/pooling governance changed")

    inference = _mapping(raw["inference"], name="inference")
    _exact_keys(
        inference,
        {
            "output_grid",
            "inverse_transform_a0_mm",
            "censor_threshold_mm",
            "ensemble_mean_after_member_inverse",
            "ensemble_median",
            "output_each_lead_independently",
            "cross_lead_member_trajectory_claim",
        },
        name="inference",
    )
    if (
        _integer_sequence(inference.get("output_grid"), name="inference output_grid")
        != (PRODUCTION_INPUT_SIZE, PRODUCTION_INPUT_SIZE)
        or not _fixed_number(inference.get("inverse_transform_a0_mm"), 1.0)
        or not _fixed_number(inference.get("censor_threshold_mm"), 0.1)
        or inference.get("ensemble_mean_after_member_inverse") is not True
        or inference.get("ensemble_median") != "empirical_lower"
        or inference.get("output_each_lead_independently") is not True
        or inference.get("cross_lead_member_trajectory_claim") is not False
    ):
        raise ValueError("Stage 3 inference contract differs from the fixed implementation")

    tracking = _mapping(raw["tracking"], name="tracking")
    _exact_keys(
        tracking,
        {"backend", "project", "job_type", "mode", "log_gpu_system_metrics"},
        name="tracking",
    )
    if tracking.get("backend") != "wandb" or tracking.get("log_gpu_system_metrics") is not True:
        raise ValueError("rank-zero W&B system tracking must remain enabled")
    mode = str(tracking.get("mode"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("unsupported W&B mode")
    project = tracking.get("project")
    job_type = tracking.get("job_type")
    if (
        not isinstance(project, str)
        or not project.strip()
        or not isinstance(job_type, str)
        or not job_type.strip()
    ):
        raise ValueError("W&B project/job type must be non-empty strings")
    tracking_config = Stage3TrackingConfig(
        project=project,
        job_type=job_type,
        mode=mode,
    )

    publication = _mapping(raw["publication"], name="publication")
    _exact_keys(
        publication,
        {
            "require_stage2_hash_lineage",
            "require_model_selection_decision_hash",
            "require_edm_checkpoint",
            "require_independent_calibration",
            "require_sampling_profile_smoke",
            "atomic_stage_manifest",
            "immutable_release_requires_manual_promotion",
        },
        name="publication",
    )
    if any(value is not True for value in publication.values()):
        raise ValueError("every Stage 3 publication requirement must remain enabled")

    config_sha256 = _semantic_sha256(raw)
    frozen_raw = _mapping(
        _deep_freeze_config(raw), name="frozen Stage 3 config"
    )
    result = Stage3Config(
        raw=frozen_raw,
        sha256=config_sha256,
        protocol_version=PROTOCOL_VERSION,
        research_track=str(raw["research_track"]),
        seed=seed,
        target_widths=target_widths,
        context_widths=context_widths,
        input_size=PRODUCTION_INPUT_SIZE,
        activation_checkpoint=True,
        query_chunk_size=query_chunk_size,
        sigma_data=1.0,
        noise=noise_config,
        optimization=optimization_config,
        selected_num_workers=workers,
        selected_prefetch_factor=prefetch,
        pin_memory=True,
        persistent_workers=True,
        condition_augmentation_sha256=condition_augmentation_sha256,
        condition_signatures=condition_signatures,
        tracking=tracking_config,
        selection_decision_file=Path(selection_decision_file_raw).resolve(),
        probability_mapping_family=MONOTONE_LOGIT_LINEAR_FAMILY,
        pooling_order=POOLING_ORDER,
    )
    result.validate_topology(world_size=2)
    return result


@dataclass(frozen=True, slots=True)
class CounterKeyedNoise:
    """One batch of counter-keyed training sigma/noise on a requested device."""

    sigma: Tensor
    gaussian: Tensor

    def __post_init__(self) -> None:
        if self.sigma.dtype is not torch.float32 or self.sigma.ndim != 1:
            raise TypeError("counter-keyed sigma must be float32 [B]")
        if (
            self.gaussian.dtype is not torch.float32
            or self.gaussian.ndim != 4
            or self.gaussian.shape[:2] != (self.sigma.shape[0], 1)
            or self.gaussian.device != self.sigma.device
        ):
            raise TypeError("counter-keyed Gaussian noise must be float32 [B,1,H,W]")
        if not torch.isfinite(self.sigma).all() or bool((self.sigma <= 0.0).any().item()):
            raise ValueError("counter-keyed sigma must be finite/positive")
        if not torch.isfinite(self.gaussian).all():
            raise ValueError("counter-keyed Gaussian noise must be finite")


def _philox_generator(
    *,
    training_seed: int,
    purpose_id: str,
    global_example_index: int,
    sample_id: str,
    lead_hours: float,
    stream: Literal["sigma", "gaussian"],
) -> np.random.Generator:
    """Build a portable counter generator whose key has no topology/model ID."""

    if training_seed not in REPEAT_TRAINING_SEEDS:
        raise ValueError("training seed is outside the preregistered Stage 3 set")
    if purpose_id != TRAINING_NOISE_PURPOSE:
        raise ValueError("training-noise purpose ID changed")
    if isinstance(global_example_index, bool) or global_example_index < 0:
        raise ValueError("global_example_index must be non-negative")
    if not sample_id or stream not in {"sigma", "gaussian"}:
        raise ValueError("counter key has an empty/unsupported field")
    if not math.isfinite(float(lead_hours)) or not math.isclose(
        float(lead_hours) * 2.0,
        round(float(lead_hours) * 2.0),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("lead_hours must be a canonical half-hour lead")
    key = {
        "format": "kcorrdiff.edm-training-counter.v1",
        "training_seed": int(training_seed),
        "purpose_id": purpose_id,
        "global_example_index": int(global_example_index),
        "sample_id": sample_id,
        "lead_half_hours": int(round(float(lead_hours) * 2.0)),
        "stream": stream,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    entropy = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    return np.random.Generator(np.random.Philox(np.random.SeedSequence(entropy)))


def counter_keyed_training_noise(
    *,
    training_seed: int,
    noise_config: Stage3NoiseConfig,
    global_example_indices: Sequence[int],
    sample_ids: Sequence[str],
    lead_hours: Sequence[float],
    spatial_shape: tuple[int, int],
    device: torch.device | str,
) -> CounterKeyedNoise:
    """Generate per-draw lognormal sigma and full-grid Gaussian noise.

    Generation happens after collation and is keyed by immutable draw identity.
    Rank, worker, world size, batch position, EDM variant, and checkpoint ID are
    intentionally absent, so re-sharding/reordering cannot change a draw's
    stochastic training state and EDM-A/B receive common random numbers.
    """

    indices = tuple(global_example_indices)
    samples = tuple(sample_ids)
    leads = tuple(float(value) for value in lead_hours)
    if not indices or len(indices) != len(samples) or len(indices) != len(leads):
        raise ValueError("counter-keyed batch identities must be non-empty/equal length")
    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != 2
        or any(isinstance(value, bool) or value <= 0 for value in spatial_shape)
    ):
        raise ValueError("spatial_shape must contain two positive integers")
    sigmas: list[np.float32] = []
    fields: list[np.ndarray] = []
    for index, sample_id, lead in zip(indices, samples, leads, strict=True):
        sigma_generator = _philox_generator(
            training_seed=training_seed,
            purpose_id=noise_config.purpose_id,
            global_example_index=index,
            sample_id=sample_id,
            lead_hours=lead,
            stream="sigma",
        )
        normal = float(sigma_generator.standard_normal())
        log_sigma = noise_config.log_sigma_mean + (
            noise_config.log_sigma_standard_deviation * normal
        )
        sigma = min(
            max(math.exp(log_sigma), noise_config.minimum_sigma),
            noise_config.maximum_sigma,
        )
        sigmas.append(np.float32(sigma))
        noise_generator = _philox_generator(
            training_seed=training_seed,
            purpose_id=noise_config.purpose_id,
            global_example_index=index,
            sample_id=sample_id,
            lead_hours=lead,
            stream="gaussian",
        )
        fields.append(
            noise_generator.standard_normal(spatial_shape, dtype=np.float32)
        )
    selected = torch.device(device)
    return CounterKeyedNoise(
        sigma=torch.from_numpy(np.asarray(sigmas, dtype=np.float32)).to(selected),
        gaussian=torch.from_numpy(np.stack(fields, axis=0))[:, None].to(selected),
    )


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    variant: str
    training_seed: int
    sha256: str

    def __post_init__(self) -> None:
        if self.variant not in EDM_VARIANTS:
            raise ValueError("checkpoint variant is not EDM-A/B")
        if self.training_seed not in REPEAT_TRAINING_SEEDS:
            raise ValueError("checkpoint seed is not preregistered")
        _sha256(self.sha256, name="checkpoint SHA-256")


def _checkpoint_map(
    identities: Iterable[CheckpointIdentity],
) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for identity in identities:
        key = (identity.variant, identity.training_seed)
        if key in result:
            raise ValueError(f"duplicate checkpoint identity: {key}")
        result[key] = identity.sha256
    return result


def _verify_checkpoint_files(
    identities: Sequence[CheckpointIdentity], *, output_dir: Path
) -> None:
    """Bind external evidence to immutable checkpoint bytes in this run root."""

    root = Path(output_dir).resolve() / "checkpoints"
    root_stat = root.stat() if root.exists() else None
    for identity in identities:
        path = (
            root
            / f"{identity.variant}-seed-{identity.training_seed}"
            / "final.pt"
        ).resolve()
        if not path.is_file():
            raise ValueError(
                "external model-selection evidence references a missing/changed "
                f"checkpoint: {identity.variant}, seed={identity.training_seed}"
            )
        if root_stat is not None and path.stat().st_dev != root_stat.st_dev:
            raise ValueError("external checkpoint escaped the Stage 3 output filesystem")
        if sha256_file(path) != identity.sha256:
            raise ValueError(
                "external model-selection evidence references a missing/changed "
                f"checkpoint: {identity.variant}, seed={identity.training_seed}"
            )


@dataclass(frozen=True, slots=True)
class ScreeningEvaluationResult:
    path: Path
    file_sha256: str
    evaluation_result_sha256: str
    screening_training_manifest_sha256: str
    checkpoint_sha256s: tuple[CheckpointIdentity, ...]
    finalist_variants: tuple[str, ...]
    evaluator_id: str


@dataclass(frozen=True, slots=True)
class FinalModelSelectionDecision:
    path: Path
    file_sha256: str
    evaluation_result_sha256: str
    finalist_training_manifest_sha256: str
    checkpoint_sha256s: tuple[CheckpointIdentity, ...]
    selected_variant: str
    architecture_sha256: str
    d_enabled: bool
    probability_mapping_family: str
    pooling_order: tuple[str, ...]
    evaluator_id: str

    def frozen_calibration_decision(self) -> FrozenModelSelectionDecision:
        """Convert the verified external decision to calibration's typed gate."""

        return FrozenModelSelectionDecision(
            decision_sha256=self.file_sha256,
            architecture_sha256=self.architecture_sha256,
            d_enabled=self.d_enabled,
            probability_mapping_family=self.probability_mapping_family,
            pooling_order=self.pooling_order,
        )


class ModelSelectionEvaluator(Protocol):
    """External holdout evaluator boundary; no production implementation here."""

    def evaluate(
        self,
        checkpoints: Sequence[CheckpointIdentity],
        *,
        split: Literal["model_selection"],
        members: Literal[16],
        edm_steps: Literal[8],
        common_ensemble_seed: Literal[11104],
    ) -> Path:
        """Return a canonical external evidence artifact, never training metrics."""


def stage3_architecture_sha256(config: Stage3Config, *, variant: str) -> str:
    """Canonical architecture identity an external decision must select."""

    if variant not in EDM_VARIANTS:
        raise ValueError("architecture variant is not preregistered")
    return _semantic_sha256(
        {
            "protocol_version": config.protocol_version,
            "stage3_config_sha256": config.sha256,
            "variant": variant,
            "target_widths": list(config.target_widths),
            "context_widths": list(config.context_widths),
            "input_size": config.input_size,
            "sigma_data": config.sigma_data,
            "query_chunk_size": config.query_chunk_size,
            "activation_checkpoint": config.activation_checkpoint,
            "state_channels": 1,
            "regression_conditions": ["mu_z", "probability_wet"],
            "deployment_encoder_pyramid": variant == "edm_b",
            "precision": "float32",
        }
    )


def _read_external_artifact(path: Path, *, expected_format: str) -> Mapping[str, object]:
    selected = Path(path).resolve()
    payload = selected.read_bytes()
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model-selection artifact is not valid UTF-8 JSON") from error
    root = _mapping(decoded, name="model-selection artifact")
    if payload != _canonical_json_bytes(root):
        raise ValueError("model-selection artifact must use canonical JSON bytes")
    if root.get("format_version") != expected_format:
        raise ValueError("model-selection artifact format mismatch")
    declared = _sha256(root.get("artifact_sha256"), name="artifact semantic SHA-256")
    if declared != _artifact_semantic_sha256(root):
        raise ValueError("model-selection artifact semantic SHA-256 mismatch")
    return root


def _parse_checkpoint_records(value: object) -> tuple[CheckpointIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("model-selection checkpoint records must be a non-empty list")
    parsed: list[CheckpointIdentity] = []
    for index, item in enumerate(value):
        raw = _mapping(item, name=f"checkpoint record {index}")
        _exact_keys(raw, {"variant", "training_seed", "sha256"}, name="checkpoint record")
        parsed.append(
            CheckpointIdentity(
                variant=str(raw["variant"]),
                training_seed=int(raw["training_seed"]),
                sha256=str(raw["sha256"]),
            )
        )
    result = tuple(sorted(parsed, key=lambda item: (item.variant, item.training_seed)))
    _checkpoint_map(result)
    return result


def load_screening_evaluation_result(
    path: Path,
    *,
    expected_screening_manifest_sha256: str,
    expected_checkpoints: Sequence[CheckpointIdentity],
) -> ScreeningEvaluationResult:
    """Accept only external model-selection evidence bound to both screen arms."""

    root = _read_external_artifact(path, expected_format=SCREENING_EVALUATION_FORMAT)
    _exact_keys(
        root,
        {
            "format_version",
            "protocol_version",
            "split",
            "complete",
            "screening_training_manifest_sha256",
            "evaluation_result_sha256",
            "sampling_profile",
            "members",
            "edm_steps",
            "common_ensemble_seed",
            "labels_used_for_parameter_training",
            "calibration_parameters_fit_from_model_selection_labels",
            "checkpoint_sha256s",
            "finalist_variants",
            "evaluator_id",
            "artifact_sha256",
        },
        name="screening evaluation",
    )
    expected_manifest = _sha256(
        expected_screening_manifest_sha256, name="expected screening manifest"
    )
    if (
        root["protocol_version"] != PROTOCOL_VERSION
        or root["split"] != "model_selection"
        or root["complete"] is not True
        or root["screening_training_manifest_sha256"] != expected_manifest
        or root["sampling_profile"] != "selection_signature"
        or root["members"] != SELECTION_MEMBERS
        or root["edm_steps"] != SELECTION_STEPS
        or root["common_ensemble_seed"] != COMMON_ENSEMBLE_SEED
        or root["labels_used_for_parameter_training"] is not False
        or root["calibration_parameters_fit_from_model_selection_labels"] is not False
    ):
        raise ValueError("screening evidence violates the frozen holdout boundary")
    checkpoints = _parse_checkpoint_records(root["checkpoint_sha256s"])
    expected = _checkpoint_map(expected_checkpoints)
    if _checkpoint_map(checkpoints) != expected or set(expected) != {
        ("edm_a", SCREEN_TRAINING_SEED),
        ("edm_b", SCREEN_TRAINING_SEED),
    }:
        raise ValueError("screening evidence is not bound to both exact seed-11103 checkpoints")
    finalists = tuple(str(value) for value in root["finalist_variants"])  # type: ignore[union-attr]
    # With this preregistered A/B-only comparison, EDM-A is automatic and
    # EDM-B is the reference, so both require the three-seed stability run.
    if finalists != EDM_VARIANTS:
        raise ValueError("screening must retain automatic EDM-A and reference EDM-B finalists")
    evaluator_id = str(root["evaluator_id"])
    if not evaluator_id:
        raise ValueError("external screening evaluator_id is empty")
    return ScreeningEvaluationResult(
        path=Path(path).resolve(),
        file_sha256=sha256_file(Path(path)),
        evaluation_result_sha256=_sha256(
            root["evaluation_result_sha256"], name="screening evaluation result"
        ),
        screening_training_manifest_sha256=expected_manifest,
        checkpoint_sha256s=checkpoints,
        finalist_variants=finalists,
        evaluator_id=evaluator_id,
    )


def load_final_model_selection_decision(
    path: Path,
    *,
    expected_finalist_manifest_sha256: str,
    expected_checkpoints: Sequence[CheckpointIdentity],
    expected_architecture_sha256s: Mapping[str, str],
) -> FinalModelSelectionDecision:
    """Load a frozen 3-seed holdout decision without implementing selection."""

    root = _read_external_artifact(path, expected_format=FINAL_SELECTION_FORMAT)
    _exact_keys(
        root,
        {
            "format_version",
            "protocol_version",
            "split",
            "complete",
            "finalist_training_manifest_sha256",
            "evaluation_result_sha256",
            "sampling_profile",
            "members",
            "edm_steps",
            "common_ensemble_seed",
            "repeat_training_seeds",
            "deployment_training_seed",
            "labels_used_for_parameter_training",
            "calibration_parameters_fit_from_model_selection_labels",
            "checkpoint_sha256s",
            "selected_variant",
            "architecture_sha256",
            "d_enabled",
            "probability_mapping_family",
            "pooling_order",
            "mandatory_guardrails_passed",
            "absolute_gates_passed",
            "edm_b_dispersion_noninferiority_passed",
            "evaluator_id",
            "artifact_sha256",
        },
        name="final model-selection decision",
    )
    expected_manifest = _sha256(
        expected_finalist_manifest_sha256, name="expected finalist manifest"
    )
    if (
        root["protocol_version"] != PROTOCOL_VERSION
        or root["split"] != "model_selection"
        or root["complete"] is not True
        or root["finalist_training_manifest_sha256"] != expected_manifest
        or root["sampling_profile"] != "selection_signature"
        or root["members"] != SELECTION_MEMBERS
        or root["edm_steps"] != SELECTION_STEPS
        or root["common_ensemble_seed"] != COMMON_ENSEMBLE_SEED
        or tuple(root["repeat_training_seeds"]) != REPEAT_TRAINING_SEEDS  # type: ignore[arg-type]
        or root["deployment_training_seed"] != DEPLOYMENT_TRAINING_SEED
        or root["labels_used_for_parameter_training"] is not False
        or root["calibration_parameters_fit_from_model_selection_labels"] is not False
        or root["mandatory_guardrails_passed"] is not True
        or root["absolute_gates_passed"] is not True
    ):
        raise ValueError("final decision violates split, seed, sampler, or gate governance")
    checkpoints = _parse_checkpoint_records(root["checkpoint_sha256s"])
    expected = _checkpoint_map(expected_checkpoints)
    required = {(variant, seed) for variant in EDM_VARIANTS for seed in REPEAT_TRAINING_SEEDS}
    if set(expected) != required or _checkpoint_map(checkpoints) != expected:
        raise ValueError("final decision is not bound to all exact A/B three-seed checkpoints")
    selected_variant = str(root["selected_variant"])
    if selected_variant not in EDM_VARIANTS:
        raise ValueError("final decision selected an undeclared architecture")
    dispersion = root["edm_b_dispersion_noninferiority_passed"]
    if not isinstance(dispersion, bool) or (selected_variant == "edm_b" and not dispersion):
        raise ValueError("EDM-B cannot be selected without dispersion non-inferiority")
    if not isinstance(root["d_enabled"], bool):
        raise TypeError("d_enabled must be a frozen boolean")
    if (
        root["probability_mapping_family"] != MONOTONE_LOGIT_LINEAR_FAMILY
        or tuple(root["pooling_order"]) != POOLING_ORDER  # type: ignore[arg-type]
    ):
        raise ValueError("final decision changed calibration family/pooling order")
    if set(expected_architecture_sha256s) != set(EDM_VARIANTS):
        raise ValueError("expected architecture hash mapping must cover exact EDM-A/B")
    expected_architecture = _sha256(
        expected_architecture_sha256s[selected_variant],
        name="expected selected architecture",
    )
    if root["architecture_sha256"] != expected_architecture:
        raise ValueError("final decision selected architecture hash is not config-derived")
    evaluator_id = str(root["evaluator_id"])
    if not evaluator_id:
        raise ValueError("external final evaluator_id is empty")
    return FinalModelSelectionDecision(
        path=Path(path).resolve(),
        file_sha256=sha256_file(Path(path)),
        evaluation_result_sha256=_sha256(
            root["evaluation_result_sha256"], name="final evaluation result"
        ),
        finalist_training_manifest_sha256=expected_manifest,
        checkpoint_sha256s=checkpoints,
        selected_variant=selected_variant,
        architecture_sha256=expected_architecture,
        d_enabled=bool(root["d_enabled"]),
        probability_mapping_family=MONOTONE_LOGIT_LINEAR_FAMILY,
        pooling_order=POOLING_ORDER,
        evaluator_id=evaluator_id,
    )


@dataclass(frozen=True, slots=True)
class Stage3CheckpointProvenance:
    protocol_version: str
    config_sha256: str
    stage3_data_sha256: str
    stage2_deployment_checkpoint_sha256: str
    stage2_launch_identity_sha256: str
    stage2_source_tree_sha256: str
    stage2_container_image_sha256: str
    stage2_runtime_report_sha256: str
    stage2_data_contract_sha256: str
    stage3_source_tree_sha256: str
    container_image_sha256: str
    runtime_report_sha256: str
    draw_manifest_sha256: str
    plan_sha256: str
    variant: str
    training_seed: int
    world_size: int
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    wandb_mode: str
    wandb_run_id: str
    precision: str = "float32"
    tf32: bool = False
    cursor_semantics: str = CURSOR_SEMANTICS

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Stage 3 checkpoint protocol mismatch")
        for name in (
            "config_sha256",
            "stage3_data_sha256",
            "stage2_deployment_checkpoint_sha256",
            "stage2_launch_identity_sha256",
            "stage2_source_tree_sha256",
            "stage2_container_image_sha256",
            "stage2_runtime_report_sha256",
            "stage2_data_contract_sha256",
            "stage3_source_tree_sha256",
            "container_image_sha256",
            "runtime_report_sha256",
            "draw_manifest_sha256",
            "plan_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if self.variant not in EDM_VARIANTS or self.training_seed not in REPEAT_TRAINING_SEEDS:
            raise ValueError("Stage 3 checkpoint candidate/seed is not preregistered")
        if self.world_size != 2:
            raise ValueError("Stage 3 checkpoint topology must contain two ranks")
        _positive_power_of_two(
            self.per_rank_microbatch_size, name="checkpoint microbatch"
        )
        _positive_int(self.gradient_accumulation_steps, name="checkpoint accumulation")
        if self.precision != "float32" or self.tf32 is not False:
            raise ValueError("Stage 3 checkpoint precision fallback is forbidden")
        if self.cursor_semantics != CURSOR_SEMANTICS:
            raise ValueError("Stage 3 checkpoint cursor semantics changed")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("Stage 3 checkpoint has an unsupported W&B mode")
        if not self.wandb_run_id:
            raise ValueError("Stage 3 checkpoint W&B run ID cannot be empty")


def _validate_float32_state(model: nn.Module) -> None:
    invalid = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_floating_point() and value.dtype is not torch.float32
    ]
    if invalid:
        raise TypeError(f"Stage 3 checkpoint contains non-float32 state: {invalid[:8]}")


def save_stage3_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    cursor: TrainingCursor,
    provenance: Stage3CheckpointProvenance,
    rank_rng_states: Sequence[object],
    complete: bool,
    plan_optimizer_steps: int,
) -> str:
    """Atomically persist exact model/optimizer/scheduler and every rank RNG."""

    cursor.validate()
    provenance.validate()
    _validate_float32_state(model)
    if len(rank_rng_states) != provenance.world_size or any(
        not isinstance(value, Mapping) for value in rank_rng_states
    ):
        raise ValueError("checkpoint requires one RNG mapping per rank")
    if isinstance(complete, bool) is False:
        raise TypeError("checkpoint complete flag must be boolean")
    _positive_int(plan_optimizer_steps, name="plan optimizer steps")
    if complete and cursor.optimizer_step != plan_optimizer_steps:
        raise ValueError("a complete checkpoint must end at the exact plan boundary")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    state = {
        "format_version": STAGE3_CHECKPOINT_FORMAT,
        "provenance": asdict(provenance),
        "cursor": asdict(cursor),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),  # type: ignore[attr-defined]
        "rank_rng_states": list(rank_rng_states),
        "complete": complete,
        "plan_optimizer_steps": plan_optimizer_steps,
    }
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _torch_load_weights(path: Path) -> Mapping[str, object]:
    try:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported torch.
        state = torch.load(Path(path), map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError("Stage 3 checkpoint root must be a mapping")
    return state


def load_stage3_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    expected_provenance: Stage3CheckpointProvenance,
    restore_rank_rng: int | None,
) -> tuple[TrainingCursor, bool, int]:
    """Load only an exact lineage/topology checkpoint and optionally rank RNG."""

    expected_provenance.validate()
    state = _torch_load_weights(path)
    _exact_keys(
        state,
        {
            "format_version",
            "provenance",
            "cursor",
            "model",
            "optimizer",
            "scheduler",
            "rank_rng_states",
            "complete",
            "plan_optimizer_steps",
        },
        name="Stage 3 checkpoint",
    )
    if state["format_version"] != STAGE3_CHECKPOINT_FORMAT:
        raise ValueError("unsupported Stage 3 checkpoint format")
    actual = Stage3CheckpointProvenance(**state["provenance"])  # type: ignore[arg-type]
    actual.validate()
    if actual != expected_provenance:
        raise ValueError("Stage 3 checkpoint provenance mismatch")
    cursor = TrainingCursor(**state["cursor"])  # type: ignore[arg-type]
    cursor.validate()
    plan_steps = _positive_int(state["plan_optimizer_steps"], name="plan optimizer steps")
    complete = state["complete"]
    if not isinstance(complete, bool):
        raise TypeError("checkpoint complete flag must be boolean")
    if cursor.optimizer_step > plan_steps or (complete and cursor.optimizer_step != plan_steps):
        raise ValueError("checkpoint cursor is inconsistent with the plan")
    model.load_state_dict(state["model"], strict=True)  # type: ignore[arg-type]
    _validate_float32_state(model)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])  # type: ignore[attr-defined,arg-type]
    rank_states = state["rank_rng_states"]
    if not isinstance(rank_states, list) or len(rank_states) != expected_provenance.world_size:
        raise ValueError("checkpoint per-rank RNG coverage mismatch")
    if restore_rank_rng is not None:
        if not 0 <= restore_rank_rng < expected_provenance.world_size:
            raise ValueError("restore rank is outside checkpoint topology")
        selected = rank_states[restore_rank_rng]
        if not isinstance(selected, Mapping):
            raise TypeError("selected checkpoint RNG state is not a mapping")
        restore_rng_state(selected)
    return cursor, complete, plan_steps


def _connected_zero(model: nn.Module) -> Tensor:
    result: Tensor | None = None
    for parameter in model.parameters():
        term = parameter.sum() * 0.0
        result = term if result is None else result + term
    if result is None:
        raise ValueError("Stage 3 training model has no parameters")
    return result


def _sync_cuda(runtime: DistributedRuntime) -> None:
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


def _distributed_tracking_log(
    tracking: TrackingRun,
    runtime: DistributedRuntime,
    data: Mapping[str, object],
    *,
    step: int,
) -> None:
    failure: str | None = None
    try:
        # Equal steps are intentional: the loss/timing record and the
        # immediately following checkpoint identity belong to one optimizer
        # step.  The append-only tracker requires nondecreasing, not unique,
        # steps.  A crash after tracking but before checkpointing may replay
        # one equal step; stateless draw noise makes that replay exact.
        tracking.log(data, step=step)
    except BaseException as error:
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = tuple(
        value for value in runtime.all_gather_objects(failure) if value is not None
    )
    if failures:
        raise RuntimeError(f"distributed Stage 3 tracking failed: {failures}")


def _distributed_tracking_initialize(
    runtime: DistributedRuntime,
    factory: Callable[[], TrackingRun],
) -> TrackingRun:
    """Make a rank-zero W&B initialization failure visible to every rank."""

    tracking: TrackingRun | None = None
    failure: str | None = None
    try:
        tracking = factory()
    except BaseException as error:
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = tuple(
        value for value in runtime.all_gather_objects(failure) if value is not None
    )
    if failures:
        if tracking is not None:
            try:
                tracking.finish(exit_code=1)
            except BaseException:
                pass
        raise RuntimeError(f"distributed Stage 3 tracking init failed: {failures}")
    if tracking is None:  # pragma: no cover - guarded by the failure gather.
        raise AssertionError("tracking initialization returned no run")
    return tracking


def _distributed_tracking_finish(
    tracking: TrackingRun,
    runtime: DistributedRuntime,
    *,
    exit_code: int,
) -> None:
    """Finish rank-zero W&B without stranding the other NCCL rank."""

    failure: str | None = None
    try:
        tracking.finish(exit_code=exit_code)
    except BaseException as error:
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = tuple(
        value for value in runtime.all_gather_objects(failure) if value is not None
    )
    if failures:
        raise RuntimeError(f"distributed Stage 3 tracking finish failed: {failures}")


@dataclass(frozen=True, slots=True)
class EDMForwardResult:
    loss_contribution: Tensor
    weighted_square_error_sum: Tensor
    valid_importance_mass: Tensor
    samples: int
    h2d_seconds: float = 0.0
    forward_seconds: float = 0.0


class EDMBatchOperations(Protocol):
    def is_padding(self, batch: object) -> bool: ...

    def valid_importance_mass(
        self, batch: object, *, device: torch.device
    ) -> Tensor: ...

    def forward(
        self,
        *,
        model: nn.Module,
        deployment: nn.Module,
        batch: object,
        global_valid_importance_mass: Tensor,
        training_seed: int,
    ) -> EDMForwardResult: ...


class ProductionEDMBatchOperations:
    """Strict target/condition separation for the real Stage 3 objective."""

    def __init__(self, noise_config: Stage3NoiseConfig) -> None:
        self.noise_config = noise_config

    def is_padding(self, batch: object) -> bool:
        if not isinstance(batch, Stage3TrainingBatch):
            raise TypeError("production Stage 3 loader returned an incompatible batch")
        return batch.is_zero_loss_sentinel

    def valid_importance_mass(
        self, batch: object, *, device: torch.device
    ) -> Tensor:
        if not isinstance(batch, Stage3TrainingBatch):
            raise TypeError("production Stage 3 loader returned an incompatible batch")
        if batch.is_zero_loss_sentinel:
            return torch.zeros((), dtype=torch.float32, device=device)
        assert batch.diffusion_target is not None
        target = batch.diffusion_target
        # This reduction is label-only and never enters model conditions.
        local = (
            target.target_validity.to(torch.float32)
            * target.omega[:, None, None, None]
        ).sum(dtype=torch.float32)
        return local.to(device=device)

    def forward(
        self,
        *,
        model: nn.Module,
        deployment: nn.Module,
        batch: object,
        global_valid_importance_mass: Tensor,
        training_seed: int,
    ) -> EDMForwardResult:
        if not isinstance(model, ResidualEDM) or not isinstance(deployment, RegressionSystem):
            raise TypeError("production Stage 3 forward requires ResidualEDM/RegressionSystem")
        if not isinstance(batch, Stage3TrainingBatch) or batch.is_zero_loss_sentinel:
            raise TypeError("production Stage 3 forward requires a real Stage3TrainingBatch")
        _sync_device = next(model.parameters()).device
        _sync_start = time.perf_counter()
        selected = batch.to(_sync_device, non_blocking=True)
        if _sync_device.type == "cuda":
            torch.cuda.synchronize(_sync_device)
        h2d_seconds = time.perf_counter() - _sync_start
        assert selected.model_conditions is not None
        assert selected.diffusion_target is not None
        assert selected.provenance is not None
        conditions = selected.model_conditions
        target = selected.diffusion_target
        # The complete Stage 2 deployment model is frozen before this call.
        # no_grad prevents its in-sample feature pyramid from being adapted.
        forward_start = time.perf_counter()
        with torch.no_grad():
            deployment_output = deployment(conditions.regression)
        edm_conditions = conditions.residual_edm_conditions(deployment_output)
        keyed = counter_keyed_training_noise(
            training_seed=training_seed,
            noise_config=self.noise_config,
            global_example_indices=selected.provenance.global_example_indices,
            sample_ids=selected.provenance.sample_ids,
            lead_hours=selected.provenance.lead_hours,
            spatial_shape=tuple(target.clean_normalized_residual.shape[-2:]),
            device=_sync_device,
        )
        noisy = build_noisy_residual_training_state(
            clean_normalized_residual=target.clean_normalized_residual,
            target_validity=target.target_validity,
            sigma=keyed.sigma,
            noise=keyed.gaussian,
        )
        prediction = model(
            noisy.noisy_residual,
            keyed.sigma,
            edm_conditions,
        )
        terms = masked_edm_loss_terms(
            denoised_normalized_residual=prediction,
            clean_normalized_residual=target.clean_normalized_residual,
            target_validity=target.target_validity,
            omega=target.omega,
            sigma=keyed.sigma,
            sigma_data=1.0,
        )
        if _sync_device.type == "cuda":
            torch.cuda.synchronize(_sync_device)
        forward_seconds = time.perf_counter() - forward_start
        return EDMForwardResult(
            loss_contribution=(
                terms.weighted_square_error_sum
                / global_valid_importance_mass.clamp_min(1.0)
            ),
            weighted_square_error_sum=terms.weighted_square_error_sum.detach(),
            valid_importance_mass=terms.valid_importance_mass.detach(),
            samples=target.target_z.shape[0],
            h2d_seconds=h2d_seconds,
            forward_seconds=forward_seconds,
        )


@dataclass(frozen=True, slots=True)
class EDMOptimizerLoopResult:
    cursor: TrainingCursor
    stopped_by_limit: bool
    optimizer_steps_run: int


def run_edm_optimizer_loop(
    *,
    model: nn.Module,
    deployment: nn.Module,
    loader: Iterable[object],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    config: Stage3Config | object,
    runtime: DistributedRuntime,
    start_cursor: TrainingCursor,
    max_optimizer_steps: int | None,
    training_seed: int,
    tracking: TrackingRun,
    role_name: str,
    checkpoint_callback: Callable[[TrainingCursor, bool], None],
    batch_operations: EDMBatchOperations,
) -> EDMOptimizerLoopResult:
    """Run one accumulation-aligned plan with exact globally normalized EDM loss."""

    optimization = config.optimization  # type: ignore[attr-defined]
    accumulation = int(optimization.gradient_accumulation_steps)
    microbatch = int(optimization.per_rank_microbatch_size)
    if accumulation <= 0 or microbatch <= 0:
        raise ValueError("Stage 3 accumulation/microbatch must be positive")
    if microbatch & (microbatch - 1):
        raise ValueError("Stage 3 per-rank microbatch must be a positive power of two")
    if training_seed not in REPEAT_TRAINING_SEEDS:
        raise ValueError("Stage 3 role seed is not preregistered")
    if start_cursor.microbatches_since_step != 0:
        raise ValueError("Stage 3 resume must lie on an optimizer-step boundary")
    expected_slots = start_cursor.optimizer_step * microbatch * accumulation
    if start_cursor.global_example_index != expected_slots:
        raise ValueError("Stage 3 resume cursor rank-local slots disagree with optimizer steps")
    completed_batches = start_cursor.global_example_index // microbatch
    iterator_rng = capture_rng_state()
    try:
        iterator = iter(loader)
        for _ in range(completed_batches):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError("Stage 3 resume cursor exceeds the rank-local plan") from error
    finally:
        restore_rng_state(iterator_rng)

    optimizer_steps_run = 0
    optimizer_step = start_cursor.optimizer_step
    consumed_batches = completed_batches
    stopped = False
    pending_metrics: tuple[dict[str, object], int] | None = None
    while True:
        if max_optimizer_steps is not None and optimizer_steps_run >= max_optimizer_steps:
            stopped = True
            break
        window: list[object] = []
        wait_start = time.perf_counter()
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        data_wait = time.perf_counter() - wait_start
        if not window:
            break
        if len(window) != accumulation:
            raise RuntimeError("Stage 3 draw plan ended inside an accumulation window")
        local_mass = torch.zeros((), dtype=torch.float32, device=runtime.device)
        for value in window:
            local_mass = local_mass + batch_operations.valid_importance_mass(
                value, device=runtime.device
            )
        global_mass = runtime.reduce_sum(local_mass)
        if not torch.isfinite(global_mass) or float(global_mass.item()) <= 0.0:
            raise FloatingPointError("Stage 3 accumulation window has no global valid mass")

        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        optimizer.zero_grad(set_to_none=True)
        local_numerator = torch.zeros((), dtype=torch.float32, device=runtime.device)
        local_reported_mass = torch.zeros((), dtype=torch.float32, device=runtime.device)
        h2d_seconds = 0.0
        forward_seconds = 0.0
        backward_seconds = 0.0
        local_samples = 0
        for value in window:
            if batch_operations.is_padding(value):
                backward_start = time.perf_counter()
                _connected_zero(model).backward()
                _sync_cuda(runtime)
                backward_seconds += time.perf_counter() - backward_start
                consumed_batches += 1
                continue
            result = batch_operations.forward(
                model=model,
                deployment=deployment,
                batch=value,
                global_valid_importance_mass=global_mass,
                training_seed=training_seed,
            )
            loss = result.loss_contribution + _connected_zero(model)
            backward_start = time.perf_counter()
            loss.backward()
            _sync_cuda(runtime)
            backward_seconds += time.perf_counter() - backward_start
            local_numerator += result.weighted_square_error_sum.to(runtime.device)
            local_reported_mass += result.valid_importance_mass.to(runtime.device)
            local_samples += result.samples
            h2d_seconds += result.h2d_seconds
            forward_seconds += result.forward_seconds
            consumed_batches += 1

        gradient_reduce_start = time.perf_counter()
        reduce_gradients_sum(model, runtime)
        _sync_cuda(runtime)
        gradient_reduce_seconds = time.perf_counter() - gradient_reduce_start
        optimizer_start = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimization.gradient_clip_norm)
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite Stage 3 gradient norm")
        optimizer.step()
        scheduler.step()  # type: ignore[attr-defined]
        _sync_cuda(runtime)
        optimizer_seconds = time.perf_counter() - optimizer_start
        optimizer_step += 1
        optimizer_steps_run += 1
        cursor = TrainingCursor(
            epoch=0,
            global_example_index=consumed_batches * microbatch,
            optimizer_step=optimizer_step,
            microbatches_since_step=0,
        )
        if optimizer_step % int(optimization.checkpoint_every_optimizer_steps) == 0:
            checkpoint_callback(cursor, False)

        global_numerator = runtime.reduce_sum(local_numerator)
        reported_mass = runtime.reduce_sum(local_reported_mass)
        if not torch.allclose(reported_mass, global_mass, rtol=1.0e-5, atol=1.0):
            raise RuntimeError("Stage 3 forward/loss valid mass disagrees with pre-reduction")
        timings = runtime.reduce_max(
            torch.tensor(
                [
                    data_wait,
                    h2d_seconds,
                    forward_seconds,
                    backward_seconds,
                    gradient_reduce_seconds,
                    optimizer_seconds,
                ],
                dtype=torch.float64,
                device=runtime.device,
            )
        )
        samples = runtime.reduce_sum(
            torch.tensor(float(local_samples), dtype=torch.float64, device=runtime.device)
        )
        if runtime.device.type == "cuda":
            memory = runtime.reduce_max(
                torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(runtime.device),
                        torch.cuda.max_memory_reserved(runtime.device),
                    ],
                    dtype=torch.float64,
                    device=runtime.device,
                )
            )
        else:
            memory = torch.zeros(2, dtype=torch.float64, device=runtime.device)
        total_seconds = float(timings.sum().item())
        metrics: dict[str, object] = {
            f"{role_name}/optimizer_step": optimizer_step,
            f"{role_name}/loss_edm": float((global_numerator / global_mass).item()),
            f"{role_name}/global_valid_importance_mass": float(global_mass.item()),
            f"{role_name}/gradient_norm": float(gradient_norm.item()),
            f"{role_name}/learning_rate": float(optimizer.param_groups[0]["lr"]),
            f"{role_name}/data_wait_s": float(timings[0].item()),
            f"{role_name}/h2d_s": float(timings[1].item()),
            f"{role_name}/forward_s": float(timings[2].item()),
            f"{role_name}/backward_s": float(timings[3].item()),
            f"{role_name}/gradient_reduce_s": float(timings[4].item()),
            f"{role_name}/optimizer_s": float(timings[5].item()),
            f"{role_name}/samples_per_s": float(samples.item()) / max(total_seconds, 1.0e-12),
            f"{role_name}/gpu_peak_allocated_bytes": int(memory[0].item()),
            f"{role_name}/gpu_peak_reserved_bytes": int(memory[1].item()),
        }
        pending_metrics = (metrics, optimizer_step)
        if (
            optimizer_step % int(optimization.log_every_optimizer_steps) == 0
            or (max_optimizer_steps is not None and optimizer_steps_run >= max_optimizer_steps)
        ):
            _distributed_tracking_log(
                tracking, runtime, metrics, step=optimizer_step
            )
            pending_metrics = None
    if pending_metrics is not None:
        metrics, step = pending_metrics
        _distributed_tracking_log(tracking, runtime, metrics, step=step)
    return EDMOptimizerLoopResult(
        cursor=TrainingCursor(
            epoch=0,
            global_example_index=consumed_batches * microbatch,
            optimizer_step=optimizer_step,
            microbatches_since_step=0,
        ),
        stopped_by_limit=stopped,
        optimizer_steps_run=optimizer_steps_run,
    )


@dataclass(frozen=True, slots=True)
class Stage3StoragePreflight:
    checkpoint_bytes: int
    checkpoint_count: int
    atomic_temporary_bytes: int
    manifest_bytes: int
    reserve_bytes: int
    required_bytes: int
    free_bytes: int


def preflight_stage3_storage(
    *,
    output_dir: Path,
    checkpoint_parameter_count: int,
    checkpoint_count: int,
    reserve_bytes: int,
    manifest_bytes: int = 8 * 1024**2,
) -> Stage3StoragePreflight:
    """Conservatively budget all retained FP32 Adam states plus atomic scratch."""

    for name, value, minimum in (
        ("checkpoint_parameter_count", checkpoint_parameter_count, 1),
        ("checkpoint_count", checkpoint_count, 0),
        ("reserve_bytes", reserve_bytes, 0),
        ("manifest_bytes", manifest_bytes, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    selected = Path(output_dir).resolve()
    selected.mkdir(parents=True, exist_ok=True)
    # FP32 parameter + Adam parameter/first/second moments and serialization
    # overhead are bounded at 16 bytes per parameter.  One extra complete file
    # is required while atomically replacing role-local latest/final state.
    per_checkpoint = checkpoint_parameter_count * 16
    checkpoint_bytes = per_checkpoint * checkpoint_count
    atomic_temporary = per_checkpoint if checkpoint_count else 0
    required = checkpoint_bytes + atomic_temporary + manifest_bytes + reserve_bytes
    free = shutil.disk_usage(selected).free
    if required > free:
        raise OSError(
            "Stage 3 checkpoint/storage byte budget exceeds free space: "
            f"required={required}, free={free}"
        )
    return Stage3StoragePreflight(
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_count=checkpoint_count,
        atomic_temporary_bytes=atomic_temporary,
        manifest_bytes=manifest_bytes,
        reserve_bytes=reserve_bytes,
        required_bytes=required,
        free_bytes=free,
    )


def _load_stage2_deployment_weights(
    model: RegressionSystem,
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_config_sha256: str,
    expected_draw_manifest_sha256: str,
    expected_launch_identity_sha256: str,
    expected_source_tree_sha256: str,
    expected_container_image_sha256: str,
    expected_runtime_report_sha256: str,
    expected_data_contract_sha256: str,
    expected_global_step: int,
    per_rank_microbatch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    selected = Path(checkpoint_path).resolve()
    if sha256_file(selected) != _sha256(expected_sha256, name="deployment checkpoint"):
        raise ValueError("Stage 2 deployment checkpoint SHA-256 changed")
    expected = CheckpointProvenance(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=expected_config_sha256,
        draw_manifest_sha256=expected_draw_manifest_sha256,
        launch_identity_sha256=expected_launch_identity_sha256,
        source_tree_sha256=expected_source_tree_sha256,
        container_image_sha256=expected_container_image_sha256,
        runtime_report_sha256=expected_runtime_report_sha256,
        data_contract_sha256=expected_data_contract_sha256,
        role="deployment",
        fold_id=None,
        world_size=2,
        per_rank_microbatch_size=per_rank_microbatch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    cursor, extra = load_training_checkpoint(
        selected,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_provenance=expected,
        restore_rng=False,
    )
    if (
        extra.get("complete") is not True
        or cursor.optimizer_step != expected_global_step
        or cursor.microbatches_since_step != 0
    ):
        raise ValueError("Stage 2 deployment checkpoint is not an exact complete final")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _validate_float32_state(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Stage 2 deployment front-end did not freeze")


def _new_edm_model(
    *, variant: str, config: Stage3Config, device: torch.device
) -> ResidualEDM:
    model = ResidualEDM(
        ResidualEDMConfig(
            variant=variant,  # type: ignore[arg-type]
            target_widths=config.target_widths,
            context_widths=config.context_widths,
            input_size=config.input_size,
            sigma_data=config.sigma_data,
            query_chunk_size=config.query_chunk_size,
            activation_checkpoint=config.activation_checkpoint,
        )
    )
    expected = (
        EDM_A_PRODUCTION_PARAMETER_COUNT if variant == "edm_a" else EDM_B_PRODUCTION_PARAMETER_COUNT
    )
    if model.parameter_count != expected:
        raise RuntimeError(
            f"{variant} full-width parameter count mismatch: {model.parameter_count} != {expected}"
        )
    model.to(device=device, dtype=torch.float32)
    assert_float32_tree(model)
    return model


def _training_seed_for_initialization(variant: str, training_seed: int) -> int:
    # Variant is included in weight initialization (the architectures differ),
    # but is deliberately absent from sigma/noise counter keys.
    payload = f"{PROTOCOL_VERSION}\0stage3\0{variant}\0{training_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _stage3_dataloader(
    *,
    bundle: Stage3ArtifactBundle,
    plan: Stage3DistributedPlan,
    rank: int,
    factory: KCorrDiffDataFactory,
    config: Stage3Config,
) -> DataLoader[Stage3TrainingBatch]:
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=rank,
        target_loader_factory=Stage2FactoryTargetLoaderFactory(factory),
    )
    collator = Stage3BatchCollator(
        artifact_provenance=bundle.provenance,
        plan_sha256=plan.semantic_sha256,
        training_collator=TrainingBatchCollator(
            factory.artifacts.geometry,
            era_normalization=factory.artifacts.era_normalization,
            device="cpu",
        ),
    )
    options = dataloader_options(
        num_workers=config.selected_num_workers,
        prefetch_factor=config.selected_prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )
    return DataLoader(
        dataset,
        batch_size=config.optimization.per_rank_microbatch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        **options,
    )


def _shutdown_dataloader(loader: DataLoader[object]) -> None:
    """Eagerly release persistent workers before the next full-width arm."""

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None  # type: ignore[attr-defined]


def _validated_tracking_audit_sha256(
    path: Path,
    *,
    run_id: str,
    config_sha256: str | None = None,
    launch_identity_sha256: str | None = None,
) -> str:
    """Prove a durable W&B mirror has one identity and a clean finish record."""

    selected = Path(path).resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"durable Stage 3 tracking audit is missing: {selected}")
    last_step = -1
    last_event: object = None
    last_finish_exit_code: object = None
    saw_identity = False
    observed_identity: tuple[object, object] | None = None
    with selected.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"tracking audit has invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, Mapping) or record.get("run_id") != run_id:
                raise ValueError("tracking audit run identity mismatch")
            event = record.get("event")
            step = record.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < last_step:
                raise ValueError("tracking audit steps are not monotonic")
            if event in {"start", "resume"}:
                if step != last_step:
                    raise ValueError("tracking lifecycle step disagrees with metric history")
                identity = (
                    record.get("config_sha256"),
                    record.get("launch_identity_sha256"),
                )
                if observed_identity is None:
                    observed_identity = identity
                elif identity != observed_identity:
                    raise ValueError("tracking audit lifecycle identity changed")
                if config_sha256 is not None and identity[0] != config_sha256:
                    raise ValueError("tracking audit config SHA-256 mismatch")
                if (
                    launch_identity_sha256 is not None
                    and identity[1] != launch_identity_sha256
                ):
                    raise ValueError("tracking audit launch identity SHA-256 mismatch")
                saw_identity = True
            elif event == "finish":
                if step != last_step:
                    raise ValueError("tracking finish step disagrees with metric history")
                last_finish_exit_code = record.get("exit_code")
            elif event != "metrics":
                raise ValueError(f"unsupported tracking audit event: {event!r}")
            last_step = step
            last_event = event
    if not saw_identity or last_event != "finish":
        raise ValueError("complete Stage 3 tracking audit has no clean finish")
    if last_finish_exit_code != 0:
        raise ValueError("complete Stage 3 tracking audit has a failed exit code")
    return sha256_file(selected)


@dataclass(frozen=True, slots=True)
class CandidateTrainingResult:
    variant: str
    training_seed: int
    complete: bool
    checkpoint_path: Path
    checkpoint_sha256: str
    cursor: TrainingCursor
    optimizer_steps_run: int
    wandb_mode: str
    wandb_run_id: str
    tracking_audit_path: Path | None
    tracking_audit_sha256: str | None


def _candidate_tracking_config(
    *,
    config: Stage3Config,
    bundle: Stage3ArtifactBundle,
    plan: Stage3DistributedPlan,
    variant: str,
    training_seed: int,
    parameter_count: int,
    stage3_source_tree_sha256: str,
    container_image_sha256: str,
    runtime_report_sha256: str,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage3_config_sha256": config.sha256,
        "stage3_data_sha256": bundle.provenance.semantic_sha256,
        "stage2_deployment_checkpoint_sha256": (
            bundle.provenance.deployment_checkpoint_sha256
        ),
        "stage2_launch_identity_sha256": (
            bundle.provenance.stage2_launch_identity_sha256
        ),
        "stage2_source_tree_sha256": bundle.provenance.stage2_source_tree_sha256,
        "stage2_container_image_sha256": (
            bundle.provenance.stage2_container_image_sha256
        ),
        "stage2_runtime_report_sha256": (
            bundle.provenance.stage2_runtime_report_sha256
        ),
        "stage2_data_contract_sha256": (
            bundle.provenance.stage2_data_contract_sha256
        ),
        "stage3_source_tree_sha256": stage3_source_tree_sha256,
        "container_image_sha256": container_image_sha256,
        "runtime_report_sha256": runtime_report_sha256,
        "plan_sha256": plan.semantic_sha256,
        "variant": variant,
        "training_seed": training_seed,
        "training_noise_purpose": config.noise.purpose_id,
        "precision": "float32",
        "tf32": False,
        "world_size": 2,
        "activation_checkpoint": True,
        "parameter_count": parameter_count,
    }


def _candidate_tracking_identity_sha256(
    *,
    config: Stage3Config,
    bundle: Stage3ArtifactBundle,
    plan: Stage3DistributedPlan,
    variant: str,
    training_seed: int,
    stage3_source_tree_sha256: str,
    container_image_sha256: str,
    runtime_report_sha256: str,
) -> str:
    return _semantic_sha256(
        {
            "stage3_config_sha256": config.sha256,
            "stage3_data_sha256": bundle.provenance.semantic_sha256,
            "plan_sha256": plan.semantic_sha256,
            "variant": variant,
            "training_seed": training_seed,
            "stage3_source_tree_sha256": stage3_source_tree_sha256,
            "container_image_sha256": container_image_sha256,
            "runtime_report_sha256": runtime_report_sha256,
        }
    )


def train_edm_candidate(
    *,
    variant: str,
    training_seed: int,
    config: Stage3Config,
    bundle: Stage3ArtifactBundle,
    factory: KCorrDiffDataFactory,
    deployment: RegressionSystem,
    runtime: DistributedRuntime,
    output_dir: Path,
    max_optimizer_steps: int | None,
    wandb_mode: str,
    run_id: str,
    stage3_source_tree_sha256: str,
    container_image_sha256: str,
    runtime_report_sha256: str,
    verify_launch_identity: Callable[[], None],
) -> CandidateTrainingResult:
    """Train/resume one exact EDM candidate; complete means the whole draw plan."""

    if variant not in EDM_VARIANTS or training_seed not in REPEAT_TRAINING_SEEDS:
        raise ValueError("candidate/seed is not preregistered")
    plan = build_stage3_distributed_plan(
        bundle,
        world_size=runtime.world_size,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
    )
    loader = _stage3_dataloader(
        bundle=bundle,
        plan=plan,
        rank=runtime.rank,
        factory=factory,
        config=config,
    )
    identity = f"{variant}-seed-{training_seed}"
    role_dir = Path(output_dir).resolve() / "checkpoints" / identity
    role_dir.mkdir(parents=True, exist_ok=True)
    latest = role_dir / "latest.pt"
    final = role_dir / "final.pt"
    provenance = Stage3CheckpointProvenance(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=config.sha256,
        stage3_data_sha256=bundle.provenance.semantic_sha256,
        stage2_deployment_checkpoint_sha256=bundle.provenance.deployment_checkpoint_sha256,
        stage2_launch_identity_sha256=bundle.provenance.stage2_launch_identity_sha256,
        stage2_source_tree_sha256=bundle.provenance.stage2_source_tree_sha256,
        stage2_container_image_sha256=bundle.provenance.stage2_container_image_sha256,
        stage2_runtime_report_sha256=bundle.provenance.stage2_runtime_report_sha256,
        stage2_data_contract_sha256=bundle.provenance.stage2_data_contract_sha256,
        stage3_source_tree_sha256=stage3_source_tree_sha256,
        container_image_sha256=container_image_sha256,
        runtime_report_sha256=runtime_report_sha256,
        draw_manifest_sha256=bundle.provenance.draw_manifest_sha256,
        plan_sha256=plan.semantic_sha256,
        variant=variant,
        training_seed=training_seed,
        world_size=runtime.world_size,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        wandb_mode=wandb_mode,
        wandb_run_id=run_id,
    )
    torch.manual_seed(_training_seed_for_initialization(variant, training_seed))
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed(
            _training_seed_for_initialization(variant, training_seed)
        )
    model = _new_edm_model(variant=variant, config=config, device=runtime.device)
    runtime.broadcast_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=config.optimization.betas,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=plan.optimizer_steps,
        eta_min=config.optimization.learning_rate
        * config.optimization.scheduler_minimum_ratio,
    )
    tracking_audit_path = (
        Path(output_dir).resolve() / "wandb" / identity / "metrics.jsonl"
    )
    tracking_config = _candidate_tracking_config(
        config=config,
        bundle=bundle,
        plan=plan,
        variant=variant,
        training_seed=training_seed,
        parameter_count=model.parameter_count,
        stage3_source_tree_sha256=stage3_source_tree_sha256,
        container_image_sha256=container_image_sha256,
        runtime_report_sha256=runtime_report_sha256,
    )
    tracking_config_sha256 = _semantic_sha256(tracking_config)
    tracking_launch_identity_sha256 = _candidate_tracking_identity_sha256(
        config=config,
        bundle=bundle,
        plan=plan,
        variant=variant,
        training_seed=training_seed,
        stage3_source_tree_sha256=stage3_source_tree_sha256,
        container_image_sha256=container_image_sha256,
        runtime_report_sha256=runtime_report_sha256,
    )

    def initialize_candidate_tracking(
        *, resume: Literal["never", "allow"]
    ) -> TrackingRun:
        return _distributed_tracking_initialize(
            runtime,
            lambda: initialize_tracking(
                enabled=wandb_mode != "disabled",
                rank=runtime.rank,
                run_dir=tracking_audit_path.parent,
                run_id=run_id,
                project=config.tracking.project,
                job_type=config.tracking.job_type,
                config=tracking_config,
                config_sha256=tracking_config_sha256,
                launch_identity_sha256=tracking_launch_identity_sha256,
                mode=wandb_mode,
                resume=resume,
            ),
        )

    cursor = TrainingCursor(0, 0, 0, 0)
    if final.exists():
        cursor, complete, plan_steps = load_stage3_checkpoint(
            final,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_provenance=provenance,
            restore_rank_rng=None,
        )
        if not complete or plan_steps != plan.optimizer_steps:
            raise ValueError("final Stage 3 checkpoint is not complete for this plan")
        if runtime.is_rank_zero:
            latest.unlink(missing_ok=True)
        runtime.barrier()
        audit_path: Path | None = None
        audit_sha256: str | None = None
        if wandb_mode != "disabled":
            audit_path = tracking_audit_path
            if not audit_path.is_file():
                raise FileNotFoundError(
                    "complete Stage 3 checkpoint has no durable W&B audit"
                )
            try:
                audit_sha256 = _validated_tracking_audit_sha256(
                    audit_path,
                    run_id=run_id,
                    config_sha256=tracking_config_sha256,
                    launch_identity_sha256=tracking_launch_identity_sha256,
                )
            except ValueError:
                # The checkpoint is already an exact complete final, so never
                # train it again.  A process may have died after its atomic
                # save but before the terminal W&B record; resume that same
                # append-only run solely to bind/finalize the checkpoint.
                tracking = initialize_candidate_tracking(resume="allow")
                digest = sha256_file(final)
                try:
                    _distributed_tracking_log(
                        tracking,
                        runtime,
                        {
                            f"{identity}/checkpoint_sha256": digest,
                            f"{identity}/complete": True,
                            f"{identity}/plan_sha256": plan.semantic_sha256,
                            f"{identity}/tracking_recovered_after_final_save": True,
                        },
                        step=cursor.optimizer_step,
                    )
                    _distributed_tracking_finish(
                        tracking, runtime, exit_code=0
                    )
                except BaseException:
                    try:
                        tracking.finish(exit_code=1)
                    except BaseException:
                        pass
                    raise
                audit_sha256 = _validated_tracking_audit_sha256(
                    audit_path,
                    run_id=run_id,
                    config_sha256=tracking_config_sha256,
                    launch_identity_sha256=tracking_launch_identity_sha256,
                )
        return CandidateTrainingResult(
            variant,
            training_seed,
            True,
            final,
            sha256_file(final),
            cursor,
            0,
            wandb_mode,
            run_id,
            audit_path,
            audit_sha256,
        )
    if latest.exists():
        cursor, complete, plan_steps = load_stage3_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=provenance,
            restore_rank_rng=runtime.rank,
        )
        if complete or plan_steps != plan.optimizer_steps:
            raise ValueError("latest Stage 3 checkpoint has inconsistent completion/plan")
    expected_slots = cursor.optimizer_step * (
        config.optimization.per_rank_microbatch_size
        * config.optimization.gradient_accumulation_steps
    )
    if cursor.global_example_index != expected_slots:
        raise ValueError("Stage 3 checkpoint cursor is not accumulation aligned")

    tracking = initialize_candidate_tracking(
        resume="allow" if latest.exists() else "never"
    )

    def save(cursor_value: TrainingCursor, complete: bool) -> None:
        verify_launch_identity()
        states = runtime.all_gather_objects(capture_rng_state())
        destination = final if complete else latest
        if runtime.is_rank_zero:
            if complete and destination.exists():
                raise FileExistsError(destination)
            save_stage3_checkpoint(
                destination,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor_value,
                provenance=provenance,
                rank_rng_states=states,
                complete=complete,
                plan_optimizer_steps=plan.optimizer_steps,
            )
        runtime.barrier()

    try:
        model.train()
        loop = run_edm_optimizer_loop(
            model=model,
            deployment=deployment,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            runtime=runtime,
            start_cursor=cursor,
            max_optimizer_steps=max_optimizer_steps,
            training_seed=training_seed,
            tracking=tracking,
            role_name=identity,
            checkpoint_callback=save,
            batch_operations=ProductionEDMBatchOperations(config.noise),
        )
        complete = loop.cursor.optimizer_step == plan.optimizer_steps
        save(loop.cursor, complete)
        if complete and runtime.is_rank_zero:
            latest.unlink(missing_ok=True)
        runtime.barrier()
        checkpoint_path = final if complete else latest
        digest = sha256_file(checkpoint_path)
        if len(set(runtime.all_gather_objects(digest))) != 1:
            raise RuntimeError("ranks observed different Stage 3 checkpoint bytes")
        _distributed_tracking_log(
            tracking,
            runtime,
            {
                f"{identity}/checkpoint_sha256": digest,
                f"{identity}/complete": complete,
                f"{identity}/plan_sha256": plan.semantic_sha256,
                f"{identity}/partial_cannot_publish_complete": not complete,
            },
            step=loop.cursor.optimizer_step,
        )
        _distributed_tracking_finish(tracking, runtime, exit_code=0)
        audit_path = None if wandb_mode == "disabled" else tracking_audit_path
        audit_sha256 = (
            None
            if audit_path is None
            else _validated_tracking_audit_sha256(
                audit_path,
                run_id=run_id,
                config_sha256=tracking_config_sha256,
                launch_identity_sha256=tracking_launch_identity_sha256,
            )
        )
        return CandidateTrainingResult(
            variant=variant,
            training_seed=training_seed,
            complete=complete,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=digest,
            cursor=loop.cursor,
            optimizer_steps_run=loop.optimizer_steps_run,
            wandb_mode=wandb_mode,
            wandb_run_id=run_id,
            tracking_audit_path=audit_path,
            tracking_audit_sha256=audit_sha256,
        )
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    finally:
        _shutdown_dataloader(loader)  # type: ignore[arg-type]


def _checkpoint_json(result: CandidateTrainingResult) -> dict[str, object]:
    return {
        "variant": result.variant,
        "training_seed": result.training_seed,
        "complete": result.complete,
        "path": str(result.checkpoint_path.resolve()),
        "sha256": result.checkpoint_sha256,
        "cursor": asdict(result.cursor),
        "optimizer_steps_run_this_invocation": result.optimizer_steps_run,
        "wandb_mode": result.wandb_mode,
        "wandb_run_id": result.wandb_run_id,
        "tracking_audit_path": (
            str(result.tracking_audit_path.resolve())
            if result.tracking_audit_path is not None
            else None
        ),
        "tracking_audit_sha256": result.tracking_audit_sha256,
    }


def _candidate_results_from_manifest(
    value: object,
) -> tuple[CandidateTrainingResult, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("training manifest checkpoint results must be non-empty")
    results: list[CandidateTrainingResult] = []
    for index, item in enumerate(value):
        raw = _mapping(item, name=f"training result {index}")
        _exact_keys(
            raw,
            {
                "variant",
                "training_seed",
                "complete",
                "path",
                "sha256",
                "cursor",
                "optimizer_steps_run_this_invocation",
                "wandb_mode",
                "wandb_run_id",
                "tracking_audit_path",
                "tracking_audit_sha256",
            },
            name=f"training result {index}",
        )
        if raw["complete"] is not True:
            raise ValueError("immutable training manifest contains a partial checkpoint")
        path = Path(str(raw["path"])).resolve()
        digest = _sha256(raw["sha256"], name="manifest checkpoint SHA-256")
        if sha256_file(path) != digest:
            raise ValueError("manifest checkpoint file changed before governance continuation")
        cursor_raw = _mapping(raw["cursor"], name="checkpoint cursor")
        _exact_keys(
            cursor_raw,
            {"epoch", "global_example_index", "optimizer_step", "microbatches_since_step"},
            name="checkpoint cursor",
        )
        cursor = TrainingCursor(**cursor_raw)  # type: ignore[arg-type]
        cursor.validate()
        wandb_mode = str(raw["wandb_mode"])
        wandb_run_id = str(raw["wandb_run_id"])
        if wandb_mode != "online" or not wandb_run_id:
            raise ValueError("immutable training manifest must bind an online W&B run")
        audit_path_raw = raw["tracking_audit_path"]
        audit_hash_raw = raw["tracking_audit_sha256"]
        if not isinstance(audit_path_raw, str) or not audit_path_raw:
            raise ValueError("immutable training result has no durable tracking audit path")
        audit_path = Path(audit_path_raw).resolve()
        audit_hash = _sha256(audit_hash_raw, name="tracking audit SHA-256")
        if (
            _validated_tracking_audit_sha256(audit_path, run_id=wandb_run_id)
            != audit_hash
        ):
            raise ValueError("durable tracking audit changed before governance continuation")
        results.append(
            CandidateTrainingResult(
                variant=str(raw["variant"]),
                training_seed=int(raw["training_seed"]),
                complete=True,
                checkpoint_path=path,
                checkpoint_sha256=digest,
                cursor=cursor,
                optimizer_steps_run=_nonnegative_int(
                    raw["optimizer_steps_run_this_invocation"],
                    name="optimizer steps run",
                ),
                wandb_mode=wandb_mode,
                wandb_run_id=wandb_run_id,
                tracking_audit_path=audit_path,
                tracking_audit_sha256=audit_hash,
            )
        )
    identities = tuple(
        CheckpointIdentity(item.variant, item.training_seed, item.checkpoint_sha256)
        for item in results
    )
    _checkpoint_map(identities)
    return tuple(results)


def _manifest_envelope(
    *,
    format_version: str,
    config: Stage3Config,
    bundle: Stage3ArtifactBundle,
    plan: Stage3DistributedPlan,
    results: Sequence[CandidateTrainingResult],
    complete: bool,
    partial: bool,
    stopped_by_max_optimizer_steps: bool,
    phase: str,
    stage3_source_tree_sha256: str,
    container_image_sha256: str,
    runtime_report_sha256: str,
    external_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": format_version,
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "complete": complete,
        "partial": partial,
        "stopped_by_max_optimizer_steps": stopped_by_max_optimizer_steps,
        "config_sha256": config.sha256,
        "stage3_data_sha256": bundle.provenance.semantic_sha256,
        "stage2_manifest_sha256": bundle.provenance.stage2_manifest_sha256,
        "stage2_config_sha256": bundle.provenance.stage2_config_sha256,
        "stage2_deployment_checkpoint_sha256": bundle.provenance.deployment_checkpoint_sha256,
        "stage2_launch_identity_sha256": bundle.provenance.stage2_launch_identity_sha256,
        "stage2_source_tree_sha256": bundle.provenance.stage2_source_tree_sha256,
        "stage2_container_image_sha256": bundle.provenance.stage2_container_image_sha256,
        "stage2_runtime_report_sha256": bundle.provenance.stage2_runtime_report_sha256,
        "stage2_data_contract_sha256": bundle.provenance.stage2_data_contract_sha256,
        "stage3_source_tree_sha256": stage3_source_tree_sha256,
        "container_image_sha256": container_image_sha256,
        "runtime_report_sha256": runtime_report_sha256,
        "draw_manifest_sha256": bundle.provenance.draw_manifest_sha256,
        "oof_manifest_sha256": bundle.provenance.oof_manifest_sha256,
        "residual_scales_sha256": bundle.provenance.residual_scales_sha256,
        "fold_checkpoint_set_sha256": bundle.provenance.fold_checkpoint_set_sha256,
        "plan_sha256": plan.semantic_sha256,
        "topology": {
            "world_size": 2,
            "per_rank_microbatch_size": config.optimization.per_rank_microbatch_size,
            "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
            "precision": "float32",
            "tf32": False,
            "cursor_global_example_index_semantics": CURSOR_SEMANTICS,
        },
        "screen_training_seed": SCREEN_TRAINING_SEED,
        "common_ensemble_seed": COMMON_ENSEMBLE_SEED,
        "repeat_training_seeds": list(REPEAT_TRAINING_SEEDS),
        "deployment_training_seed": DEPLOYMENT_TRAINING_SEED,
        "selection_signature": {"members": SELECTION_MEMBERS, "edm_steps": SELECTION_STEPS},
        "training_noise_purpose": TRAINING_NOISE_PURPOSE,
        "results": [_checkpoint_json(value) for value in results],
        "external_evidence": dict(external_evidence or {}),
        "labels_used_for_parameter_training": False,
        "training_metrics_used_for_model_selection": False,
        "calibration_performed": False,
        "wandb_online_required_for_complete_training": True,
        "manual_release_required": True,
    }
    payload["artifact_sha256"] = _artifact_semantic_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class Stage3RunResult:
    phase: str
    complete: bool
    partial: bool
    manifest_path: Path
    manifest_sha256: str
    results: tuple[CandidateTrainingResult, ...]


def _retained_checkpoint_count(
    *, phase: str, output_dir: Path, max_optimizer_steps: int | None
) -> int:
    """Count all final/latest checkpoints that can coexist at this invocation."""

    selected = Path(output_dir).resolve()
    if phase == "screen":
        planned = 2
    elif phase == "finalists":
        # Screening A/B remain retained while the six three-seed finalist
        # checkpoints are produced.
        planned = 8
    elif phase == "bind-decision":
        # No model/optimizer checkpoint is written in this governance-only
        # phase; storage was already preflighted during training.
        planned = 0
    else:
        raise ValueError(f"unsupported Stage 3 phase: {phase}")
    existing = len(
        {
            path.resolve()
            for pattern in ("checkpoints/*/final.pt", "checkpoints/*/latest.pt")
            for path in selected.glob(pattern)
            if path.is_file()
        }
    )
    # A bounded run can cross role boundaries; it does not justify budgeting
    # only one checkpoint.  Existing files from earlier phases also remain.
    del max_optimizer_steps
    return max(planned, existing)


def run_stage3(
    *,
    phase: Literal["screen", "finalists", "bind-decision"],
    config: Stage3Config,
    bundle: Stage3ArtifactBundle,
    factory: KCorrDiffDataFactory,
    deployment: RegressionSystem,
    runtime: DistributedRuntime,
    output_dir: Path,
    max_optimizer_steps: int | None,
    wandb_mode: str,
    run_id: str,
    stage3_source_tree_sha256: str,
    container_image_sha256: str,
    runtime_report_sha256: str,
    source_root: Path,
    screening_evaluation: Path | None = None,
    final_decision: Path | None = None,
) -> Stage3RunResult:
    """Execute one governance phase without crossing an unevaluated boundary."""

    config.validate_topology(world_size=runtime.world_size)
    if max_optimizer_steps is not None and max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    output = Path(output_dir).resolve()
    verify_launch = lambda: verify_stage3_launch_identity(
        source_root=source_root,
        expected_source_tree_sha256=stage3_source_tree_sha256,
        expected_container_image_sha256=container_image_sha256,
        expected_runtime_report_sha256=runtime_report_sha256,
    )
    verify_launch()
    if (
        phase in {"screen", "finalists"}
        and max_optimizer_steps is None
        and wandb_mode != "online"
    ):
        raise ValueError(
            "complete Stage 3 training publication requires online W&B tracking"
        )
    output.mkdir(parents=True, exist_ok=True)
    plan = build_stage3_distributed_plan(
        bundle,
        world_size=runtime.world_size,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
    )
    results: list[CandidateTrainingResult] = []
    evidence: dict[str, object] = {}
    immutable_path: Path | None = None
    format_version = STAGE3_PARTIAL_MANIFEST_FORMAT
    stopped = False

    if phase == "screen":
        if screening_evaluation is not None or final_decision is not None:
            raise ValueError("screen training cannot consume later evaluation/decision artifacts")
        for variant in EDM_VARIANTS:
            remaining = (
                None
                if max_optimizer_steps is None
                else max_optimizer_steps
                - sum(item.optimizer_steps_run for item in results)
            )
            if remaining is not None and remaining <= 0:
                stopped = True
                break
            result = train_edm_candidate(
                variant=variant,
                training_seed=SCREEN_TRAINING_SEED,
                config=config,
                bundle=bundle,
                factory=factory,
                deployment=deployment,
                runtime=runtime,
                output_dir=output,
                max_optimizer_steps=remaining,
                wandb_mode=wandb_mode,
                run_id=f"{run_id}-{variant}-seed-{SCREEN_TRAINING_SEED}",
                stage3_source_tree_sha256=stage3_source_tree_sha256,
                container_image_sha256=container_image_sha256,
                runtime_report_sha256=runtime_report_sha256,
                verify_launch_identity=verify_launch,
            )
            results.append(result)
            if not result.complete:
                stopped = True
                break
        training_complete = len(results) == 2 and all(item.complete for item in results)
        complete = training_complete and max_optimizer_steps is None
        if max_optimizer_steps is not None:
            stopped = True
        if complete:
            format_version = STAGE3_SCREENING_MANIFEST_FORMAT
            immutable_path = output / "screening-training-manifest.json"

    elif phase == "finalists":
        if screening_evaluation is None or final_decision is not None:
            raise FileNotFoundError(
                "finalist training requires a separate external screening evaluation artifact"
            )
        screening_manifest_path = output / "screening-training-manifest.json"
        if not screening_manifest_path.is_file():
            raise FileNotFoundError("complete screening training manifest is missing")
        screening_root = _read_external_artifact(
            screening_manifest_path, expected_format=STAGE3_SCREENING_MANIFEST_FORMAT
        )
        if screening_root.get("complete") is not True or screening_root.get("partial") is not False:
            raise ValueError("screening training manifest is not complete")
        screen_results = _candidate_results_from_manifest(screening_root["results"])
        screen_records = tuple(
            CheckpointIdentity(item.variant, item.training_seed, item.checkpoint_sha256)
            for item in screen_results
        )
        screening = load_screening_evaluation_result(
            screening_evaluation,
            expected_screening_manifest_sha256=sha256_file(screening_manifest_path),
            expected_checkpoints=screen_records,
        )
        _verify_checkpoint_files(screen_records, output_dir=output)
        evidence = {
            "screening_evaluation_path": str(screening.path),
            "screening_evaluation_file_sha256": screening.file_sha256,
            "screening_evaluation_result_sha256": screening.evaluation_result_sha256,
            "screening_training_manifest_sha256": screening.screening_training_manifest_sha256,
            "external_evaluator_id": screening.evaluator_id,
        }
        required = tuple(
            (variant, seed)
            for variant in screening.finalist_variants
            for seed in REPEAT_TRAINING_SEEDS
        )
        for variant, seed in required:
            remaining = (
                None
                if max_optimizer_steps is None
                else max_optimizer_steps
                - sum(item.optimizer_steps_run for item in results)
            )
            if remaining is not None and remaining <= 0:
                stopped = True
                break
            result = train_edm_candidate(
                variant=variant,
                training_seed=seed,
                config=config,
                bundle=bundle,
                factory=factory,
                deployment=deployment,
                runtime=runtime,
                output_dir=output,
                max_optimizer_steps=remaining,
                wandb_mode=wandb_mode,
                run_id=f"{run_id}-{variant}-seed-{seed}",
                stage3_source_tree_sha256=stage3_source_tree_sha256,
                container_image_sha256=container_image_sha256,
                runtime_report_sha256=runtime_report_sha256,
                verify_launch_identity=verify_launch,
            )
            results.append(result)
            if not result.complete:
                stopped = True
                break
        training_complete = len(results) == len(required) and all(
            item.complete for item in results
        )
        complete = training_complete and max_optimizer_steps is None
        if max_optimizer_steps is not None:
            stopped = True
        if complete:
            format_version = STAGE3_FINALIST_MANIFEST_FORMAT
            immutable_path = output / "finalist-training-manifest.json"

    elif phase == "bind-decision":
        if (
            final_decision is None
            or screening_evaluation is not None
            or max_optimizer_steps is not None
        ):
            raise ValueError(
                "bind-decision requires only the external final decision and cannot be step-bounded"
            )
        finalist_manifest_path = output / "finalist-training-manifest.json"
        if not finalist_manifest_path.is_file():
            raise FileNotFoundError("complete finalist training manifest is missing")
        finalist_root = _read_external_artifact(
            finalist_manifest_path, expected_format=STAGE3_FINALIST_MANIFEST_FORMAT
        )
        if finalist_root.get("complete") is not True or finalist_root.get("partial") is not False:
            raise ValueError("finalist training manifest is not complete")
        finalist_results = _candidate_results_from_manifest(finalist_root["results"])
        results.extend(finalist_results)
        checkpoint_records = tuple(
            CheckpointIdentity(item.variant, item.training_seed, item.checkpoint_sha256)
            for item in finalist_results
        )
        decision = load_final_model_selection_decision(
            final_decision,
            expected_finalist_manifest_sha256=sha256_file(finalist_manifest_path),
            expected_checkpoints=checkpoint_records,
            expected_architecture_sha256s={
                variant: stage3_architecture_sha256(config, variant=variant)
                for variant in EDM_VARIANTS
            },
        )
        _verify_checkpoint_files(checkpoint_records, output_dir=output)
        deployment_hash = _checkpoint_map(decision.checkpoint_sha256s)[
            (decision.selected_variant, DEPLOYMENT_TRAINING_SEED)
        ]
        deployment_path = next(
            item.checkpoint_path
            for item in finalist_results
            if (
                item.variant == decision.selected_variant
                and item.training_seed == DEPLOYMENT_TRAINING_SEED
            )
        )
        frozen_calibration_decision = decision.frozen_calibration_decision()
        evidence = {
            "model_selection_decision_path": str(decision.path),
            "model_selection_decision_file_sha256": decision.file_sha256,
            "model_selection_decision_sha256": decision.file_sha256,
            "model_selection_evaluation_result_sha256": decision.evaluation_result_sha256,
            "finalist_training_manifest_sha256": decision.finalist_training_manifest_sha256,
            "selected_variant": decision.selected_variant,
            "selected_architecture_sha256": decision.architecture_sha256,
            "d_enabled": decision.d_enabled,
            "probability_mapping_family": decision.probability_mapping_family,
            "pooling_order": list(decision.pooling_order),
            "deployment_diffusion_checkpoint_sha256": deployment_hash,
            "deployment_diffusion_checkpoint_path": str(deployment_path.resolve()),
            "deployment_training_seed": DEPLOYMENT_TRAINING_SEED,
            "frozen_calibration_decision": frozen_calibration_decision.to_dict(),
            "external_evaluator_id": decision.evaluator_id,
            "calibration_required_next": True,
            "calibration_split": "calibration",
        }
        complete = True
        format_version = STAGE3_TRAINING_MANIFEST_FORMAT
        immutable_path = output / "stage3-training-manifest.json"
    else:  # pragma: no cover - protected by Literal/CLI.
        raise ValueError(f"unsupported Stage 3 phase: {phase}")

    partial = not complete
    verify_launch()
    payload = _manifest_envelope(
        format_version=format_version,
        config=config,
        bundle=bundle,
        plan=plan,
        results=results,
        complete=complete,
        partial=partial,
        stopped_by_max_optimizer_steps=stopped,
        phase=phase,
        stage3_source_tree_sha256=stage3_source_tree_sha256,
        container_image_sha256=container_image_sha256,
        runtime_report_sha256=runtime_report_sha256,
        external_evidence=evidence,
    )
    if runtime.is_rank_zero:
        partial_path = output / "stage3-training-partial.json"
        if complete:
            assert immutable_path is not None
            manifest_path = immutable_path
            digest = _atomic_json(manifest_path, payload, replace=False)
            partial_path.unlink(missing_ok=True)
        else:
            manifest_path = partial_path
            digest = _atomic_json(manifest_path, payload, replace=True)
    else:
        manifest_path = immutable_path or output / "stage3-training-partial.json"
        digest = ""
    runtime.barrier()
    observed = sha256_file(manifest_path)
    hashes = runtime.all_gather_objects(observed)
    if len(set(hashes)) != 1:
        raise RuntimeError("ranks observed different Stage 3 manifest bytes")
    return Stage3RunResult(
        phase=phase,
        complete=complete,
        partial=partial,
        manifest_path=manifest_path,
        manifest_sha256=observed,
        results=tuple(results),
    )


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _environment_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"invalid boolean environment value: {value!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("screen", "finalists", "bind-decision"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage2-release-index", type=Path, required=True)
    parser.add_argument("--stage2-release-index-sha256", required=True)
    parser.add_argument("--stage2-release-root", type=Path, required=True)
    parser.add_argument("--oof-remote-credentials", type=Path, required=True)
    parser.add_argument("--oof-remote-ca-certificate", type=Path, required=True)
    parser.add_argument("--oof-remote-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screening-evaluation", type=Path)
    parser.add_argument("--final-decision", type=Path)
    parser.add_argument("--require-world-size", type=int, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--target-widths", type=_csv_ints, required=True)
    parser.add_argument("--context-widths", type=_csv_ints, required=True)
    parser.add_argument("--era-latent-channels", type=int, required=True)
    parser.add_argument("--era-grid-size", type=int, required=True)
    parser.add_argument("--per-rank-microbatch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument(
        "--wandb-mode", choices=("config", "online", "offline", "disabled"), default="config"
    )
    parser.add_argument(
        "--run-id", default=os.environ.get("WANDB_RUN_ID") or os.environ.get("RUN_ID")
    )
    parser.add_argument("--storage-reserve-gib", type=float, default=2.0)
    parser.add_argument("--container-image-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--runtime-report-sha256", required=True)
    return parser.parse_args(argv)


def validate_launch_arguments(
    arguments: argparse.Namespace,
    config: Stage3Config,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Bind every production CLI/env choice to the hashed Stage 3 contract."""

    environment = os.environ if environ is None else environ
    config.validate_topology(world_size=arguments.require_world_size)
    if arguments.require_world_size != 2:
        raise ValueError("Stage 3 requires exactly two torchrun ranks")
    if arguments.precision != "float32" or not arguments.disable_tf32:
        raise ValueError("Stage 3 requires explicit float32 and --disable-tf32")
    if not arguments.fail_on_fallback:
        raise ValueError("--fail-on-fallback is required")
    expected = {
        "target_widths": config.target_widths,
        "context_widths": config.context_widths,
        "era_latent_channels": 128,
        "era_grid_size": 33,
        "per_rank_microbatch_size": config.optimization.per_rank_microbatch_size,
        "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
        "num_workers": config.selected_num_workers,
        "prefetch_factor": config.selected_prefetch_factor,
    }
    for name, value in expected.items():
        actual = getattr(arguments, name)
        if isinstance(value, tuple):
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"CLI {name} disagrees with hashed Stage 3 config")
    for name in _FALLBACK_ENVIRONMENT:
        raw = environment.get(name)
        if raw is not None and _environment_flag(raw):
            raise ValueError(f"fallback environment flag must be disabled: {name}")
    required = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    for name, expected_value in required.items():
        value = environment.get(name)
        if value is None or value.strip().lower() != expected_value:
            raise ValueError(f"strict environment contract mismatch: {name}")
    if arguments.max_optimizer_steps is not None and arguments.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if not arguments.run_id:
        raise ValueError("run ID is required")
    for name in (
        "container_image_sha256",
        "source_tree_sha256",
        "runtime_report_sha256",
    ):
        _sha256(getattr(arguments, name), name=f"CLI {name}")
    if arguments.runtime_report_sha256 == arguments.source_tree_sha256:
        raise ValueError("runtime report and source tree identities must be distinct")
    image_reference = environment.get("KCORRDIFF_CONTAINER_IMAGE")
    if image_reference is None:
        raise ValueError("KCORRDIFF_CONTAINER_IMAGE is required")
    validate_container_image_identity(
        image_reference, expected_sha256=arguments.container_image_sha256
    )
    if arguments.phase == "screen":
        if arguments.screening_evaluation is not None or arguments.final_decision is not None:
            raise ValueError("screen phase cannot consume evaluation/decision artifacts")
    elif arguments.phase == "finalists":
        if arguments.screening_evaluation is None or arguments.final_decision is not None:
            raise ValueError("finalists require only --screening-evaluation")
    elif arguments.phase == "bind-decision":
        if (
            arguments.final_decision is None
            or arguments.screening_evaluation is not None
            or arguments.max_optimizer_steps is not None
        ):
            raise ValueError("bind-decision requires only --final-decision and no step bound")
        if arguments.final_decision.resolve() != config.selection_decision_file:
            raise ValueError("--final-decision path disagrees with hashed Stage 3 config")


def _factory_inputs_from_arguments(
    arguments: argparse.Namespace, stage2_config: Stage2Config
) -> FactoryInputs:
    root = json.loads(arguments.data_contract.read_text(encoding="utf-8"))
    contract = _mapping(root, name="data contract")
    if (
        contract.get("protocol_version") != stage2_config.protocol_version
        or contract.get("research_track") != stage2_config.research_track
    ):
        raise ValueError("data contract protocol/research-track mismatch")
    geometry = _mapping(contract.get("geometry"), name="data-contract geometry")
    target_hash = geometry.get("target_coordinates_sha256")
    context_hash = geometry.get("condition_coordinates_sha256")
    static_hash = geometry.get("target_static_sha256")
    if not all(isinstance(value, str) for value in (target_hash, context_hash, static_hash)):
        raise ValueError("data contract geometry hashes are incomplete")
    if sha256_file(arguments.static_path) != static_hash:
        raise ValueError("target-static hash disagrees with data contract")
    return FactoryInputs(
        radar_cache_root=arguments.radar_cache_root,
        era5_cache_root=arguments.era5_cache_root,
        target_static=arguments.static_path,
        candidate_manifest=arguments.candidate_manifest,
        draw_manifest=arguments.draw_manifest,
        bundle_metadata=arguments.bundle_metadata,
        expected_target_coordinates_sha256=str(target_hash),
        expected_context_coordinates_sha256=str(context_hash),
        verify_cache_hashes=bool(arguments.verify_cache_hashes),
    )


def _bind_stage2_release_arguments(
    arguments: argparse.Namespace,
) -> Stage2ReleaseIndex:
    """Resolve every Stage 2 input from one externally pinned portable index."""

    release = load_stage2_release_index(
        arguments.stage2_release_index,
        expected_sha256=arguments.stage2_release_index_sha256,
        containment_root=arguments.stage2_release_root,
    )
    resolved = release.stage3_cli_inputs()
    for name, value in (
        ("stage2_config", resolved.stage2_config),
        ("stage2_manifest", resolved.stage2_manifest),
        ("stage2_manifest_sha256", resolved.stage2_manifest_sha256),
        ("oof_artifact", resolved.oof_artifact),
        ("residual_scales", resolved.residual_scales),
        ("radar_cache_root", resolved.radar_cache_root),
        ("era5_cache_root", resolved.era5_cache_root),
        ("draw_manifest", resolved.draw_manifest),
        ("candidate_manifest", resolved.candidate_manifest),
        ("bundle_metadata", resolved.bundle_metadata),
        ("data_contract", resolved.data_contract),
        ("static_path", resolved.static_path),
    ):
        setattr(arguments, name, value)
    return release


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    stage2_release = _bind_stage2_release_arguments(arguments)
    config = load_stage3_config(arguments.config)
    validate_launch_arguments(arguments, config)
    runtime = initialize_distributed_runtime(require_world_size=arguments.require_world_size)
    try:
        repository_root = Path(__file__).resolve().parents[2]
        actual_source_tree = source_tree_sha256(repository_root)
        if actual_source_tree != arguments.source_tree_sha256:
            raise ValueError("Stage 3 source tree changed after launch identity capture")
        actual_runtime = _runtime_report_sha256()
        if actual_runtime != arguments.runtime_report_sha256:
            raise ValueError("Stage 3 runtime report changed after launch identity capture")
        stage2_config = load_stage2_config(arguments.stage2_config)
        if (
            stage2_config.protocol_version != config.protocol_version
            or stage2_config.research_track != config.research_track
        ):
            raise ValueError("Stage 2/3 config lineage mismatch")
        if (
            stage2_config.data.condition_augmentation_sha256
            != config.condition_augmentation_sha256
            or stage2_config.data.condition_signatures
            != config.condition_signatures
        ):
            raise ValueError(
                "Stage 2/3 condition augmentation lineage mismatch"
            )
        stage3_data_raw = _mapping(config.raw["data"], name="Stage 3 data")
        for name, producer_path in (
            ("target_coordinates", stage2_config.data.target_coordinates),
            ("context_coordinates", stage2_config.data.context_coordinates),
            ("normalization", stage2_config.data.normalization),
        ):
            if Path(str(stage3_data_raw[name])).resolve() != producer_path.resolve():
                raise ValueError(
                    f"Stage 2/3 producer {name} paths disagree before relocation"
                )
        release_inputs = stage2_release.stage3_cli_inputs()
        stage2_config = replace(
            stage2_config,
            data=replace(
                stage2_config.data,
                target_coordinates=release_inputs.target_coordinates,
                context_coordinates=release_inputs.context_coordinates,
                normalization=release_inputs.normalization,
            ),
        )
        factory = build_data_factory(
            stage2_config, _factory_inputs_from_arguments(arguments, stage2_config)
        )
        remote_store = AuthenticatedHTTPSShardStore(
            endpoint=stage2_config.crossfit.remote_spill.endpoint,
            remote_prefix=stage2_config.crossfit.remote_spill.remote_prefix,
            credential_file=arguments.oof_remote_credentials,
            ca_certificate=arguments.oof_remote_ca_certificate,
            timeout_seconds=stage2_config.crossfit.remote_spill.timeout_seconds,
        )
        bundle = load_stage3_artifacts(
            Stage3ArtifactInputs(
                stage2_manifest=arguments.stage2_manifest,
                expected_stage2_manifest_sha256=arguments.stage2_manifest_sha256,
                draw_manifest=arguments.draw_manifest,
                oof_artifact=arguments.oof_artifact,
                residual_scales=arguments.residual_scales,
                checkpoint_path_overrides=tuple(
                    (record.role, record.fold_id, record.path)
                    for record in release_inputs.checkpoints
                ),
                imported_fold_set_manifest_override=(
                    None
                    if release_inputs.imported_fold_set is None
                    else release_inputs.imported_fold_set.manifest
                ),
                imported_fold_partial_manifest_overrides=(
                    ()
                    if release_inputs.imported_fold_set is None
                    else release_inputs.imported_fold_set.partials
                ),
                remote_store=remote_store,
                remote_cache_root=arguments.oof_remote_cache_root,
            )
        )
        if stage2_config.sha256 != bundle.provenance.stage2_config_sha256:
            raise ValueError("loaded Stage 2 config hash disagrees with complete Stage 2 manifest")
        # Stage 2/3 YAML paths remain semantic producer metadata (the shared
        # coordinate/normalization producer identities were checked above).
        # Runtime
        # deployment paths are deliberately relocated by the externally
        # pinned release index, whose role/content bindings were revalidated
        # above.  Requiring the producer's absolute paths here would defeat
        # the portable release boundary.
        contract = regression_system_config_from_stage2(stage2_config)
        deployment = RegressionSystem(contract.system).to(
            device=runtime.device, dtype=torch.float32
        )
        _load_stage2_deployment_weights(
            deployment,
            Path(bundle.deployment_checkpoint.path),
            expected_sha256=bundle.provenance.deployment_checkpoint_sha256,
            expected_config_sha256=bundle.provenance.stage2_config_sha256,
            expected_draw_manifest_sha256=bundle.provenance.draw_manifest_sha256,
            expected_launch_identity_sha256=(
                bundle.provenance.stage2_launch_identity_sha256
            ),
            expected_source_tree_sha256=(
                bundle.provenance.stage2_source_tree_sha256
            ),
            expected_container_image_sha256=(
                bundle.provenance.stage2_container_image_sha256
            ),
            expected_runtime_report_sha256=(
                bundle.provenance.stage2_runtime_report_sha256
            ),
            expected_data_contract_sha256=(
                bundle.provenance.stage2_data_contract_sha256
            ),
            expected_global_step=bundle.deployment_checkpoint.global_step,
            per_rank_microbatch_size=stage2_config.optimization.per_rank_microbatch_size,
            gradient_accumulation_steps=stage2_config.optimization.gradient_accumulation_steps,
        )
        checkpoint_count = _retained_checkpoint_count(
            phase=arguments.phase,
            output_dir=arguments.output_dir,
            max_optimizer_steps=arguments.max_optimizer_steps,
        )
        preflight_stage3_storage(
            output_dir=arguments.output_dir,
            checkpoint_parameter_count=max(
                EDM_A_PRODUCTION_PARAMETER_COUNT, EDM_B_PRODUCTION_PARAMETER_COUNT
            ),
            checkpoint_count=checkpoint_count,
            reserve_bytes=int(arguments.storage_reserve_gib * 1024**3),
        )
        wandb_mode = (
            config.tracking.mode
            if arguments.wandb_mode == "config"
            else arguments.wandb_mode
        )
        result = run_stage3(
            phase=arguments.phase,
            config=config,
            bundle=bundle,
            factory=factory,
            deployment=deployment,
            runtime=runtime,
            output_dir=arguments.output_dir,
            max_optimizer_steps=arguments.max_optimizer_steps,
            wandb_mode=wandb_mode,
            run_id=arguments.run_id,
            stage3_source_tree_sha256=arguments.source_tree_sha256,
            container_image_sha256=arguments.container_image_sha256,
            runtime_report_sha256=arguments.runtime_report_sha256,
            source_root=repository_root,
            screening_evaluation=arguments.screening_evaluation,
            final_decision=arguments.final_decision,
        )
        if runtime.is_rank_zero:
            print(
                json.dumps(
                    {
                        "phase": result.phase,
                        "complete": result.complete,
                        "partial": result.partial,
                        "manifest": str(result.manifest_path),
                        "manifest_sha256": result.manifest_sha256,
                    },
                    sort_keys=True,
                )
            )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateTrainingResult",
    "CheckpointIdentity",
    "COMMON_ENSEMBLE_SEED",
    "CounterKeyedNoise",
    "DEPLOYMENT_TRAINING_SEED",
    "EDMBatchOperations",
    "EDMForwardResult",
    "EDMOptimizerLoopResult",
    "FINAL_SELECTION_FORMAT",
    "FinalModelSelectionDecision",
    "ModelSelectionEvaluator",
    "ProductionEDMBatchOperations",
    "REPEAT_TRAINING_SEEDS",
    "SCREENING_EVALUATION_FORMAT",
    "SCREEN_TRAINING_SEED",
    "STAGE3_FINALIST_MANIFEST_FORMAT",
    "STAGE3_SCREENING_MANIFEST_FORMAT",
    "STAGE3_TRAINING_MANIFEST_FORMAT",
    "ScreeningEvaluationResult",
    "Stage3CheckpointProvenance",
    "Stage3Config",
    "Stage3NoiseConfig",
    "Stage3RunResult",
    "Stage3StoragePreflight",
    "counter_keyed_training_noise",
    "load_final_model_selection_decision",
    "load_screening_evaluation_result",
    "load_stage3_checkpoint",
    "load_stage3_config",
    "main",
    "preflight_stage3_storage",
    "run_edm_optimizer_loop",
    "run_stage3",
    "save_stage3_checkpoint",
    "stage3_architecture_sha256",
    "train_edm_candidate",
    "validate_launch_arguments",
    "validate_container_image_identity",
    "verify_stage3_launch_identity",
    "write_runtime_report",
]

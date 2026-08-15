"""Fail-closed runtime contract for full-width float32 K-CorrDiff training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

import torch

from kcorrdiff.models.common import (
    CONDITION_DIM as CONDITION_DIMENSION,
    CONTEXT_WIDTHS,
    ERA_GRID_SIZE,
    ERA_LATENT_CHANNELS,
    TARGET_WIDTHS,
    FullWidthRuntimeContract,
    assert_strict_fp32_runtime,
    configure_strict_fp32_runtime,
)


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    target_widths: tuple[int, ...] = TARGET_WIDTHS
    context_widths: tuple[int, ...] = CONTEXT_WIDTHS
    era_latent_channels: int = ERA_LATENT_CHANNELS
    era_grid_size: int = ERA_GRID_SIZE
    precision: str = "float32"
    tf32: bool = False
    allow_cpu_fallback: bool = False
    allow_model_width_fallback: bool = False
    allow_precision_fallback: bool = False
    allow_era_grid_fallback: bool = False

    def validate(self) -> None:
        FullWidthRuntimeContract(
            target_widths=self.target_widths,
            context_widths=self.context_widths,
            era_latent_channels=self.era_latent_channels,
            era_grid_size=self.era_grid_size,
            precision=self.precision,
            tf32_enabled=self.tf32,
            allow_cpu_fallback=self.allow_cpu_fallback,
            allow_model_width_fallback=self.allow_model_width_fallback,
            allow_precision_fallback=self.allow_precision_fallback,
            allow_era_grid_fallback=self.allow_era_grid_fallback,
        ).validate()


def contract_from_mapping(raw: Mapping[str, object]) -> RuntimeContract:
    """Parse a strict model/runtime mapping without accepting silent defaults."""

    required = {
        "target_widths",
        "context_widths",
        "era_latent_channels",
        "era_grid_size",
        "precision",
        "tf32",
        "allow_cpu_fallback",
        "allow_model_width_fallback",
        "allow_precision_fallback",
        "allow_era_grid_fallback",
    }
    missing = sorted(required - set(raw))
    extra = sorted(set(raw) - required)
    if missing or extra:
        raise ValueError(
            f"runtime contract schema mismatch: missing={missing}, extra={extra}"
        )
    result = RuntimeContract(
        target_widths=_integer_sequence(raw["target_widths"], "target_widths"),
        context_widths=_integer_sequence(raw["context_widths"], "context_widths"),
        era_latent_channels=_strict_int(
            raw["era_latent_channels"], "era_latent_channels"
        ),
        era_grid_size=_strict_int(raw["era_grid_size"], "era_grid_size"),
        precision=_strict_string(raw["precision"], "precision"),
        tf32=_strict_bool(raw["tf32"], "tf32"),
        allow_cpu_fallback=_strict_bool(
            raw["allow_cpu_fallback"], "allow_cpu_fallback"
        ),
        allow_model_width_fallback=_strict_bool(
            raw["allow_model_width_fallback"], "allow_model_width_fallback"
        ),
        allow_precision_fallback=_strict_bool(
            raw["allow_precision_fallback"], "allow_precision_fallback"
        ),
        allow_era_grid_fallback=_strict_bool(
            raw["allow_era_grid_fallback"], "allow_era_grid_fallback"
        ),
    )
    result.validate()
    return result


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("width declarations must be sequences")
    return value


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _integer_sequence(value: object, name: str) -> tuple[int, ...]:
    selected = _sequence(value)
    return tuple(_strict_int(item, f"{name}[{index}]") for index, item in enumerate(selected))


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def configure_strict_float32(
    *,
    require_cuda: bool,
    required_visible_gpus: int | None = None,
) -> torch.device:
    """Disable reduced-precision paths and validate the selected accelerator."""

    configure_strict_fp32_runtime()
    assert_strict_fp32_runtime()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") not in (None, "0"):
        raise RuntimeError("NVIDIA_TF32_OVERRIDE conflicts with the float32 contract")
    visible = torch.cuda.device_count()
    if required_visible_gpus is not None and visible != required_visible_gpus:
        raise RuntimeError(
            f"expected exactly {required_visible_gpus} visible GPU(s), got {visible}"
        )
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required and CPU fallback is forbidden")
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def assert_float32_tree(module: torch.nn.Module) -> None:
    """Reject checkpoints/modules that downcast any floating parameter or buffer."""

    offenders = [
        name
        for name, value in (*module.named_parameters(), *module.named_buffers())
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    if offenders:
        raise TypeError(f"non-float32 model state is forbidden: {offenders[:8]}")


__all__ = [
    "CONDITION_DIMENSION",
    "CONTEXT_WIDTHS",
    "ERA_GRID_SIZE",
    "ERA_LATENT_CHANNELS",
    "TARGET_WIDTHS",
    "RuntimeContract",
    "assert_float32_tree",
    "configure_strict_float32",
    "contract_from_mapping",
]

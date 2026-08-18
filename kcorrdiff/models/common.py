"""Shared building blocks for configurable five-level condition banks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


HISTORY_STEPS: Final[int] = 12
PRODUCTION_INPUT_SIZE: Final[int] = 256
TARGET_WIDTHS: Final[tuple[int, ...]] = (64, 128, 256, 384, 512)
CONTEXT_WIDTHS: Final[tuple[int, ...]] = (32, 64, 128, 256, 384)
ERA_GRID_SIZE: Final[int] = 33
ERA_LATENT_CHANNELS: Final[int] = 128
CONDITION_DIM: Final[int] = 512


def _autocast_enabled() -> bool:
    enabled = bool(torch.is_autocast_enabled())
    try:
        enabled = enabled or bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # pragma: no cover - compatibility with older torch.
        enabled = enabled or bool(torch.is_autocast_cpu_enabled())
    return enabled


def require_no_autocast() -> None:
    """Reject implicit mixed precision at every model boundary."""

    if _autocast_enabled():
        raise RuntimeError(
            "K-CorrDiff full-width encoders require autocast to be disabled"
        )


def require_float32_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int | None = None,
    finite: bool = True,
) -> None:
    """Validate the strict floating-point contract without converting data."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if finite and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a non-finite value")


def require_bool_tensor(name: str, value: Tensor, *, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must be bool, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")


def require_module_float32(name: str, module: nn.Module) -> None:
    """Reject an explicitly down-cast module before an opaque kernel error."""

    for parameter_name, parameter in module.named_parameters():
        if parameter.is_floating_point() and parameter.dtype is not torch.float32:
            raise TypeError(
                f"{name}.{parameter_name} must remain float32, got "
                f"{parameter.dtype}"
            )
    for buffer_name, buffer in module.named_buffers():
        if buffer.is_floating_point() and buffer.dtype is not torch.float32:
            raise TypeError(
                f"{name}.{buffer_name} must remain float32, got {buffer.dtype}"
            )


def validate_widths(
    name: str,
    widths: Sequence[int],
    *,
    production: Sequence[int],
    allow_test_override: bool,
) -> tuple[int, ...]:
    raw = tuple(widths)
    if any(isinstance(width, bool) or not isinstance(width, Integral) for width in raw):
        raise TypeError(f"{name} must contain integer widths")
    result = tuple(int(width) for width in raw)
    if len(result) != 5:
        raise ValueError(f"{name} must contain exactly five levels")
    if any(width <= 0 for width in result):
        raise ValueError(f"{name} must contain only positive widths")
    if any(right < left for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be non-decreasing")
    if result[0] % 2:
        raise ValueError(f"{name}[0] must be even for temporal/static stems")
    del production, allow_test_override
    return result


def validate_input_size(size: int, *, allow_test_override: bool) -> int:
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError("input_size must be an integer")
    result = int(size)
    if result <= 0 or result % 16:
        raise ValueError("input_size must be positive and divisible by 16")
    del allow_test_override
    return result


@dataclass(frozen=True, slots=True)
class FullWidthRuntimeContract:
    """Runtime values used by a model run; experiment knobs remain configurable."""

    precision: str = "float32"
    target_widths: tuple[int, ...] = TARGET_WIDTHS
    context_widths: tuple[int, ...] = CONTEXT_WIDTHS
    era_latent_channels: int = ERA_LATENT_CHANNELS
    era_grid_size: int = ERA_GRID_SIZE
    tf32_enabled: bool = False
    allow_cpu_fallback: bool = False
    allow_model_width_fallback: bool = False
    allow_precision_fallback: bool = False
    allow_era_grid_fallback: bool = False

    def validate(
        self,
        *,
        device: torch.device | str | None = None,
        require_cuda: bool = False,
    ) -> None:
        if self.precision != "float32":
            raise ValueError("precision fallback is forbidden; expected float32")
        validate_widths(
            "target_widths",
            self.target_widths,
            production=TARGET_WIDTHS,
            allow_test_override=True,
        )
        validate_widths(
            "context_widths",
            self.context_widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=True,
        )
        if self.era_latent_channels <= 0 or self.era_grid_size <= 0:
            raise ValueError("ERA dimensions must be positive")
        if self.tf32_enabled:
            raise ValueError("TF32 must be disabled for the strict FP32 arm")
        if require_cuda:
            if device is None:
                raise ValueError("device is required when require_cuda=True")
            if torch.device(device).type != "cuda":
                raise RuntimeError("CUDA is required; CPU fallback is forbidden")


def configure_strict_fp32_runtime() -> None:
    """Configure PyTorch's global matmul/convolution policy for strict FP32."""

    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def assert_strict_fp32_runtime() -> None:
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("float32 matmul precision must be 'highest'")
    if hasattr(torch.backends, "cuda") and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("CUDA matmul TF32 must be disabled")
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.allow_tf32:
        raise RuntimeError("cuDNN TF32 must be disabled")
    require_no_autocast()


def _normalization_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return groups
    return 1


class ResidualBlock2d(nn.Module):
    """Bias-free pre-activation residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _normalization_groups(in_channels), in_channels
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(
            _normalization_groups(out_channels), out_channels
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return residual + hidden


class ResidualStage2d(nn.Module):
    def __init__(self, channels: int, *, blocks: int = 2) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        self.blocks = nn.ModuleList(
            ResidualBlock2d(channels, channels) for _ in range(blocks)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def run_checkpointed(
    module: nn.Module,
    inputs: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    """Run a deterministic block with non-reentrant activation checkpointing."""

    if enabled and module.training and torch.is_grad_enabled():
        return checkpoint(
            module,
            inputs,
            use_reentrant=False,
            preserve_rng_state=True,
        )
    return module(inputs)


class TemporalSpatialStem(nn.Module):
    """Factorized spatial/temporal stem followed by temporal attention pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        history_steps: int = HISTORY_STEPS,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("stem channel counts must be positive")
        if history_steps != HISTORY_STEPS:
            raise ValueError("the v1.1.3b radar history requires exactly 12 steps")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.history_steps = history_steps
        self.activation_checkpoint = bool(activation_checkpoint)
        self.frame_projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.frame_residual = ResidualBlock2d(out_channels, out_channels)
        self.temporal_norm = nn.GroupNorm(
            _normalization_groups(out_channels), out_channels
        )
        self.temporal_depthwise = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=out_channels,
            bias=False,
        )
        self.temporal_pointwise = nn.Conv3d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.temporal_position = nn.Parameter(
            torch.empty(1, out_channels, history_steps, 1, 1)
        )
        self.temporal_score = nn.Conv3d(
            out_channels, 1, kernel_size=1, bias=True
        )
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)

    def forward(
        self,
        frames: Tensor,
        *,
        pooling_validity: Tensor | None = None,
    ) -> Tensor:
        require_no_autocast()
        require_float32_tensor("frames", frames, ndim=5)
        batch, steps, channels, height, width = frames.shape
        if steps != self.history_steps or channels != self.in_channels:
            raise ValueError(
                "frames must have shape "
                f"[B,{self.history_steps},{self.in_channels},H,W], got "
                f"{tuple(frames.shape)}"
            )
        flattened = frames.reshape(batch * steps, channels, height, width)
        hidden = self.frame_projection(flattened)
        hidden = run_checkpointed(
            self.frame_residual,
            hidden,
            enabled=self.activation_checkpoint,
        )
        hidden = hidden.reshape(
            batch, steps, self.out_channels, height, width
        ).permute(0, 2, 1, 3, 4)
        mixed = F.silu(self.temporal_norm(hidden))
        mixed = self.temporal_depthwise(mixed)
        mixed = self.temporal_pointwise(F.silu(mixed))
        hidden = hidden + mixed + self.temporal_position
        scores = self.temporal_score(F.silu(hidden))

        validity: Tensor | None = None
        if pooling_validity is not None:
            require_bool_tensor("pooling_validity", pooling_validity, ndim=5)
            expected = (batch, steps, 1, height, width)
            if tuple(pooling_validity.shape) != expected:
                raise ValueError(
                    f"pooling_validity must have shape {expected}, got "
                    f"{tuple(pooling_validity.shape)}"
                )
            validity = pooling_validity.permute(0, 2, 1, 3, 4)
            scores = scores.masked_fill(
                ~validity, torch.finfo(scores.dtype).min
            )

        weights = torch.softmax(scores, dim=2)
        if validity is not None:
            weights = weights * validity.to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-12)
        return torch.sum(hidden * weights, dim=2)


class StaticSpatialStem(nn.Module):
    """Encode static fields once, without replicating them over time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.residual = ResidualBlock2d(out_channels, out_channels)

    def forward(self, fields: Tensor) -> Tensor:
        require_no_autocast()
        require_float32_tensor("static fields", fields, ndim=4)
        if fields.shape[1] != self.in_channels:
            raise ValueError(
                f"static fields require {self.in_channels} channels, got "
                f"{fields.shape[1]}"
            )
        hidden = self.projection(fields)
        return run_checkpointed(
            self.residual,
            hidden,
            enabled=self.activation_checkpoint,
        )


class FiveLevelPyramidEncoder(nn.Module):
    """Five-level CNN pyramid with two residual blocks at every level."""

    def __init__(
        self,
        widths: Sequence[int],
        *,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.widths = tuple(int(value) for value in widths)
        if len(self.widths) != 5:
            raise ValueError("five pyramid widths are required")
        self.activation_checkpoint = bool(activation_checkpoint)
        self.stages = nn.ModuleList(
            ResidualStage2d(width, blocks=2) for width in self.widths
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(
                current,
                following,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            for current, following in zip(self.widths, self.widths[1:])
        )

    def forward(self, level_zero: Tensor) -> "FeaturePyramid":
        require_float32_tensor("level_zero", level_zero, ndim=4)
        if level_zero.shape[1] != self.widths[0]:
            raise ValueError("level_zero channel count does not match widths[0]")
        levels: list[Tensor] = []
        hidden = level_zero
        for index, stage in enumerate(self.stages):
            hidden = run_checkpointed(
                stage, hidden, enabled=self.activation_checkpoint
            )
            levels.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        return FeaturePyramid(*levels, widths=self.widths)


@dataclass(frozen=True, slots=True)
class FeaturePyramid:
    """Immutable handle to five cached feature tensors."""

    l0: Tensor
    l1: Tensor
    l2: Tensor
    l3: Tensor
    l4: Tensor
    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        levels = self.levels
        if len(self.widths) != 5:
            raise ValueError("feature widths must contain five entries")
        batch = levels[0].shape[0]
        device = levels[0].device
        dtype = levels[0].dtype
        expected_height, expected_width = levels[0].shape[-2:]
        for index, (level, channels) in enumerate(zip(levels, self.widths)):
            if level.ndim != 4:
                raise ValueError(f"L{index} must have shape [B,C,H,W]")
            if level.dtype is not torch.float32 or level.dtype != dtype:
                raise TypeError("all feature levels must be float32")
            if level.device != device or level.shape[0] != batch:
                raise ValueError("feature levels must share device and batch")
            if level.shape[1] != channels:
                raise ValueError(
                    f"L{index} expected {channels} channels, got {level.shape[1]}"
                )
            if tuple(level.shape[-2:]) != (expected_height, expected_width):
                raise ValueError(f"L{index} has the wrong spatial shape")
            expected_height //= 2
            expected_width //= 2

    @property
    def levels(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.l0, self.l1, self.l2, self.l3, self.l4

    def detached(self) -> "FeaturePyramid":
        return FeaturePyramid(
            *(level.detach() for level in self.levels), widths=self.widths
        )


@dataclass(frozen=True, slots=True)
class ValidityPyramid:
    """Area/fraction validity cached at the same five spatial levels."""

    l0: Tensor
    l1: Tensor
    l2: Tensor
    l3: Tensor
    l4: Tensor

    def __post_init__(self) -> None:
        levels = self.levels
        batch = levels[0].shape[0]
        expected = levels[0].shape[-2:]
        for index, level in enumerate(levels):
            require_float32_tensor(f"validity L{index}", level, ndim=4)
            if level.shape[:2] != (batch, 1):
                raise ValueError("validity levels must have shape [B,1,H,W]")
            if tuple(level.shape[-2:]) != tuple(expected):
                raise ValueError(f"validity L{index} has the wrong spatial shape")
            if bool(((level < 0.0) | (level > 1.0)).any().item()):
                raise ValueError("validity fractions must lie in [0,1]")
            expected = (expected[0] // 2, expected[1] // 2)

    @property
    def levels(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.l0, self.l1, self.l2, self.l3, self.l4

    def detached(self) -> "ValidityPyramid":
        return ValidityPyramid(*(level.detach() for level in self.levels))


def build_validity_pyramid(level_zero: Tensor) -> ValidityPyramid:
    require_float32_tensor("level-zero validity", level_zero, ndim=4)
    if level_zero.shape[1] != 1:
        raise ValueError("level-zero validity must have one channel")
    if bool(((level_zero < 0.0) | (level_zero > 1.0)).any().item()):
        raise ValueError("level-zero validity must lie in [0,1]")
    # Match the actual repeated k3/s2/p1 feature lattice.  Each level is
    # computed from L0 over the complete nominal receptive field, rather than
    # over disjoint 2x2 pooling blocks.  Padded positions are unsupported and
    # therefore remain in the denominator.
    levels = [level_zero]
    for downsampling_levels in range(1, 5):
        jump = 1 << downsampling_levels
        receptive_field = 2 * jump - 1
        levels.append(
            F.avg_pool2d(
                level_zero,
                kernel_size=receptive_field,
                stride=jump,
                padding=jump - 1,
                count_include_pad=True,
            )
        )
    return ValidityPyramid(*levels)


def sample_repeated_stride2_lattice(
    level_zero: Tensor,
    *,
    downsampling_levels: int,
    expected_shape: tuple[int, int],
) -> Tensor:
    """Sample L0 at the centres of repeated k3/s2/p1 CNN features.

    Neither ``interpolate(..., align_corners=True)`` nor its half-pixel
    counterpart lands on this lattice for a reduction such as 256 -> 32.
    Repeated k3/s2/p1 centres are exactly input indices ``0, 2**L, ...``.
    """

    require_float32_tensor("level-zero lattice field", level_zero, ndim=4)
    if isinstance(downsampling_levels, bool) or not isinstance(
        downsampling_levels, int
    ):
        raise TypeError("downsampling_levels must be an integer")
    if downsampling_levels < 0:
        raise ValueError("downsampling_levels must be non-negative")
    if (
        len(expected_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in expected_shape
        )
    ):
        raise ValueError("expected_shape must contain two positive integers")
    stride = 1 << downsampling_levels
    sampled = level_zero[..., ::stride, ::stride]
    if tuple(sampled.shape[-2:]) != tuple(expected_shape):
        raise ValueError(
            "repeated-stride2 lattice shape mismatch: "
            f"sampled {tuple(sampled.shape[-2:])}, expected {tuple(expected_shape)}"
        )
    return sampled


def count_parameters(modules: nn.Module | Iterable[nn.Module]) -> int:
    selected = (modules,) if isinstance(modules, nn.Module) else tuple(modules)
    return sum(
        parameter.numel()
        for module in selected
        for parameter in module.parameters()
    )


__all__ = [
    "CONDITION_DIM",
    "CONTEXT_WIDTHS",
    "ERA_GRID_SIZE",
    "ERA_LATENT_CHANNELS",
    "FeaturePyramid",
    "FiveLevelPyramidEncoder",
    "FullWidthRuntimeContract",
    "HISTORY_STEPS",
    "PRODUCTION_INPUT_SIZE",
    "ResidualBlock2d",
    "StaticSpatialStem",
    "TARGET_WIDTHS",
    "TemporalSpatialStem",
    "ValidityPyramid",
    "assert_strict_fp32_runtime",
    "build_validity_pyramid",
    "configure_strict_fp32_runtime",
    "count_parameters",
    "require_bool_tensor",
    "require_float32_tensor",
    "require_module_float32",
    "require_no_autocast",
    "sample_repeated_stride2_lattice",
    "validate_input_size",
    "validate_widths",
]

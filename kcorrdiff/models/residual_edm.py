"""Full-width target-grid residual EDM for K-CorrDiff v1.1.3b.

The denoiser has a deliberately narrow causal boundary.  Its stochastic state
is one normalized residual field on the uniform target grid and its only
regression-output conditions are ``mu_z`` and ``probability_wet``.  Future
target validity remains exclusively in :mod:`kcorrdiff.training.edm_loss`.

Two pre-registered condition arms are represented by separate model
instances:

``edm_a``
    Excludes the full-data deployment feature pyramid and context pyramid.  It
    retains the frozen static stem, causal advection, ERA, and ``[mu_z, p]``.
``edm_b``
    Adds detached full-data deployment target/context pyramids through
    diffusion-owned adapters and diffusion-owned physical attention.

Source K/V and physical bias are prepared without ``sigma`` and may therefore
be reused for every ensemble member and denoising step sharing one input,
lead, and condition signature.  Query projection and the zero-initialized
source gate remain noise-conditioned and are evaluated on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral
from typing import Final, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from kcorrdiff.data.advection import ADVECTION_MODEL_CHANNEL_NAMES
from kcorrdiff.training.edm_loss import SIGMA_DATA, edm_preconditioning

from .advection import AdvAdapter
from .common import (
    CONDITION_DIM,
    CONTEXT_WIDTHS,
    PRODUCTION_INPUT_SIZE,
    TARGET_WIDTHS,
    assert_strict_fp32_runtime,
    count_parameters,
    require_float32_tensor,
    require_module_float32,
    require_no_autocast,
    sample_repeated_stride2_lattice,
    validate_input_size,
    validate_widths,
)
from .condition_bank import ConditionBankCache
from .embeddings import DiffusionConditionFusionMLP, SigmaEmbedding
from .era_encoder import ERA_NATIVE_SHAPE, ERA_OUTPUT_CHANNELS, EraQueryResult
from .physical_attention import (
    PhysicalCrossAttention,
    PhysicalTokenGeometry,
)
from .regression import ConditionalResidualBlock, RegressionGeometry


EDMVariant = Literal["edm_a", "edm_b"]
ADVECTION_CHANNELS: Final[int] = len(ADVECTION_MODEL_CHANNEL_NAMES)

# Filled from the exact implementation below and guarded by tests.  These are
# checkpoint-interface constants, not tunable approximations.
EDM_A_PRODUCTION_PARAMETER_COUNT: Final[int] = 52_111_457
EDM_B_PRODUCTION_PARAMETER_COUNT: Final[int] = 52_605_025

_FUTURE_OR_RAW_AMOUNT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "future",
        "future_target",
        "label",
        "m_target",
        "m_target_tau",
        "m_tau",
        "raw_positive_amount",
        "target_validity",
        "target_wet",
        "target_z",
        "training_label",
        "wet_label",
        "z_tau",
    }
)


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return groups
    return 1


def _conditioned_checkpoint(
    module: nn.Module,
    value: Tensor,
    e_diff: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    if enabled and module.training and torch.is_grad_enabled():
        return checkpoint(
            module,
            value,
            e_diff,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return module(value, e_diff)


@dataclass(frozen=True, slots=True)
class ResidualEDMConfig:
    """Architecture and runtime contract for one residual-EDM candidate."""

    variant: EDMVariant = "edm_b"
    target_widths: tuple[int, ...] = TARGET_WIDTHS
    context_widths: tuple[int, ...] = CONTEXT_WIDTHS
    input_size: int = PRODUCTION_INPUT_SIZE
    sigma_data: float = SIGMA_DATA
    query_chunk_size: int = 128
    activation_checkpoint: bool = False
    allow_test_override: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"edm_a", "edm_b"}:
            raise ValueError("variant must be 'edm_a' or 'edm_b'")
        target = validate_widths(
            "residual EDM target widths",
            self.target_widths,
            production=TARGET_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        context = validate_widths(
            "residual EDM context widths",
            self.context_widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        size = validate_input_size(
            self.input_size, allow_test_override=self.allow_test_override
        )
        if not math.isfinite(float(self.sigma_data)) or float(self.sigma_data) != 1.0:
            raise ValueError(
                "v1.1.3b normalized-residual sigma_data must equal 1.0"
            )
        if isinstance(self.query_chunk_size, bool) or not isinstance(
            self.query_chunk_size, Integral
        ):
            raise TypeError("query_chunk_size must be an integer")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        object.__setattr__(self, "target_widths", target)
        object.__setattr__(self, "context_widths", context)
        object.__setattr__(self, "input_size", size)
        object.__setattr__(self, "sigma_data", float(self.sigma_data))
        object.__setattr__(self, "query_chunk_size", int(self.query_chunk_size))

    @property
    def uses_deployment_pyramid(self) -> bool:
        return self.variant == "edm_b"


@dataclass(frozen=True, slots=True)
class ResidualEDMConditions:
    """All inference-time conditions for one requested lead.

    The schema intentionally has no future target mask, label, target ``z``,
    or raw hurdle amount.  ``mu_z`` and ``probability_wet`` are the complete
    regression-output boundary.
    """

    condition_bank: ConditionBankCache
    era: EraQueryResult
    advection_features: Tensor
    mu_z: Tensor
    probability_wet: Tensor
    e_cond: Tensor
    geometry: RegressionGeometry
    sample_ids: tuple[str, ...]
    condition_signatures: tuple[str, ...]
    lead_indices: Tensor

    def __post_init__(self) -> None:
        names = {field.name.lower() for field in fields(self)}
        forbidden = sorted(names & _FUTURE_OR_RAW_AMOUNT_FIELDS)
        if forbidden:
            raise ValueError(
                "future labels/raw amount are forbidden in ResidualEDM conditions: "
                + ", ".join(forbidden)
            )
        if not isinstance(self.condition_bank, ConditionBankCache):
            raise TypeError("condition_bank must be a ConditionBankCache")
        if not isinstance(self.era, EraQueryResult):
            raise TypeError("era must be an EraQueryResult")
        if not isinstance(self.geometry, RegressionGeometry):
            raise TypeError("geometry must be a RegressionGeometry")
        batch = self.condition_bank.batch_size
        size = self.condition_bank.input_size
        for name, value in (
            ("advection_features", self.advection_features),
            ("mu_z", self.mu_z),
            ("probability_wet", self.probability_wet),
            ("e_cond", self.e_cond),
        ):
            require_float32_tensor(f"residual EDM {name}", value)
        if tuple(self.advection_features.shape) != (
            batch,
            ADVECTION_CHANNELS,
            size,
            size,
        ):
            raise ValueError(
                "advection_features must have shape "
                f"[B,{ADVECTION_CHANNELS},H_target,W_target]"
            )
        expected_field = (batch, 1, size, size)
        if tuple(self.mu_z.shape) != expected_field:
            raise ValueError(f"mu_z must have shape {expected_field}")
        if tuple(self.probability_wet.shape) != expected_field:
            raise ValueError(f"probability_wet must have shape {expected_field}")
        if tuple(self.e_cond.shape) != (batch, CONDITION_DIM):
            raise ValueError(f"e_cond must have shape [B,{CONDITION_DIM}]")
        if bool(
            ((self.probability_wet < 0.0) | (self.probability_wet > 1.0))
            .any()
            .item()
        ):
            raise ValueError("probability_wet must lie in [0,1]")
        if tuple(self.era.features.shape) != (
            batch,
            ERA_OUTPUT_CHANNELS,
            *ERA_NATIVE_SHAPE,
        ):
            raise ValueError("ERA query must retain [B,128,33,33]")
        require_float32_tensor("residual EDM ERA features", self.era.features, ndim=4)
        for name, value in (
            ("used_source_null", self.era.used_source_null),
            ("used_masked_null", self.era.used_masked_null),
        ):
            if value.shape != (batch,) or value.dtype is not torch.bool:
                raise TypeError(f"ERA {name} must be bool [B]")
            if value.device != self.era.features.device:
                raise ValueError(f"ERA {name} must share the feature device")
        if self.lead_indices.shape != (batch,):
            raise ValueError("lead_indices must have shape [B]")
        if self.lead_indices.dtype is not torch.int64:
            raise TypeError("lead_indices must use int64")
        if bool(((self.lead_indices < 0) | (self.lead_indices > 11)).any().item()):
            raise ValueError("lead_indices must lie in [0,11]")
        if len(self.sample_ids) != batch or any(
            not isinstance(value, str) or not value for value in self.sample_ids
        ):
            raise ValueError("one non-empty sample identity is required per item")
        if len(self.condition_signatures) != batch or any(
            not isinstance(value, str) or not value for value in self.condition_signatures
        ):
            raise ValueError("one non-empty condition signature is required per item")
        devices = (
            self.condition_bank.device,
            self.era.features.device,
            self.advection_features.device,
            self.mu_z.device,
            self.probability_wet.device,
            self.e_cond.device,
            self.lead_indices.device,
        )
        if any(device != devices[0] for device in devices):
            raise ValueError("all residual EDM conditions must share one device")


def _source_cache_tensors(
    conditions: ResidualEDMConditions,
) -> tuple[tuple[str, Tensor], ...]:
    """Return every tensor that determines cached source K/V or geometry."""

    bank = conditions.condition_bank
    selected: list[tuple[str, Tensor]] = [
        ("context_l3", bank.context.l3),
        ("context_l4", bank.context.l4),
        ("context_validity_l3", bank.context.validity.l3),
        ("context_validity_l4", bank.context.validity.l4),
        ("era_features", conditions.era.features),
        ("era_valid_token_mask", conditions.era.valid_token_mask),
        ("era_tp_token_mask", conditions.era.tp_token_mask),
        ("era_used_source_null", conditions.era.used_source_null),
        ("era_used_masked_null", conditions.era.used_masked_null),
    ]
    for geometry_name in (
        "target_l3",
        "target_l4",
        "context_l3",
        "context_l4",
        "era_native",
    ):
        geometry = getattr(conditions.geometry, geometry_name)
        selected.extend(
            (
                (f"geometry_{geometry_name}_x", geometry.x_shared),
                (f"geometry_{geometry_name}_y", geometry.y_shared),
                (
                    f"geometry_{geometry_name}_footprint_width",
                    geometry.footprint_width,
                ),
                (
                    f"geometry_{geometry_name}_footprint_height",
                    geometry.footprint_height,
                ),
            )
        )
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class _TensorCacheBinding:
    name: str
    tensor: Tensor
    version: int | None
    snapshot: Tensor | None

    @classmethod
    def capture(cls, name: str, tensor: Tensor) -> "_TensorCacheBinding":
        try:
            version: int | None = int(tensor._version)
        except RuntimeError:
            version = None
        # `_version` catches normal in-place writes cheaply, but PyTorch's
        # legacy `.data`/raw-storage paths can bypass it.  Cache correctness is
        # more important than that shortcut: keep an exact private snapshot of
        # every relatively small source/KV-dependent tensor and compare it on
        # reuse.  This also covers inference tensors, which have no counter.
        snapshot = tensor.detach().clone()
        return cls(name=name, tensor=tensor, version=version, snapshot=snapshot)

    def validate(self, name: str, tensor: Tensor) -> None:
        if name != self.name or tensor is not self.tensor:
            raise ValueError(f"source K/V cache {self.name} tensor identity mismatch")
        if self.version is not None:
            if int(tensor._version) != self.version:
                raise ValueError(f"source K/V cache {self.name} tensor was mutated")
        if self.snapshot is None or not torch.equal(self.snapshot, tensor.detach()):
            raise ValueError(f"source K/V cache {self.name} tensor was mutated")


@dataclass(frozen=True, slots=True)
class _AttentionSourceKV:
    owner_id: int
    query_shape: tuple[int, int]
    source_shape: tuple[int, int]
    key: Tensor | None
    value: Tensor | None
    valid: Tensor
    any_valid: Tensor
    physical_bias: Tensor | None
    source_connection: Tensor
    e_cond_connection: Tensor

    @property
    def is_exactly_absent(self) -> bool:
        return self.key is None


@dataclass(frozen=True, slots=True)
class ResidualEDMSourceKVCache:
    """Sigma-independent, per-input/lead/signature physical source cache."""

    owner_id: int
    variant: EDMVariant
    lead_indices: Tensor
    sample_ids: tuple[str, ...]
    condition_signatures: tuple[str, ...]
    e_cond_key: Tensor
    source_bindings: tuple[_TensorCacheBinding, ...]
    model_bindings: tuple[_TensorCacheBinding, ...]
    context_l3: _AttentionSourceKV
    era_l3: _AttentionSourceKV
    context_l4: _AttentionSourceKV
    era_l4: _AttentionSourceKV

    def __post_init__(self) -> None:
        # A schema-level assertion prevents a later refactor from quietly
        # making this reusable cache noise-level dependent.
        if any("sigma" in field.name.lower() for field in fields(self)):
            raise AssertionError("source K/V cache must not contain sigma")
        if self.lead_indices.dtype is not torch.int64 or self.lead_indices.ndim != 1:
            raise TypeError("cached lead_indices must be int64 [B]")
        if len(self.sample_ids) != self.lead_indices.shape[0]:
            raise ValueError("cached sample identity count must match the batch")
        if (
            self.e_cond_key.dtype is not torch.float32
            or self.e_cond_key.ndim != 2
            or self.e_cond_key.shape[1:] != (CONDITION_DIM,)
        ):
            raise TypeError("cached e_cond key must be float32 [B,512]")


class _DiffusionResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.activation_checkpoint = bool(activation_checkpoint)
        self.blocks = nn.ModuleList(
            (
                ConditionalResidualBlock(in_channels, out_channels),
                ConditionalResidualBlock(out_channels, out_channels),
            )
        )

    def forward(self, value: Tensor, e_diff: Tensor) -> Tensor:
        hidden = value
        for block in self.blocks:
            hidden = _conditioned_checkpoint(
                block,
                hidden,
                e_diff,
                enabled=self.activation_checkpoint,
            )
        return hidden


class _CachedDiffusionPhysicalAttention(nn.Module):
    """Physical attention with e_cond-only K/V and e_diff-only query/gate."""

    def __init__(
        self,
        *,
        target_channels: int,
        source_channels: int,
        source_name: str,
        query_chunk_size: int,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        # Every instance owns independent diffusion Q/K/V, output projection,
        # geometry bias, and gate parameters.  The existing physical-attention
        # implementation remains the single geometry/attention kernel.
        self.core = PhysicalCrossAttention(
            target_channels=target_channels,
            source_channels=source_channels,
            condition_dim=CONDITION_DIM,
            source_name=source_name,
            query_chunk_size=query_chunk_size,
            activation_checkpoint=activation_checkpoint,
        )

    @property
    def gate_projection(self) -> nn.Linear:
        return self.core.gate_projection

    @property
    def key_projection(self) -> nn.Linear:
        return self.core.key_projection

    @property
    def value_projection(self) -> nn.Linear:
        return self.core.value_projection

    @property
    def query_projection(self) -> nn.Linear:
        return self.core.query_projection

    @property
    def activation_checkpoint(self) -> bool:
        return self.core.activation_checkpoint

    def prepare_source(
        self,
        source: Tensor,
        *,
        query_geometry: PhysicalTokenGeometry,
        source_geometry: PhysicalTokenGeometry,
        query_shape: tuple[int, int],
        source_validity: Tensor,
        source_present: Tensor,
        e_cond: Tensor,
    ) -> _AttentionSourceKV:
        require_no_autocast()
        source_tokens = self.core._feature_tokens(
            source, channels=self.core.source_channels, name="diffusion source"
        )
        batch, _, source_height, source_width = source.shape
        source_shape = (source_height, source_width)
        if source_validity.shape != (batch, *source_shape):
            raise ValueError("source_validity must have shape [B,H_source,W_source]")
        if source_validity.dtype is not torch.bool or source_validity.device != source.device:
            raise TypeError("source_validity must be bool on the source device")
        if source_present.shape != (batch,):
            raise ValueError("source_present must have shape [B]")
        if source_present.dtype is not torch.bool or source_present.device != source.device:
            raise TypeError("source_present must be bool on the source device")
        if e_cond.shape != (batch, CONDITION_DIM):
            raise ValueError(f"e_cond must have shape [B,{CONDITION_DIM}]")
        if e_cond.dtype is not torch.float32 or e_cond.device != source.device:
            raise TypeError("e_cond must be float32 on the source device")
        if not bool(torch.isfinite(e_cond).all().item()):
            raise ValueError("e_cond must be finite")

        # Validate both geometries even when the source is absent, but avoid
        # constructing the Q*K dense physical bias in that exact-off case.
        self.core._geometry_tokens(
            query_geometry,
            batch=batch,
            shape=query_shape,
            device=source.device,
            name="query",
        )
        self.core._geometry_tokens(
            source_geometry,
            batch=batch,
            shape=source_shape,
            device=source.device,
            name="source",
        )
        valid = source_validity.flatten(1) & source_present[:, None]
        any_valid = valid.any(dim=1)
        if not bool(any_valid.any().item()):
            return _AttentionSourceKV(
                owner_id=id(self),
                query_shape=query_shape,
                source_shape=source_shape,
                key=None,
                value=None,
                valid=valid,
                any_valid=any_valid,
                physical_bias=None,
                source_connection=source,
                e_cond_connection=e_cond,
            )

        source_modulated = self.core.source_norm(source_tokens, e_cond)
        key = self.core.key_projection(source_modulated).view(
            batch, -1, self.core.heads, self.core.head_dim
        ).permute(0, 2, 1, 3)
        value = self.core.value_projection(source_modulated).view(
            batch, -1, self.core.heads, self.core.head_dim
        ).permute(0, 2, 1, 3)
        physical_bias = self.core.compute_physical_bias(
            query_geometry,
            source_geometry,
            batch_size=batch,
            query_shape=query_shape,
            source_shape=source_shape,
            device=source.device,
        )
        return _AttentionSourceKV(
            owner_id=id(self),
            query_shape=query_shape,
            source_shape=source_shape,
            key=key,
            value=value,
            valid=valid,
            any_valid=any_valid,
            physical_bias=physical_bias,
            source_connection=source,
            e_cond_connection=e_cond,
        )

    def _connected_zero(self, target: Tensor, e_diff: Tensor, cache: _AttentionSourceKV) -> Tensor:
        if not torch.is_grad_enabled():
            return torch.zeros_like(target)
        connected = (
            cache.source_connection.reshape(-1)[0] * 0.0
            + cache.e_cond_connection.reshape(-1)[0] * 0.0
            + e_diff.reshape(-1)[0] * 0.0
        )
        for parameter in self.parameters():
            if parameter.requires_grad:
                connected = connected + parameter.reshape(-1)[0] * 0.0
        return torch.zeros_like(target) + connected

    def forward(
        self,
        target: Tensor,
        e_diff: Tensor,
        cache: _AttentionSourceKV,
    ) -> tuple[Tensor, Tensor]:
        require_no_autocast()
        if cache.owner_id != id(self):
            raise ValueError("physical source cache belongs to another attention module")
        query_tokens = self.core._feature_tokens(
            target, channels=self.core.target_channels, name="diffusion query"
        )
        batch, _, height, width = target.shape
        if (height, width) != cache.query_shape:
            raise ValueError("cached source query shape does not match target")
        if e_diff.shape != (batch, CONDITION_DIM):
            raise ValueError(f"e_diff must have shape [B,{CONDITION_DIM}]")
        if e_diff.dtype is not torch.float32 or e_diff.device != target.device:
            raise TypeError("e_diff must be float32 on the query device")
        if cache.valid.device != target.device:
            raise ValueError("source cache and query must share one device")

        if cache.is_exactly_absent:
            residual = self._connected_zero(target, e_diff, cache)
            gate = self.core.gate_projection(e_diff)
            gated = residual * gate[:, :, None, None]
            return target + gated, gated
        if cache.key is None or cache.value is None or cache.physical_bias is None:
            raise AssertionError("present source cache is incomplete")

        query = self.core.query_projection(
            self.core.query_norm(query_tokens, e_diff)
        ).view(batch, -1, self.core.heads, self.core.head_dim)
        attended_chunks: list[Tensor] = []
        total_queries = query.shape[1]
        for start, end in self.core._chunks(total_queries, self.core.query_chunk_size):
            query_chunk = query[:, start:end].permute(0, 2, 1, 3)
            attended = self.core._run_attention_chunk(
                self.core._finish_attention_chunk,
                query_chunk,
                cache.key,
                cache.value,
                cache.valid,
                cache.any_valid,
                cache.physical_bias[:, :, start:end],
            )
            attended_chunks.append(attended.permute(0, 2, 1, 3))
        attended = torch.cat(attended_chunks, dim=1).reshape(
            batch, total_queries, self.core.attention_dim
        )
        residual = self.core.output_projection(attended).transpose(1, 2).reshape_as(
            target
        )
        gate = self.core.gate_projection(e_diff)
        # ``any_valid`` is per item, so a missing row is a bitwise identity even
        # when another row in the same batch has a live source.
        gated = (
            residual
            * gate[:, :, None, None]
            * cache.any_valid[:, None, None, None]
        )
        return target + gated, gated


class ResidualEDM(nn.Module):
    """EDM-preconditioned full-width denoiser on the 500 m target grid."""

    def __init__(self, config: ResidualEDMConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ResidualEDMConfig()
        if not isinstance(self.config, ResidualEDMConfig):
            raise TypeError("config must be a ResidualEDMConfig")
        widths = self.config.target_widths
        context_widths = self.config.context_widths
        checkpointing = self.config.activation_checkpoint

        self.sigma_embedding = SigmaEmbedding()
        self.diffusion_condition_fusion = DiffusionConditionFusionMLP()
        self.state_adapter = nn.Conv2d(1, widths[0], 3, padding=1, bias=False)
        self.c_noise_adapter = nn.Linear(1, widths[0])
        self.regression_condition_adapter = nn.Conv2d(
            2, widths[0], 3, padding=1, bias=False
        )
        self.static_adapter: nn.Conv2d | None
        self.deployment_adapters: nn.ModuleList | None
        if self.config.variant == "edm_a":
            self.static_adapter = nn.Conv2d(
                widths[0] // 2, widths[0], 1, bias=False
            )
            self.deployment_adapters = None
        else:
            self.static_adapter = None
            self.deployment_adapters = nn.ModuleList(
                nn.Conv2d(width, width, 1, bias=False) for width in widths
            )

        self.encoder_stages = nn.ModuleList(
            _DiffusionResidualStage(
                width,
                width,
                activation_checkpoint=checkpointing,
            )
            for width in widths
        )
        self.downsample_projections = nn.ModuleList(
            nn.Conv2d(current, following, 3, stride=2, padding=1, bias=False)
            for current, following in zip(widths, widths[1:])
        )
        self.advection_adapters = nn.ModuleList(
            AdvAdapter(
                target_channels=width,
                advection_channels=ADVECTION_CHANNELS,
                condition_dim=CONDITION_DIM,
            )
            for width in widths
        )
        self.upsample_projections = nn.ModuleList(
            nn.Conv2d(high, low, 3, padding=1, bias=False)
            for high, low in zip(widths[:0:-1], widths[-2::-1])
        )
        self.decoder_stages = nn.ModuleList(
            _DiffusionResidualStage(
                2 * width,
                width,
                activation_checkpoint=checkpointing,
            )
            for width in widths[-2::-1]
        )
        self.output_norm = nn.GroupNorm(_groups(widths[0]), widths[0])
        self.output_head = nn.Conv2d(widths[0], 1, 3, padding=1)

        def attention(target: int, source: int, name: str) -> _CachedDiffusionPhysicalAttention:
            return _CachedDiffusionPhysicalAttention(
                target_channels=target,
                source_channels=source,
                source_name=name,
                query_chunk_size=self.config.query_chunk_size,
                activation_checkpoint=checkpointing,
            )

        # Keep four explicitly named, independent attention modules.  EDM-A's
        # context branches are exact-off, but retaining the same named boundary
        # makes candidate diagnostics and checkpoint audits unambiguous.
        self.context_attention_l3 = attention(
            widths[3], context_widths[3], "diffusion_context_l3"
        )
        self.era_attention_l3 = attention(
            widths[3], ERA_OUTPUT_CHANNELS, "diffusion_era_l3"
        )
        self.context_attention_l4 = attention(
            widths[4], context_widths[4], "diffusion_context_l4"
        )
        self.era_attention_l4 = attention(
            widths[4], ERA_OUTPUT_CHANNELS, "diffusion_era_l4"
        )

    def _source_cache_model_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        """Return model state that determines cached source K/V or bias.

        Query normalization/projection, output projection, and the gate are
        deliberately excluded: they are evaluated after cache validation on
        every denoising call.  All source normalization, K/V projection, and
        physical-bias state is bound for every independent source/level arm.
        """

        selected: list[tuple[str, Tensor]] = []
        cached_parameter_prefixes = (
            "source_norm.",
            "key_projection.",
            "value_projection.",
            "geometry_bias.",
        )
        for attention_name in (
            "context_attention_l3",
            "era_attention_l3",
            "context_attention_l4",
            "era_attention_l4",
        ):
            core = getattr(self, attention_name).core
            for state_name, parameter in core.named_parameters():
                if state_name.startswith(cached_parameter_prefixes):
                    selected.append(
                        (f"model_{attention_name}.{state_name}", parameter)
                    )
            for state_name, buffer in core.named_buffers():
                if state_name == "geometry_frequencies":
                    selected.append((f"model_{attention_name}.{state_name}", buffer))
        return tuple(selected)

    def _validate_conditions(self, conditions: ResidualEDMConditions) -> None:
        if not isinstance(conditions, ResidualEDMConditions):
            raise TypeError("ResidualEDM expects ResidualEDMConditions")
        cache = conditions.condition_bank
        if cache.input_size != self.config.input_size:
            raise ValueError("condition-bank and residual-EDM spatial sizes differ")
        if cache.target_widths != self.config.target_widths:
            raise ValueError("condition-bank and residual-EDM target widths differ")
        if cache.context_widths != self.config.context_widths:
            raise ValueError("condition-bank and residual-EDM context widths differ")
        static = cache.target.static_level_zero
        expected_static = (
            cache.batch_size,
            self.config.target_widths[0] // 2,
            self.config.input_size,
            self.config.input_size,
        )
        if tuple(static.shape) != expected_static:
            raise ValueError("frozen target static stem has the wrong shape")

    def _prepare_attention(
        self,
        module: _CachedDiffusionPhysicalAttention,
        source: Tensor,
        *,
        query_geometry: PhysicalTokenGeometry,
        source_geometry: PhysicalTokenGeometry,
        query_shape: tuple[int, int],
        source_validity: Tensor,
        source_present: Tensor,
        e_cond: Tensor,
    ) -> _AttentionSourceKV:
        return module.prepare_source(
            source.detach(),
            query_geometry=query_geometry,
            source_geometry=source_geometry,
            query_shape=query_shape,
            source_validity=source_validity.detach(),
            source_present=source_present.detach(),
            e_cond=e_cond.detach(),
        )

    def prepare_source_cache(
        self, conditions: ResidualEDMConditions
    ) -> ResidualEDMSourceKVCache:
        """Prepare source K/V once, before any sigma-dependent denoising call.

        In training this cache belongs to the current autograd graph and should
        be consumed by one backward pass.  In evaluation it can be reused for
        all members and solver steps of this exact condition batch.
        """

        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("ResidualEDM", self)
        self._validate_conditions(conditions)
        bank = conditions.condition_bank
        batch = bank.batch_size
        e_cond = conditions.e_cond.detach()
        context_present_by_level: list[Tensor] = []
        for level in (3, 4):
            validity = bank.context.validity.levels[level][:, 0] > 0.0
            if self.config.uses_deployment_pyramid:
                context_present_by_level.append(validity.flatten(1).any(dim=1))
            else:
                # EDM-A explicitly excludes the deployment context pyramid.
                context_present_by_level.append(
                    torch.zeros(batch, dtype=torch.bool, device=bank.device)
                )
        era_present = ~(
            conditions.era.used_source_null | conditions.era.used_masked_null
        )
        if era_present.shape != (batch,) or era_present.dtype is not torch.bool:
            raise TypeError("ERA null diagnostics must be bool [B]")
        era_validity = torch.ones(
            (batch, *ERA_NATIVE_SHAPE), dtype=torch.bool, device=bank.device
        )

        context_l3 = self._prepare_attention(
            self.context_attention_l3,
            bank.context.l3,
            query_geometry=conditions.geometry.target_l3,
            source_geometry=conditions.geometry.context_l3,
            query_shape=tuple(bank.target.l3.shape[-2:]),
            source_validity=bank.context.validity.l3[:, 0] > 0.0,
            source_present=context_present_by_level[0],
            e_cond=e_cond,
        )
        era_l3 = self._prepare_attention(
            self.era_attention_l3,
            conditions.era.features,
            query_geometry=conditions.geometry.target_l3,
            source_geometry=conditions.geometry.era_native,
            query_shape=tuple(bank.target.l3.shape[-2:]),
            source_validity=era_validity,
            source_present=era_present,
            e_cond=e_cond,
        )
        context_l4 = self._prepare_attention(
            self.context_attention_l4,
            bank.context.l4,
            query_geometry=conditions.geometry.target_l4,
            source_geometry=conditions.geometry.context_l4,
            query_shape=tuple(bank.target.l4.shape[-2:]),
            source_validity=bank.context.validity.l4[:, 0] > 0.0,
            source_present=context_present_by_level[1],
            e_cond=e_cond,
        )
        era_l4 = self._prepare_attention(
            self.era_attention_l4,
            conditions.era.features,
            query_geometry=conditions.geometry.target_l4,
            source_geometry=conditions.geometry.era_native,
            query_shape=tuple(bank.target.l4.shape[-2:]),
            source_validity=era_validity,
            source_present=era_present,
            e_cond=e_cond,
        )
        return ResidualEDMSourceKVCache(
            owner_id=id(self),
            variant=self.config.variant,
            lead_indices=conditions.lead_indices.detach().clone(),
            sample_ids=tuple(conditions.sample_ids),
            condition_signatures=tuple(conditions.condition_signatures),
            e_cond_key=e_cond.clone(),
            source_bindings=tuple(
                _TensorCacheBinding.capture(name, tensor)
                for name, tensor in _source_cache_tensors(conditions)
            ),
            model_bindings=tuple(
                _TensorCacheBinding.capture(name, tensor)
                for name, tensor in self._source_cache_model_tensors()
            ),
            context_l3=context_l3,
            era_l3=era_l3,
            context_l4=context_l4,
            era_l4=era_l4,
        )

    def _validate_source_cache(
        self,
        conditions: ResidualEDMConditions,
        cache: ResidualEDMSourceKVCache,
    ) -> None:
        if not isinstance(cache, ResidualEDMSourceKVCache):
            raise TypeError("source_cache must be a ResidualEDMSourceKVCache")
        if cache.owner_id != id(self) or cache.variant != self.config.variant:
            raise ValueError("source K/V cache belongs to another residual EDM")
        if cache.sample_ids != conditions.sample_ids:
            raise ValueError("source K/V cache sample identity mismatch")
        if cache.condition_signatures != conditions.condition_signatures:
            raise ValueError("source K/V cache condition signature mismatch")
        if not torch.equal(cache.lead_indices, conditions.lead_indices):
            raise ValueError("source K/V cache lead mismatch")
        if not torch.equal(cache.e_cond_key, conditions.e_cond.detach()):
            raise ValueError("source K/V cache e_cond/input mismatch")
        current = _source_cache_tensors(conditions)
        if len(cache.source_bindings) != len(current):
            raise ValueError("source K/V cache tensor binding schema mismatch")
        for binding, (name, tensor) in zip(
            cache.source_bindings, current, strict=True
        ):
            binding.validate(name, tensor)
        current_model = self._source_cache_model_tensors()
        if len(cache.model_bindings) != len(current_model):
            raise ValueError("source K/V cache model binding schema mismatch")
        for binding, (name, tensor) in zip(
            cache.model_bindings, current_model, strict=True
        ):
            binding.validate(name, tensor)

    def _conditioned_level(
        self,
        level: int,
        hidden: Tensor,
        conditions: ResidualEDMConditions,
        e_diff: Tensor,
    ) -> Tensor:
        if self.deployment_adapters is not None:
            deployment = conditions.condition_bank.target.features.levels[level].detach()
            hidden = hidden + self.deployment_adapters[level](deployment)
        resized = sample_repeated_stride2_lattice(
            conditions.advection_features.detach(),
            downsampling_levels=level,
            expected_shape=tuple(hidden.shape[-2:]),
        )
        confidence = resized[:, 6:7].clamp(0.0, 1.0)
        return self.advection_adapters[level](hidden, resized, e_diff, confidence)

    def _raw_network(
        self,
        scaled_noisy_residual: Tensor,
        c_noise: Tensor,
        conditions: ResidualEDMConditions,
        e_diff: Tensor,
        source_cache: ResidualEDMSourceKVCache,
    ) -> Tensor:
        regression_condition = torch.cat(
            (
                conditions.mu_z.detach(),
                conditions.probability_wet.detach(),
            ),
            dim=1,
        )
        hidden = self.state_adapter(scaled_noisy_residual)
        hidden = hidden + self.c_noise_adapter(c_noise.flatten(1))[:, :, None, None]
        hidden = hidden + self.regression_condition_adapter(regression_condition)
        if self.static_adapter is not None:
            hidden = hidden + self.static_adapter(
                conditions.condition_bank.target.static_level_zero.detach()
            )

        encoded: list[Tensor] = []
        for level, stage in enumerate(self.encoder_stages):
            if level:
                hidden = self.downsample_projections[level - 1](hidden)
            hidden = self._conditioned_level(level, hidden, conditions, e_diff)
            hidden = stage(hidden, e_diff)
            if level == 3:
                hidden, _ = self.context_attention_l3(
                    hidden, e_diff, source_cache.context_l3
                )
                hidden, _ = self.era_attention_l3(
                    hidden, e_diff, source_cache.era_l3
                )
            elif level == 4:
                hidden, _ = self.context_attention_l4(
                    hidden, e_diff, source_cache.context_l4
                )
                hidden, _ = self.era_attention_l4(
                    hidden, e_diff, source_cache.era_l4
                )
            encoded.append(hidden)

        for projection, stage, skip in zip(
            self.upsample_projections,
            self.decoder_stages,
            encoded[-2::-1],
            strict=True,
        ):
            hidden = F.interpolate(
                hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            hidden = projection(hidden)
            hidden = stage(torch.cat((hidden, skip), dim=1), e_diff)
        return self.output_head(F.silu(self.output_norm(hidden)))

    def denoise(
        self,
        noisy_normalized_residual: Tensor,
        sigma: Tensor,
        conditions: ResidualEDMConditions,
        *,
        source_cache: ResidualEDMSourceKVCache | None = None,
    ) -> Tensor:
        """Return the EDM clean estimate ``[B,1,H_target,W_target]``."""

        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("ResidualEDM", self)
        self._validate_conditions(conditions)
        batch = conditions.condition_bank.batch_size
        expected = (
            batch,
            1,
            self.config.input_size,
            self.config.input_size,
        )
        require_float32_tensor(
            "noisy normalized residual", noisy_normalized_residual, ndim=4
        )
        if tuple(noisy_normalized_residual.shape) != expected:
            raise ValueError(
                f"noisy normalized residual must have shape {expected}"
            )
        if noisy_normalized_residual.device != conditions.condition_bank.device:
            raise ValueError("noisy residual and conditions must share one device")
        require_float32_tensor("sigma", sigma, ndim=1)
        if sigma.shape != (batch,) or sigma.device != noisy_normalized_residual.device:
            raise ValueError("sigma must have shape [B] on the residual device")
        if bool((sigma <= 0.0).any().item()):
            raise ValueError("sigma must be strictly positive")

        selected_cache = (
            self.prepare_source_cache(conditions)
            if source_cache is None
            else source_cache
        )
        self._validate_source_cache(conditions, selected_cache)
        coefficients = edm_preconditioning(sigma, sigma_data=self.config.sigma_data)
        e_sigma = self.sigma_embedding(sigma)
        e_diff = self.diffusion_condition_fusion(
            conditions.e_cond.detach(), e_sigma
        )
        prediction = self._raw_network(
            coefficients.c_in * noisy_normalized_residual,
            coefficients.c_noise,
            conditions,
            e_diff,
            selected_cache,
        )
        denoised = (
            coefficients.c_skip * noisy_normalized_residual
            + coefficients.c_out * prediction
        )
        if denoised.dtype is not torch.float32 or tuple(denoised.shape) != expected:
            raise RuntimeError("ResidualEDM violated its target-grid FP32 contract")
        if not bool(torch.isfinite(denoised).all().item()):
            raise RuntimeError("ResidualEDM produced a non-finite value")
        return denoised

    def forward(
        self,
        noisy_normalized_residual: Tensor,
        sigma: Tensor,
        conditions: ResidualEDMConditions,
        *,
        source_cache: ResidualEDMSourceKVCache | None = None,
    ) -> Tensor:
        return self.denoise(
            noisy_normalized_residual,
            sigma,
            conditions,
            source_cache=source_cache,
        )

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)


__all__ = [
    "ADVECTION_CHANNELS",
    "EDM_A_PRODUCTION_PARAMETER_COUNT",
    "EDM_B_PRODUCTION_PARAMETER_COUNT",
    "EDMVariant",
    "ResidualEDM",
    "ResidualEDMConditions",
    "ResidualEDMConfig",
    "ResidualEDMSourceKVCache",
]

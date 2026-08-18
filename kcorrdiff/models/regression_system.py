"""End-to-end strict-FP32 causal regression composition.

``RegressionSystem`` owns the shared condition bank, the single central
condition embedding, the lead-independent ERA frame encoder/cache, the
lead-specific ERA query, streaming causal advection, and the regression
U-Net.  Its public forward accepts only :class:`RegressionModelBatch`; the
loss-only labels produced by the collator are structurally inexpressible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from kcorrdiff.data.advection import (
    ADVECTION_MODEL_CHANNEL_NAMES,
    AdvectionProvenance,
    DenseFlow,
    FlowConfig,
    build_causal_advection,
)
from kcorrdiff.training.batch import RegressionModelBatch

from .common import (
    CONDITION_DIM,
    CONTEXT_WIDTHS,
    PRODUCTION_INPUT_SIZE,
    TARGET_WIDTHS,
    assert_strict_fp32_runtime,
    count_parameters,
    require_module_float32,
    require_no_autocast,
    validate_input_size,
    validate_widths,
)
from .condition_bank import (
    ConditionBank,
    ConditionBankCache,
    ConditionBankConfig,
)
from .era_encoder import EraEncoder, EraFrameCache, EraQueryResult
from .regression import (
    DirectPhysicalRegression,
    DirectPhysicalRegressionOutput,
    RegressionInputs,
    RegressionOutput,
    RegressionPhysicalBiasCache,
    RegressionUNet,
)


@dataclass(frozen=True, slots=True)
class RegressionSystemConfig:
    """Architecture parameters for a regression experiment."""

    target_widths: tuple[int, ...] = TARGET_WIDTHS
    context_widths: tuple[int, ...] = CONTEXT_WIDTHS
    input_size: int = PRODUCTION_INPUT_SIZE
    activation_checkpoint: bool = False
    era_stem_channels: int = 64
    era_spatial_blocks: int = 2
    era_temporal_heads: int = 8
    era_time_frequencies: int = 8
    regression_query_chunk_size: int = 128
    flow_config: FlowConfig = FlowConfig()
    allow_test_override: bool = False

    def __post_init__(self) -> None:
        target = validate_widths(
            "target widths",
            self.target_widths,
            production=TARGET_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        context = validate_widths(
            "context widths",
            self.context_widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        size = validate_input_size(
            self.input_size, allow_test_override=self.allow_test_override
        )
        if min(
            self.era_stem_channels,
            self.era_temporal_heads,
            self.era_time_frequencies,
            self.regression_query_chunk_size,
        ) <= 0 or self.era_spatial_blocks < 0:
            raise ValueError("ERA/regression dimensions must be positive")
        if not isinstance(self.flow_config, FlowConfig):
            raise TypeError("flow_config must be a FlowConfig")
        object.__setattr__(self, "target_widths", target)
        object.__setattr__(self, "context_widths", context)
        object.__setattr__(self, "input_size", size)


@dataclass(frozen=True, slots=True)
class RequestedAdvection:
    """Only the requested lead's eight channels plus causal flow provenance."""

    features: Tensor
    lead_indices: Tensor
    flow: DenseFlow
    provenance: AdvectionProvenance
    channel_names: tuple[str, ...] = ADVECTION_MODEL_CHANNEL_NAMES

    def __post_init__(self) -> None:
        if self.features.ndim != 4 or self.features.shape[1] != len(
            ADVECTION_MODEL_CHANNEL_NAMES
        ):
            raise ValueError("requested advection must have shape [B,8,H,W]")
        if self.features.dtype is not torch.float32 or not torch.isfinite(
            self.features
        ).all():
            raise TypeError("requested advection must be finite float32")
        if self.lead_indices.shape != (self.features.shape[0],):
            raise ValueError("requested advection lead indices must have shape [B]")
        if self.lead_indices.dtype is not torch.int64:
            raise TypeError("requested advection lead indices must be int64")
        if self.lead_indices.device != self.features.device:
            raise ValueError("requested advection tensors must share a device")
        if tuple(self.channel_names) != ADVECTION_MODEL_CHANNEL_NAMES:
            raise ValueError("advection channel schema/order mismatch")
        if self.provenance.used_future_observations:
            raise ValueError("future observations entered causal advection")
        if self.provenance.full_trajectory_cached:
            raise ValueError("full advection trajectory caching is forbidden")


@dataclass(frozen=True, slots=True)
class _TensorMutationBinding:
    """Identity/version stamp for one source or cached tensor."""

    name: str
    tensor: Tensor
    version: int

    @classmethod
    def capture(cls, name: str, tensor: Tensor) -> "_TensorMutationBinding":
        try:
            version = int(tensor._version)
        except RuntimeError as exc:
            raise ValueError(
                f"issue-time cache tensor {name!r} must track mutations"
            ) from exc
        return cls(name=name, tensor=tensor, version=version)

    def validate(self, name: str | None = None, tensor: Tensor | None = None) -> None:
        if name is not None and name != self.name:
            raise ValueError(f"issue-time cache tensor {self.name!r} name changed")
        if tensor is not None and tensor is not self.tensor:
            raise ValueError(f"issue-time cache tensor {self.name!r} identity changed")
        try:
            current = int(self.tensor._version)
        except RuntimeError as exc:
            raise RuntimeError(
                f"issue-time cache source {self.name!r} lost mutation tracking"
            ) from exc
        if current != self.version:
            raise RuntimeError(
                f"issue-time cache tensor {self.name!r} changed after preparation"
            )


def _physical_geometry_tensors(batch: RegressionModelBatch) -> tuple[tuple[str, Tensor], ...]:
    selected: list[tuple[str, Tensor]] = []
    for level in (
        "target_l3",
        "target_l4",
        "context_l3",
        "context_l4",
        "era_native",
    ):
        geometry = getattr(batch.geometry, level)
        selected.extend(
            (
                (f"geometry.{level}.x_shared", geometry.x_shared),
                (f"geometry.{level}.y_shared", geometry.y_shared),
                (f"geometry.{level}.footprint_width", geometry.footprint_width),
                (f"geometry.{level}.footprint_height", geometry.footprint_height),
            )
        )
    return tuple(selected)


def _shared_issue_tensors(batch: RegressionModelBatch) -> tuple[tuple[str, Tensor], ...]:
    target = batch.condition_bank.target
    context = batch.condition_bank.context
    era = batch.era
    causal = batch.advection.causal
    geometry = batch.advection.geometry
    return (
        ("target.radar_history", target.radar_history),
        ("target.history_validity", target.history_validity),
        ("target.static_fields", target.static_fields),
        ("target.static_coverage", target.static_coverage),
        ("context.dynamic_fields", context.dynamic_fields),
        ("context.detail_validity", context.detail_validity),
        ("context.static_fields", context.static_fields),
        ("era.values", era.values),
        ("era.delta_hours", era.delta_hours),
        (
            "era.tp_interval_center_delta_hours",
            era.tp_interval_center_delta_hours,
        ),
        ("era.data_valid_inst", era.data_valid_inst),
        ("era.tp_valid", era.tp_valid),
        ("era.trajectory_window_mask", era.trajectory_window_mask),
        ("era.era_present", era.era_present),
        ("era.tp_present", era.tp_present),
        ("advection.target_rate", causal.target_rate_mm_per_hour),
        ("advection.target_valid", causal.target_valid),
        ("advection.context_rate", causal.context_rate_mm_per_hour),
        ("advection.context_valid", causal.context_valid_fraction),
        (
            "advection.context_confidence",
            causal.context_interpolation_confidence,
        ),
        ("advection_geometry.target_x", geometry.target_x_km),
        ("advection_geometry.target_y", geometry.target_y_km),
        ("advection_geometry.context_x", geometry.context_x_km),
        ("advection_geometry.context_y", geometry.context_y_km),
        *_physical_geometry_tensors(batch),
    )


def _shared_issue_metadata(batch: RegressionModelBatch) -> tuple[object, ...]:
    target = batch.condition_bank.target
    context = batch.condition_bank.context
    era = batch.era
    provenance = batch.provenance
    causal = batch.advection.causal
    return (
        target.static_channel_names,
        context.dynamic_channel_names,
        context.static_channel_names,
        era.valid_times_utc,
        era.tp_intervals_utc,
        era.condition_signatures,
        era.provenance,
        era.channel_names,
        era.value_space,
        era.normalization_artifact_id,
        causal.t0_utc,
        causal.history_times_utc,
        batch.advection.context_channels_alias_condition_bank,
        provenance.block_ids,
        provenance.fold_ids,
        provenance.t0_utc,
        provenance.history_times_utc,
        provenance.condition_signatures,
        provenance.explicit_history_copies,
    )


def _validate_same_issue_time(
    reference: RegressionModelBatch, candidate: RegressionModelBatch
) -> None:
    if reference.batch_size != candidate.batch_size:
        raise ValueError("12-lead issue-time batches must share one batch size")
    if reference.device != candidate.device:
        raise ValueError("12-lead issue-time batches must share one device")
    if _shared_issue_metadata(reference) != _shared_issue_metadata(candidate):
        raise ValueError("12-lead issue-time batch metadata changed across leads")
    reference_tensors = _shared_issue_tensors(reference)
    candidate_tensors = _shared_issue_tensors(candidate)
    if tuple(name for name, _ in reference_tensors) != tuple(
        name for name, _ in candidate_tensors
    ):
        raise AssertionError("shared issue-time tensor schema changed")
    for (name, left), (_, right) in zip(
        reference_tensors, candidate_tensors, strict=True
    ):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
            or not torch.equal(left, right)
        ):
            raise ValueError(
                f"12-lead issue-time input {name!r} changed across leads"
            )


def _cached_issue_tensors(
    *,
    condition_bank: ConditionBankCache,
    era_frames: EraFrameCache,
    advection_features_all_leads: Tensor,
    flow: DenseFlow,
    physical_bias: RegressionPhysicalBiasCache,
) -> tuple[tuple[str, Tensor], ...]:
    selected: list[tuple[str, Tensor]] = []
    for source_name, source in (
        ("condition.target", condition_bank.target),
        ("condition.context", condition_bank.context),
    ):
        selected.extend(
            (f"{source_name}.feature_l{index}", tensor)
            for index, tensor in enumerate(source.features.levels)
        )
        selected.extend(
            (f"{source_name}.validity_l{index}", tensor)
            for index, tensor in enumerate(source.validity.levels)
        )
        selected.extend(
            (
                (f"{source_name}.temporal_level_zero", source.temporal_level_zero),
                (f"{source_name}.static_level_zero", source.static_level_zero),
            )
        )
    selected.extend(
        (
            ("era.instantaneous", era_frames.instantaneous),
            ("era.precipitation", era_frames.precipitation),
            ("era.encoded", era_frames.encoded),
            ("era.tp_null", era_frames.tp_null),
            ("era.delta_hours", era_frames.delta_hours),
            (
                "era.tp_interval_center_delta_hours",
                era_frames.tp_interval_center_delta_hours,
            ),
            ("era.data_valid_inst", era_frames.data_valid_inst),
            ("era.tp_valid", era_frames.tp_valid),
            ("era.trajectory_window_mask", era_frames.trajectory_window_mask),
            ("era.era_present", era_frames.era_present),
            ("era.tp_present", era_frames.tp_present),
            ("advection.all_leads", advection_features_all_leads),
            ("flow.velocity", flow.velocity_km_per_hour),
            ("flow.backward_velocity", flow.backward_velocity_km_per_hour),
            ("flow.confidence", flow.confidence),
            ("flow.forward_backward_error", flow.forward_backward_error_km),
            ("flow.valid_mask", flow.valid_mask),
            ("physical_bias.context_l3", physical_bias.context_l3),
            ("physical_bias.era_l3", physical_bias.era_l3),
            ("physical_bias.context_l4", physical_bias.context_l4),
            ("physical_bias.era_l4", physical_bias.era_l4),
        )
    )
    return tuple(selected)


def _model_tensors(model: nn.Module) -> tuple[tuple[str, Tensor], ...]:
    return tuple(
        (f"parameter.{name}", value)
        for name, value in model.named_parameters(remove_duplicate=False)
    ) + tuple(
        (f"buffer.{name}", value)
        for name, value in model.named_buffers(remove_duplicate=False)
    )


@dataclass(frozen=True, slots=True, init=False)
class _RegressionIssueTimeCache:
    """Inference-only lead-independent state shared by all twelve leads."""

    _owner_model: nn.Module
    reference_batch: RegressionModelBatch
    source_bindings: tuple[_TensorMutationBinding, ...]
    cache_bindings: tuple[_TensorMutationBinding, ...]
    model_bindings: tuple[_TensorMutationBinding, ...]
    module_bindings: tuple[tuple[str, int], ...]
    condition_bank: ConditionBankCache
    era_frames: EraFrameCache
    advection_features_all_leads: Tensor
    flow: DenseFlow
    advection_provenance: AdvectionProvenance
    physical_bias: RegressionPhysicalBiasCache

    @classmethod
    def _create(
        cls,
        *,
        owner_model: nn.Module,
        reference_batch: RegressionModelBatch,
        source_bindings: tuple[_TensorMutationBinding, ...],
        condition_bank: ConditionBankCache,
        era_frames: EraFrameCache,
        advection_features_all_leads: Tensor,
        flow: DenseFlow,
        advection_provenance: AdvectionProvenance,
        physical_bias: RegressionPhysicalBiasCache,
    ) -> "_RegressionIssueTimeCache":
        cached_tensors = _cached_issue_tensors(
            condition_bank=condition_bank,
            era_frames=era_frames,
            advection_features_all_leads=advection_features_all_leads,
            flow=flow,
            physical_bias=physical_bias,
        )
        values = {
            "_owner_model": owner_model,
            "reference_batch": reference_batch,
            "source_bindings": source_bindings,
            "cache_bindings": tuple(
                _TensorMutationBinding.capture(name, tensor)
                for name, tensor in cached_tensors
            ),
            "model_bindings": tuple(
                _TensorMutationBinding.capture(name, tensor)
                for name, tensor in _model_tensors(owner_model)
            ),
            "module_bindings": tuple(
                (name, id(module)) for name, module in owner_model.named_modules()
            ),
            "condition_bank": condition_bank,
            "era_frames": era_frames,
            "advection_features_all_leads": advection_features_all_leads,
            "flow": flow,
            "advection_provenance": advection_provenance,
            "physical_bias": physical_bias,
        }
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        batch = self.reference_batch.batch_size
        size = self.condition_bank.input_size
        if self._owner_model is None:
            raise ValueError("issue-time cache owner model is missing")
        if self.condition_bank.batch_size != batch or self.era_frames.batch_size != batch:
            raise ValueError("issue-time cache components must share one batch size")
        expected = (batch, 12, len(ADVECTION_MODEL_CHANNEL_NAMES), size, size)
        value = self.advection_features_all_leads
        if tuple(value.shape) != expected:
            raise ValueError(f"all-lead advection cache must have shape {expected}")
        if (
            value.dtype is not torch.float32
            or value.device != self.reference_batch.device
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError("all-lead advection cache must be finite float32")
        autograd_names = tuple(
            name
            for name, tensor in _cached_issue_tensors(
                condition_bank=self.condition_bank,
                era_frames=self.era_frames,
                advection_features_all_leads=self.advection_features_all_leads,
                flow=self.flow,
                physical_bias=self.physical_bias,
            )
            if tensor.requires_grad
        )
        if autograd_names:
            raise ValueError(
                "issue-time inference cache cannot retain autograd state: "
                + ", ".join(autograd_names)
            )
        if self.advection_provenance.used_future_observations:
            raise ValueError("future observations entered the issue-time cache")
        if self.advection_provenance.full_trajectory_cached:
            raise ValueError("the streaming issue-time cache cannot retain a trajectory")

    def validate(self, model: nn.Module) -> None:
        if model is not self._owner_model:
            raise ValueError("issue-time cache belongs to another regression model")
        if tuple((name, id(module)) for name, module in model.named_modules()) != (
            self.module_bindings
        ):
            raise RuntimeError("issue-time cache model module identity changed")
        model_tensors = _model_tensors(model)
        if len(model_tensors) != len(self.model_bindings):
            raise RuntimeError("issue-time cache model tensor schema changed")
        for binding, (name, tensor) in zip(
            self.model_bindings, model_tensors, strict=True
        ):
            binding.validate(name, tensor)
        for binding in self.source_bindings:
            binding.validate()
        cached_tensors = _cached_issue_tensors(
            condition_bank=self.condition_bank,
            era_frames=self.era_frames,
            advection_features_all_leads=self.advection_features_all_leads,
            flow=self.flow,
            physical_bias=self.physical_bias,
        )
        if len(cached_tensors) != len(self.cache_bindings):
            raise ValueError("issue-time cache tensor schema changed")
        for binding, (name, tensor) in zip(
            self.cache_bindings, cached_tensors, strict=True
        ):
            binding.validate(name, tensor)


@dataclass(frozen=True, slots=True)
class RegressionSystemOutput:
    """Full forward artifacts and the official occurrence/mean outputs."""

    regression: RegressionOutput
    e_cond: Tensor
    condition_cache: ConditionBankCache
    era_frame_cache: EraFrameCache
    era_query: EraQueryResult
    advection: RequestedAdvection
    condition_signatures: tuple[str, ...]

    @property
    def occurrence_logits(self) -> Tensor:
        return self.regression.occurrence_logits

    @property
    def probability_wet(self) -> Tensor:
        return self.regression.probability_wet

    @property
    def wet_amount(self) -> Tensor:
        return self.regression.wet_amount

    @property
    def mu_z(self) -> Tensor:
        """Official transformed-space conditional mean ``p * m``."""

        return self.regression.transformed_mean


class RegressionSystem(nn.Module):
    """Compose the complete causal regression path exactly once per batch."""

    def __init__(self, config: RegressionSystemConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else RegressionSystemConfig()
        if not isinstance(self.config, RegressionSystemConfig):
            raise TypeError("config must be RegressionSystemConfig")
        bank_config = ConditionBankConfig(
            target_widths=self.config.target_widths,
            context_widths=self.config.context_widths,
            input_size=self.config.input_size,
            activation_checkpoint=self.config.activation_checkpoint,
            allow_test_override=self.config.allow_test_override,
        )
        self.condition_bank = ConditionBank(bank_config)
        self.era_encoder = EraEncoder(
            condition_dim=CONDITION_DIM,
            stem_channels=self.config.era_stem_channels,
            temporal_heads=self.config.era_temporal_heads,
            time_frequencies=self.config.era_time_frequencies,
            spatial_blocks=self.config.era_spatial_blocks,
        )
        self.regression = RegressionUNet(
            target_widths=self.config.target_widths,
            context_widths=self.config.context_widths,
            query_chunk_size=self.config.regression_query_chunk_size,
            activation_checkpoint=self.config.activation_checkpoint,
            allow_test_override=self.config.allow_test_override,
        )

    def _validate_batch(self, batch: RegressionModelBatch) -> None:
        if not isinstance(batch, RegressionModelBatch):
            raise TypeError("RegressionSystem expects RegressionModelBatch, never labels")
        batch.validate()
        parameter = next(self.parameters())
        if parameter.dtype is not torch.float32:
            raise TypeError("RegressionSystem parameters must remain float32")
        if batch.device != parameter.device:
            raise ValueError(
                f"model batch is on {batch.device}, system is on {parameter.device}"
            )
        target_shape = tuple(batch.condition_bank.target.radar_history.shape[-2:])
        if target_shape != (self.config.input_size, self.config.input_size):
            raise ValueError("model batch spatial size differs from system config")
        if batch.provenance.condition_signatures != batch.era.condition_signatures:
            raise ValueError("batch/ERA condition signatures differ")

    def _validate_issue_cache_runtime(self, batch: RegressionModelBatch) -> None:
        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("RegressionSystem", self)
        if self.training or any(module.training for module in self.modules()):
            raise ValueError("issue-time caching requires RegressionSystem.eval()")
        if not torch.is_inference_mode_enabled():
            raise ValueError("issue-time caching requires torch.inference_mode()")
        self._validate_batch(batch)

    def _physical_bias_cache(
        self, batch: RegressionModelBatch, condition_cache: ConditionBankCache
    ) -> RegressionPhysicalBiasCache:
        target_l3_shape = tuple(condition_cache.target.l3.shape[-2:])
        target_l4_shape = tuple(condition_cache.target.l4.shape[-2:])
        context_l3_shape = tuple(condition_cache.context.l3.shape[-2:])
        context_l4_shape = tuple(condition_cache.context.l4.shape[-2:])
        common = {
            "batch_size": batch.batch_size,
            "device": batch.device,
        }
        return RegressionPhysicalBiasCache(
            context_l3=self.regression.context_attention_l3.compute_physical_bias(
                batch.geometry.target_l3,
                batch.geometry.context_l3,
                query_shape=target_l3_shape,
                source_shape=context_l3_shape,
                **common,
            ),
            era_l3=self.regression.era_attention_l3.compute_physical_bias(
                batch.geometry.target_l3,
                batch.geometry.era_native,
                query_shape=target_l3_shape,
                source_shape=(33, 33),
                **common,
            ),
            context_l4=self.regression.context_attention_l4.compute_physical_bias(
                batch.geometry.target_l4,
                batch.geometry.context_l4,
                query_shape=target_l4_shape,
                source_shape=context_l4_shape,
                **common,
            ),
            era_l4=self.regression.era_attention_l4.compute_physical_bias(
                batch.geometry.target_l4,
                batch.geometry.era_native,
                query_shape=target_l4_shape,
                source_shape=(33, 33),
                **common,
            ),
        )

    def _forward_from_shared_components(
        self,
        batch: RegressionModelBatch,
        *,
        condition_cache: ConditionBankCache,
        era_frame_cache: EraFrameCache,
        advection_features_all_leads: Tensor,
        flow: DenseFlow,
        advection_provenance: AdvectionProvenance,
        physical_bias: RegressionPhysicalBiasCache | None,
    ) -> RegressionSystemOutput:
        # This is the only central lead/time embedding call. The exact same
        # tensor instance is consumed by both ERA query and regression.
        e_cond = self.condition_bank.embed_condition(batch.embedding)
        era_query = self.era_encoder.query(
            era_frame_cache,
            temporal_access_mask=batch.era.temporal_access_mask,
            e_cond=e_cond,
        )
        row = torch.arange(batch.batch_size, device=batch.device)
        requested = RequestedAdvection(
            features=advection_features_all_leads[
                row, batch.advection.lead_indices
            ],
            lead_indices=batch.advection.lead_indices,
            flow=flow,
            provenance=advection_provenance,
        )
        regression = self.regression(
            RegressionInputs(
                condition_bank=condition_cache,
                era=era_query,
                advection_features=requested.features,
                e_cond=e_cond,
                geometry=batch.geometry,
                condition_signatures=batch.provenance.condition_signatures,
                physical_bias=physical_bias,
            )
        )
        return RegressionSystemOutput(
            regression=regression,
            e_cond=e_cond,
            condition_cache=condition_cache,
            era_frame_cache=era_frame_cache,
            era_query=era_query,
            advection=requested,
            condition_signatures=batch.provenance.condition_signatures,
        )

    def _prepare_issue_time_cache(
        self,
        batch: RegressionModelBatch,
        *,
        flow_override: DenseFlow | None = None,
    ) -> _RegressionIssueTimeCache:
        """Encode one issue-time batch once for reuse by all twelve leads."""

        self._validate_issue_cache_runtime(batch)
        bindings = tuple(
            _TensorMutationBinding.capture(name, tensor)
            for name, tensor in _shared_issue_tensors(batch)
        )
        # Cache tensors need ordinary PyTorch version counters so mutation is
        # detectable on every reuse. Disable inference-tensor creation while
        # retaining the surrounding inference-only/no-autograd contract.
        with torch.inference_mode(False), torch.no_grad():
            selected_flow = (
                None
                if flow_override is None
                else DenseFlow(
                    velocity_km_per_hour=flow_override.velocity_km_per_hour.clone(),
                    backward_velocity_km_per_hour=(
                        flow_override.backward_velocity_km_per_hour.clone()
                    ),
                    confidence=flow_override.confidence.clone(),
                    forward_backward_error_km=(
                        flow_override.forward_backward_error_km.clone()
                    ),
                    valid_mask=flow_override.valid_mask.clone(),
                    config_hash=flow_override.config_hash,
                )
            )
            condition_cache = self.condition_bank.encode_shared(batch.condition_bank)
            era = batch.era
            era_frame_cache = self.era_encoder.encode_frames(
                era.instantaneous,
                era.precipitation,
                delta_hours=era.delta_hours,
                data_valid_inst=era.data_valid_inst,
                tp_valid=era.tp_valid,
                trajectory_window_mask=era.trajectory_window_mask,
                era_present=era.era_present,
                tp_present=era.tp_present,
            )
            # ``tp_null`` is an expanded parameter view and therefore keeps
            # requires_grad even inside no_grad; the inference cache owns a
            # detached view of the same immutable values.
            era_frame_cache = EraFrameCache(
                instantaneous=era_frame_cache.instantaneous,
                precipitation=era_frame_cache.precipitation,
                encoded=era_frame_cache.encoded,
                tp_null=era_frame_cache.tp_null.detach(),
                delta_hours=era_frame_cache.delta_hours,
                tp_interval_center_delta_hours=(
                    era_frame_cache.tp_interval_center_delta_hours
                ),
                data_valid_inst=era_frame_cache.data_valid_inst,
                tp_valid=era_frame_cache.tp_valid,
                trajectory_window_mask=era_frame_cache.trajectory_window_mask,
                era_present=era_frame_cache.era_present,
                tp_present=era_frame_cache.tp_present,
            )
            artifact = build_causal_advection(
                batch.advection.causal,
                batch.advection.geometry,
                config=self.config.flow_config,
                flow=selected_flow,
                materialize_trajectory=False,
                track_grad=False,
            )
            advection_features = artifact.leads.as_model_tensor()
            physical_bias = self._physical_bias_cache(batch, condition_cache)
        cache = _RegressionIssueTimeCache._create(
            owner_model=self,
            reference_batch=batch,
            source_bindings=bindings,
            condition_bank=condition_cache,
            era_frames=era_frame_cache,
            advection_features_all_leads=advection_features,
            flow=artifact.flow,
            advection_provenance=artifact.provenance,
            physical_bias=physical_bias,
        )
        cache.validate(self)
        return cache

    def _forward_from_issue_time_cache(
        self,
        batch: RegressionModelBatch,
        cache: _RegressionIssueTimeCache,
    ) -> RegressionSystemOutput:
        """Run one lead from an internally prepared, exact issue-time cache."""

        self._validate_issue_cache_runtime(batch)
        if not isinstance(cache, _RegressionIssueTimeCache):
            raise TypeError("cache must be an internal issue-time cache")
        cache.validate(self)
        _validate_same_issue_time(cache.reference_batch, batch)
        output = self._forward_from_shared_components(
            batch,
            condition_cache=cache.condition_bank,
            era_frame_cache=cache.era_frames,
            advection_features_all_leads=cache.advection_features_all_leads,
            flow=cache.flow,
            advection_provenance=cache.advection_provenance,
            physical_bias=cache.physical_bias,
        )
        cache.validate(self)
        return output

    def _validate_issue_time_batches(
        self, batches: Sequence[RegressionModelBatch]
    ) -> tuple[RegressionModelBatch, ...]:
        """Fail before cache construction when any all-lead input differs."""

        selected = tuple(batches)
        if len(selected) != 12:
            raise ValueError("all-lead cached regression requires exactly 12 batches")
        reference = selected[0]
        for batch in selected:
            self._validate_issue_cache_runtime(batch)
            _validate_same_issue_time(reference, batch)
        return selected

    def forward(
        self,
        batch: RegressionModelBatch,
        *,
        flow_override: DenseFlow | None = None,
    ) -> RegressionSystemOutput:
        """Run shared caches, one ``e_cond``, one ERA query, and regression.

        Supplying ``flow_override`` is an explicit experiment choice; otherwise
        flow is estimated from the twelve causal context frames. Advection uses
        the streaming path and retains only the requested eight-channel lead.
        """

        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("RegressionSystem", self)
        self._validate_batch(batch)

        condition_cache = self.condition_bank.encode_shared(batch.condition_bank)
        era = batch.era
        era_frame_cache = self.era_encoder.encode_frames(
            era.instantaneous,
            era.precipitation,
            delta_hours=era.delta_hours,
            data_valid_inst=era.data_valid_inst,
            tp_valid=era.tp_valid,
            trajectory_window_mask=era.trajectory_window_mask,
            era_present=era.era_present,
            tp_present=era.tp_present,
        )
        artifact = build_causal_advection(
            batch.advection.causal,
            batch.advection.geometry,
            config=self.config.flow_config,
            flow=flow_override,
            materialize_trajectory=False,
            track_grad=False,
        )
        return self._forward_from_shared_components(
            batch,
            condition_cache=condition_cache,
            era_frame_cache=era_frame_cache,
            advection_features_all_leads=artifact.leads.as_model_tensor(),
            flow=artifact.flow,
            advection_provenance=artifact.provenance,
            physical_bias=None,
        )

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)

    @property
    def component_parameter_counts(self) -> dict[str, int]:
        return {
            "condition_bank": count_parameters(self.condition_bank),
            "era_encoder": count_parameters(self.era_encoder),
            "regression": count_parameters(self.regression),
        }


@dataclass(frozen=True, slots=True)
class DirectPhysicalRegressionSystemOutput:
    """Direct comparison output plus the same causal front-end artifacts."""

    direct: DirectPhysicalRegressionOutput
    e_cond: Tensor
    condition_cache: ConditionBankCache
    era_frame_cache: EraFrameCache
    era_query: EraQueryResult
    advection: RequestedAdvection
    condition_signatures: tuple[str, ...]

    @property
    def prediction_mm(self) -> Tensor:
        return self.direct.prediction_mm

    @property
    def statistic(self) -> Literal["mean", "q50"]:
        return self.direct.statistic


class DirectPhysicalRegressionSystem(nn.Module):
    """Independent full-capacity physical mean/q50 comparison system.

    The condition bank, condition embedding, ERA encoder/query and causal
    advection path are structurally identical to :class:`RegressionSystem`.
    Only the complete regression module is replaced by a separately trained
    :class:`DirectPhysicalRegression`; no hurdle checkpoint weights are shared.
    """

    def __init__(
        self,
        *,
        statistic: Literal["mean", "q50"],
        config: RegressionSystemConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else RegressionSystemConfig()
        if not isinstance(self.config, RegressionSystemConfig):
            raise TypeError("config must be RegressionSystemConfig")
        if statistic not in {"mean", "q50"}:
            raise ValueError("direct physical statistic must be mean or q50")
        self.statistic: Literal["mean", "q50"] = statistic
        self.condition_bank = ConditionBank(
            ConditionBankConfig(
                target_widths=self.config.target_widths,
                context_widths=self.config.context_widths,
                input_size=self.config.input_size,
                activation_checkpoint=self.config.activation_checkpoint,
                allow_test_override=self.config.allow_test_override,
            )
        )
        self.era_encoder = EraEncoder(
            condition_dim=CONDITION_DIM,
            stem_channels=self.config.era_stem_channels,
            temporal_heads=self.config.era_temporal_heads,
            time_frequencies=self.config.era_time_frequencies,
            spatial_blocks=self.config.era_spatial_blocks,
        )
        self.regression = DirectPhysicalRegression(
            statistic=statistic,
            target_widths=self.config.target_widths,
            context_widths=self.config.context_widths,
            query_chunk_size=self.config.regression_query_chunk_size,
            activation_checkpoint=self.config.activation_checkpoint,
            allow_test_override=self.config.allow_test_override,
        )

    def _validate_batch(self, batch: RegressionModelBatch) -> None:
        if not isinstance(batch, RegressionModelBatch):
            raise TypeError(
                "DirectPhysicalRegressionSystem expects RegressionModelBatch, never labels"
            )
        batch.validate()
        parameter = next(self.parameters())
        if parameter.dtype is not torch.float32:
            raise TypeError("DirectPhysicalRegressionSystem parameters must remain float32")
        if batch.device != parameter.device:
            raise ValueError(
                f"model batch is on {batch.device}, system is on {parameter.device}"
            )
        if tuple(batch.condition_bank.target.radar_history.shape[-2:]) != (
            self.config.input_size,
            self.config.input_size,
        ):
            raise ValueError("model batch spatial size differs from system config")
        if batch.provenance.condition_signatures != batch.era.condition_signatures:
            raise ValueError("batch/ERA condition signatures differ")

    def forward(
        self,
        batch: RegressionModelBatch,
        *,
        flow_override: DenseFlow | None = None,
    ) -> DirectPhysicalRegressionSystemOutput:
        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("DirectPhysicalRegressionSystem", self)
        self._validate_batch(batch)

        condition_cache = self.condition_bank.encode_shared(batch.condition_bank)
        e_cond = self.condition_bank.embed_condition(batch.embedding)
        era = batch.era
        era_frame_cache = self.era_encoder.encode_frames(
            era.instantaneous,
            era.precipitation,
            delta_hours=era.delta_hours,
            data_valid_inst=era.data_valid_inst,
            tp_valid=era.tp_valid,
            trajectory_window_mask=era.trajectory_window_mask,
            era_present=era.era_present,
            tp_present=era.tp_present,
        )
        era_query = self.era_encoder.query(
            era_frame_cache,
            temporal_access_mask=era.temporal_access_mask,
            e_cond=e_cond,
        )
        artifact = build_causal_advection(
            batch.advection.causal,
            batch.advection.geometry,
            config=self.config.flow_config,
            flow=flow_override,
            materialize_trajectory=False,
            track_grad=False,
        )
        all_leads = artifact.leads.as_model_tensor()
        row = torch.arange(batch.batch_size, device=batch.device)
        requested = RequestedAdvection(
            features=all_leads[row, batch.advection.lead_indices],
            lead_indices=batch.advection.lead_indices,
            flow=artifact.flow,
            provenance=artifact.provenance,
        )
        direct = self.regression(
            RegressionInputs(
                condition_bank=condition_cache,
                era=era_query,
                advection_features=requested.features,
                e_cond=e_cond,
                geometry=batch.geometry,
                condition_signatures=batch.provenance.condition_signatures,
            )
        )
        return DirectPhysicalRegressionSystemOutput(
            direct=direct,
            e_cond=e_cond,
            condition_cache=condition_cache,
            era_frame_cache=era_frame_cache,
            era_query=era_query,
            advection=requested,
            condition_signatures=batch.provenance.condition_signatures,
        )

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)

    @property
    def component_parameter_counts(self) -> dict[str, int]:
        return {
            "condition_bank": count_parameters(self.condition_bank),
            "era_encoder": count_parameters(self.era_encoder),
            "regression": count_parameters(self.regression),
        }


__all__ = [
    "DirectPhysicalRegressionSystem",
    "DirectPhysicalRegressionSystemOutput",
    "RegressionSystem",
    "RegressionSystemConfig",
    "RegressionSystemOutput",
    "RequestedAdvection",
]

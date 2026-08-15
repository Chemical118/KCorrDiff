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
    RegressionUNet,
)


@dataclass(frozen=True, slots=True)
class RegressionSystemConfig:
    """Frozen production architecture, with small models behind one flag."""

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
        if not self.allow_test_override and (
            self.era_stem_channels != 64
            or self.era_spatial_blocks != 2
            or self.era_temporal_heads != 8
            or self.era_time_frequencies != 8
        ):
            raise ValueError("production ERA architecture fallback is forbidden")
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

    def forward(
        self,
        batch: RegressionModelBatch,
        *,
        flow_override: DenseFlow | None = None,
    ) -> RegressionSystemOutput:
        """Run shared caches, one ``e_cond``, one ERA query, and regression.

        ``flow_override`` is available only for explicit synthetic tests.  A
        production call always estimates flow from the twelve causal context
        frames.  Advection uses the streaming path and only the requested
        eight-channel lead is retained by this system boundary.
        """

        require_no_autocast()
        assert_strict_fp32_runtime()
        require_module_float32("RegressionSystem", self)
        if flow_override is not None and not self.config.allow_test_override:
            raise ValueError("flow_override is restricted to explicit test models")
        self._validate_batch(batch)

        condition_cache = self.condition_bank.encode_shared(batch.condition_bank)
        # This is the only central lead/time embedding call.  The exact same
        # tensor instance is consumed by both ERA query and regression.
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
        requested_features = all_leads[row, batch.advection.lead_indices]
        requested = RequestedAdvection(
            features=requested_features,
            lead_indices=batch.advection.lead_indices,
            flow=artifact.flow,
            provenance=artifact.provenance,
        )
        regression = self.regression(
            RegressionInputs(
                condition_bank=condition_cache,
                era=era_query,
                advection_features=requested.features,
                e_cond=e_cond,
                geometry=batch.geometry,
                condition_signatures=batch.provenance.condition_signatures,
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
        if flow_override is not None and not self.config.allow_test_override:
            raise ValueError("flow_override is restricted to explicit test models")
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

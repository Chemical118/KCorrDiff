"""Artifact-bound, end-to-end inference for one homogeneous forecast cell.

The public boundary deliberately accepts no free-form model, checkpoint, calibration
key, condition, or residual-scale arguments. Those values are all carried by a
verified production identity or by the explicitly non-production development
identity. This makes it structurally impossible for a normal caller to join a
regression output to an unrelated diffusion/calibration artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

import torch
from torch import Tensor

from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.data.time_features import OFFICIAL_LEAD_HOURS
from kcorrdiff.inference.cache import prepare_residual_edm_sampler
from kcorrdiff.inference.calibration_audit import (
    CalibrationApplicationAudit,
    CalibrationApplicationAuditBundle,
    ResidualCalibrationApplicationAudit,
    apply_ensemble_probability_with_audit,
    apply_regression_probability_with_audit,
    apply_residual_calibration_with_audit,
    build_calibration_application_audit_bundle,
)
from kcorrdiff.inference.identity import (
    DevelopmentForecastIdentity,
    DevelopmentScaleAudit,
    VerifiedForecastIdentity,
)
from kcorrdiff.models.regression_system import RegressionSystem, RegressionSystemOutput
from kcorrdiff.models.residual_edm import ResidualEDM, ResidualEDMConditions
from kcorrdiff.training.batch import RegressionModelBatch
from kcorrdiff.training.calibration import (
    empirical_lower_median,
    transformed_members_to_physical,
)
from kcorrdiff.training.edm_sampling import (
    OFFICIAL_Q_THRESHOLDS_MM,
    LeadForecast,
    sample_normalized_residual_ensemble,
)
from kcorrdiff.training.stage3_data import ResidualScaleRecord


ForecastIdentity: TypeAlias = VerifiedForecastIdentity | DevelopmentForecastIdentity
ResidualScaleAudit: TypeAlias = ResidualScaleRecord | DevelopmentScaleAudit


@dataclass(frozen=True, slots=True)
class TwelveLeadForecastRequest:
    """Canonical artifact-bound identities and batches for one issue time."""

    identities: tuple[ForecastIdentity, ...]
    batches: tuple[RegressionModelBatch, ...]

    @classmethod
    def from_mixed_batch(
        cls,
        *,
        identities: tuple[ForecastIdentity, ...],
        batch: RegressionModelBatch,
    ) -> "TwelveLeadForecastRequest":
        """Split one canonical shared-t0 batch into storage-sharing lead views."""

        if not isinstance(batch, RegressionModelBatch):
            raise TypeError("batch must be a RegressionModelBatch")
        if batch.batch_size != len(OFFICIAL_LEAD_HOURS):
            raise ValueError("mixed twelve-lead batch must contain exactly 12 rows")
        observed = tuple(float(value) for value in batch.embedding.lead_hours.tolist())
        if observed != OFFICIAL_LEAD_HOURS:
            raise ValueError("mixed batch rows must use canonical lead order")
        return cls(
            identities=tuple(identities),
            batches=tuple(
                batch.select_rows((index,))
                for index in range(len(OFFICIAL_LEAD_HOURS))
            ),
        )

    def __post_init__(self) -> None:
        identities = tuple(self.identities)
        batches = tuple(self.batches)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "batches", batches)
        if len(identities) != len(OFFICIAL_LEAD_HOURS) or len(batches) != len(
            OFFICIAL_LEAD_HOURS
        ):
            raise ValueError("a twelve-lead request requires exactly 12 cells")
        if any(
            not isinstance(
                identity, (VerifiedForecastIdentity, DevelopmentForecastIdentity)
            )
            for identity in identities
        ):
            raise TypeError("every cell requires a verified or development identity")
        if any(not isinstance(batch, RegressionModelBatch) for batch in batches):
            raise TypeError("every twelve-lead cell requires a RegressionModelBatch")
        if any(batch.batch_size != 1 for batch in batches):
            raise ValueError("one twelve-lead request represents exactly one issue time")
        if tuple(identity.lead_hours for identity in identities) != OFFICIAL_LEAD_HOURS:
            raise ValueError("twelve-lead identities must use canonical lead order")

        first = identities[0]
        for identity in identities[1:]:
            if type(identity) is not type(first) or identity.mode != first.mode:
                raise ValueError("production and development identities cannot be mixed")
            if identity.condition_signature != first.condition_signature:
                raise ValueError("twelve-lead identities must share one condition signature")
            if identity.regression_binding is not first.regression_binding:
                raise ValueError("twelve-lead identities must share one regression binding")
            if identity.residual_binding is not first.residual_binding:
                raise ValueError("twelve-lead identities must share one residual binding")
            if identity.calibration is not first.calibration:
                raise ValueError("twelve-lead identities must share one calibration resolver")
            if identity.ensemble_signature != first.ensemble_signature:
                raise ValueError("twelve-lead identities must share one ensemble signature")
            if identity.fold_checkpoint_sha256s != first.fold_checkpoint_sha256s:
                raise ValueError("twelve-lead identities must share regression fold lineage")
        if isinstance(first, VerifiedForecastIdentity):
            for identity in identities[1:]:
                assert isinstance(identity, VerifiedForecastIdentity)
                if (
                    identity.stage3_data_semantic_sha256
                    != first.stage3_data_semantic_sha256
                    or identity.calibration_semantic_sha256
                    != first.calibration_semantic_sha256
                    or identity.model_selection is not first.model_selection
                ):
                    raise ValueError(
                        "production twelve-lead identities must share one release"
                    )

    def validate(self) -> None:
        """Revalidate a frozen request after possible low-level mutation."""

        self.__post_init__()


def _probability_tensor(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()) or bool(
        ((value < 0.0) | (value > 1.0)).any().item()
    ):
        raise ValueError(f"{name} must be finite and lie in [0,1]")


def _forecast_tensor(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _residual_audit_is_complete(audit: ResidualCalibrationApplicationAudit) -> bool:
    """Treat a frozen d=0 selection as supported, but never a fallback identity."""

    return (
        audit.location_scale.calibrated
        and (
            audit.sampler_bias.calibrated
            or audit.sampler_bias.mode == "selection_disabled"
        )
        and audit.spread.calibrated
    )


@dataclass(frozen=True, slots=True)
class EnsembleLeadForecastResult:
    """A diffusion ensemble plus values and audits actually reported to users.

    ``reported_*`` is intentionally neutral terminology. A terminal or
    development identity is a raw pass-through and is never described as a
    calibrated value merely because it crossed the calibration boundary.
    """

    marginal: LeadForecast
    raw_regression_probability_wet: Tensor
    reported_regression_probability_wet: Tensor
    reported_q_fractions: Tensor
    calibration_audit: CalibrationApplicationAuditBundle
    residual_scale_audit: ResidualScaleAudit
    identity_mode: Literal["production", "development"]
    mode: Literal["regression_diffusion"] = "regression_diffusion"

    def __post_init__(self) -> None:
        if not isinstance(self.marginal, LeadForecast):
            raise TypeError("marginal must be a LeadForecast")
        batch, _members, _channel, height, width = (
            self.marginal.physical_members_mm.shape
        )
        probability_shape = (batch, 1, height, width)
        _probability_tensor(
            self.raw_regression_probability_wet,
            name="raw_regression_probability_wet",
            shape=probability_shape,
        )
        _probability_tensor(
            self.reported_regression_probability_wet,
            name="reported_regression_probability_wet",
            shape=probability_shape,
        )
        _probability_tensor(
            self.reported_q_fractions,
            name="reported_q_fractions",
            shape=tuple(self.marginal.q_fractions.shape),
        )
        if not isinstance(
            self.calibration_audit, CalibrationApplicationAuditBundle
        ):
            raise TypeError(
                "calibration_audit must be a CalibrationApplicationAuditBundle"
            )
        if self.identity_mode not in {"production", "development"}:
            raise ValueError("identity_mode must be production or development")
        if self.identity_mode == "production" and not isinstance(
            self.residual_scale_audit, ResidualScaleRecord
        ):
            raise TypeError("production result requires an artifact scale audit")
        if self.identity_mode == "development" and not isinstance(
            self.residual_scale_audit, DevelopmentScaleAudit
        ):
            raise TypeError("development result requires a development scale audit")
        if self.residual_scale_audit.lead_hours != self.marginal.lead_hours:
            raise ValueError("residual scale audit lead differs from the forecast")
        if self.mode != "regression_diffusion":
            raise ValueError("ensemble result mode must be regression_diffusion")

        p_thresholds = tuple(
            item.threshold_mm
            for item in self.calibration_audit.regression_probability
        )
        q_thresholds = tuple(
            item.threshold_mm for item in self.calibration_audit.ensemble_probability
        )
        if p_thresholds != (A_WET_MM,):
            raise ValueError("regression probability audit must cover exact A_wet")
        if q_thresholds != OFFICIAL_Q_THRESHOLDS_MM:
            raise ValueError("ensemble probability audit must cover official q cells")

    @property
    def fully_calibrated(self) -> bool:
        """Whether every applicable fitted mapping was used without fallback."""

        return (
            self.identity_mode == "production"
            and _residual_audit_is_complete(self.calibration_audit.residual)
            and all(
                item.calibrated
                for item in self.calibration_audit.regression_probability
            )
            and all(
                item.calibrated for item in self.calibration_audit.ensemble_probability
            )
        )


@dataclass(frozen=True, slots=True)
class RegressionOnlyLeadForecastResult:
    """Explicit fallback when an OOF residual scale cannot support diffusion."""

    lead_hours: float
    condition_signature: str
    occurrence_logits: Tensor
    transformed_mean: Tensor
    wet_amount: Tensor
    raw_regression_probability_wet: Tensor
    reported_regression_probability_wet: Tensor
    probability_calibration_audit: CalibrationApplicationAudit
    residual_scale_audit: ResidualScaleAudit
    identity_mode: Literal["production"]
    reason: Literal["diffusion_scale_unsupported"] = (
        "diffusion_scale_unsupported"
    )
    mode: Literal["regression_only"] = "regression_only"

    def __post_init__(self) -> None:
        if self.lead_hours not in OFFICIAL_LEAD_HOURS:
            raise ValueError("lead_hours must be an official lead")
        if (
            not isinstance(self.condition_signature, str)
            or not self.condition_signature
        ):
            raise ValueError("condition_signature must be non-empty")
        if (
            not isinstance(self.transformed_mean, Tensor)
            or self.transformed_mean.ndim != 4
        ):
            raise TypeError("transformed_mean must be a float32 [B,1,H,W] tensor")
        point_shape = tuple(self.transformed_mean.shape)
        if len(point_shape) != 4 or point_shape[1] != 1:
            raise ValueError("transformed_mean must have shape [B,1,H,W]")
        _forecast_tensor(
            self.occurrence_logits,
            name="occurrence_logits",
            shape=point_shape,
        )
        _forecast_tensor(
            self.transformed_mean,
            name="transformed_mean",
            shape=point_shape,
        )
        _forecast_tensor(self.wet_amount, name="wet_amount", shape=point_shape)
        _probability_tensor(
            self.raw_regression_probability_wet,
            name="raw_regression_probability_wet",
            shape=point_shape,
        )
        _probability_tensor(
            self.reported_regression_probability_wet,
            name="reported_regression_probability_wet",
            shape=point_shape,
        )
        if not isinstance(
            self.probability_calibration_audit, CalibrationApplicationAudit
        ):
            raise TypeError(
                "probability_calibration_audit must be CalibrationApplicationAudit"
            )
        if not isinstance(self.residual_scale_audit, ResidualScaleRecord) or not (
            self.residual_scale_audit.diffusion_scale_unsupported
        ):
            raise ValueError(
                "regression-only result requires an unsupported artifact scale audit"
            )
        if self.identity_mode != "production":
            raise ValueError("unsupported residual scales exist only in production")
        if self.reason != "diffusion_scale_unsupported":
            raise ValueError("unknown regression-only reason")
        if self.mode != "regression_only":
            raise ValueError("regression-only result mode must be regression_only")

    @property
    def fully_calibrated(self) -> bool:
        """A regression-only fallback is never a fully calibrated ensemble."""

        return False


ForecastResult: TypeAlias = (
    EnsembleLeadForecastResult | RegressionOnlyLeadForecastResult
)


@dataclass(frozen=True, slots=True)
class TwelveLeadForecastResult:
    """Canonical marginal results from one shared issue-time cache."""

    results: tuple[ForecastResult, ...]
    condition_signature: str
    identity_mode: Literal["production", "development"]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        object.__setattr__(self, "results", results)
        if len(results) != len(OFFICIAL_LEAD_HOURS):
            raise ValueError("twelve-lead result requires exactly 12 forecasts")
        if any(
            not isinstance(
                result,
                (EnsembleLeadForecastResult, RegressionOnlyLeadForecastResult),
            )
            for result in results
        ):
            raise TypeError("twelve-lead result contains an invalid forecast type")
        observed = tuple(
            result.marginal.lead_hours
            if isinstance(result, EnsembleLeadForecastResult)
            else result.lead_hours
            for result in results
        )
        if observed != OFFICIAL_LEAD_HOURS:
            raise ValueError("twelve-lead results must use canonical lead order")
        if not self.condition_signature:
            raise ValueError("twelve-lead result condition signature cannot be empty")
        if self.identity_mode not in {"production", "development"}:
            raise ValueError("twelve-lead result identity mode is invalid")
        for result in results:
            if (
                result.residual_scale_audit.condition_signature
                != self.condition_signature
            ):
                raise ValueError("twelve-lead result condition signature changed")
            if result.identity_mode != self.identity_mode:
                raise ValueError("twelve-lead result identity modes differ")

    @property
    def by_lead(self) -> Mapping[float, ForecastResult]:
        return MappingProxyType(dict(zip(OFFICIAL_LEAD_HOURS, self.results, strict=True)))

    @property
    def fully_calibrated(self) -> bool:
        return all(result.fully_calibrated for result in self.results)


def _validate_identity_batch(
    identity: ForecastIdentity, batch: RegressionModelBatch
) -> None:
    lead_values = batch.embedding.lead_hours.detach()
    if lead_values.ndim != 1 or lead_values.numel() == 0:
        raise ValueError("one forecast call requires a non-empty lead vector")
    if not bool(torch.isfinite(lead_values).all().item()) or not bool(
        torch.all(lead_values == lead_values[0]).item()
    ):
        raise ValueError("one forecast call requires one finite homogeneous lead")
    if float(lead_values[0].item()) != identity.lead_hours:
        raise ValueError("batch lead does not match the verified forecast identity")

    expected_lead_index = OFFICIAL_LEAD_HOURS.index(identity.lead_hours)
    if not bool((batch.advection.lead_indices == expected_lead_index).all().item()):
        raise ValueError("batch lead indices disagree with the verified lead")
    signatures = tuple(batch.provenance.condition_signatures)
    if not signatures or len(set(signatures)) != 1:
        raise ValueError("one forecast call requires one condition signature")
    if signatures[0] != identity.condition_signature:
        raise ValueError(
            "batch condition signature does not match the verified forecast identity"
        )
    if batch.era.condition_signatures != signatures:
        raise ValueError("batch ERA and provenance condition signatures disagree")

    if identity.location_key.lead_hours != identity.lead_hours or (
        identity.location_key.condition_signature != identity.condition_signature
    ):
        raise RuntimeError("forecast identity location key changed")
    if (
        identity.location_key.fold_checkpoint_sha256s
        != identity.fold_checkpoint_sha256s
        or identity.location_key.full_checkpoint_sha256
        != identity.regression_binding.checkpoint_sha256
    ):
        raise RuntimeError("forecast identity regression checkpoint key changed")
    if (
        identity.bias_key.lead_hours != identity.lead_hours
        or identity.bias_key.condition_signature != identity.condition_signature
        or identity.bias_key.sampler_core != identity.ensemble_signature.sampler_core
        or identity.spread_key.lead_hours != identity.lead_hours
        or identity.spread_key.condition_signature != identity.condition_signature
        or identity.spread_key.ensemble_signature != identity.ensemble_signature
        or identity.residual_binding.checkpoint_sha256
        != identity.ensemble_signature.sampler_core.checkpoint_id
    ):
        raise RuntimeError("forecast identity sampler key changed")
    if (
        identity.regression_probability_key.lead_hours != identity.lead_hours
        or identity.regression_probability_key.condition_signature
        != identity.condition_signature
        or identity.regression_probability_key.threshold_mm != A_WET_MM
        or identity.regression_probability_key.regression_checkpoint_sha256
        != identity.regression_binding.checkpoint_sha256
    ):
        raise RuntimeError("forecast identity regression probability key changed")
    if tuple(
        key.threshold_mm for key in identity.ensemble_probability_keys
    ) != OFFICIAL_Q_THRESHOLDS_MM:
        raise RuntimeError("forecast identity q keys changed")
    for key in identity.ensemble_probability_keys:
        if (
            key.lead_hours != identity.lead_hours
            or key.condition_signature != identity.condition_signature
            or key.ensemble_signature != identity.ensemble_signature
        ):
            raise RuntimeError("forecast identity q key changed")
    if identity.diffusion_scale_supported != (identity.residual_scale is not None):
        raise RuntimeError("forecast identity residual-scale support state changed")
    scale_audit = identity.scale_audit
    if (
        scale_audit.lead_hours != identity.lead_hours
        or scale_audit.condition_signature != identity.condition_signature
    ):
        raise RuntimeError("forecast identity residual-scale audit changed")

    if isinstance(identity, VerifiedForecastIdentity):
        if identity.mode != "production":
            raise RuntimeError("verified forecast identity mode changed")
        if not identity.regression_binding.is_production or not (
            identity.residual_binding.is_production
        ):
            raise RuntimeError("production identity lost its production model binding")
    elif isinstance(identity, DevelopmentForecastIdentity):
        if identity.mode != "development":
            raise RuntimeError("development forecast identity mode changed")
        if not identity.diffusion_scale_supported or identity.residual_scale is None:
            raise RuntimeError("development identity cannot disable diffusion scale")
    else:  # pragma: no cover - guarded at the public boundary.
        raise TypeError("identity must be a verified or development forecast identity")


def _conditions_from_regression(
    *, batch: RegressionModelBatch, deployment: RegressionSystemOutput
) -> ResidualEDMConditions:
    # Keeping construction in this module prevents a caller from substituting
    # independently prepared conditions after the verified regression forward.
    return ResidualEDMConditions(
        condition_bank=deployment.condition_cache,
        era=deployment.era_query,
        advection_features=deployment.advection.features,
        mu_z=deployment.mu_z,
        probability_wet=deployment.probability_wet,
        e_cond=deployment.e_cond,
        geometry=batch.geometry,
        sample_ids=batch.provenance.sample_ids,
        condition_signatures=deployment.condition_signatures,
        lead_indices=deployment.advection.lead_indices,
    )


def _restore_residual_scale(normalized: Tensor, scale: Real) -> Tensor:
    # Identity construction accepts only a scalar. Revalidate here so a
    # corrupted object cannot silently restore a spatially varying field.
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("verified OOF residual scale must be a real scalar")
    scalar = float(scale)
    scale_tensor = torch.tensor(
        scalar, dtype=torch.float32, device=normalized.device
    )
    if not bool(torch.isfinite(scale_tensor).item()) or scalar <= 0.0:
        raise ValueError("verified OOF residual scale must be finite and positive")
    restored = normalized * scale_tensor
    if not bool(torch.isfinite(restored).all().item()):
        raise OverflowError("OOF residual restoration overflowed float32")
    return restored


def _sample_identity_lead(
    *,
    identity: ForecastIdentity,
    residual_model: ResidualEDM,
    conditions: ResidualEDMConditions,
) -> EnsembleLeadForecastResult:
    if identity.residual_scale is None:
        raise RuntimeError("diffusion sampling requires a supported residual scale")
    if tuple(conditions.condition_signatures) != (
        identity.condition_signature,
    ) * len(conditions.condition_signatures):
        raise ValueError("regression conditions differ from the verified signature")
    expected_lead_index = OFFICIAL_LEAD_HOURS.index(identity.lead_hours)
    if not bool((conditions.lead_indices == expected_lead_index).all().item()):
        raise ValueError("regression conditions differ from the verified lead")

    adapter, cache = prepare_residual_edm_sampler(
        residual_model,
        conditions,
        sample_ids=conditions.sample_ids,
    )
    normalized = sample_normalized_residual_ensemble(
        denoise=adapter,
        condition_cache=cache,
        sample_ids=cache.sample_ids,
        lead_hours=identity.lead_hours,
        spatial_shape=tuple(conditions.mu_z.shape[-2:]),
        ensemble_signature=identity.ensemble_signature,
        device=conditions.mu_z.device,
    )
    restored = _restore_residual_scale(normalized, identity.residual_scale)
    reported_residual, residual_audit = apply_residual_calibration_with_audit(
        identity.calibration,
        restored,
        location_key=identity.location_key,
        bias_key=identity.bias_key,
        spread_key=identity.spread_key,
    )
    transformed = conditions.mu_z[:, None] + reported_residual
    if not bool(torch.isfinite(transformed).all().item()):
        raise OverflowError("reported transformed members overflowed float32")
    physical = transformed_members_to_physical(
        conditions.mu_z, reported_residual
    )
    q_fractions = torch.stack(
        [
            (physical >= threshold).to(torch.float32).mean(dim=1)
            for threshold in OFFICIAL_Q_THRESHOLDS_MM
        ],
        dim=1,
    )
    marginal = LeadForecast(
        lead_hours=identity.lead_hours,
        ensemble_signature=identity.ensemble_signature,
        normalized_residual_members=normalized,
        restored_residual_members=restored,
        # LeadForecast predates the audit boundary. This field contains the
        # reported residual values; its audit states whether a fit or an
        # identity was actually applied.
        calibrated_residual_members=reported_residual,
        transformed_members=transformed,
        physical_members_mm=physical,
        ensemble_mean_mm=physical.mean(dim=1),
        ensemble_lower_median_mm=empirical_lower_median(physical),
        q_thresholds_mm=OFFICIAL_Q_THRESHOLDS_MM,
        q_fractions=q_fractions,
    )

    reported_p, p_audit = apply_regression_probability_with_audit(
        identity.calibration,
        conditions.probability_wet,
        key=identity.regression_probability_key,
    )
    reported_q: list[Tensor] = []
    q_audits: dict[float, CalibrationApplicationAudit] = {}
    q_key_map = identity.ensemble_probability_key_map
    for index, threshold in enumerate(OFFICIAL_Q_THRESHOLDS_MM):
        value, audit = apply_ensemble_probability_with_audit(
            identity.calibration,
            q_fractions[:, index],
            key=q_key_map[threshold],
        )
        reported_q.append(value)
        q_audits[threshold] = audit
    audit_bundle = build_calibration_application_audit_bundle(
        residual=residual_audit,
        regression_probability={A_WET_MM: p_audit},
        ensemble_probability=q_audits,
    )
    return EnsembleLeadForecastResult(
        marginal=marginal,
        raw_regression_probability_wet=conditions.probability_wet,
        reported_regression_probability_wet=reported_p,
        reported_q_fractions=torch.stack(reported_q, dim=1),
        calibration_audit=audit_bundle,
        residual_scale_audit=identity.scale_audit,
        identity_mode=identity.mode,
    )


def _forecast_from_deployment(
    *,
    identity: ForecastIdentity,
    batch: RegressionModelBatch,
    deployment: RegressionSystemOutput,
    residual: ResidualEDM,
) -> ForecastResult:
    signatures = tuple(deployment.condition_signatures)
    if signatures != (identity.condition_signature,) * batch.batch_size:
        raise RuntimeError("regression output condition signature changed")
    if not bool(
        (
            deployment.advection.lead_indices
            == OFFICIAL_LEAD_HOURS.index(identity.lead_hours)
        )
        .all()
        .item()
    ):
        raise RuntimeError("regression output lead changed")

    if not identity.diffusion_scale_supported:
        if not isinstance(identity, VerifiedForecastIdentity):
            raise RuntimeError("only a verified artifact may disable diffusion")
        if identity.residual_scale is not None:
            raise RuntimeError("unsupported residual scale unexpectedly has a value")
        reported_p, p_audit = apply_regression_probability_with_audit(
            identity.calibration,
            deployment.probability_wet,
            key=identity.regression_probability_key,
        )
        return RegressionOnlyLeadForecastResult(
            lead_hours=identity.lead_hours,
            condition_signature=identity.condition_signature,
            occurrence_logits=deployment.occurrence_logits,
            transformed_mean=deployment.mu_z,
            wet_amount=deployment.wet_amount,
            raw_regression_probability_wet=deployment.probability_wet,
            reported_regression_probability_wet=reported_p,
            probability_calibration_audit=p_audit,
            residual_scale_audit=identity.scale_audit,
            identity_mode="production",
        )

    conditions = _conditions_from_regression(batch=batch, deployment=deployment)
    return _sample_identity_lead(
        identity=identity,
        residual_model=residual,
        conditions=conditions,
    )


def _forecast_no_context(
    identity: ForecastIdentity, batch: RegressionModelBatch
) -> ForecastResult:
    regression, residual = identity.validate_model_bindings()
    _validate_identity_batch(identity, batch)
    if next(regression.parameters()).device != batch.device:
        raise ValueError("regression model and batch must share a device")
    if next(residual.parameters()).device != batch.device:
        raise ValueError("residual model and batch must share a device")

    deployment = regression(batch)
    return _forecast_from_deployment(
        identity=identity,
        batch=batch,
        deployment=deployment,
        residual=residual,
    )


def forecast_regression_diffusion_lead(
    *,
    identity: ForecastIdentity,
    batch: RegressionModelBatch,
) -> ForecastResult:
    """Run one exact artifact-bound lead with autograd and autocast disabled.

    The function is the sole public regression-to-diffusion forecast path. A
    caller cannot override the bound models, checkpoint IDs, calibration keys,
    condition signature, or scalar OOF residual scale.
    """

    if not isinstance(
        identity, (VerifiedForecastIdentity, DevelopmentForecastIdentity)
    ):
        raise TypeError(
            "identity must be a VerifiedForecastIdentity or "
            "DevelopmentForecastIdentity"
        )
    if not isinstance(batch, RegressionModelBatch):
        raise TypeError("batch must be a RegressionModelBatch")
    with torch.inference_mode(), torch.autocast(
        device_type=batch.device.type, enabled=False
    ):
        batch.validate()
        return _forecast_no_context(identity, batch)


def forecast_regression_diffusion_12_leads(
    *, request: TwelveLeadForecastRequest
) -> TwelveLeadForecastResult:
    """Forecast all official leads from one internally bound issue-time cache."""

    if not isinstance(request, TwelveLeadForecastRequest):
        raise TypeError("request must be a TwelveLeadForecastRequest")
    request.validate()
    device = request.batches[0].device
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=False
    ):
        regression: RegressionSystem | None = None
        residual: ResidualEDM | None = None
        for identity, batch in zip(
            request.identities, request.batches, strict=True
        ):
            batch.validate()
            _validate_identity_batch(identity, batch)
            selected_regression, selected_residual = identity.validate_model_bindings()
            if next(selected_regression.parameters()).device != batch.device:
                raise ValueError("regression model and batch must share a device")
            if next(selected_residual.parameters()).device != batch.device:
                raise ValueError("residual model and batch must share a device")
            if batch.device != device:
                raise ValueError("all twelve lead batches must share one device")
            if regression is None:
                regression = selected_regression
                residual = selected_residual
            elif selected_regression is not regression or selected_residual is not residual:
                raise RuntimeError("twelve-lead model bindings changed during validation")

        if regression is None or residual is None:  # pragma: no cover - fixed length.
            raise AssertionError("twelve-lead request unexpectedly contained no cells")
        validated_batches = regression._validate_issue_time_batches(request.batches)
        cache = regression._prepare_issue_time_cache(validated_batches[0])
        results: list[ForecastResult] = []
        for identity, batch in zip(
            request.identities, validated_batches, strict=True
        ):
            current_regression, current_residual = identity.validate_model_bindings()
            if current_regression is not regression or current_residual is not residual:
                raise RuntimeError("twelve-lead model binding changed before regression")
            batch.validate()
            _validate_identity_batch(identity, batch)
            deployment = regression._forward_from_issue_time_cache(batch, cache)
            batch.validate()
            _validate_identity_batch(identity, batch)
            current_regression, current_residual = identity.validate_model_bindings()
            if current_regression is not regression or current_residual is not residual:
                raise RuntimeError("twelve-lead model binding changed before sampling")
            results.append(
                _forecast_from_deployment(
                    identity=identity,
                    batch=batch,
                    deployment=deployment,
                    residual=residual,
                )
            )
            final_regression, final_residual = identity.validate_model_bindings()
            if final_regression is not regression or final_residual is not residual:
                raise RuntimeError("twelve-lead model binding changed during sampling")
        return TwelveLeadForecastResult(
            results=tuple(results),
            condition_signature=request.identities[0].condition_signature,
            identity_mode=request.identities[0].mode,
        )


__all__ = [
    "EnsembleLeadForecastResult",
    "ForecastIdentity",
    "ForecastResult",
    "RegressionOnlyLeadForecastResult",
    "TwelveLeadForecastRequest",
    "TwelveLeadForecastResult",
    "forecast_regression_diffusion_12_leads",
    "forecast_regression_diffusion_lead",
]

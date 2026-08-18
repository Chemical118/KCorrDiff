"""Forecast identities that assemble exact calibration keys in one place.

The public forecast functions operate on several calibration keys.  Building
them by hand is error prone, so this module is the single construction
boundary: a production identity derives its keys from the Stage 3 artifact
bundle and the bound models, while a development identity derives checkpoint
IDs from the bindings themselves.  Provenance objects carried by artifacts and
checkpoints are recorded as metadata and are not cross-verified here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Real
from types import MappingProxyType
from typing import Literal, Mapping

from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.inference.model_binding import (
    VerifiedRegressionModel,
    VerifiedResidualEDMModel,
)
from kcorrdiff.models.regression_system import RegressionSystem
from kcorrdiff.models.residual_edm import ResidualEDM
from kcorrdiff.training.calibration_artifact import (
    CalibrationResolver,
    EnsembleProbabilityKey,
    LocationScaleKey,
    OFFICIAL_ENSEMBLE_THRESHOLDS_MM,
    OFFICIAL_LEADS_HOURS,
    RegressionProbabilityKey,
    SamplerBiasKey,
    SpreadKey,
)
from kcorrdiff.training.config import Stage2Config
from kcorrdiff.training.crossfit import CheckpointRecord
from kcorrdiff.training.edm_sampling import EnsembleSignature
from kcorrdiff.training.stage3_data import (
    ResidualScaleRecord,
    Stage3ArtifactBundle,
    Stage3DataProvenance,
)
from kcorrdiff.training.train_stage3 import (
    EDM_VARIANTS,
    CheckpointIdentity,
    FinalModelSelectionDecision,
    Stage3Config,
)


_DEVELOPMENT_FOLD_DOMAIN = b"kcorrdiff.development-forecast-fold.v1\0"


def _forecast_cell(
    *, lead_hours: float, condition_signature: str
) -> tuple[float, str]:
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, Real):
        raise TypeError("lead_hours must be a real scalar")
    lead = float(lead_hours)
    if lead not in OFFICIAL_LEADS_HOURS:
        raise ValueError("lead_hours must be one of the 12 official half-hour leads")
    return lead, parse_condition_signature(condition_signature).key


def _fold_checkpoint_sha256s(
    bundle: Stage3ArtifactBundle,
) -> tuple[tuple[str, ...], CheckpointRecord]:
    """Collect the fold checkpoint digests and the one deployment record."""

    records = tuple(bundle.checkpoints)
    folds = tuple(
        sorted(
            (record for record in records if record.role == "fold"),
            key=lambda record: -1 if record.fold_id is None else record.fold_id,
        )
    )
    fold_ids = tuple(record.fold_id for record in folds)
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Stage 3 artifacts contain duplicate fold checkpoints")
    deployments = tuple(record for record in records if record.role == "deployment")
    if len(deployments) != 1:
        raise ValueError(
            "Stage 3 artifacts must contain exactly one deployment checkpoint"
        )
    return tuple(record.sha256 for record in folds), deployments[0]


def _decision_checkpoint_map(
    decision: FinalModelSelectionDecision,
) -> Mapping[tuple[str, int], str]:
    """Validate the decision structure and index its checkpoint identities."""

    if not isinstance(decision, FinalModelSelectionDecision):
        raise TypeError("model_selection must be a FinalModelSelectionDecision")
    checkpoints: dict[tuple[str, int], str] = {}
    for checkpoint in decision.checkpoint_sha256s:
        if not isinstance(checkpoint, CheckpointIdentity):
            raise TypeError(
                "model-selection checkpoints must be CheckpointIdentity values"
            )
        key = (checkpoint.variant, checkpoint.training_seed)
        if key in checkpoints:
            raise ValueError("model selection contains a duplicate checkpoint identity")
        checkpoints[key] = checkpoint.sha256
    if decision.selected_variant not in EDM_VARIANTS:
        raise ValueError("model selection chose an unsupported residual variant")
    return MappingProxyType(checkpoints)


def _calibration_keys(
    *,
    lead_hours: float,
    condition_signature: str,
    fold_checkpoint_sha256s: tuple[str, ...],
    deployment_checkpoint_sha256: str,
    ensemble_signature: EnsembleSignature,
) -> tuple[
    LocationScaleKey,
    SamplerBiasKey,
    SpreadKey,
    RegressionProbabilityKey,
    tuple[EnsembleProbabilityKey, ...],
]:
    location = LocationScaleKey(
        lead_hours=lead_hours,
        condition_signature=condition_signature,
        fold_checkpoint_sha256s=fold_checkpoint_sha256s,
        full_checkpoint_sha256=deployment_checkpoint_sha256,
    )
    bias = SamplerBiasKey(
        lead_hours=lead_hours,
        condition_signature=condition_signature,
        sampler_core=ensemble_signature.sampler_core,
    )
    spread = SpreadKey(
        lead_hours=lead_hours,
        condition_signature=condition_signature,
        ensemble_signature=ensemble_signature,
    )
    regression_probability = RegressionProbabilityKey(
        lead_hours=lead_hours,
        threshold_mm=A_WET_MM,
        condition_signature=condition_signature,
        regression_checkpoint_sha256=deployment_checkpoint_sha256,
    )
    ensemble_probability = tuple(
        EnsembleProbabilityKey(
            lead_hours=lead_hours,
            threshold_mm=threshold,
            condition_signature=condition_signature,
            ensemble_signature=ensemble_signature,
        )
        for threshold in OFFICIAL_ENSEMBLE_THRESHOLDS_MM
    )
    return location, bias, spread, regression_probability, ensemble_probability


class _ModelBindingAccess:
    regression_binding: VerifiedRegressionModel
    residual_binding: VerifiedResidualEDMModel
    ensemble_signature: EnsembleSignature
    location_key: LocationScaleKey
    ensemble_probability_keys: tuple[EnsembleProbabilityKey, ...]
    diffusion_scale_supported: bool

    def validate_model_bindings(self) -> tuple[RegressionSystem, ResidualEDM]:
        """Return the bound models (retained for API compatibility)."""

        return self.regression_binding.model, self.residual_binding.model

    @property
    def regression_model(self) -> RegressionSystem:
        return self.validate_model_bindings()[0]

    @property
    def residual_model(self) -> ResidualEDM:
        return self.validate_model_bindings()[1]

    @property
    def ensemble_probability_key_map(
        self,
    ) -> Mapping[float, EnsembleProbabilityKey]:
        return MappingProxyType(
            {key.threshold_mm: key for key in self.ensemble_probability_keys}
        )

    @property
    def scale_supported(self) -> bool:
        return bool(self.diffusion_scale_supported)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedForecastIdentity(_ModelBindingAccess):
    """One production lead/cell whose keys come from Stage 3 artifacts."""

    mode: Literal["production"]
    regression_binding: VerifiedRegressionModel
    residual_binding: VerifiedResidualEDMModel
    stage2_config: Stage2Config
    stage3_config: Stage3Config
    model_selection: FinalModelSelectionDecision
    stage3_data_provenance: Stage3DataProvenance
    calibration: CalibrationResolver
    calibration_semantic_sha256: str | None
    stage3_data_semantic_sha256: str
    ensemble_signature: EnsembleSignature
    lead_hours: float
    condition_signature: str
    fold_checkpoint_sha256s: tuple[str, ...]
    residual_scale: float | None
    diffusion_scale_supported: bool
    residual_scale_record: ResidualScaleRecord
    location_key: LocationScaleKey
    bias_key: SamplerBiasKey
    spread_key: SpreadKey
    regression_probability_key: RegressionProbabilityKey
    ensemble_probability_keys: tuple[EnsembleProbabilityKey, ...]

    @classmethod
    def create(
        cls,
        *,
        stage3_artifacts: Stage3ArtifactBundle,
        stage2_config: Stage2Config,
        stage3_config: Stage3Config,
        regression_binding: VerifiedRegressionModel,
        residual_binding: VerifiedResidualEDMModel,
        model_selection: FinalModelSelectionDecision,
        calibration: CalibrationResolver,
        ensemble_signature: EnsembleSignature,
        lead_hours: float,
        condition_signature: str,
    ) -> "VerifiedForecastIdentity":
        if not isinstance(stage3_artifacts, Stage3ArtifactBundle):
            raise TypeError("stage3_artifacts must be a Stage3ArtifactBundle")
        if not isinstance(stage3_artifacts.provenance, Stage3DataProvenance):
            raise TypeError("Stage 3 artifact provenance is invalid")
        if not isinstance(stage2_config, Stage2Config):
            raise TypeError("stage2_config must be a Stage2Config")
        if not isinstance(stage3_config, Stage3Config):
            raise TypeError("stage3_config must be a Stage3Config")
        if not isinstance(regression_binding, VerifiedRegressionModel):
            raise TypeError("regression_binding must be a VerifiedRegressionModel")
        if not isinstance(residual_binding, VerifiedResidualEDMModel):
            raise TypeError("residual_binding must be a VerifiedResidualEDMModel")
        if not isinstance(calibration, CalibrationResolver):
            raise TypeError("calibration must be a CalibrationResolver")
        if not isinstance(ensemble_signature, EnsembleSignature):
            raise TypeError("ensemble_signature must be an EnsembleSignature")
        _decision_checkpoint_map(model_selection)
        lead, condition = _forecast_cell(
            lead_hours=lead_hours, condition_signature=condition_signature
        )
        folds, deployment = _fold_checkpoint_sha256s(stage3_artifacts)
        provenance = stage3_artifacts.provenance
        scale_record = stage3_artifacts.residual_scale_record(
            lead_hours=lead, condition_signature=condition
        )
        if (
            scale_record.lead_hours != lead
            or scale_record.condition_signature != condition
        ):
            raise ValueError("residual-scale record does not match the forecast cell")
        supported = not scale_record.diffusion_scale_unsupported
        if supported != (scale_record.scale is not None):
            raise ValueError("residual-scale support flag and value disagree")
        if supported and (
            isinstance(scale_record.scale, bool)
            or not isinstance(scale_record.scale, Real)
            or not math.isfinite(float(scale_record.scale))
            or float(scale_record.scale) <= 0.0
        ):
            raise ValueError("supported residual scale must be finite and positive")

        location, bias, spread, regression_probability, ensemble_probability = (
            _calibration_keys(
                lead_hours=lead,
                condition_signature=condition,
                fold_checkpoint_sha256s=folds,
                deployment_checkpoint_sha256=deployment.sha256,
                ensemble_signature=ensemble_signature,
            )
        )
        artifact = calibration.artifact
        if artifact is not None and not calibration.development_mode:
            # Look up the exact requested cell now so a genuinely missing
            # calibration record fails here rather than mid-forecast.
            calibration.location(location)
            calibration.bias(bias)
            calibration.gamma(spread)
            regression_probability_digests = {
                record.key.semantic_sha256
                for record in artifact.regression_probability
            }
            if (
                regression_probability.semantic_sha256
                not in regression_probability_digests
            ):
                raise KeyError("calibration artifact has no p_cal forecast cell")
            ensemble_probability_digests = {
                record.key.semantic_sha256 for record in artifact.ensemble_probability
            }
            for key in ensemble_probability:
                if key.semantic_sha256 not in ensemble_probability_digests:
                    raise KeyError("calibration artifact has no q_cal forecast cell")

        values = {
            "mode": "production",
            "regression_binding": regression_binding,
            "residual_binding": residual_binding,
            "stage2_config": stage2_config,
            "stage3_config": stage3_config,
            "model_selection": model_selection,
            "stage3_data_provenance": provenance,
            "calibration": calibration,
            "calibration_semantic_sha256": (
                None if artifact is None else artifact.semantic_sha256
            ),
            "stage3_data_semantic_sha256": provenance.semantic_sha256,
            "ensemble_signature": ensemble_signature,
            "lead_hours": lead,
            "condition_signature": condition,
            "fold_checkpoint_sha256s": folds,
            "residual_scale": scale_record.scale,
            "diffusion_scale_supported": supported,
            "residual_scale_record": scale_record,
            "location_key": location,
            "bias_key": bias,
            "spread_key": spread,
            "regression_probability_key": regression_probability,
            "ensemble_probability_keys": ensemble_probability,
        }
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance

    from_verified_artifacts = create

    @property
    def scale_audit(self) -> ResidualScaleRecord:
        return self.residual_scale_record


@dataclass(frozen=True, slots=True)
class DevelopmentScaleAudit:
    """Explicit, unpublished residual scale accepted only in development mode."""

    lead_hours: float
    condition_signature: str
    scale: float
    source: Literal["explicit_development_scalar"] = "explicit_development_scalar"


def _development_fold_ids(
    regression_checkpoint_sha256: str, *, fold_count: int
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            _DEVELOPMENT_FOLD_DOMAIN
            + regression_checkpoint_sha256.encode("ascii")
            + fold_id.to_bytes(1, "big")
        ).hexdigest()
        for fold_id in range(fold_count)
    )


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentForecastIdentity(_ModelBindingAccess):
    """Clearly tagged, unpublished identity for local research inference.

    Any model bindings may be used, including checkpoint-loaded ones; the
    identity simply derives its keys from the bindings' checkpoint IDs.
    """

    mode: Literal["development"]
    regression_binding: VerifiedRegressionModel
    residual_binding: VerifiedResidualEDMModel
    calibration: CalibrationResolver
    ensemble_signature: EnsembleSignature
    lead_hours: float
    condition_signature: str
    fold_checkpoint_sha256s: tuple[str, ...]
    residual_scale: float
    diffusion_scale_supported: Literal[True]
    scale_audit: DevelopmentScaleAudit
    location_key: LocationScaleKey
    bias_key: SamplerBiasKey
    spread_key: SpreadKey
    regression_probability_key: RegressionProbabilityKey
    ensemble_probability_keys: tuple[EnsembleProbabilityKey, ...]

    @classmethod
    def create(
        cls,
        *,
        regression_binding: VerifiedRegressionModel,
        residual_binding: VerifiedResidualEDMModel,
        calibration: CalibrationResolver,
        ensemble_signature: EnsembleSignature,
        lead_hours: float,
        condition_signature: str,
        residual_scale: float,
        fold_count: int = 3,
    ) -> "DevelopmentForecastIdentity":
        if not isinstance(regression_binding, VerifiedRegressionModel):
            raise TypeError("regression_binding must be a VerifiedRegressionModel")
        if not isinstance(residual_binding, VerifiedResidualEDMModel):
            raise TypeError("residual_binding must be a VerifiedResidualEDMModel")
        if not isinstance(calibration, CalibrationResolver):
            raise TypeError("calibration must be a CalibrationResolver")
        if not isinstance(ensemble_signature, EnsembleSignature):
            raise TypeError("ensemble_signature must be an EnsembleSignature")
        lead, condition = _forecast_cell(
            lead_hours=lead_hours, condition_signature=condition_signature
        )
        if isinstance(residual_scale, bool) or not isinstance(residual_scale, Real):
            raise TypeError("development residual scale must be a real scalar")
        scale = float(residual_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("development residual scale must be finite and positive")
        if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
            raise ValueError("development fold_count must be an integer of at least two")

        # The caller chooses only a declared member/step profile.  Its
        # checkpoint label is discarded and replaced by the binding's ID.
        derived_ensemble = EnsembleSignature(
            sampler_core=replace(
                ensemble_signature.sampler_core,
                checkpoint_id=residual_binding.checkpoint_sha256,
            ),
            member_count=ensemble_signature.member_count,
        )
        folds = _development_fold_ids(
            regression_binding.checkpoint_sha256, fold_count=fold_count
        )
        location, bias, spread, regression_probability, ensemble_probability = (
            _calibration_keys(
                lead_hours=lead,
                condition_signature=condition,
                fold_checkpoint_sha256s=folds,
                deployment_checkpoint_sha256=regression_binding.checkpoint_sha256,
                ensemble_signature=derived_ensemble,
            )
        )
        audit = DevelopmentScaleAudit(lead, condition, scale)
        values = {
            "mode": "development",
            "regression_binding": regression_binding,
            "residual_binding": residual_binding,
            "calibration": calibration,
            "ensemble_signature": derived_ensemble,
            "lead_hours": lead,
            "condition_signature": condition,
            "fold_checkpoint_sha256s": folds,
            "residual_scale": scale,
            "diffusion_scale_supported": True,
            "scale_audit": audit,
            "location_key": location,
            "bias_key": bias,
            "spread_key": spread,
            "regression_probability_key": regression_probability,
            "ensemble_probability_keys": ensemble_probability,
        }
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance

    from_development_bindings = create


def build_verified_forecast_identity(
    **kwargs: object,
) -> VerifiedForecastIdentity:
    """Named production factory; see :meth:`VerifiedForecastIdentity.create`."""

    return VerifiedForecastIdentity.create(**kwargs)  # type: ignore[arg-type]


def build_development_forecast_identity(
    **kwargs: object,
) -> DevelopmentForecastIdentity:
    """Named development factory; see :meth:`DevelopmentForecastIdentity.create`."""

    return DevelopmentForecastIdentity.create(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "DevelopmentForecastIdentity",
    "DevelopmentScaleAudit",
    "VerifiedForecastIdentity",
    "build_development_forecast_identity",
    "build_verified_forecast_identity",
]

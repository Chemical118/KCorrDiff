"""Artifact-derived identities for calibrated production inference.

The public forecast functions operate on several exact calibration keys.  It
is unsafe for a caller to assemble those keys from checkpoint labels or SHA
strings: a syntactically valid key can still join artifacts from different
training lineages.  This module provides the single construction boundary for
those keys.  Production identities are derived only from already-verified
Stage 2/3 objects, and development identities are conspicuously separate and
derive their checkpoint IDs from captured model state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
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
from kcorrdiff.models.residual_edm import ResidualEDM, ResidualEDMConfig
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
from kcorrdiff.training.checkpoints import CheckpointProvenance
from kcorrdiff.training.config import Stage2Config, thaw_config_mapping
from kcorrdiff.training.crossfit import CheckpointRecord, checkpoint_set_hash
from kcorrdiff.training.edm_sampling import EnsembleSignature
from kcorrdiff.training.stage3_data import (
    ResidualScaleRecord,
    Stage3ArtifactBundle,
    Stage3DataProvenance,
)
from kcorrdiff.training.runtime import contract_from_mapping
from kcorrdiff.training.train_stage3 import (
    DEPLOYMENT_TRAINING_SEED,
    EDM_VARIANTS,
    REPEAT_TRAINING_SEEDS,
    CheckpointIdentity,
    FinalModelSelectionDecision,
    Stage3Config,
    Stage3CheckpointProvenance,
    stage3_architecture_sha256,
)
from kcorrdiff.training.train_stage2 import regression_system_config_from_stage2


_DEVELOPMENT_FOLD_DOMAIN = b"kcorrdiff.development-forecast-fold.v1\0"


def _config_raw_sha256(raw: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            thaw_config_mapping(raw),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _forecast_cell(
    *, lead_hours: float, condition_signature: str
) -> tuple[float, str]:
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, Real):
        raise TypeError("lead_hours must be a real scalar")
    lead = float(lead_hours)
    if lead not in OFFICIAL_LEADS_HOURS:
        raise ValueError("lead_hours must be one of the 12 official half-hour leads")
    return lead, parse_condition_signature(condition_signature).key


def _require_production_regression(
    binding: VerifiedRegressionModel,
    *,
    provenance: Stage3DataProvenance,
    deployment_record: CheckpointRecord,
) -> None:
    if not isinstance(binding, VerifiedRegressionModel):
        raise TypeError("regression_binding must be a VerifiedRegressionModel")
    if not binding.is_production or not isinstance(
        binding.provenance, CheckpointProvenance
    ):
        raise ValueError("production forecast requires a production regression binding")
    binding.validate()
    if (
        binding.checkpoint_sha256 != provenance.deployment_checkpoint_sha256
        or deployment_record.sha256 != provenance.deployment_checkpoint_sha256
    ):
        raise ValueError(
            "regression checkpoint is not the Stage 3 deployment checkpoint"
        )

    checkpoint = binding.provenance
    expected = {
        "protocol_version": provenance.protocol_version,
        "config_sha256": provenance.stage2_config_sha256,
        "draw_manifest_sha256": provenance.draw_manifest_sha256,
        "launch_identity_sha256": provenance.stage2_launch_identity_sha256,
        "source_tree_sha256": provenance.stage2_source_tree_sha256,
        "container_image_sha256": provenance.stage2_container_image_sha256,
        "runtime_report_sha256": provenance.stage2_runtime_report_sha256,
        "data_contract_sha256": provenance.stage2_data_contract_sha256,
    }
    if any(getattr(checkpoint, name) != value for name, value in expected.items()):
        raise ValueError("regression checkpoint Stage 2 lineage mismatch")
    if checkpoint.role != "deployment" or checkpoint.fold_id is not None:
        raise ValueError(
            "regression checkpoint provenance is not deployment provenance"
        )


def _fold_checkpoint_sha256s(
    bundle: Stage3ArtifactBundle,
) -> tuple[tuple[str, ...], CheckpointRecord]:
    records = tuple(bundle.checkpoints)
    folds = tuple(
        sorted(
            (record for record in records if record.role == "fold"),
            key=lambda record: -1 if record.fold_id is None else record.fold_id,
        )
    )
    if tuple(record.fold_id for record in folds) != (0, 1, 2):
        raise ValueError("Stage 3 artifacts must contain exactly folds 0, 1, and 2")
    deployments = tuple(record for record in records if record.role == "deployment")
    if len(deployments) != 1:
        raise ValueError(
            "Stage 3 artifacts must contain exactly one deployment checkpoint"
        )

    provenance = bundle.provenance
    if checkpoint_set_hash(folds) != provenance.fold_checkpoint_set_sha256:
        raise ValueError("fold checkpoint set no longer matches Stage 3 provenance")
    for record in (*folds, deployments[0]):
        if (
            record.config_sha256 != provenance.stage2_config_sha256
            or record.draw_manifest_sha256 != provenance.draw_manifest_sha256
        ):
            raise ValueError("regression checkpoint record Stage 2 lineage mismatch")
    return tuple(record.sha256 for record in folds), deployments[0]


def _decision_checkpoint_map(
    decision: FinalModelSelectionDecision,
) -> Mapping[tuple[str, int], str]:
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
    required = {
        (variant, seed)
        for variant in EDM_VARIANTS
        for seed in REPEAT_TRAINING_SEEDS
    }
    if set(checkpoints) != required:
        raise ValueError("model selection is not bound to the exact A/B three-seed set")
    if decision.selected_variant not in EDM_VARIANTS:
        raise ValueError("model selection chose an unsupported residual variant")
    return MappingProxyType(checkpoints)


def _require_production_residual(
    binding: VerifiedResidualEDMModel,
    *,
    provenance: Stage3DataProvenance,
    decision: FinalModelSelectionDecision,
) -> None:
    if not isinstance(binding, VerifiedResidualEDMModel):
        raise TypeError("residual_binding must be a VerifiedResidualEDMModel")
    if not binding.is_production or not isinstance(
        binding.provenance, Stage3CheckpointProvenance
    ):
        raise ValueError("production forecast requires a production residual binding")
    binding.validate()

    selected = _decision_checkpoint_map(decision)
    expected_checkpoint = selected[
        (decision.selected_variant, DEPLOYMENT_TRAINING_SEED)
    ]
    checkpoint = binding.provenance
    if (
        binding.variant != decision.selected_variant
        or checkpoint.variant != decision.selected_variant
        or checkpoint.training_seed != DEPLOYMENT_TRAINING_SEED
    ):
        raise ValueError("residual binding is not the selected deployment candidate")
    if binding.checkpoint_sha256 != expected_checkpoint:
        raise ValueError(
            "residual checkpoint is not the selected seed-11103 checkpoint"
        )

    expected = {
        "protocol_version": provenance.protocol_version,
        "stage3_data_sha256": provenance.semantic_sha256,
        "stage2_deployment_checkpoint_sha256": (
            provenance.deployment_checkpoint_sha256
        ),
        "stage2_launch_identity_sha256": provenance.stage2_launch_identity_sha256,
        "stage2_source_tree_sha256": provenance.stage2_source_tree_sha256,
        "stage2_container_image_sha256": provenance.stage2_container_image_sha256,
        "stage2_runtime_report_sha256": provenance.stage2_runtime_report_sha256,
        "stage2_data_contract_sha256": provenance.stage2_data_contract_sha256,
        "draw_manifest_sha256": provenance.draw_manifest_sha256,
    }
    if any(getattr(checkpoint, name) != value for name, value in expected.items()):
        raise ValueError("residual checkpoint Stage 2/3 lineage mismatch")


def _require_model_configurations(
    *,
    stage2_config: Stage2Config,
    stage3_config: Stage3Config,
    provenance: Stage3DataProvenance,
    regression_binding: VerifiedRegressionModel,
    residual_binding: VerifiedResidualEDMModel,
    decision: FinalModelSelectionDecision,
    condition_signature: str,
) -> None:
    """Bind non-state architecture/runtime choices to verified config bytes."""

    if not isinstance(stage2_config, Stage2Config):
        raise TypeError("stage2_config must be a verified Stage2Config")
    if not isinstance(stage3_config, Stage3Config):
        raise TypeError("stage3_config must be a verified Stage3Config")
    if _config_raw_sha256(stage2_config.raw) != stage2_config.sha256:
        raise ValueError("Stage 2 config is not bound to its canonical raw bytes")
    if _config_raw_sha256(stage3_config.raw) != stage3_config.sha256:
        raise ValueError("Stage 3 config is not bound to its canonical raw bytes")
    if stage2_config.sha256 != provenance.stage2_config_sha256:
        raise ValueError("Stage 2 config does not match Stage 3 data provenance")
    stage2_runtime = stage2_config.raw.get("runtime")
    if not isinstance(stage2_runtime, Mapping) or (
        contract_from_mapping(stage2_runtime) != stage2_config.runtime
    ):
        raise ValueError("typed Stage 2 runtime differs from its verified raw config")
    expected_regression = regression_system_config_from_stage2(stage2_config).system
    if regression_binding.model.config != expected_regression:
        raise ValueError("regression model configuration differs from verified Stage 2")

    checkpoint = residual_binding.provenance
    if not isinstance(checkpoint, Stage3CheckpointProvenance):
        raise TypeError("residual binding lacks Stage 3 checkpoint provenance")
    if stage3_config.sha256 != checkpoint.config_sha256:
        raise ValueError("Stage 3 config does not match residual checkpoint provenance")
    stage3_runtime = stage3_config.raw.get("runtime")
    stage3_model = stage3_config.raw.get("model")
    stage3_data = stage3_config.raw.get("data")
    if not all(
        isinstance(value, Mapping)
        for value in (stage3_runtime, stage3_model, stage3_data)
    ):
        raise TypeError("verified Stage 3 raw config sections must be mappings")
    assert isinstance(stage3_runtime, Mapping)
    assert isinstance(stage3_model, Mapping)
    assert isinstance(stage3_data, Mapping)
    typed_stage3_contract = (
        tuple(stage3_runtime.get("target_widths", ())),
        tuple(stage3_runtime.get("context_widths", ())),
        stage3_runtime.get("target_grid_size"),
        stage3_model.get("activation_checkpoint"),
        stage3_model.get("sigma_data"),
        stage3_model.get("physical_attention_query_chunk_size"),
        tuple(stage3_data.get("condition_signatures", ())),
    )
    if typed_stage3_contract != (
        stage3_config.target_widths,
        stage3_config.context_widths,
        stage3_config.input_size,
        stage3_config.activation_checkpoint,
        stage3_config.sigma_data,
        stage3_config.query_chunk_size,
        stage3_config.condition_signatures,
    ):
        raise ValueError("typed Stage 3 fields differ from the verified raw config")
    expected_residual = ResidualEDMConfig(
        variant=decision.selected_variant,  # type: ignore[arg-type]
        target_widths=stage3_config.target_widths,
        context_widths=stage3_config.context_widths,
        input_size=stage3_config.input_size,
        sigma_data=stage3_config.sigma_data,
        query_chunk_size=stage3_config.query_chunk_size,
        activation_checkpoint=stage3_config.activation_checkpoint,
    )
    if residual_binding.model.config != expected_residual:
        raise ValueError("residual model configuration differs from verified Stage 3")
    if (
        stage3_architecture_sha256(
            stage3_config, variant=decision.selected_variant
        )
        != decision.architecture_sha256
    ):
        raise ValueError("selected architecture hash does not match Stage 3 config")
    if condition_signature not in stage3_config.condition_signatures:
        raise ValueError("forecast condition is outside the verified Stage 3 config")


def _require_complete_calibration(
    calibration: CalibrationResolver,
    *,
    decision: FinalModelSelectionDecision,
    provenance: Stage3DataProvenance,
    fold_checkpoint_sha256s: tuple[str, ...],
    deployment_checkpoint_sha256: str,
    ensemble_signature: EnsembleSignature,
    condition_signature: str,
) -> str:
    if not isinstance(calibration, CalibrationResolver):
        raise TypeError("calibration must be a CalibrationResolver")
    artifact = calibration.artifact
    if (
        calibration.development_mode
        or artifact is None
        or artifact.release_status != "complete"
        or artifact.calibration_absent_identity
        or artifact.coverage is None
    ):
        raise ValueError("production forecast requires complete-release calibration")

    frozen_decision = decision.frozen_calibration_decision()
    if (
        calibration.model_selection != frozen_decision
        or artifact.model_selection != frozen_decision
    ):
        raise ValueError(
            "calibration is not bound to the final model-selection decision"
        )
    coverage = artifact.coverage
    if (
        coverage.fold_checkpoint_sha256s != fold_checkpoint_sha256s
        or coverage.full_checkpoint_sha256 != deployment_checkpoint_sha256
    ):
        raise ValueError("calibration regression checkpoint coverage mismatch")
    if condition_signature not in coverage.condition_signatures:
        raise ValueError("condition signature is outside calibration coverage")
    if ensemble_signature not in coverage.ensemble_signatures:
        raise ValueError("ensemble signature is outside calibration coverage")
    if (
        artifact.provenance.mapping.get("residual_scales_sha256")
        != provenance.residual_scales_sha256
    ):
        raise ValueError("calibration residual-scale provenance mismatch")
    return artifact.semantic_sha256


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
        """Revalidate immutable tensor captures before every forecast call."""

        regression = self.regression_binding.validate()
        residual = self.residual_binding.validate()
        if (
            self.regression_binding.checkpoint_sha256
            != self.location_key.full_checkpoint_sha256
            or self.residual_binding.checkpoint_sha256
            != self.ensemble_signature.sampler_core.checkpoint_id
        ):
            raise RuntimeError("forecast identity/model checkpoint binding changed")
        return regression, residual

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
    """One production lead/cell bound across Stage 2, Stage 3, and calibration."""

    mode: Literal["production"]
    regression_binding: VerifiedRegressionModel
    residual_binding: VerifiedResidualEDMModel
    stage2_config: Stage2Config
    stage3_config: Stage3Config
    model_selection: FinalModelSelectionDecision
    stage3_data_provenance: Stage3DataProvenance
    calibration: CalibrationResolver
    calibration_semantic_sha256: str
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
        if not isinstance(ensemble_signature, EnsembleSignature):
            raise TypeError("ensemble_signature must be an EnsembleSignature")
        lead, condition = _forecast_cell(
            lead_hours=lead_hours, condition_signature=condition_signature
        )
        folds, deployment = _fold_checkpoint_sha256s(stage3_artifacts)
        provenance = stage3_artifacts.provenance
        _require_production_regression(
            regression_binding,
            provenance=provenance,
            deployment_record=deployment,
        )
        _require_production_residual(
            residual_binding,
            provenance=provenance,
            decision=model_selection,
        )
        _require_model_configurations(
            stage2_config=stage2_config,
            stage3_config=stage3_config,
            provenance=provenance,
            regression_binding=regression_binding,
            residual_binding=residual_binding,
            decision=model_selection,
            condition_signature=condition,
        )
        if (
            ensemble_signature.sampler_core.checkpoint_id
            != residual_binding.checkpoint_sha256
        ):
            raise ValueError("ensemble sampler is not bound to the residual checkpoint")
        calibration_sha256 = _require_complete_calibration(
            calibration,
            decision=model_selection,
            provenance=provenance,
            fold_checkpoint_sha256s=folds,
            deployment_checkpoint_sha256=deployment.sha256,
            ensemble_signature=ensemble_signature,
            condition_signature=condition,
        )
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
        # A complete artifact validates global coverage at load time; these
        # lookups additionally prove the exact requested cell before inference.
        calibration.location(location)
        calibration.bias(bias)
        calibration.gamma(spread)
        artifact = calibration.artifact
        assert artifact is not None
        regression_probability_digests = {
            record.key.semantic_sha256 for record in artifact.regression_probability
        }
        if regression_probability.semantic_sha256 not in regression_probability_digests:
            raise KeyError("complete calibration lost the exact p_cal forecast cell")
        ensemble_probability_digests = {
            record.key.semantic_sha256 for record in artifact.ensemble_probability
        }
        for key in ensemble_probability:
            if key.semantic_sha256 not in ensemble_probability_digests:
                raise KeyError("complete calibration lost an exact q_cal forecast cell")

        values = {
            "mode": "production",
            "regression_binding": regression_binding,
            "residual_binding": residual_binding,
            "stage2_config": stage2_config,
            "stage3_config": stage3_config,
            "model_selection": model_selection,
            "stage3_data_provenance": provenance,
            "calibration": calibration,
            "calibration_semantic_sha256": calibration_sha256,
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

    def validate_model_bindings(self) -> tuple[RegressionSystem, ResidualEDM]:
        models = _ModelBindingAccess.validate_model_bindings(self)
        _require_model_configurations(
            stage2_config=self.stage2_config,
            stage3_config=self.stage3_config,
            provenance=self.stage3_data_provenance,
            regression_binding=self.regression_binding,
            residual_binding=self.residual_binding,
            decision=self.model_selection,
            condition_signature=self.condition_signature,
        )
        return models


@dataclass(frozen=True, slots=True)
class DevelopmentScaleAudit:
    """Explicit, unpublished residual scale accepted only in development mode."""

    lead_hours: float
    condition_signature: str
    scale: float
    source: Literal["explicit_development_scalar"] = "explicit_development_scalar"


def _development_fold_ids(regression_checkpoint_sha256: str) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            _DEVELOPMENT_FOLD_DOMAIN
            + regression_checkpoint_sha256.encode("ascii")
            + fold_id.to_bytes(1, "big")
        ).hexdigest()
        for fold_id in range(3)
    )


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentForecastIdentity(_ModelBindingAccess):
    """Clearly tagged, unpublished identity for state-bound local inference."""

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
    ) -> "DevelopmentForecastIdentity":
        if not isinstance(regression_binding, VerifiedRegressionModel):
            raise TypeError("regression_binding must be a VerifiedRegressionModel")
        if not isinstance(residual_binding, VerifiedResidualEDMModel):
            raise TypeError("residual_binding must be a VerifiedResidualEDMModel")
        if regression_binding.is_production or residual_binding.is_production:
            raise ValueError(
                "development identity cannot accept production model bindings"
            )
        if (
            regression_binding.checkpoint_sha256
            != regression_binding._capture.state_sha256
            or residual_binding.checkpoint_sha256
            != residual_binding._capture.state_sha256
        ):
            raise ValueError(
                "development checkpoint IDs must be derived from model state"
            )
        regression_binding.validate()
        residual_binding.validate()
        if not isinstance(calibration, CalibrationResolver):
            raise TypeError("calibration must be a CalibrationResolver")
        artifact = calibration.artifact
        if (
            not calibration.development_mode
            or calibration.model_selection is None
            or (
                artifact is not None
                and (
                    artifact.release_status != "development"
                    or not artifact.calibration_absent_identity
                )
            )
        ):
            raise ValueError(
                "development identity requires absent development calibration"
            )
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

        # The caller chooses only a declared member/step profile.  Its
        # checkpoint label is discarded and replaced by captured model state.
        derived_ensemble = EnsembleSignature(
            sampler_core=replace(
                ensemble_signature.sampler_core,
                checkpoint_id=residual_binding.checkpoint_sha256,
            ),
            member_count=ensemble_signature.member_count,
        )
        folds = _development_fold_ids(regression_binding.checkpoint_sha256)
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

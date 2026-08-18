from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcorrdiff.inference.identity import (
    DevelopmentForecastIdentity,
    VerifiedForecastIdentity,
)
from kcorrdiff.inference.model_binding import (
    VerifiedRegressionModel,
    VerifiedResidualEDMModel,
    bind_development_regression_model,
    bind_development_residual_edm_model,
)
from kcorrdiff.models.regression_system import RegressionSystem, RegressionSystemConfig
from kcorrdiff.models.residual_edm import ResidualEDM, ResidualEDMConfig
from kcorrdiff.training.calibration import (
    IndependentBlockSupport,
    LocationScaleCalibration,
    POOLING_ORDER,
    ProbabilityCalibration,
)
from kcorrdiff.training.calibration_artifact import (
    CalibrationArtifact,
    CalibrationCoverage,
    CalibrationProvenance,
    CalibrationResolver,
    EnsembleProbabilityRecord,
    FrozenModelSelectionDecision,
    LocationScaleRecord,
    PoolingAudit,
    PoolingLevelAudit,
    RegressionProbabilityRecord,
    SamplerBiasRecord,
    SpreadRecord,
)
from kcorrdiff.training.checkpoints import CheckpointProvenance
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.crossfit import CheckpointRecord, checkpoint_set_hash
from kcorrdiff.training.edm_sampling import (
    EnsembleSignature,
    SamplerCoreSignature,
)
from kcorrdiff.training.stage3_data import (
    PROTOCOL_VERSION,
    STAGE2_TARGET_BUILDER_VERSION,
    STAGE3_DATA_FORMAT,
    ResidualScaleRecord,
    Stage3ArtifactBundle,
    Stage3DataProvenance,
    _semantic_sha256,
)
from kcorrdiff.training.train_stage3 import (
    DEPLOYMENT_TRAINING_SEED,
    REPEAT_TRAINING_SEEDS,
    CheckpointIdentity,
    FinalModelSelectionDecision,
    Stage3CheckpointProvenance,
    load_stage3_config,
    stage3_architecture_sha256,
)
from kcorrdiff.training.train_stage2 import regression_system_config_from_stage2


CONDITION = "era5_oracle:era=1:tp=1:full_trajectory"
TARGET_WIDTHS = (4, 8, 12, 16, 20)
CONTEXT_WIDTHS = (4, 8, 12, 16, 20)
REPOSITORY = Path(__file__).resolve().parents[2]
STAGE2_CONFIG = load_stage2_config(REPOSITORY / "configs" / "stage2-full-width.yaml")
STAGE3_CONFIG = load_stage3_config(REPOSITORY / "configs" / "stage3-full-width.yaml")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


FOLD_HASHES = tuple(_digest(f"fold-{index}") for index in range(3))
DEPLOYMENT_HASH = _digest("stage2-deployment")
RESIDUAL_HASH = _digest("stage3-edm-a-11103")
DECISION_HASH = _digest("final-selection")
ARCHITECTURE_HASH = stage3_architecture_sha256(STAGE3_CONFIG, variant="edm_a")
RESIDUAL_SCALES_HASH = _digest("residual-scales")
DRAW_HASH = _digest("draw-manifest")
STAGE2_CONFIG_HASH = STAGE2_CONFIG.sha256
LAUNCH_HASH = _digest("stage2-launch")
SOURCE_HASH = _digest("stage2-source")
CONTAINER_HASH = _digest("stage2-container")
RUNTIME_HASH = _digest("stage2-runtime")
DATA_HASH = _digest("stage2-data")


def _regression_model() -> RegressionSystem:
    return RegressionSystem(
        RegressionSystemConfig(
            target_widths=TARGET_WIDTHS,
            context_widths=CONTEXT_WIDTHS,
            input_size=16,
            era_stem_channels=8,
            era_spatial_blocks=0,
            regression_query_chunk_size=2,
            allow_test_override=True,
        )
    )


def _residual_model() -> ResidualEDM:
    return ResidualEDM(
        ResidualEDMConfig(
            variant="edm_a",
            target_widths=TARGET_WIDTHS,
            context_widths=CONTEXT_WIDTHS,
            input_size=16,
            query_chunk_size=2,
            allow_test_override=True,
        )
    )


def _production_configured_models() -> tuple[RegressionSystem, ResidualEDM]:
    regression = _regression_model()
    regression.config = regression_system_config_from_stage2(STAGE2_CONFIG).system
    residual = _residual_model()
    residual.config = ResidualEDMConfig(
        variant="edm_a",
        target_widths=STAGE3_CONFIG.target_widths,
        context_widths=STAGE3_CONFIG.context_widths,
        input_size=STAGE3_CONFIG.input_size,
        sigma_data=STAGE3_CONFIG.sigma_data,
        query_chunk_size=STAGE3_CONFIG.query_chunk_size,
        activation_checkpoint=STAGE3_CONFIG.activation_checkpoint,
    )
    return regression, residual


def _checkpoint_records() -> tuple[CheckpointRecord, ...]:
    common = {
        "training_blocks_sha256": _digest("training-blocks"),
        "config_sha256": STAGE2_CONFIG_HASH,
        "draw_manifest_sha256": DRAW_HASH,
        "global_step": 10,
    }
    return tuple(
        CheckpointRecord(
            fold_id=fold_id,
            role="fold",
            path=f"/verified/fold-{fold_id}.pt",
            sha256=FOLD_HASHES[fold_id],
            **common,
        )
        for fold_id in range(3)
    ) + (
        CheckpointRecord(
            fold_id=None,
            role="deployment",
            path="/verified/deployment.pt",
            sha256=DEPLOYMENT_HASH,
            **common,
        ),
    )


def _stage3_provenance(records: tuple[CheckpointRecord, ...]) -> Stage3DataProvenance:
    folds = tuple(record for record in records if record.role == "fold")
    values: dict[str, object] = {
        "format_version": STAGE3_DATA_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "stage2_manifest_sha256": _digest("stage2-manifest"),
        "stage2_config_sha256": STAGE2_CONFIG_HASH,
        "stage2_launch_identity_sha256": LAUNCH_HASH,
        "stage2_source_tree_sha256": SOURCE_HASH,
        "stage2_container_image_sha256": CONTAINER_HASH,
        "stage2_runtime_report_sha256": RUNTIME_HASH,
        "stage2_data_contract_sha256": DATA_HASH,
        "stage2_artifact_hashes": (("draw_manifest", DRAW_HASH),),
        "draw_manifest_sha256": DRAW_HASH,
        "oof_manifest_sha256": _digest("oof-manifest"),
        "residual_scales_sha256": RESIDUAL_SCALES_HASH,
        "fold_checkpoint_set_sha256": checkpoint_set_hash(folds),
        "deployment_checkpoint_sha256": DEPLOYMENT_HASH,
        "target_builder_version": STAGE2_TARGET_BUILDER_VERSION,
        "grid_shape": (16, 16),
    }
    semantic = {
        **values,
        "stage2_artifact_hashes": [
            list(item) for item in values["stage2_artifact_hashes"]
        ],
        "grid_shape": list(values["grid_shape"]),
    }
    values["semantic_sha256"] = _semantic_sha256(semantic)
    return Stage3DataProvenance(**values)  # type: ignore[arg-type]


def _scale_record(*, unsupported: bool = False) -> ResidualScaleRecord:
    return ResidualScaleRecord(
        lead_hours=1.0,
        condition_signature=CONDITION,
        scale=None if unsupported else 0.75,
        raw_rms=None if unsupported else 0.75,
        epsilon_applied=False,
        items=100,
        valid_pixels=1000,
        importance_weight_sum=500.0,
        pooling_level=None if unsupported else "full_cell",
        pooling_ladder=(),
        terminal_fallback=unsupported,
        diffusion_scale_unsupported=unsupported,
    )


def _bundle(*, unsupported: bool = False) -> Stage3ArtifactBundle:
    records = _checkpoint_records()
    scale = _scale_record(unsupported=unsupported)
    bundle = object.__new__(Stage3ArtifactBundle)
    bundle.rows = ()
    bundle.oof = None
    bundle.scales = (scale,)
    bundle.provenance = _stage3_provenance(records)
    bundle.checkpoints = records
    bundle._scale_lookup = {(1.0, CONDITION): scale}
    return bundle


def _regression_provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=STAGE2_CONFIG_HASH,
        draw_manifest_sha256=DRAW_HASH,
        launch_identity_sha256=LAUNCH_HASH,
        source_tree_sha256=SOURCE_HASH,
        container_image_sha256=CONTAINER_HASH,
        runtime_report_sha256=RUNTIME_HASH,
        data_contract_sha256=DATA_HASH,
        role="deployment",
        fold_id=None,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )


def _residual_provenance(stage3_data_sha256: str) -> Stage3CheckpointProvenance:
    return Stage3CheckpointProvenance(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=STAGE3_CONFIG.sha256,
        stage3_data_sha256=stage3_data_sha256,
        stage2_deployment_checkpoint_sha256=DEPLOYMENT_HASH,
        stage2_launch_identity_sha256=LAUNCH_HASH,
        stage2_source_tree_sha256=SOURCE_HASH,
        stage2_container_image_sha256=CONTAINER_HASH,
        stage2_runtime_report_sha256=RUNTIME_HASH,
        stage2_data_contract_sha256=DATA_HASH,
        stage3_source_tree_sha256=_digest("stage3-source"),
        container_image_sha256=_digest("stage3-container"),
        runtime_report_sha256=_digest("stage3-runtime"),
        draw_manifest_sha256=DRAW_HASH,
        plan_sha256=_digest("stage3-plan"),
        variant="edm_a",
        training_seed=DEPLOYMENT_TRAINING_SEED,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
        wandb_mode="disabled",
        wandb_run_id="identity-test",
    )


def _selection() -> FinalModelSelectionDecision:
    checkpoints = []
    for variant in ("edm_a", "edm_b"):
        for seed in REPEAT_TRAINING_SEEDS:
            digest = (
                RESIDUAL_HASH
                if (variant, seed) == ("edm_a", DEPLOYMENT_TRAINING_SEED)
                else _digest(f"{variant}-{seed}")
            )
            checkpoints.append(CheckpointIdentity(variant, seed, digest))
    return FinalModelSelectionDecision(
        path=Path("/verified/final-selection.json"),
        file_sha256=DECISION_HASH,
        evaluation_result_sha256=_digest("evaluation-result"),
        finalist_training_manifest_sha256=_digest("finalist-manifest"),
        checkpoint_sha256s=tuple(checkpoints),
        selected_variant="edm_a",
        architecture_sha256=ARCHITECTURE_HASH,
        d_enabled=False,
        probability_mapping_family="monotone_logit_linear",
        pooling_order=POOLING_ORDER,
        evaluator_id="external-test-evaluator",
    )


def _terminal_audit(*, probability: bool) -> PoolingAudit:
    support = IndependentBlockSupport(0, 0.0, 0, 0)
    ladder = tuple(
        PoolingLevelAudit(level, 0, 0, support) for level in POOLING_ORDER
    )
    return PoolingAudit(probability, ladder, None, terminal_fallback=True)


def _complete_calibration(ensemble: EnsembleSignature) -> CalibrationResolver:
    decision = FrozenModelSelectionDecision(
        decision_sha256=DECISION_HASH,
        architecture_sha256=ARCHITECTURE_HASH,
        d_enabled=False,
    )
    coverage = CalibrationCoverage(
        condition_signatures=(CONDITION,),
        fold_checkpoint_sha256s=FOLD_HASHES,
        full_checkpoint_sha256=DEPLOYMENT_HASH,
        ensemble_signatures=(ensemble,),
    )
    provenance = CalibrationProvenance.from_mapping(
        {
            "calibration_coverage_sha256": coverage.semantic_sha256,
            "model_selection_decision_sha256": DECISION_HASH,
            "residual_scales_sha256": RESIDUAL_SCALES_HASH,
            "selected_architecture_sha256": ARCHITECTURE_HASH,
        }
    )
    provenance_sha256 = provenance.semantic_sha256
    residual_audit = _terminal_audit(probability=False)
    probability_audit = _terminal_audit(probability=True)
    location_calibration = LocationScaleCalibration(
        location_b=0.0,
        total_scale_c=1.0,
        fold_weights=(1.0 / 3.0,) * 3,
        fold_mixture_mean=0.0,
        fold_mixture_variance=1.0,
        full_mean=0.0,
        full_variance=1.0,
    )
    probability_calibration = ProbabilityCalibration(
        alpha=0.0,
        beta_raw=math.log(math.expm1(1.0)),
        slope=1.0,
        weighted_log_loss=0.5,
        iterations=0,
        pooling=None,
        identity=True,
    )
    locations = []
    biases = []
    spreads = []
    regression_probabilities = []
    ensemble_probabilities = []
    for location_key in coverage.required_location_scale_keys():
        locations.append(
            LocationScaleRecord(
                location_key,
                location_calibration,
                residual_audit,
                provenance_sha256,
            )
        )
    location_by_cell = {
        (record.key.lead_hours, record.key.condition_signature): record
        for record in locations
    }
    for bias_key in coverage.required_sampler_bias_keys():
        location = location_by_cell[
            (bias_key.lead_hours, bias_key.condition_signature)
        ]
        biases.append(
            SamplerBiasRecord(
                key=bias_key,
                sampler_bias_d=0.0,
                d_enabled=False,
                bias_fraction=0.0,
                location_scale_key_sha256=location.key.semantic_sha256,
                pooling=residual_audit,
                provenance_sha256=provenance_sha256,
            )
        )
    bias_by_cell = {
        (record.key.lead_hours, record.key.condition_signature): record
        for record in biases
    }
    for spread_key in coverage.required_spread_keys():
        bias = bias_by_cell[(spread_key.lead_hours, spread_key.condition_signature)]
        spreads.append(
            SpreadRecord(
                key=spread_key,
                spread_gamma=1.0,
                sampler_bias_key_sha256=bias.key.semantic_sha256,
                pooling=residual_audit,
                provenance_sha256=provenance_sha256,
            )
        )
    regression_probabilities.extend(
        RegressionProbabilityRecord(
            key,
            probability_calibration,
            probability_audit,
            provenance_sha256,
        )
        for key in coverage.required_regression_probability_keys()
    )
    ensemble_probabilities.extend(
        EnsembleProbabilityRecord(
            key,
            probability_calibration,
            probability_audit,
            provenance_sha256,
        )
        for key in coverage.required_ensemble_probability_keys()
    )

    def ordered(values: list[object]) -> tuple[object, ...]:
        def record_digest(record: object) -> str:
            return record.key.semantic_sha256  # type: ignore[attr-defined]

        return tuple(sorted(values, key=record_digest))

    artifact = CalibrationArtifact(
        split="calibration",
        release_status="complete",
        model_selection=decision,
        provenance=provenance,
        coverage=coverage,
        location_scale=ordered(locations),  # type: ignore[arg-type]
        sampler_bias=ordered(biases),  # type: ignore[arg-type]
        spread=ordered(spreads),  # type: ignore[arg-type]
        regression_probability=ordered(  # type: ignore[arg-type]
            regression_probabilities
        ),
        ensemble_probability=ordered(ensemble_probabilities),  # type: ignore[arg-type]
    )
    return CalibrationResolver.for_complete_release(artifact)


@pytest.fixture(scope="module")
def verified_context() -> SimpleNamespace:
    bundle = _bundle()
    regression_model, residual_model = _production_configured_models()
    development_regression = bind_development_regression_model(regression_model)
    development_residual = bind_development_residual_edm_model(residual_model)
    regression = VerifiedRegressionModel(
        model=development_regression.model,
        checkpoint_sha256=DEPLOYMENT_HASH,
        _capture=development_regression._capture,
        provenance=_regression_provenance(),
    )
    residual = VerifiedResidualEDMModel(
        model=development_residual.model,
        checkpoint_sha256=RESIDUAL_HASH,
        variant="edm_a",
        _capture=development_residual._capture,
        provenance=_residual_provenance(bundle.provenance.semantic_sha256),
    )
    ensemble = EnsembleSignature(
        SamplerCoreSignature(checkpoint_id=RESIDUAL_HASH, edm_steps=12), 32
    )
    return SimpleNamespace(
        bundle=bundle,
        stage2_config=STAGE2_CONFIG,
        stage3_config=STAGE3_CONFIG,
        development_regression=development_regression,
        development_residual=development_residual,
        regression=regression,
        residual=residual,
        ensemble=ensemble,
        selection=_selection(),
        calibration=_complete_calibration(ensemble),
    )


def _verified_identity(context: SimpleNamespace, **updates: object):
    inputs = {
        "stage3_artifacts": context.bundle,
        "stage2_config": context.stage2_config,
        "stage3_config": context.stage3_config,
        "regression_binding": context.regression,
        "residual_binding": context.residual,
        "model_selection": context.selection,
        "calibration": context.calibration,
        "ensemble_signature": context.ensemble,
        "lead_hours": 1.0,
        "condition_signature": CONDITION,
    }
    inputs.update(updates)
    return VerifiedForecastIdentity.create(**inputs)  # type: ignore[arg-type]


def test_production_identity_derives_all_exact_keys_and_scale(
    verified_context: SimpleNamespace,
) -> None:
    identity = _verified_identity(verified_context)

    assert identity.mode == "production"
    assert identity.fold_checkpoint_sha256s == FOLD_HASHES
    assert identity.location_key.fold_checkpoint_sha256s == FOLD_HASHES
    assert identity.location_key.full_checkpoint_sha256 == DEPLOYMENT_HASH
    assert identity.bias_key.sampler_core.checkpoint_id == RESIDUAL_HASH
    assert identity.spread_key.ensemble_signature == verified_context.ensemble
    assert identity.regression_probability_key.regression_checkpoint_sha256 == (
        DEPLOYMENT_HASH
    )
    assert tuple(identity.ensemble_probability_key_map) == (0.1, 1.0, 5.0)
    assert identity.residual_scale == 0.75
    assert identity.diffusion_scale_supported
    assert identity.scale_supported
    assert identity.residual_scale_record is verified_context.bundle.scales[0]
    assert identity.scale_audit is identity.residual_scale_record
    regression, residual = identity.validate_model_bindings()
    assert regression is verified_context.regression.model
    assert residual is verified_context.residual.model
    with pytest.raises(FrozenInstanceError):
        identity.lead_hours = 2.0  # type: ignore[misc]


def test_production_identity_keeps_unsupported_scale_as_auditable_state(
    verified_context: SimpleNamespace,
) -> None:
    unsupported = _bundle(unsupported=True)
    residual = replace(
        verified_context.residual,
        provenance=_residual_provenance(unsupported.provenance.semantic_sha256),
    )
    identity = _verified_identity(
        verified_context,
        stage3_artifacts=unsupported,
        residual_binding=residual,
    )

    assert identity.residual_scale is None
    assert identity.diffusion_scale_supported is False
    assert identity.residual_scale_record.diffusion_scale_unsupported


def test_production_identity_accepts_development_calibration_resolver(
    verified_context: SimpleNamespace,
) -> None:
    development = CalibrationResolver.development_identity(
        verified_context.selection.frozen_calibration_decision()
    )
    identity = _verified_identity(verified_context, calibration=development)

    assert identity.mode == "production"
    assert identity.calibration is development
    assert identity.calibration_semantic_sha256 is None


def test_development_identity_discards_caller_checkpoint_label_and_is_deterministic(
    verified_context: SimpleNamespace,
) -> None:
    calibration = CalibrationResolver.development_identity(
        verified_context.selection.frozen_calibration_decision()
    )
    caller_ensemble = EnsembleSignature(
        SamplerCoreSignature(checkpoint_id="caller-label", edm_steps=6), 4
    )
    first = DevelopmentForecastIdentity.create(
        regression_binding=verified_context.development_regression,
        residual_binding=verified_context.development_residual,
        calibration=calibration,
        ensemble_signature=caller_ensemble,
        lead_hours=np.float32(1.0),
        condition_signature=CONDITION,
        residual_scale=np.float32(0.5),
    )
    second = DevelopmentForecastIdentity.create(
        regression_binding=verified_context.development_regression,
        residual_binding=verified_context.development_residual,
        calibration=calibration,
        ensemble_signature=caller_ensemble,
        lead_hours=1.0,
        condition_signature=CONDITION,
        residual_scale=0.5,
    )

    assert first.mode == "development"
    assert first.ensemble_signature.sampler_core.checkpoint_id == (
        verified_context.development_residual.checkpoint_sha256
    )
    assert first.fold_checkpoint_sha256s == second.fold_checkpoint_sha256s
    assert len(set(first.fold_checkpoint_sha256s)) == 3
    assert first.location_key.full_checkpoint_sha256 == (
        verified_context.development_regression.checkpoint_sha256
    )
    assert first.scale_audit.source == "explicit_development_scalar"
    first.validate_model_bindings()


def test_development_identity_accepts_checkpoint_loaded_bindings(
    verified_context: SimpleNamespace,
) -> None:
    caller_ensemble = EnsembleSignature(
        SamplerCoreSignature(checkpoint_id="caller-label", edm_steps=6), 4
    )
    identity = DevelopmentForecastIdentity.create(
        regression_binding=verified_context.regression,
        residual_binding=verified_context.residual,
        calibration=verified_context.calibration,
        ensemble_signature=caller_ensemble,
        lead_hours=1.0,
        condition_signature=CONDITION,
        residual_scale=1.0,
    )

    assert identity.mode == "development"
    assert identity.regression_binding.is_production
    assert identity.residual_binding.is_production
    assert identity.ensemble_signature.sampler_core.checkpoint_id == RESIDUAL_HASH
    assert identity.location_key.full_checkpoint_sha256 == DEPLOYMENT_HASH

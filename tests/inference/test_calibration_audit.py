from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.inference.calibration_audit import (
    CalibrationApplicationAudit,
    CalibrationApplicationAuditBundle,
    ResidualCalibrationApplicationAudit,
    ThresholdCalibrationApplicationAudit,
    apply_ensemble_probability_with_audit,
    apply_regression_probability_with_audit,
    apply_residual_calibration_with_audit,
    build_calibration_application_audit_bundle,
)
from kcorrdiff.training.calibration import FoldCalibrationMoments, POOLING_ORDER
from kcorrdiff.training.calibration_artifact import (
    CalibrationArtifactBuilder,
    CalibrationResolver,
    EnsembleProbabilityKey,
    FrozenModelSelectionDecision,
    LocationScaleKey,
    PoolingEvidence,
    RegressionProbabilityKey,
    SamplerBiasKey,
    SpreadKey,
)
from kcorrdiff.training.edm_sampling import EnsembleSignature, SamplerCoreSignature


CONDITION = "era5_oracle:era=1:tp=1:full_trajectory"


def _decision(*, d_enabled: bool = False) -> FrozenModelSelectionDecision:
    return FrozenModelSelectionDecision(
        decision_sha256="a" * 64,
        architecture_sha256="b" * 64,
        d_enabled=d_enabled,
    )


def _keys() -> tuple[
    LocationScaleKey,
    SamplerBiasKey,
    SpreadKey,
    RegressionProbabilityKey,
    EnsembleProbabilityKey,
]:
    core = SamplerCoreSignature(checkpoint_id="f" * 64, edm_steps=12)
    ensemble = EnsembleSignature(core, 32)
    return (
        LocationScaleKey(
            1.0,
            CONDITION,
            ("1" * 64, "2" * 64, "3" * 64),
            "4" * 64,
        ),
        SamplerBiasKey(1.0, CONDITION, core),
        SpreadKey(1.0, CONDITION, ensemble),
        RegressionProbabilityKey(1.0, A_WET_MM, CONDITION, "4" * 64),
        EnsembleProbabilityKey(1.0, A_WET_MM, CONDITION, ensemble),
    )


def _evidence(
    *, count: int, probability: bool
) -> dict[str, PoolingEvidence]:
    observation = np.asarray([index % 2 for index in range(count)], dtype=np.float64)
    rows = tuple(f"row-{index:03d}" for index in range(count))
    return {
        level: PoolingEvidence(
            block_id=tuple(f"{level}-{index:03d}" for index in range(count)),
            weight=np.ones(count, dtype=np.float64),
            observation=observation if probability else None,
            row_id=rows,
        )
        for level in POOLING_ORDER
    }


def _artifact_fixture(
    *, count: int, d_enabled: bool = False
) -> tuple[CalibrationResolver, tuple[object, ...]]:
    builder = CalibrationArtifactBuilder(
        split="calibration",
        model_selection=_decision(d_enabled=d_enabled),
        provenance_hashes={"calibration_manifest_sha256": "c" * 64},
    )
    location_key, bias_key, spread_key, p_key, q_key = _keys()
    rows = tuple(f"row-{index:03d}" for index in range(count))
    residual = np.linspace(-1.0, 1.0, count, dtype=np.float64)[:, None]
    weight = np.ones_like(residual)
    builder.fit_location_scale(
        location_key,
        folds=(
            FoldCalibrationMoments(0, 10.0, -0.2, 1.0),
            FoldCalibrationMoments(1, 20.0, 0.0, 0.9),
            FoldCalibrationMoments(2, 30.0, 0.2, 1.1),
        ),
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=_evidence(count=count, probability=False),
        fit_row_id=rows,
    )
    members = np.random.default_rng(13).normal(
        0.0, 0.5, size=(count, 32, 1)
    )
    builder.fit_sampler(
        bias_key=bias_key,
        spread_key=spread_key,
        location_scale_key=location_key,
        restored_members=members,
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=_evidence(count=count, probability=False),
        fit_row_id=rows,
    )
    probability = np.linspace(0.05, 0.95, count, dtype=np.float64)
    observation = np.asarray([index % 2 for index in range(count)], dtype=np.float64)
    probability_evidence = _evidence(count=count, probability=True)
    builder.fit_regression_probability(
        p_key,
        probability=probability,
        observation=observation,
        weight=np.ones(count, dtype=np.float64),
        pooling_evidence=probability_evidence,
        fit_row_id=rows,
    )
    builder.fit_ensemble_probability(
        q_key,
        probability=probability[::-1].copy(),
        observation=observation,
        weight=np.ones(count, dtype=np.float64),
        pooling_evidence=probability_evidence,
        fit_row_id=rows,
    )
    artifact = builder.build(release_status="development")
    return CalibrationResolver(artifact), (
        location_key,
        bias_key,
        spread_key,
        p_key,
        q_key,
    )


def test_application_audit_has_strict_derived_calibrated_semantics() -> None:
    fitted = CalibrationApplicationAudit(
        "1" * 64, "fitted", "full_cell", False
    )
    disabled = CalibrationApplicationAudit(
        "2" * 64, "selection_disabled", "lead_only", False
    )
    terminal = CalibrationApplicationAudit(
        "3" * 64, "terminal_identity", None, True
    )
    development = CalibrationApplicationAudit(
        "4" * 64, "development_identity", None, False
    )
    assert fitted.calibrated
    assert not disabled.calibrated
    assert not terminal.calibrated
    assert not development.calibrated
    with pytest.raises(FrozenInstanceError):
        terminal.mode = "fitted"  # type: ignore[misc]
    with pytest.raises(ValueError, match="terminal identity"):
        CalibrationApplicationAudit(
            "5" * 64, "terminal_identity", None, False
        )
    with pytest.raises(ValueError, match="not a terminal fallback"):
        CalibrationApplicationAudit(
            "6" * 64, "selection_disabled", None, True
        )


def test_fitted_application_reports_selected_disabled_d_separately() -> None:
    resolver, raw_keys = _artifact_fixture(count=40, d_enabled=False)
    location_key, bias_key, spread_key, p_key, q_key = raw_keys
    assert isinstance(location_key, LocationScaleKey)
    assert isinstance(bias_key, SamplerBiasKey)
    assert isinstance(spread_key, SpreadKey)
    assert isinstance(p_key, RegressionProbabilityKey)
    assert isinstance(q_key, EnsembleProbabilityKey)
    members = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(
        1, 32, 1, 1, 1
    )
    expected = resolver.apply_residual(
        members,
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    actual, audit = apply_residual_calibration_with_audit(
        resolver,
        members,
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    assert torch.equal(actual, expected)
    assert audit.location_scale.mode == "fitted"
    assert audit.location_scale.calibrated
    assert audit.sampler_bias.mode == "selection_disabled"
    assert not audit.sampler_bias.calibrated
    assert not audit.sampler_bias.terminal_fallback
    assert audit.spread.mode == "fitted"

    probability = torch.tensor([0.2, 0.8], dtype=torch.float32)
    fitted_p, p_audit = apply_regression_probability_with_audit(
        resolver, probability, key=p_key
    )
    fitted_q, q_audit = apply_ensemble_probability_with_audit(
        resolver, probability, key=q_key
    )
    assert torch.equal(
        fitted_p, resolver.apply_regression_probability(probability, key=p_key)
    )
    assert torch.equal(
        fitted_q, resolver.apply_ensemble_probability(probability, key=q_key)
    )
    assert p_audit.mode == q_audit.mode == "fitted"
    assert p_audit.key_sha256 == p_key.semantic_sha256
    assert q_audit.key_sha256 == q_key.semantic_sha256


def test_terminal_probability_is_raw_and_never_labeled_calibrated() -> None:
    resolver, raw_keys = _artifact_fixture(count=10, d_enabled=False)
    location_key, bias_key, spread_key, p_key, q_key = raw_keys
    assert isinstance(location_key, LocationScaleKey)
    assert isinstance(bias_key, SamplerBiasKey)
    assert isinstance(spread_key, SpreadKey)
    assert isinstance(p_key, RegressionProbabilityKey)
    assert isinstance(q_key, EnsembleProbabilityKey)
    members = torch.randn(1, 32, 1, 2, 2, dtype=torch.float32)
    _result, residual_audit = apply_residual_calibration_with_audit(
        resolver,
        members,
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    assert residual_audit.location_scale.mode == "terminal_identity"
    assert residual_audit.location_scale.terminal_fallback
    assert residual_audit.sampler_bias.mode == "selection_disabled"
    assert not residual_audit.sampler_bias.terminal_fallback
    assert residual_audit.spread.mode == "terminal_identity"

    probability = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float32)
    calibrated_p, p_audit = apply_regression_probability_with_audit(
        resolver, probability, key=p_key
    )
    calibrated_q, q_audit = apply_ensemble_probability_with_audit(
        resolver, probability, key=q_key
    )
    assert calibrated_p is probability
    assert calibrated_q is probability
    assert p_audit.mode == q_audit.mode == "terminal_identity"
    assert p_audit.terminal_fallback and q_audit.terminal_fallback
    assert not p_audit.calibrated and not q_audit.calibrated


def test_development_resolver_always_reports_explicit_development_identity() -> None:
    resolver = CalibrationResolver.development_identity(_decision(d_enabled=False))
    location_key, bias_key, spread_key, p_key, q_key = _keys()
    members = torch.randn(1, 32, 1, 2, 2, dtype=torch.float32)
    result, residual_audit = apply_residual_calibration_with_audit(
        resolver,
        members,
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    assert result is members
    assert {
        residual_audit.location_scale.mode,
        residual_audit.sampler_bias.mode,
        residual_audit.spread.mode,
    } == {"development_identity"}
    assert not residual_audit.sampler_bias.terminal_fallback

    probability = torch.tensor([0.1, 0.9], dtype=torch.float32)
    result_p, p_audit = apply_regression_probability_with_audit(
        resolver, probability, key=p_key
    )
    result_q, q_audit = apply_ensemble_probability_with_audit(
        resolver, probability, key=q_key
    )
    assert result_p is probability and result_q is probability
    assert p_audit.mode == q_audit.mode == "development_identity"


def test_audit_bundle_freezes_probability_thresholds_in_canonical_order() -> None:
    base = CalibrationApplicationAudit(
        "1" * 64, "fitted", "full_cell", False
    )
    residual = ResidualCalibrationApplicationAudit(base, base, base)
    bundle = build_calibration_application_audit_bundle(
        residual=residual,
        regression_probability={A_WET_MM: base},
        ensemble_probability={5.0: base, A_WET_MM: base, 1.0: base},
    )
    assert isinstance(bundle, CalibrationApplicationAuditBundle)
    assert tuple(
        item.threshold_mm for item in bundle.ensemble_probability
    ) == (A_WET_MM, 1.0, 5.0)
    with pytest.raises(ValueError, match="canonical order"):
        CalibrationApplicationAuditBundle(
            residual,
            (ThresholdCalibrationApplicationAudit(A_WET_MM, base),),
            (
                ThresholdCalibrationApplicationAudit(5.0, base),
                ThresholdCalibrationApplicationAudit(1.0, base),
            ),
        )


def test_residual_helper_fails_before_lookup_on_cross_signature_keys() -> None:
    resolver = CalibrationResolver.development_identity(_decision())
    location_key, bias_key, spread_key, _p_key, _q_key = _keys()
    wrong_bias = SamplerBiasKey(1.5, CONDITION, bias_key.sampler_core)
    with pytest.raises(ValueError, match="mismatched exact signatures"):
        apply_residual_calibration_with_audit(
            resolver,
            torch.zeros(1, 32, 1, 1, 1, dtype=torch.float32),
            location_key=location_key,
            bias_key=wrong_bias,
            spread_key=spread_key,
        )

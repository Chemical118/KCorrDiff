from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.training.calibration import FoldCalibrationMoments, POOLING_ORDER
from kcorrdiff.training.calibration_artifact import (
    CalibrationArtifact,
    CalibrationArtifactBuilder,
    CalibrationResolver,
    EnsembleProbabilityKey,
    FrozenModelSelectionDecision,
    LocationScaleKey,
    PoolingEvidence,
    RegressionProbabilityKey,
    SamplerBiasKey,
    SpreadKey,
    build_pooling_audit,
    publish_calibration_artifact,
    read_calibration_artifact,
)
from kcorrdiff.training.edm_sampling import EnsembleSignature, SamplerCoreSignature


CONDITION = "era5_oracle:era=1:tp=1:full_trajectory"


def _decision(*, d_enabled: bool = False) -> FrozenModelSelectionDecision:
    return FrozenModelSelectionDecision(
        decision_sha256="a" * 64,
        architecture_sha256="b" * 64,
        d_enabled=d_enabled,
    )


def _builder(*, d_enabled: bool = False) -> CalibrationArtifactBuilder:
    return CalibrationArtifactBuilder(
        split="calibration",
        model_selection=_decision(d_enabled=d_enabled),
        provenance_hashes={
            "calibration_config_sha256": "c" * 64,
            "calibration_manifest_sha256": "d" * 64,
            "residual_scales_sha256": "e" * 64,
        },
    )


def _evidence(
    *,
    probability: bool,
    counts: tuple[int, int, int] = (40, 40, 40),
) -> dict[str, PoolingEvidence]:
    result: dict[str, PoolingEvidence] = {}
    for level, count in zip(POOLING_ORDER, counts):
        block_id = tuple(f"{level}-{index:03d}" for index in range(count))
        weight = np.ones(count, dtype=np.float64)
        observation = (
            np.asarray([index % 2 for index in range(count)], dtype=np.float64)
            if probability
            else None
        )
        result[level] = PoolingEvidence(block_id, weight, observation)
    return result


def _keys(
    *, lead_hours: float = 1.0
) -> tuple[
    LocationScaleKey,
    SamplerBiasKey,
    SpreadKey,
    RegressionProbabilityKey,
    EnsembleProbabilityKey,
]:
    core = SamplerCoreSignature(checkpoint_id="f" * 64, edm_steps=12)
    ensemble = EnsembleSignature(core, 32)
    location = LocationScaleKey(
        lead_hours=lead_hours,
        condition_signature=CONDITION,
        fold_checkpoint_sha256s=("1" * 64, "2" * 64, "3" * 64),
        full_checkpoint_sha256="4" * 64,
    )
    return (
        location,
        SamplerBiasKey(lead_hours, CONDITION, core),
        SpreadKey(lead_hours, CONDITION, ensemble),
        RegressionProbabilityKey(lead_hours, A_WET_MM, CONDITION, "4" * 64),
        EnsembleProbabilityKey(lead_hours, A_WET_MM, CONDITION, ensemble),
    )


def _fit_complete() -> tuple[CalibrationArtifact, tuple[object, ...]]:
    builder = _builder(d_enabled=False)
    location_key, bias_key, spread_key, p_key, q_key = _keys()
    residual = np.linspace(-1.0, 1.0, 40, dtype=np.float64)[:, None]
    weight = np.ones_like(residual)
    location = builder.fit_location_scale(
        location_key,
        folds=(
            FoldCalibrationMoments(0, 10.0, -0.2, 1.0),
            FoldCalibrationMoments(1, 20.0, 0.0, 0.9),
            FoldCalibrationMoments(2, 30.0, 0.2, 1.1),
        ),
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=_evidence(probability=False),
    )
    rng = np.random.default_rng(17)
    members = rng.normal(0.0, 0.7, size=(40, 32, 1))
    bias, spread = builder.fit_sampler(
        bias_key=bias_key,
        spread_key=spread_key,
        location_scale_key=location_key,
        restored_members=members,
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=_evidence(probability=False),
    )
    probability = np.linspace(0.04, 0.96, 40, dtype=np.float64)
    # Both classes are present and overlap across the predictor range.
    observation = np.asarray(
        [int((index % 5) not in (0, 1)) for index in range(40)],
        dtype=np.float64,
    )
    probability_weight = np.ones(40, dtype=np.float64)
    p_record = builder.fit_regression_probability(
        p_key,
        probability=probability,
        observation=observation,
        weight=probability_weight,
        pooling_evidence=_evidence(probability=True),
    )
    q_record = builder.fit_ensemble_probability(
        q_key,
        probability=probability[::-1].copy(),
        observation=observation,
        weight=probability_weight,
        pooling_evidence=_evidence(probability=True),
    )
    artifact = builder.build(release_status="complete")
    return artifact, (
        location_key,
        bias_key,
        spread_key,
        p_key,
        q_key,
        location,
        bias,
        spread,
        p_record,
        q_record,
    )


def test_pooling_ladder_is_deterministic_and_records_all_support() -> None:
    audit = build_pooling_audit(
        _evidence(probability=False, counts=(10, 35, 40)),
        probability_gate=False,
    )
    assert audit.decision.level == "lead_provider"
    assert tuple(level.level for level in audit.ladder) == POOLING_ORDER
    assert [level.record_count for level in audit.ladder] == [10, 35, 40]
    assert [level.positive_weight_record_count for level in audit.ladder] == [10, 35, 40]
    assert audit.ladder[1].support.block_count == 35
    assert audit.ladder[1].support.block_ess == pytest.approx(35.0)
    assert audit.ladder[1].support.positive_support_blocks == 0
    assert audit.ladder[1].support.negative_support_blocks == 0


def test_probability_pooling_records_both_class_block_counts() -> None:
    audit = build_pooling_audit(
        _evidence(probability=True), probability_gate=True
    )
    assert audit.decision.level == "lead_provider_era_present"
    assert audit.decision.support.positive_support_blocks == 20
    assert audit.decision.support.negative_support_blocks == 20
    with pytest.raises(ValueError, match="binary observations"):
        build_pooling_audit(_evidence(probability=False), probability_gate=True)


def test_split_and_frozen_decision_are_mandatory_before_fit() -> None:
    with pytest.raises(ValueError, match="split='calibration'"):
        CalibrationArtifactBuilder(
            split="model_selection",
            model_selection=_decision(),
            provenance_hashes={"calibration_manifest_sha256": "c" * 64},
        )
    with pytest.raises(ValueError, match="conflicts"):
        CalibrationArtifactBuilder(
            split="calibration",
            model_selection=_decision(),
            provenance_hashes={
                "model_selection_decision_sha256": "9" * 64,
            },
        )
    with pytest.raises(ValueError, match="SHA-256"):
        CalibrationArtifactBuilder(
            split="calibration",
            model_selection=_decision(),
            provenance_hashes={"calibration_manifest_sha256": "not-a-hash"},
        )


def test_exact_keys_reject_noncanonical_or_incomplete_signatures() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LocationScaleKey(1.0, CONDITION, ("1" * 64,), "4" * 64)
    with pytest.raises(ValueError, match="official half-hour"):
        _keys(lead_hours=0.75)
    with pytest.raises(ValueError, match="exactly A_wet"):
        RegressionProbabilityKey(1.0, 1.0, CONDITION, "4" * 64)
    core = SamplerCoreSignature(checkpoint_id="f" * 64, edm_steps=12)
    with pytest.raises(ValueError, match="official or"):
        EnsembleProbabilityKey(
            1.0, 2.0, CONDITION, EnsembleSignature(core, 32)
        )
    with pytest.raises(ValueError, match="not canonical"):
        LocationScaleKey(
            1.0,
            "anything:era=0:tp=1:no_era_access",
            ("1" * 64, "2" * 64),
            "4" * 64,
        )


def test_builder_fits_each_exact_table_and_d_is_not_label_selected() -> None:
    artifact, values = _fit_complete()
    location_key, bias_key, spread_key, p_key, q_key = values[:5]
    assert artifact.release_status == "complete"
    assert len(artifact.location_scale) == 1
    assert len(artifact.sampler_bias) == 1
    assert len(artifact.spread) == 1
    assert len(artifact.regression_probability) == 1
    assert len(artifact.ensemble_probability) == 1
    assert artifact.sampler_bias[0].sampler_bias_d == 0.0
    assert artifact.sampler_bias[0].d_enabled is False
    assert artifact.sampler_bias[0].location_scale_key_sha256 == (
        location_key.semantic_sha256
    )
    assert artifact.spread[0].sampler_bias_key_sha256 == bias_key.semantic_sha256
    assert spread_key.ensemble_signature.member_count == 32
    assert p_key.threshold_mm == A_WET_MM
    assert q_key.threshold_mm == A_WET_MM
    assert artifact.provenance.mapping["model_selection_decision_sha256"] == "a" * 64
    assert artifact.provenance.mapping["selected_architecture_sha256"] == "b" * 64
    assert all(
        record.provenance_sha256 == artifact.provenance.semantic_sha256
        for records in (
            artifact.location_scale,
            artifact.sampler_bias,
            artifact.spread,
            artifact.regression_probability,
            artifact.ensemble_probability,
        )
        for record in records
    )


def test_atomic_publication_round_trip_is_canonical_and_no_overwrite(
    tmp_path: Path,
) -> None:
    artifact, _ = _fit_complete()
    destination = tmp_path / "calibration.json"
    semantic_hash = publish_calibration_artifact(destination, artifact)
    assert semantic_hash == artifact.semantic_sha256
    assert len(semantic_hash) == 64
    assert destination.read_bytes().endswith(b"\n")
    loaded = read_calibration_artifact(destination)
    assert loaded == artifact
    assert loaded.to_dict() == json.loads(destination.read_text())
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_calibration_artifact(destination, artifact)
    assert destination.read_bytes() == original


def test_read_rejects_digest_tamper_and_noncanonical_json(tmp_path: Path) -> None:
    artifact, _ = _fit_complete()
    tampered = tmp_path / "tampered.json"
    raw = artifact.to_dict()
    raw["release_status"] = "development"
    tampered.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        read_calibration_artifact(tampered)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")
    with pytest.raises(ValueError, match="not canonical"):
        read_calibration_artifact(noncanonical)


def test_resolver_applies_linked_records_and_fails_closed_on_any_key_change() -> None:
    artifact, values = _fit_complete()
    location_key, bias_key, spread_key, p_key, q_key = values[:5]
    resolver = CalibrationResolver.for_complete_release(artifact)
    members = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(
        1, 32, 1, 1, 1
    )
    calibrated = resolver.apply_residual(
        members,
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    assert calibrated.shape == members.shape
    assert not torch.equal(calibrated, members)
    probability = torch.tensor([0.2, 0.8], dtype=torch.float32)
    assert not torch.equal(
        resolver.apply_regression_probability(probability, key=p_key), probability
    )
    assert not torch.equal(
        resolver.apply_ensemble_probability(probability, key=q_key), probability
    )

    wrong_location = LocationScaleKey(
        location_key.lead_hours,
        location_key.condition_signature,
        location_key.fold_checkpoint_sha256s,
        "9" * 64,
    )
    with pytest.raises(KeyError, match="exact b/c"):
        resolver.apply_residual(
            members,
            location_key=wrong_location,
            bias_key=bias_key,
            spread_key=spread_key,
        )
    wrong_q = EnsembleProbabilityKey(
        q_key.lead_hours,
        q_key.threshold_mm,
        q_key.condition_signature,
        EnsembleSignature(
            SamplerCoreSignature(checkpoint_id="8" * 64, edm_steps=12), 32
        ),
    )
    with pytest.raises(KeyError, match="exact q_cal"):
        resolver.apply_ensemble_probability(probability, key=wrong_q)

    other_lead_bias = SamplerBiasKey(1.5, CONDITION, bias_key.sampler_core)
    with pytest.raises(ValueError, match="mismatched exact signatures"):
        resolver.apply_residual(
            members,
            location_key=location_key,
            bias_key=other_lead_bias,
            spread_key=spread_key,
        )


def test_absent_identity_requires_explicit_development_and_never_completes() -> None:
    with pytest.raises(FileNotFoundError, match="required"):
        CalibrationResolver(None)
    with pytest.raises(FileNotFoundError, match="requires"):
        CalibrationResolver.for_complete_release(None)

    resolver = CalibrationResolver.development_identity(_decision(d_enabled=True))
    location_key, bias_key, spread_key, p_key, q_key = _keys()
    members = torch.randn(1, 32, 1, 2, 2, dtype=torch.float32)
    assert torch.equal(
        resolver.apply_residual(
            members,
            location_key=location_key,
            bias_key=bias_key,
            spread_key=spread_key,
        ),
        members,
    )
    probability = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float32)
    assert resolver.apply_regression_probability(probability, key=p_key) is probability
    assert resolver.apply_ensemble_probability(probability, key=q_key) is probability

    empty_builder = _builder()
    with pytest.raises(ValueError, match="complete release"):
        empty_builder.build(release_status="complete")
    with pytest.raises(ValueError, match="explicitly declare"):
        empty_builder.build(release_status="development")
    development = empty_builder.build(
        release_status="development", calibration_absent_identity=True
    )
    assert development.calibration_absent_identity
    with pytest.raises(ValueError, match="explicit development_mode"):
        CalibrationResolver(development)
    serialized_identity = CalibrationResolver(development, development_mode=True)
    assert torch.equal(
        serialized_identity.apply_residual(
            members,
            location_key=location_key,
            bias_key=bias_key,
            spread_key=spread_key,
        ),
        members,
    )
    with pytest.raises(ValueError, match="development identity"):
        CalibrationResolver.for_complete_release(development)


def test_complete_artifact_typed_roundtrip_rejects_unknown_or_wrong_links() -> None:
    artifact, _ = _fit_complete()
    raw = artifact.to_dict()
    raw["unexpected"] = True
    semantic = dict(raw)
    semantic.pop("semantic_sha256")
    raw["semantic_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="round trip"):
        CalibrationArtifact.from_dict(raw)

    linked = artifact.to_dict()
    linked["spread"][0]["sampler_bias_key_sha256"] = "0" * 64
    semantic = dict(linked)
    semantic.pop("semantic_sha256")
    linked["semantic_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="lacks its exact sampler-bias"):
        CalibrationArtifact.from_dict(linked)

from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.training.calibration import FoldCalibrationMoments, POOLING_ORDER
from kcorrdiff.training.calibration_artifact import (
    CalibrationArtifact,
    CalibrationArtifactBuilder,
    CalibrationCoverage,
    CalibrationResolver,
    EnsembleProbabilityKey,
    FrozenModelSelectionDecision,
    LocationScaleKey,
    PoolingEvidence,
    RegressionProbabilityKey,
    SamplerBiasKey,
    SpreadKey,
    OFFICIAL_ENSEMBLE_THRESHOLDS_MM,
    OFFICIAL_LEADS_HOURS,
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


def _coverage() -> CalibrationCoverage:
    ensemble = EnsembleSignature(
        SamplerCoreSignature(checkpoint_id="f" * 64, edm_steps=12), 32
    )
    return CalibrationCoverage(
        condition_signatures=(CONDITION,),
        fold_checkpoint_sha256s=("1" * 64, "2" * 64, "3" * 64),
        full_checkpoint_sha256="4" * 64,
        ensemble_signatures=(ensemble,),
    )


def _builder(
    *, d_enabled: bool = False, complete_coverage: bool = False
) -> CalibrationArtifactBuilder:
    return CalibrationArtifactBuilder(
        split="calibration",
        model_selection=_decision(d_enabled=d_enabled),
        provenance_hashes={
            "calibration_config_sha256": "c" * 64,
            "calibration_manifest_sha256": "d" * 64,
            "residual_scales_sha256": "e" * 64,
        },
        coverage=_coverage() if complete_coverage else None,
    )


def _evidence(
    *,
    probability: bool,
    counts: tuple[int, int, int, int] = (40, 40, 40, 40),
    observation: np.ndarray | None = None,
) -> dict[str, PoolingEvidence]:
    result: dict[str, PoolingEvidence] = {}
    for level, count in zip(POOLING_ORDER, counts):
        block_id = tuple(f"{level}-{index:03d}" for index in range(count))
        weight = np.ones(count, dtype=np.float64)
        level_observation = None
        if probability:
            level_observation = (
                np.asarray(observation, dtype=np.float64)
                if observation is not None and len(observation) == count
                else np.asarray(
                    [index % 2 for index in range(count)], dtype=np.float64
                )
            )
        row_id = tuple(f"row-{index:03d}" for index in range(count))
        result[level] = PoolingEvidence(block_id, weight, level_observation, row_id)
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
    builder = _builder(d_enabled=False, complete_coverage=True)
    residual = np.linspace(-1.0, 1.0, 60, dtype=np.float64)[:, None]
    weight = np.ones_like(residual)
    rng = np.random.default_rng(17)
    members = rng.normal(0.0, 0.7, size=(60, 32, 1))
    probability = np.linspace(0.04, 0.96, 60, dtype=np.float64)
    # Both classes are present and overlap across the predictor range.
    observation = np.asarray(
        [int((index % 5) not in (0, 1)) for index in range(60)],
        dtype=np.float64,
    )
    probability_weight = np.ones(60, dtype=np.float64)
    rows = tuple(f"row-{index:03d}" for index in range(60))
    selected: tuple[object, ...] | None = None
    for lead_hours in OFFICIAL_LEADS_HOURS:
        location_key, bias_key, spread_key, p_key, wet_q_key = _keys(
            lead_hours=lead_hours
        )
        location = builder.fit_location_scale(
            location_key,
            folds=(
                FoldCalibrationMoments(0, 10.0, -0.2, 1.0),
                FoldCalibrationMoments(1, 20.0, 0.0, 0.9),
                FoldCalibrationMoments(2, 30.0, 0.2, 1.1),
            ),
            full_residual=residual,
            calibration_weight=weight,
            pooling_evidence=_evidence(probability=False, counts=(60, 60, 60, 60)),
            fit_row_id=rows,
        )
        bias, spread = builder.fit_sampler(
            bias_key=bias_key,
            spread_key=spread_key,
            location_scale_key=location_key,
            restored_members=members,
            full_residual=residual,
            calibration_weight=weight,
            pooling_evidence=_evidence(probability=False, counts=(60, 60, 60, 60)),
            fit_row_id=rows,
        )
        p_record = builder.fit_regression_probability(
            p_key,
            probability=probability,
            observation=observation,
            weight=probability_weight,
            pooling_evidence=_evidence(
                probability=True,
                counts=(60, 60, 60, 60),
                observation=observation,
            ),
            fit_row_id=rows,
        )
        wet_q_record = None
        for threshold in OFFICIAL_ENSEMBLE_THRESHOLDS_MM:
            q_key = EnsembleProbabilityKey(
                lead_hours,
                threshold,
                CONDITION,
                spread_key.ensemble_signature,
            )
            q_record = builder.fit_ensemble_probability(
                q_key,
                probability=probability[::-1].copy(),
                observation=observation,
                weight=probability_weight,
                pooling_evidence=_evidence(
                    probability=True,
                    counts=(60, 60, 60, 60),
                    observation=observation,
                ),
                fit_row_id=rows,
            )
            if threshold == A_WET_MM:
                wet_q_record = q_record
        if lead_hours == 1.0:
            assert wet_q_record is not None
            selected = (
                location_key,
                bias_key,
                spread_key,
                p_key,
                wet_q_key,
                location,
                bias,
                spread,
                p_record,
                wet_q_record,
            )
    artifact = builder.build(release_status="complete")
    assert selected is not None
    return artifact, selected


def test_pooling_ladder_is_deterministic_and_records_all_support() -> None:
    evidence = _evidence(probability=False, counts=(10, 10, 35, 40))
    audit = build_pooling_audit(
        evidence,
        probability_gate=False,
        fit_row_id=tuple(evidence["lead_provider"].row_id or ()),
        fit_weight=evidence["lead_provider"].weight,
    )
    assert audit.decision.level == "lead_provider"
    assert tuple(level.level for level in audit.ladder) == POOLING_ORDER
    assert [level.record_count for level in audit.ladder] == [10, 10, 35, 40]
    assert [level.positive_weight_record_count for level in audit.ladder] == [10, 10, 35, 40]
    assert audit.ladder[2].support.block_count == 35
    assert audit.ladder[2].support.block_ess == pytest.approx(35.0)
    assert audit.ladder[2].support.positive_support_blocks == 0
    assert audit.ladder[2].support.negative_support_blocks == 0


def test_zero_mass_narrow_cell_falls_through_to_supported_broader_cell() -> None:
    evidence = _evidence(probability=False, counts=(10, 10, 35, 40))
    narrow = evidence["full_cell"]
    evidence["full_cell"] = PoolingEvidence(
        block_id=narrow.block_id,
        weight=np.zeros(10),
        row_id=narrow.row_id,
    )

    audit = build_pooling_audit(
        evidence,
        probability_gate=False,
        fit_row_id=tuple(evidence["lead_provider"].row_id or ()),
        fit_weight=evidence["lead_provider"].weight,
    )

    assert audit.ladder[0].support.block_count == 0
    assert audit.decision is not None
    assert audit.decision.level == "lead_provider"


def test_probability_pooling_records_both_class_block_counts() -> None:
    evidence = _evidence(probability=True)
    audit = build_pooling_audit(
        evidence,
        probability_gate=True,
        fit_row_id=tuple(evidence["full_cell"].row_id or ()),
        fit_weight=evidence["full_cell"].weight,
        fit_observation=evidence["full_cell"].observation,
    )
    assert audit.decision.level == "full_cell"
    assert audit.decision.support.positive_support_blocks == 20
    assert audit.decision.support.negative_support_blocks == 20
    with pytest.raises(ValueError, match="binary observations"):
        build_pooling_audit(
            _evidence(probability=False),
            probability_gate=True,
            fit_row_id=tuple(f"row-{index:03d}" for index in range(40)),
            fit_weight=np.ones(40),
            fit_observation=np.zeros(40),
        )


def test_pooling_evidence_is_bound_to_selected_fit_rows_weights_and_labels() -> None:
    evidence = _evidence(probability=True)
    rows = tuple(evidence["full_cell"].row_id or ())
    weights = np.asarray(evidence["full_cell"].weight)
    observation = np.asarray(evidence["full_cell"].observation)
    with pytest.raises(ValueError, match="row IDs differ"):
        build_pooling_audit(
            evidence,
            probability_gate=True,
            fit_row_id=tuple(reversed(rows)),
            fit_weight=weights,
            fit_observation=observation,
        )
    changed_weight = weights.copy()
    changed_weight[0] = 2.0
    with pytest.raises(ValueError, match="weights differ"):
        build_pooling_audit(
            evidence,
            probability_gate=True,
            fit_row_id=rows,
            fit_weight=changed_weight,
            fit_observation=observation,
        )
    changed_observation = observation.copy()
    changed_observation[0] = 1.0 - changed_observation[0]
    with pytest.raises(ValueError, match="observations differ"):
        build_pooling_audit(
            evidence,
            probability_gate=True,
            fit_row_id=rows,
            fit_weight=weights,
            fit_observation=changed_observation,
        )


def test_terminal_pooling_publishes_explicit_identity_for_every_mapping() -> None:
    builder = _builder()
    location_key, bias_key, spread_key, p_key, q_key = _keys()
    count = 10
    rows = tuple(f"row-{index:03d}" for index in range(count))
    residual = np.linspace(-1.0, 1.0, count, dtype=np.float64)[:, None]
    weight = np.ones_like(residual)
    residual_evidence = _evidence(
        probability=False, counts=(count, count, count, count)
    )
    location = builder.fit_location_scale(
        location_key,
        folds=(
            FoldCalibrationMoments(0, 1.0, -0.1, 0.5),
            FoldCalibrationMoments(1, 1.0, 0.1, 0.5),
            FoldCalibrationMoments(2, 1.0, 0.0, 0.5),
        ),
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=residual_evidence,
        fit_row_id=rows,
    )
    members = np.stack((residual - 0.5, residual + 0.5), axis=1)
    bias, spread = builder.fit_sampler(
        bias_key=bias_key,
        spread_key=spread_key,
        location_scale_key=location_key,
        restored_members=members,
        full_residual=residual,
        calibration_weight=weight,
        pooling_evidence=residual_evidence,
        fit_row_id=rows,
    )
    probability = np.linspace(0.1, 0.9, count)
    observation = np.asarray([index % 2 for index in range(count)], dtype=np.float64)
    probability_evidence = _evidence(
        probability=True,
        counts=(count, count, count, count),
        observation=observation,
    )
    p_record = builder.fit_regression_probability(
        p_key,
        probability=probability,
        observation=observation,
        weight=np.ones(count),
        pooling_evidence=probability_evidence,
        fit_row_id=rows,
    )
    q_record = builder.fit_ensemble_probability(
        q_key,
        probability=probability,
        observation=observation,
        weight=np.ones(count),
        pooling_evidence=probability_evidence,
        fit_row_id=rows,
    )
    artifact = builder.build(release_status="development")
    assert all(
        record.pooling.terminal_fallback
        for record in (location, bias, spread, p_record, q_record)
    )
    assert (location.calibration.location_b, location.calibration.total_scale_c) == (
        0.0,
        1.0,
    )
    assert bias.sampler_bias_d == 0.0
    assert spread.spread_gamma == 1.0
    assert p_record.calibration.identity and q_record.calibration.identity
    resolver = CalibrationResolver(artifact)
    raw_probability = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float32)
    assert (
        resolver.apply_regression_probability(raw_probability, key=p_key)
        is raw_probability
    )
    assert resolver.apply_ensemble_probability(raw_probability, key=q_key) is raw_probability


def test_split_and_frozen_decision_are_mandatory_before_fit() -> None:
    with pytest.raises(ValueError, match="split='calibration'"):
        CalibrationArtifactBuilder(
            split="model_selection",
            model_selection=_decision(),
            provenance_hashes={"calibration_manifest_sha256": "c" * 64},
        )
    metadata_builder = CalibrationArtifactBuilder(
        split="calibration",
        model_selection=_decision(),
        provenance_hashes={"model_selection_decision_sha256": "9" * 64},
    )
    assert dict(metadata_builder.provenance.hashes)[
        "model_selection_decision_sha256"
    ] == "9" * 64
    builder = CalibrationArtifactBuilder(
        split="calibration",
        model_selection=_decision(),
        provenance_hashes={"calibration_manifest_sha256": "not-a-hash"},
    )
    assert dict(builder.provenance.hashes)["calibration_manifest_sha256"] == "not-a-hash"


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
    assert len(artifact.location_scale) == 12
    assert len(artifact.sampler_bias) == 12
    assert len(artifact.spread) == 12
    assert len(artifact.regression_probability) == 12
    assert len(artifact.ensemble_probability) == 36
    bias_record = next(
        item for item in artifact.sampler_bias if item.key == bias_key
    )
    spread_record = next(item for item in artifact.spread if item.key == spread_key)
    assert bias_record.sampler_bias_d == 0.0
    assert bias_record.d_enabled is False
    assert bias_record.location_scale_key_sha256 == (
        location_key.semantic_sha256
    )
    assert spread_record.sampler_bias_key_sha256 == bias_key.semantic_sha256
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


def test_complete_release_requires_exact_predeclared_key_coverage() -> None:
    artifact, _ = _fit_complete()
    with pytest.raises(ValueError, match="gamma coverage mismatch"):
        replace(artifact, spread=artifact.spread[1:])
    with pytest.raises(ValueError, match="q_cal coverage mismatch"):
        replace(artifact, ensemble_probability=artifact.ensemble_probability[1:])

    # Disabled d remains an explicit zero-valued record at every required cell.
    assert artifact.model_selection.d_enabled is False
    assert len(artifact.sampler_bias) == len(OFFICIAL_LEADS_HOURS)
    assert all(
        not record.d_enabled and record.sampler_bias_d == 0.0
        for record in artifact.sampler_bias
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


def test_read_accepts_edited_and_noncanonical_json(tmp_path: Path) -> None:
    # Research code: artifacts are plain JSON — hand-edited or re-serialized
    # files still load; the stored digest is informational.
    artifact, _ = _fit_complete()
    edited = tmp_path / "edited.json"
    raw = artifact.to_dict()
    raw["release_status"] = "development"
    edited.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    assert read_calibration_artifact(edited).release_status == "development"

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")
    assert read_calibration_artifact(noncanonical) == artifact


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
    # A resolver without an artifact is a pass-through identity resolver.
    assert CalibrationResolver(None).artifact is None
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
    # Any artifact/mode combination is accepted; an absent-identity artifact
    # simply resolves nothing.
    assert CalibrationResolver(development)._identity_mode
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
    # A development/absent artifact is accepted anywhere; its lookups simply
    # resolve nothing.
    assert CalibrationResolver.for_complete_release(development)._identity_mode


def test_complete_artifact_tolerates_unknown_keys_but_rejects_wrong_links() -> None:
    artifact, _ = _fit_complete()
    raw = artifact.to_dict()
    raw["unexpected"] = True
    assert CalibrationArtifact.from_dict(raw) == artifact

    # A spread record pointing at a nonexistent sampler-bias record is a real
    # structural error: the lookup chain would break at application time.
    linked = artifact.to_dict()
    linked["spread"][0]["sampler_bias_key_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lacks its exact sampler-bias"):
        CalibrationArtifact.from_dict(linked)


def test_artifact_deserialization_rejects_boolean_numeric_aliases() -> None:
    artifact, _ = _fit_complete()
    original = artifact.to_dict()
    mutations = (
        (("location_scale", 0, "key", "lead_hours"), True),
        (
            ("sampler_bias", 0, "key", "sampler_core", "edm_steps"),
            True,
        ),
        (
            (
                "spread",
                0,
                "key",
                "ensemble_signature",
                "member_count",
            ),
            False,
        ),
        (("regression_probability", 0, "key", "threshold_mm"), True),
        (("ensemble_probability", 0, "key", "threshold_mm"), False),
        (("location_scale", 0, "pooling", "ladder", 0, "record_count"), True),
        (("location_scale", 0, "pooling", "ladder", 0, "block_ess"), True),
        (("location_scale", 0, "parameters", "full_mean"), True),
        (("sampler_bias", 0, "sampler_bias_d"), False),
        (("spread", 0, "spread_gamma"), True),
        (("regression_probability", 0, "parameters", "alpha"), True),
        (("ensemble_probability", 0, "parameters", "iterations"), False),
    )
    for path, replacement in mutations:
        raw = deepcopy(original)
        target: object = raw
        for component in path[:-1]:
            target = target[component]  # type: ignore[index]
        target[path[-1]] = replacement  # type: ignore[index]
        semantic = dict(raw)
        semantic.pop("semantic_sha256")
        raw["semantic_sha256"] = __import__("hashlib").sha256(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with pytest.raises(TypeError):
            CalibrationArtifact.from_dict(raw)


def test_artifact_deserialization_rejects_string_scalar_aliases() -> None:
    artifact, _ = _fit_complete()
    raw = artifact.to_dict()
    raw["ensemble_probability"][0]["parameters"]["iterations"] = "12"  # type: ignore[index]
    semantic = dict(raw)
    semantic.pop("semantic_sha256")
    raw["semantic_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    with pytest.raises(TypeError, match="must be an integer"):
        CalibrationArtifact.from_dict(raw)

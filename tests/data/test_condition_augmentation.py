from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from kcorrdiff.data.condition_augmentation import (
    NORMALIZATION_CONTRACT,
    NULL_CONDITION_SIGNATURE,
    PROTOCOL_VERSION,
    ConditionAugmentationPolicy,
    ConditionAugmentationProvenance,
    ConditionAugmentationResult,
    ConditionMixtureEntry,
    canonical_draw_rows_sha256,
    expand_condition_signatures,
    expand_manifest_condition_support,
    materialize_condition_signatures,
    oof_lineage_metadata,
    validate_materialized_condition_provenance,
)
from kcorrdiff.data.sampling import (
    CandidateItem,
    bounded_draw_manifest,
)
from kcorrdiff.data.split_manifest import canonical_sample_id, make_item


FULL = "era5_oracle:era=1:tp=1:full_trajectory"
TP_OFF = "era5_oracle:era=1:tp=0:full_trajectory"
CAUSAL = "era5_oracle:era=1:tp=1:target_end_causal"
CAUSAL_TP_OFF = "era5_oracle:era=1:tp=0:target_end_causal"


def _entry(signature: str, target: float, draw: float) -> ConditionMixtureEntry:
    return ConditionMixtureEntry(
        condition_signature=signature,
        target_probability=target,
        draw_probability=draw,
    )


def _matched_policy(
    entries: tuple[ConditionMixtureEntry, ...] | None = None,
) -> ConditionAugmentationPolicy:
    return ConditionAugmentationPolicy(
        policy_id="era5-full-dropout-preregistered-v1",
        protocol_version=PROTOCOL_VERSION,
        source_condition_signature=FULL,
        temporal_access_strategy="matched_arm",
        entries=entries
        or (
            _entry(FULL, 0.5, 0.25),
            _entry(TP_OFF, 0.3, 0.25),
            _entry(NULL_CONDITION_SIGNATURE, 0.2, 0.5),
        ),
    )


def _candidate(index: int, *, signature: str = FULL) -> CandidateItem:
    t0 = datetime(2021, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return CandidateItem(
        sample_id=canonical_sample_id(t0, 0.5, signature),
        t0_utc=t0.isoformat(),
        lead_hours=0.5,
        condition_signature=signature,
        block_id=f"event-{index}",
        stratum="event",
        split="outer_train",
        p_target=0.5,
        fold_id=index,
    )


def _base_rows(*, draws: int = 200, purpose: str = "base-draw-v1"):
    candidates = (_candidate(0), _candidate(1))
    return bounded_draw_manifest(
        candidates,
        draws=draws,
        seed=11103,
        purpose_id=purpose,
        draw_probabilities=(0.8, 0.2),
    )


def _materialize(
    rows,
    *,
    policy: ConditionAugmentationPolicy | None = None,
    signature_purpose: str = "condition-signature-v1",
) -> ConditionAugmentationResult:
    return materialize_condition_signatures(
        rows,
        policy=policy or _matched_policy(),
        training_seed=11103,
        source_draw_purpose_id="base-draw-v1",
        signature_purpose_id=signature_purpose,
        source_draw_manifest_sha256=canonical_draw_rows_sha256(rows),
    )


def test_policy_has_no_implicit_dropout_defaults_and_requires_both_dropouts() -> None:
    with pytest.raises(ValueError, match="whole-ERA"):
        _matched_policy(
            (
                _entry(FULL, 0.6, 0.6),
                _entry(TP_OFF, 0.4, 0.4),
            )
        )

    with pytest.raises(ValueError, match="tp-channel"):
        _matched_policy(
            (
                _entry(FULL, 0.8, 0.8),
                _entry(NULL_CONDITION_SIGNATURE, 0.2, 0.2),
            )
        )

    with pytest.raises(ValueError, match="sum to one"):
        _matched_policy(
            (
                _entry(FULL, 0.5, 0.5),
                _entry(TP_OFF, 0.3, 0.3),
                _entry(NULL_CONDITION_SIGNATURE, 0.1, 0.1),
            )
        )

    with pytest.raises(TypeError, match="real number"):
        _entry(FULL, True, 1.0)


def test_policy_canonical_hash_is_order_independent_and_provider_locked() -> None:
    entries = (
        _entry(FULL, 0.5, 0.25),
        _entry(TP_OFF, 0.3, 0.25),
        _entry(NULL_CONDITION_SIGNATURE, 0.2, 0.5),
    )
    first = _matched_policy(entries)
    second = _matched_policy(tuple(reversed(entries)))
    assert first.entries == second.entries
    assert first.semantic_sha256 == second.semantic_sha256
    assert ConditionAugmentationPolicy.from_mapping(first.to_json()) == first

    incomplete = first.to_json()
    entries_json = list(incomplete["entries"])
    entries_json[0] = dict(entries_json[0])
    del entries_json[0]["draw_probability"]
    incomplete["entries"] = entries_json
    with pytest.raises(ValueError, match="entry 0 is missing consumed keys"):
        ConditionAugmentationPolicy.from_mapping(incomplete)

    wrong_provider = "aurora:era=1:tp=0:full_trajectory"
    with pytest.raises(ValueError, match="provider"):
        _matched_policy(
            (
                _entry(FULL, 0.5, 0.5),
                _entry(wrong_provider, 0.3, 0.3),
                _entry(NULL_CONDITION_SIGNATURE, 0.2, 0.2),
            )
        )


def test_condition_weight_clipping_is_optional_metadata_not_a_manifest_gate() -> None:
    policy = replace(_matched_policy(), weight_clipping=2.5)
    assert ConditionAugmentationPolicy.from_mapping(policy.to_json()) == policy

    result = _materialize(_base_rows(draws=8), policy=policy)
    assert result.provenance.weight_clipping == 2.5
    assert all(
        row.omega == pytest.approx(row.p_target / row.p_draw)
        for row in result.rows
    )


def test_temporal_access_mix_requires_an_explicit_strategy() -> None:
    entries = (
        _entry(FULL, 0.3, 0.3),
        _entry(TP_OFF, 0.2, 0.2),
        _entry(CAUSAL, 0.25, 0.25),
        _entry(CAUSAL_TP_OFF, 0.15, 0.15),
        _entry(NULL_CONDITION_SIGNATURE, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="matched_arm"):
        _matched_policy(entries)

    confounded = (
        _entry(FULL, 0.35, 0.35),
        _entry(TP_OFF, 0.25, 0.25),
        _entry(CAUSAL, 0.3, 0.3),
        _entry(NULL_CONDITION_SIGNATURE, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="each configured ERA temporal-access"):
        ConditionAugmentationPolicy(
            policy_id="confounded-access-tp-policy",
            protocol_version=PROTOCOL_VERSION,
            source_condition_signature=FULL,
            temporal_access_strategy="full_with_target_end_causal_augmentation",
            entries=confounded,
        )

    policy = ConditionAugmentationPolicy(
        policy_id="era5-full-with-causal-augmentation-v1",
        protocol_version=PROTOCOL_VERSION,
        source_condition_signature=FULL,
        temporal_access_strategy="full_with_target_end_causal_augmentation",
        entries=entries,
    )
    assert {entry.signature.temporal_access_mode for entry in policy.entries} == {
        "full_trajectory",
        "target_end_causal",
        "no_era_access",
    }


def test_support_expansion_keeps_base_and_signature_probabilities_separate() -> None:
    candidates = (_candidate(0), _candidate(1))
    expanded = expand_condition_signatures(candidates, policy=_matched_policy())

    assert len(expanded) == 3 * len(candidates)
    assert {row.source_sample_id for row in expanded} == {
        candidate.sample_id for candidate in candidates
    }
    for candidate in candidates:
        selected = [
            row for row in expanded if row.source_sample_id == candidate.sample_id
        ]
        assert sum(
            row.conditional_target_probability for row in selected
        ) == pytest.approx(1.0)
        assert sum(
            row.conditional_draw_probability for row in selected
        ) == pytest.approx(1.0)
        assert all(
            row.full_target_probability
            == pytest.approx(
                row.base_target_probability
                * row.conditional_target_probability
            )
            for row in selected
        )
        assert {row.fold_id for row in selected} == {candidate.fold_id}
        assert all(
            row.sample_id
            == canonical_sample_id(
                datetime.fromisoformat(row.t0_utc),
                row.lead_hours,
                row.condition_signature,
            )
            for row in selected
        )
    assert not hasattr(expanded[0], "p_target")


def test_manifest_support_preserves_base_mass_and_full_item_weights() -> None:
    first_time = datetime(2021, 1, 1, tzinfo=UTC)
    second_time = first_time + timedelta(hours=1)
    base = (
        replace(
            make_item(
                t0=first_time,
                lead_hours=0.5,
                condition_signature=FULL,
                split="outer_train",
                block_id="event-0",
                stratum="event",
                p_target=0.5,
                p_draw=0.8,
            ),
            fold_id=0,
        ),
        replace(
            make_item(
                t0=second_time,
                lead_hours=0.5,
                condition_signature=FULL,
                split="outer_train",
                block_id="event-1",
                stratum="event",
                p_target=0.5,
                p_draw=0.2,
            ),
            fold_id=1,
        ),
    )
    support = expand_manifest_condition_support(base, policy=_matched_policy())

    assert len(support) == 6
    assert sum(item.p_target for item in support) == pytest.approx(1.0)
    assert sum(item.p_draw for item in support) == pytest.approx(1.0)
    for item in base:
        selected = [row for row in support if row.block_id == item.block_id]
        assert sum(row.p_target for row in selected) == pytest.approx(item.p_target)
        assert sum(row.p_draw for row in selected) == pytest.approx(item.p_draw)
        assert all(
            row.omega == pytest.approx(row.p_target / row.p_draw)
            for row in selected
        )


def test_signature_materialization_is_purpose_keyed_prefix_stable() -> None:
    short_base = _base_rows(draws=40)
    long_base = _base_rows(draws=200)
    short = _materialize(short_base)
    long = _materialize(long_base)

    assert long.rows[: len(short.rows)] == short.rows
    assert [row.global_example_index for row in long.rows] == list(range(200))
    assert len({row.condition_signature for row in long.rows}) == 3
    changed = _materialize(long_base, signature_purpose="condition-signature-v2")
    assert [row.condition_signature for row in changed.rows] != [
        row.condition_signature for row in long.rows
    ]


def test_full_item_probabilities_and_importance_weights_are_propagated() -> None:
    policy = _matched_policy()
    base = _base_rows(draws=200)
    result = _materialize(base, policy=policy)
    by_signature = {
        str(entry.condition_signature): entry for entry in policy.entries
    }

    for source, augmented in zip(base, result.rows, strict=True):
        entry = by_signature[augmented.condition_signature]
        assert augmented.p_target == pytest.approx(
            source.p_target * entry.target_probability
        )
        assert augmented.p_draw == pytest.approx(
            source.p_draw * entry.draw_probability
        )
        assert augmented.omega == pytest.approx(
            augmented.p_target / augmented.p_draw
        )
        assert augmented.block_id == source.block_id
        assert augmented.fold_id == source.fold_id
        assert augmented.sample_id == canonical_sample_id(
            datetime.fromisoformat(augmented.t0_utc),
            augmented.lead_hours,
            augmented.condition_signature,
        )


def test_genuinely_absent_source_is_null_only_and_is_never_upgraded() -> None:
    candidate = _candidate(0, signature=NULL_CONDITION_SIGNATURE)
    candidate = replace(candidate, p_target=1.0)
    rows = bounded_draw_manifest(
        (candidate,), draws=10, seed=11103, purpose_id="base-draw-v1"
    )
    result = _materialize(rows)

    assert {row.condition_signature for row in result.rows} == {
        NULL_CONDITION_SIGNATURE
    }
    assert all(row.p_target == 1.0 and row.p_draw == 1.0 for row in result.rows)
    assert dict(result.provenance.signature_counts) == {
        FULL: 0,
        TP_OFF: 0,
        NULL_CONDITION_SIGNATURE: 10,
    }


def test_oof_lineage_is_unique_signature_exact_and_normalization_bound() -> None:
    result = _materialize(_base_rows(draws=200))
    parsed_provenance = ConditionAugmentationProvenance.from_mapping(
        result.provenance.to_json()
    )
    assert parsed_provenance == result.provenance
    assert validate_materialized_condition_provenance(
        result.rows,
        policy=_matched_policy(),
        provenance=parsed_provenance,
    ) == result.oof_requirements
    identities = {
        (row.t0_utc, row.lead_hours, row.condition_signature)
        for row in result.rows
    }
    requirements = result.oof_requirements
    assert len(requirements) == len(identities)
    assert {item.semantic_key for item in requirements} == identities
    assert all(
        item.sample_id
        == canonical_sample_id(
            datetime.fromisoformat(item.t0_utc),
            item.lead_hours,
            item.condition_signature,
        )
        for item in requirements
    )

    metadata = oof_lineage_metadata(
        result, normalization_artifact_sha256="a" * 64
    )
    assert metadata["condition_policy_sha256"] == result.provenance.policy_sha256
    assert metadata["augmented_draw_manifest_sha256"] == (
        result.provenance.augmented_draw_manifest_sha256
    )
    assert metadata["normalization_fit_split"] == "outer_train"
    assert metadata["normalization_contract"] == NORMALIZATION_CONTRACT
    assert metadata["target_label_fields_used_for_condition_assignment"] == []


def test_provenance_sidecar_tolerates_missing_digests_and_unknown_keys() -> None:
    result = _materialize(_base_rows(draws=40))
    raw = dict(result.provenance.to_json())
    for key in (
        "policy_sha256",
        "source_draw_manifest_sha256",
        "augmented_draw_manifest_sha256",
        "oof_requirements_sha256",
    ):
        del raw[key]
    raw["future_extension"] = {"anything": True}
    parsed = ConditionAugmentationProvenance.from_mapping(raw)
    assert parsed.policy_sha256 == ""
    assert parsed.draws == result.provenance.draws
    assert validate_materialized_condition_provenance(
        result.rows,
        policy=_matched_policy(),
        provenance=parsed,
    ) == result.oof_requirements


def test_fail_closed_on_nontrain_split_identity_or_purpose_lineage() -> None:
    with pytest.raises(ValueError, match="outer_train-only"):
        expand_condition_signatures(
            (replace(_candidate(0), split="model_selection"),),
            policy=_matched_policy(),
        )

    base = _base_rows(draws=4)
    with pytest.raises(ValueError, match="distinct purpose"):
        materialize_condition_signatures(
            base,
            policy=_matched_policy(),
            training_seed=11103,
            source_draw_purpose_id="same",
            signature_purpose_id="same",
            source_draw_manifest_sha256=canonical_draw_rows_sha256(base),
        )
    with pytest.raises(ValueError, match="lineage mismatch"):
        _materialize((replace(base[0], sample_id="forged"), *base[1:]))


def test_materialize_records_canonical_source_hash_when_omitted() -> None:
    base = _base_rows(draws=4)
    result = materialize_condition_signatures(
        base,
        policy=_matched_policy(),
        training_seed=11103,
        source_draw_purpose_id="base-draw-v1",
        signature_purpose_id="signature-v1",
    )
    assert result.provenance.source_draw_manifest_sha256 == (
        canonical_draw_rows_sha256(base)
    )

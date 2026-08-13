from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcorrdiff.data.availability import LEADS_HOURS, format_radar_timestamp
from kcorrdiff.data.condition_augmentation import (
    ConditionAugmentationPolicy,
    ConditionAugmentationProvenance,
    ConditionMixtureEntry,
)
from kcorrdiff.data.event_manifest import (
    ALL_12_LEADS_MASK,
    EventIndexRow,
    _validate_event_manifest_config,
    _validate_sampling_config,
    build_event_pretrain_index,
    eligible_sample_ids_from_event_index,
    event_index_main,
    expand_event_candidates,
    strict_issue_times,
    write_event_training_bundle,
)
from kcorrdiff.data.sampling import read_draw_manifest
from kcorrdiff.data.split_manifest import (
    SplitInterval,
    audit_manifest,
    canonical_sample_id,
    main as manifest_cli,
)


def sequence(start: datetime, count: int):
    return [start + i * timedelta(minutes=5) for i in range(count)]


def test_strict_issue_times_match_84_frame_contract() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    assert len(strict_issue_times(sequence(start, 84))) == 1
    assert strict_issue_times(sequence(start, 84))[0] == start + timedelta(minutes=55)
    assert len(strict_issue_times(sequence(start, 100))) == 17


def test_index_expands_only_logically_and_never_splits_event_block() -> None:
    first = sequence(datetime(2020, 1, 1, tzinfo=UTC), 100)
    second = sequence(datetime(2021, 1, 2, tzinfo=UTC), 84)
    splits = (
        SplitInterval(
            "outer_train",
            datetime(2019, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 1, tzinfo=UTC),
        ),
        SplitInterval(
            "model_selection",
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
    )
    rows, audit = build_event_pretrain_index([*first, *second], splits)
    assert len(rows) == 18
    assert audit.expanded_lead_items == 216
    assert audit.event_blocks == 2
    assert {row.available_leads_mask for row in rows} == {ALL_12_LEADS_MASK}
    block_splits = {}
    for row in rows:
        block_splits.setdefault(row.block_id, set()).add(row.split)
    assert all(len(value) == 1 for value in block_splits.values())


def test_dependency_embargo_and_cross_split_event_exclusions_are_separate() -> None:
    # One continuous event straddles the boundary. Dependency-crossing issue
    # times are embargoed; remaining lower-priority rows are excluded for the
    # block-level holdout rule and must not inflate boundary_embargo_fraction.
    start = datetime(2020, 12, 31, 12, tzinfo=UTC)
    timestamps = sequence(start, 300)
    splits = (
        SplitInterval(
            "outer_train",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 1, tzinfo=UTC),
        ),
        SplitInterval(
            "model_selection",
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
    )

    _, audit = build_event_pretrain_index(timestamps, splits)

    assert audit.dependency_embargo_exclusions > 0
    assert audit.cross_split_event_exclusions > 0
    assert audit.boundary_embargo_fraction == (
        audit.dependency_embargo_exclusions
        / audit.strict_issue_times_before_embargo
    )
    assert audit.total_exclusion_fraction > audit.boundary_embargo_fraction


def compact_rows() -> tuple[EventIndexRow, ...]:
    start = datetime(2020, 2, 1, 12, tzinfo=UTC)
    return (
        EventIndexRow(start, "event-a", "outer_train", "era5"),
        EventIndexRow(start + timedelta(days=3), "event-b", "outer_train", "era5"),
        EventIndexRow(
            start + timedelta(days=3, minutes=5),
            "event-b",
            "outer_train",
            "era5",
        ),
        EventIndexRow(start + timedelta(days=6), "event-c", "outer_train", "era5"),
        EventIndexRow(
            start + timedelta(days=6, minutes=5),
            "event-c",
            "outer_train",
            "era5",
        ),
        EventIndexRow(
            start + timedelta(days=6, minutes=10),
            "event-c",
            "outer_train",
            "era5",
        ),
    )


def outer_interval() -> tuple[SplitInterval, ...]:
    return (
        SplitInterval(
            "outer_train",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 1, tzinfo=UTC),
        ),
    )


def test_full_expansion_is_canonical_complete_and_block_balanced() -> None:
    rows = compact_rows()
    items = expand_event_candidates(rows[::-1], draw_policy="block-balanced")

    assert len(items) == len(rows) * 12
    assert items == expand_event_candidates(rows, draw_policy="block-balanced")
    assert {item.stratum for item in items} == {"event"}
    assert {item.lead_hours for item in items} == set(LEADS_HOURS)
    assert all(item.p_target == pytest.approx(1.0 / len(items)) for item in items)
    assert all(
        item.sample_id
        == canonical_sample_id(
            datetime.fromisoformat(item.t0_utc),
            item.lead_hours,
            item.condition_signature,
        )
        for item in items
    )

    block_mass: dict[str, float] = {}
    block_folds: dict[str, set[int | None]] = {}
    for item in items:
        block_mass[item.block_id] = block_mass.get(item.block_id, 0.0) + item.p_draw
        block_folds.setdefault(item.block_id, set()).add(item.fold_id)
    assert block_mass == pytest.approx(
        {"event-a": 1.0 / 3.0, "event-b": 1.0 / 3.0, "event-c": 1.0 / 3.0}
    )
    assert all(len(folds) == 1 for folds in block_folds.values())
    assert {next(iter(folds)) for folds in block_folds.values()} == {0, 1, 2}

    independently_eligible = eligible_sample_ids_from_event_index(rows)
    report = audit_manifest(
        items,
        outer_interval(),
        eligible_sample_ids=independently_eligible,
        expected_outer_folds=3,
        validate_split_probabilities=True,
    )
    assert report["unassigned_eligible_item_fraction"] == 0.0
    with pytest.raises(ValueError, match="eligible items are unassigned"):
        audit_manifest(
            items[:-1],
            outer_interval(),
            eligible_sample_ids=independently_eligible,
            expected_outer_folds=3,
        )


def test_available_lead_mask_maps_bits_and_rejects_invalid_masks() -> None:
    rows = tuple(
        EventIndexRow(
            datetime(2020, 2, day, 12, tzinfo=UTC),
            f"event-{day}",
            "outer_train",
            "era5",
            (1 << 0) | (1 << 11),
        )
        for day in (1, 3, 5)
    )
    items = expand_event_candidates(rows)
    assert len(items) == 6
    assert {item.lead_hours for item in items} == {0.5, 6.0}

    for invalid in (0, 1 << 12, -1, True):
        with pytest.raises(ValueError, match="available-leads mask"):
            eligible_sample_ids_from_event_index(
                (
                    EventIndexRow(
                        datetime(2020, 2, 1, 12, tzinfo=UTC),
                        "event-a",
                        "outer_train",
                        "era5",
                        invalid,
                    ),
                )
            )


def test_cli_config_contracts_fail_closed_on_protocol_drift() -> None:
    manifest = {
        "leads_hours": list(LEADS_HOURS),
        "target_distribution": "event_conditioned_archive",
        "boundary_embargo_hours": 7,
    }
    _validate_event_manifest_config(manifest)
    _validate_sampling_config(
        {
            "target_policy": "eligible_items_uniform",
            "weight_clipping": None,
        }
    )

    with pytest.raises(ValueError, match="official 12 leads"):
        _validate_event_manifest_config(
            {**manifest, "leads_hours": list(LEADS_HOURS[:-1])}
        )
    with pytest.raises(ValueError, match="7-hour"):
        _validate_event_manifest_config(
            {**manifest, "boundary_embargo_hours": 6}
        )
    with pytest.raises(ValueError, match="eligible_items_uniform"):
        _validate_sampling_config({"target_policy": "block_uniform"})
    with pytest.raises(ValueError, match="forbids"):
        _validate_sampling_config({"weight_clipping": 10.0})


def _three_event_timestamps() -> tuple[datetime, ...]:
    starts = (
        datetime(2020, 2, 1, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
        datetime(2020, 4, 1, tzinfo=UTC),
    )
    return tuple(
        value
        for start, length in zip(starts, (84, 85, 86), strict=True)
        for value in sequence(start, length)
    )


def test_bundle_is_atomic_hashed_folded_and_budgeted(tmp_path: Path) -> None:
    rows, event_audit = build_event_pretrain_index(
        _three_event_timestamps(), outer_interval()
    )
    output = tmp_path / "bundle"
    result = write_event_training_bundle(
        output,
        rows,
        event_audit,
        outer_interval(),
        draw_count=40,
        draw_policy="block-balanced",
        seed=17,
        purpose_id="regression",
        maximum_oof_bytes=100_000_000,
        config_sha256="a" * 64,
        protocol_version="v1.1.3b",
    )

    assert result.event_rows == 6
    assert result.candidates == 72
    assert result.draws == 40
    draws = read_draw_manifest(output / "training-draw-manifest.jsonl")
    assert all(row.fold_id in {0, 1, 2} for row in draws)
    assert any(row.omega != pytest.approx(1.0) for row in draws)
    metadata_bytes = (output / "bundle-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    assert hashlib.sha256(metadata_bytes).hexdigest() == result.metadata_sha256
    assert metadata["candidate_audit"]["unassigned_eligible_item_fraction"] == 0.0
    assert metadata["oof_preflight"]["required_bytes"] == result.oof_required_bytes
    assert metadata["sampling"]["draw_policy"] == "block-balanced"
    for artifact in metadata["artifacts"].values():
        payload = (output / artifact["path"]).read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]

    rejected = tmp_path / "rejected"
    with pytest.raises(OSError, match="exceeding"):
        write_event_training_bundle(
            rejected,
            rows,
            event_audit,
            outer_interval(),
            draw_count=1,
            draw_policy="uniform",
            seed=17,
            purpose_id="regression",
            maximum_oof_bytes=1,
            config_sha256="a" * 64,
            protocol_version="v1.1.3b",
        )
    assert not rejected.exists()


def test_bundle_materializes_sample_first_condition_signatures(tmp_path: Path) -> None:
    rows, event_audit = build_event_pretrain_index(
        _three_event_timestamps(), outer_interval()
    )
    full = "era5_oracle:era=1:tp=1:full_trajectory"
    tp_off = "era5_oracle:era=1:tp=0:full_trajectory"
    null = "null_provider:era=0:tp=0:no_era_access"
    policy = ConditionAugmentationPolicy(
        policy_id="event-full-dropout-test-v1",
        protocol_version="v1.1.3b",
        source_condition_signature=full,
        temporal_access_strategy="matched_arm",
        entries=(
            ConditionMixtureEntry(full, 0.6, 0.5),
            ConditionMixtureEntry(tp_off, 0.25, 0.25),
            ConditionMixtureEntry(null, 0.15, 0.25),
        ),
    )
    output = tmp_path / "augmented-bundle"
    result = write_event_training_bundle(
        output,
        rows,
        event_audit,
        outer_interval(),
        draw_count=80,
        draw_policy="block-balanced",
        seed=17,
        purpose_id="base-regression-v1",
        signature_purpose_id="condition-signature-v1",
        condition_augmentation=policy,
        maximum_oof_bytes=100_000_000,
        config_sha256="a" * 64,
        protocol_version="v1.1.3b",
    )

    assert result.candidates == 72 * 3
    source = read_draw_manifest(
        output / "source-training-draw-manifest.jsonl"
    )
    augmented = read_draw_manifest(output / "training-draw-manifest.jsonl")
    assert len(source) == len(augmented) == 80
    assert all(row.condition_signature == full for row in source)
    assert {
        row.condition_signature for row in augmented
    } == {full, tp_off, null}
    for base, selected in zip(source, augmented, strict=True):
        assert selected.global_example_index == base.global_example_index
        assert selected.t0_utc == base.t0_utc
        assert selected.lead_hours == base.lead_hours
        assert selected.block_id == base.block_id
        assert selected.fold_id == base.fold_id

    candidate = json.loads((output / "candidate-manifest.json").read_text())
    assert candidate["metadata"]["condition_augmentation_policy_sha256"] == (
        policy.semantic_sha256
    )
    assert sum(
        item["p_target"] for item in candidate["items"]
    ) == pytest.approx(1.0)
    assert sum(item["p_draw"] for item in candidate["items"]) == pytest.approx(
        1.0
    )

    metadata = json.loads((output / "bundle-metadata.json").read_text())
    provenance = ConditionAugmentationProvenance.from_mapping(
        metadata["condition_augmentation"]
    )
    assert metadata["condition_signatures"] == list(
        provenance.supported_signatures
    )
    assert metadata["condition_augmentation_policy_sha256"] == (
        policy.semantic_sha256
    )
    assert metadata["condition_augmentation_provenance_sha256"] == (
        provenance.semantic_sha256
    )
    assert metadata["sampling"]["condition_selection_order"] == (
        "base_sample_then_one_condition_signature"
    )
    assert sum(dict(provenance.signature_counts).values()) == 80
    assert metadata["artifacts"]["source_training_draw_manifest"][
        "purpose_id"
    ] == "base-regression-v1"


def test_full_coverage_bundle_visits_each_outer_train_base_once(tmp_path: Path) -> None:
    rows, event_audit = build_event_pretrain_index(
        _three_event_timestamps(), outer_interval()
    )
    output = tmp_path / "whole-support-bundle"
    base_support = event_audit.expanded_lead_items
    result = write_event_training_bundle(
        output,
        rows,
        event_audit,
        outer_interval(),
        draw_count=base_support,
        draw_policy="full-coverage-shuffled",
        seed=17,
        purpose_id="whole-support-production-v1",
        maximum_oof_bytes=100_000_000,
        config_sha256="a" * 64,
        protocol_version="v1.1.3b",
    )

    draws = read_draw_manifest(output / "training-draw-manifest.jsonl")
    assert result.draws == base_support
    assert len({row.sample_id for row in draws}) == base_support
    assert all(row.p_target == row.p_draw and row.omega == 1.0 for row in draws)
    metadata = json.loads((output / "bundle-metadata.json").read_text())
    assert metadata["sampling"]["coverage"] == (
        "all_outer_train_base_candidates_once"
    )
    assert metadata["sampling"]["replacement"] is False
    assert metadata["sampling"]["base_candidate_count"] == base_support

    with pytest.raises(ValueError, match="draw_count must equal"):
        write_event_training_bundle(
            tmp_path / "incomplete-whole-support",
            rows,
            event_audit,
            outer_interval(),
            draw_count=base_support - 1,
            draw_policy="full-coverage-shuffled",
            seed=17,
            purpose_id="whole-support-production-v1",
            maximum_oof_bytes=100_000_000,
            config_sha256="a" * 64,
            protocol_version="v1.1.3b",
        )


def test_installed_manifest_cli_builds_from_configured_sampler(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    timestamps = _three_event_timestamps()
    fields = {
        format_radar_timestamp(value): np.zeros((1, 1), dtype=np.float32)
        for value in timestamps
    }
    np.savez(raw / "hsr_22_SEOUL_NON_UNI_2020.npz", **fields)
    np.savez(raw / "hsr_22_SEOUL_NON_UNI_2020_cond.npz", **fields)
    config = {
        "protocol_version": "v1.1.3b",
        "era5": {"condition_signature": "era5"},
        "manifest": {
            "leads_hours": list(LEADS_HOURS),
            "target_distribution": "event_conditioned_archive",
            "event_merge_gap_hours": 12,
            "event_guard_hours": 7,
            "split_intervals": [
                {
                    "name": "outer_train",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2021-01-01T00:00:00Z",
                }
            ],
        },
        "sampling": {
            "draws": 12,
            "draw_policy": "block-balanced",
            "seed": 31,
            "purpose_id": "cli-test",
            "maximum_oof_bytes": 100_000_000,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "cli-bundle"

    assert (
        manifest_cli(
            [
                "--config",
                str(config_path),
                "--target-root",
                str(raw),
                "--condition-root",
                str(raw),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    metadata = json.loads((output / "bundle-metadata.json").read_text())
    assert metadata["sampling"]["draw_policy"] == "block-balanced"
    assert metadata["sampling"]["draw_count"] == 12
    assert metadata["population"] == "cprecnet_event_conditioned_archive"
    candidate_payload = json.loads((output / "candidate-manifest.json").read_text())
    assert len(candidate_payload["items"]) == 72
    assert {item["stratum"] for item in candidate_payload["items"]} == {"event"}


def test_cli_can_reuse_hashed_parquet_event_index_with_requested_flags(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    fields = {
        format_radar_timestamp(value): np.zeros((1, 1), dtype=np.float32)
        for value in _three_event_timestamps()
    }
    np.savez(raw / "hsr_22_SEOUL_NON_UNI_2020.npz", **fields)
    np.savez(raw / "hsr_22_SEOUL_NON_UNI_2020_cond.npz", **fields)
    config = {
        "protocol_version": "v1.1.3b",
        "era5": {"condition_signature": "era5"},
        "manifest": {
            "leads_hours": list(LEADS_HOURS),
            "target_distribution": "event_conditioned_archive",
            "event_merge_gap_hours": 12,
            "event_guard_hours": 7,
            "split_intervals": [
                {
                    "name": "outer_train",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2021-01-01T00:00:00Z",
                }
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_index = tmp_path / "source-index"
    assert (
        event_index_main(
            [
                "--config",
                str(config_path),
                "--target-root",
                str(raw),
                "--condition-root",
                str(raw),
                "--output-dir",
                str(source_index),
            ]
        )
        == 0
    )
    assert {path.name for path in source_index.iterdir()} == {
        "event-index.parquet",
        "audit.json",
    }
    reused = tmp_path / "reused"
    assert (
        manifest_cli(
            [
                "--config",
                str(config_path),
                "--event-index",
                str(source_index / "event-index.parquet"),
                "--output-dir",
                str(reused),
                "--draws",
                "8",
                "--maximum-oof-gib",
                "0.1",
                "--sampler-policy",
                "uniform",
            ]
        )
        == 0
    )
    metadata = json.loads((reused / "bundle-metadata.json").read_text())
    source = metadata["source_event_index"]
    assert source["event_index_sha256"] == hashlib.sha256(
        (source_index / "event-index.parquet").read_bytes()
    ).hexdigest()
    assert source["audit_sha256"] == hashlib.sha256(
        (source_index / "audit.json").read_bytes()
    ).hexdigest()
    assert metadata["oof_preflight"]["maximum_bytes"] == int(0.1 * 1024**3)

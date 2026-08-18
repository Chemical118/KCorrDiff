from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

import kcorrdiff.data.condition_bundle as condition_bundle
from kcorrdiff.data.condition_augmentation import ConditionAugmentationPolicy
from kcorrdiff.data.event_manifest import (
    build_event_pretrain_index,
    write_event_training_bundle,
)
from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.data.sampling import read_draw_manifest
from kcorrdiff.data.split_manifest import SplitInterval


POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "condition-augmentation-production-v1.json"
)


def _sequence(start: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(start + index * timedelta(minutes=5) for index in range(count))


def _source_bundle(
    tmp_path: Path, *, include_holdout: bool = False
) -> tuple[Path, int, int]:
    intervals: tuple[SplitInterval, ...] = (
        SplitInterval(
            "outer_train",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 1, tzinfo=UTC),
        ),
    )
    starts = [
        datetime(2020, 2, 1, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
        datetime(2020, 4, 1, tzinfo=UTC),
    ]
    lengths = [84, 85, 86]
    if include_holdout:
        intervals = (
            *intervals,
            SplitInterval(
                "model_selection",
                datetime(2021, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
            ),
        )
        starts.append(datetime(2021, 2, 1, tzinfo=UTC))
        lengths.append(84)
    timestamps = tuple(
        value
        for start, length in zip(
            starts,
            lengths,
            strict=True,
        )
        for value in _sequence(start, length)
    )
    rows, audit = build_event_pretrain_index(timestamps, intervals)
    source = tmp_path / "source"
    result = write_event_training_bundle(
        source,
        rows,
        audit,
        intervals,
        draw_count=40,
        draw_policy="block-balanced",
        seed=11_103,
        purpose_id="cprecnet-event-pilot-block-balanced-v1",
        maximum_oof_bytes=100_000_000,
        config_sha256="a" * 64,
        protocol_version="v1.1.3b",
    )
    metadata = json.loads((source / "bundle-metadata.json").read_text())
    outer_candidates = metadata["candidate_audit"]["split_counts"][
        "outer_train"
    ]
    return source, outer_candidates, result.candidates


DRAW_PURPOSE_ID = "cprecnet-event-production-full-coverage-v1"
SIGNATURE_PURPOSE_ID = "cprecnet-event-production-condition-signature-v1"


def test_production_policy_config_parses_with_expected_entries() -> None:
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = ConditionAugmentationPolicy.from_mapping(raw)

    assert policy.policy_id == "era5-full-dropout-50-25-25-v1"
    assert [
        (entry.condition_signature, entry.target_probability, entry.draw_probability)
        for entry in policy.entries
    ] == [
        ("era5_oracle:era=1:tp=0:full_trajectory", 0.25, 0.25),
        ("era5_oracle:era=1:tp=1:full_trajectory", 0.5, 0.5),
        ("null_provider:era=0:tp=0:no_era_access", 0.25, 0.25),
    ]


def test_production_builder_streams_expands_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    source, base_candidates, source_candidates = _source_bundle(
        tmp_path, include_holdout=True
    )
    test_policy = tmp_path / POLICY_PATH.name
    test_policy.write_bytes(POLICY_PATH.read_bytes())
    policy = ConditionAugmentationPolicy.from_mapping(
        json.loads(test_policy.read_text(encoding="utf-8"))
    )
    output = tmp_path / "production"
    digest = condition_bundle.augment_condition_bundle(
        source_bundle_dir=source,
        output_dir=output,
        policy_path=test_policy,
        stage="production",
        draw_policy="full-coverage-shuffled",
        training_seed=11_103,
        source_draw_purpose_id=DRAW_PURPOSE_ID,
        signature_purpose_id=SIGNATURE_PURPOSE_ID,
        expected_base_candidate_count=base_candidates,
        planned_epochs=1,
        maximum_oof_bytes=100_000_000,
    )

    metadata_path = output / "bundle-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert digest == sha256_file(metadata_path)
    assert (output / "bundle-metadata.sha256").read_text().split()[0] == digest
    assert metadata["derivation_format_version"] == (
        condition_bundle.AUGMENTED_BUNDLE_FORMAT_VERSION
    )
    assert metadata["condition_augmentation_policy_sha256"] == (
        policy.semantic_sha256
    )
    expected_sampling = {
        "stage": "production",
        "planned_epochs": 1,
        "draw_policy": "full-coverage-shuffled",
        "draw_count": base_candidates,
        "coverage": "all_outer_train_base_candidates_once",
        "replacement": False,
        "base_candidate_count": base_candidates,
    }
    assert {
        key: metadata["sampling"][key] for key in expected_sampling
    } == expected_sampling
    source_rows = read_draw_manifest(
        output / "source-training-draw-manifest.jsonl"
    )
    augmented_rows = read_draw_manifest(output / "training-draw-manifest.jsonl")
    assert len(source_rows) == len(augmented_rows) == base_candidates
    assert len({row.sample_id for row in source_rows}) == base_candidates
    assert all(row.omega == pytest.approx(1.0) for row in augmented_rows)
    assert all(row.p_target == pytest.approx(row.p_draw) for row in augmented_rows)

    candidate = json.loads((output / "candidate-manifest.json").read_text())
    holdout_candidates = source_candidates - base_candidates
    assert len(candidate["items"]) == base_candidates * 3 + holdout_candidates
    assert sum(
        item["split"] == "model_selection" for item in candidate["items"]
    ) == holdout_candidates
    assert candidate["metadata"]["candidate_semantic_sha256"] == (
        metadata["candidate_semantic_sha256"]
    )
    assert candidate["metadata"]["eligible_universe_sample_ids_sha256"] == (
        metadata["eligible_universe_sample_ids_sha256"]
    )
    assert "input_training_draw_manifest" in metadata["artifacts"]
    for record in metadata["artifacts"].values():
        path = output / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]

    with pytest.raises(FileExistsError):
        condition_bundle.augment_condition_bundle(
            source_bundle_dir=source,
            output_dir=output,
            policy_path=test_policy,
            stage="production",
            draw_policy="full-coverage-shuffled",
            training_seed=11_103,
            source_draw_purpose_id=DRAW_PURPOSE_ID,
            signature_purpose_id=SIGNATURE_PURPOSE_ID,
            expected_base_candidate_count=base_candidates,
            planned_epochs=1,
            maximum_oof_bytes=100_000_000,
        )


def test_cli_requires_every_sampling_choice() -> None:
    with pytest.raises(SystemExit):
        condition_bundle.parse_args(
            [
                "--source-bundle-dir",
                "source",
                "--output-dir",
                "output",
                "--policy-json",
                "policy.json",
            ]
        )

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.data.sampling import (
    CandidateItem,
    DrawRow,
    block_balanced_draw_probabilities,
    bounded_draw_manifest,
    canonical_counter,
    enforce_byte_budget,
    full_coverage_shuffled_draw_manifest,
    oof_dense_byte_budget,
    read_draw_manifest,
    write_draw_manifest,
)


def candidates() -> list[CandidateItem]:
    return [
        CandidateItem(
            sample_id=f"s{i}",
            t0_utc=f"2022-01-01T0{i}:00:00+00:00",
            lead_hours=0.5,
            condition_signature="era5_full",
            block_id=f"event-{i}",
            stratum="event",
            split="outer_train",
            p_target=0.5,
        )
        for i in range(2)
    ]


def test_draw_sequence_is_topology_independent_and_weighted() -> None:
    first = bounded_draw_manifest(
        candidates(), draws=20, seed=11103, purpose_id="regression", draw_probabilities=[0.8, 0.2]
    )
    second = bounded_draw_manifest(
        candidates(), draws=20, seed=11103, purpose_id="regression", draw_probabilities=[0.8, 0.2]
    )
    assert first == second
    assert [row.global_example_index for row in first] == list(range(20))
    assert all(row.omega == pytest.approx(row.p_target / row.p_draw) for row in first)
    changed = bounded_draw_manifest(candidates(), draws=20, seed=11103, purpose_id="edm")
    assert [row.sample_id for row in first] != [row.sample_id for row in changed]


def test_oof_budget_counts_unique_sample_signature() -> None:
    rows = bounded_draw_manifest(candidates(), draws=100, seed=4, purpose_id="x")
    required = oof_dense_byte_budget(rows, height=2, width=3, fields=2)
    assert required == 2 * 2 * 3 * 2 * 4
    enforce_byte_budget(required, required)
    with pytest.raises(OSError, match="exceeding"):
        enforce_byte_budget(required, required - 1)


def test_manifest_round_trip_and_hash(tmp_path: Path) -> None:
    rows = bounded_draw_manifest(candidates(), draws=5, seed=2, purpose_id="x")
    digest = write_draw_manifest(tmp_path / "draws.jsonl", rows)
    assert len(digest) == 64
    assert read_draw_manifest(tmp_path / "draws.jsonl") == rows


def test_target_distribution_must_be_normalized_and_unique() -> None:
    invalid = candidates()
    invalid[0] = replace(invalid[0], p_target=0.4)
    with pytest.raises(ValueError, match="sum to one"):
        bounded_draw_manifest(invalid, draws=2, seed=1, purpose_id="x")

    duplicate = [candidates()[0], candidates()[0]]
    with pytest.raises(ValueError, match="sample_id"):
        bounded_draw_manifest(duplicate, draws=2, seed=1, purpose_id="x")


def test_uniform_target_and_draw_probabilities_produce_unit_weight() -> None:
    rows = bounded_draw_manifest(candidates(), draws=20, seed=3, purpose_id="x")
    assert all(row.omega == pytest.approx(1.0) for row in rows)


def test_storm_balanced_probabilities_equalize_blocks_and_preserve_item_draw() -> None:
    base = candidates()
    values = [
        replace(base[0], sample_id="a0", block_id="a", p_target=1.0 / 3.0),
        replace(
            base[0],
            sample_id="a1",
            t0_utc="2022-01-01T02:00:00+00:00",
            block_id="a",
            p_target=1.0 / 3.0,
        ),
        replace(
            base[1],
            sample_id="b0",
            block_id="b",
            p_target=1.0 / 3.0,
        ),
    ]
    probabilities = block_balanced_draw_probabilities(values)

    np.testing.assert_allclose(probabilities, [0.25, 0.25, 0.5])
    assert probabilities[:2].sum() == pytest.approx(probabilities[2:].sum())
    rows = bounded_draw_manifest(
        values,
        draws=20,
        seed=4,
        purpose_id="storm-balanced",
        draw_probabilities=probabilities,
    )
    assert {row.p_draw for row in rows}.issubset({0.25, 0.5})


def test_read_manifest_rejects_tampered_importance_weight(tmp_path: Path) -> None:
    rows = bounded_draw_manifest(candidates(), draws=1, seed=2, purpose_id="x")
    path = tmp_path / "tampered.jsonl"
    payload = asdict(rows[0])
    payload["omega"] = 99.0
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="importance weight"):
        read_draw_manifest(path)


def test_draw_manifest_prefix_fold_and_probability_contracts() -> None:
    folded = [replace(item, fold_id=index) for index, item in enumerate(candidates())]
    short = bounded_draw_manifest(
        folded, draws=8, seed=9, purpose_id="prefix", draw_probabilities=[0.7, 0.3]
    )
    long = bounded_draw_manifest(
        folded, draws=20, seed=9, purpose_id="prefix", draw_probabilities=[0.7, 0.3]
    )
    assert long[: len(short)] == short
    assert {row.fold_id for row in long} <= {0, 1}

    with pytest.raises(ValueError, match="sum to one"):
        bounded_draw_manifest(
            folded,
            draws=2,
            seed=9,
            purpose_id="prefix",
            draw_probabilities=[0.7, 0.2],
        )
    with pytest.raises(ValueError, match="purpose_id"):
        bounded_draw_manifest(folded, draws=2, seed=9, purpose_id="")
    with pytest.raises(ValueError, match="seed"):
        canonical_counter(-1, "x", 0)
    with pytest.raises(ValueError, match="global_example_index"):
        canonical_counter(1, "x", -1)


def test_oof_budget_uses_semantic_identity_not_untrusted_sample_id() -> None:
    shared = dict(
        global_example_index=0,
        sample_id="colliding-id",
        lead_hours=0.5,
        condition_signature="era5",
        block_id="event-a",
        stratum="event",
        split="outer_train",
        p_target=0.5,
        p_draw=0.5,
        omega=1.0,
        fold_id=0,
    )
    first = DrawRow(t0_utc="2022-01-01T00:00:00+00:00", **shared)
    second = DrawRow(
        t0_utc="2022-01-02T00:00:00+00:00",
        **{**shared, "global_example_index": 1},
    )
    assert oof_dense_byte_budget(
        [first, second], height=2, width=3, fields=2
    ) == 2 * 2 * 3 * 2 * 4


def test_empty_draw_manifest_is_not_a_valid_training_artifact(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_draw_manifest(path)


def test_full_coverage_shuffle_is_exact_order_independent_and_prefix_free() -> None:
    template = candidates()[0]
    base = [
        replace(
            template,
            sample_id=f"s{index}",
            t0_utc=f"2022-01-{index + 1:02d}T00:00:00+00:00",
            block_id=f"event-{index}",
            p_target=0.1,
        )
        for index in range(10)
    ]
    first = full_coverage_shuffled_draw_manifest(
        base, seed=11103, purpose_id="whole-support-production-v1"
    )
    reordered = full_coverage_shuffled_draw_manifest(
        base[::-1], seed=11103, purpose_id="whole-support-production-v1"
    )
    changed = full_coverage_shuffled_draw_manifest(
        base, seed=11103, purpose_id="whole-support-production-v2"
    )

    assert first == reordered
    assert {row.sample_id for row in first} == {item.sample_id for item in base}
    assert len(first) == len({row.sample_id for row in first}) == len(base)
    assert [row.global_example_index for row in first] == list(range(len(base)))
    assert all(row.p_draw == row.p_target and row.omega == 1.0 for row in first)
    assert [row.sample_id for row in first] != [row.sample_id for row in changed]

    nonuniform = [
        replace(item, p_target=0.2 if index == 0 else 0.8 / 9)
        for index, item in enumerate(base)
    ]
    with pytest.raises(ValueError, match="item-uniform"):
        full_coverage_shuffled_draw_manifest(
            nonuniform, seed=11103, purpose_id="whole-support-production-v1"
        )

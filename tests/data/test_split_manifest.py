from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

from kcorrdiff.data.split_manifest import (
    SplitInterval,
    audit_main,
    assign_grouped_folds,
    assign_split,
    audit_manifest,
    make_item,
    write_manifest,
)


def intervals():
    return (
        SplitInterval(
            "outer_train",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
        SplitInterval(
            "model_selection",
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 1, tzinfo=UTC),
        ),
    )


def test_complete_dependency_interval_enforces_boundary_embargo() -> None:
    assert assign_split(datetime(2021, 6, 1, tzinfo=UTC), intervals()) == "outer_train"
    assert assign_split(datetime(2021, 12, 31, 20, tzinfo=UTC), intervals()) is None
    assert assign_split(datetime(2022, 1, 1, 0, 30, tzinfo=UTC), intervals()) is None
    assert assign_split(datetime(2022, 1, 1, 7, tzinfo=UTC), intervals()) == "model_selection"


def test_exact_embargo_edges_leave_seven_hours_between_issue_times() -> None:
    boundary = datetime(2022, 1, 1, tzinfo=UTC)
    last_outer = boundary - timedelta(hours=6, minutes=5)
    first_selection = boundary + timedelta(minutes=55)

    assert assign_split(last_outer, intervals()) == "outer_train"
    assert assign_split(last_outer + timedelta(minutes=5), intervals()) is None
    assert assign_split(first_selection - timedelta(minutes=5), intervals()) is None
    assert assign_split(first_selection, intervals()) == "model_selection"
    assert first_selection - last_outer == timedelta(hours=7)


def test_grouped_fold_assignment_never_splits_blocks() -> None:
    items = []
    for hour, block in [(0, "a"), (8, "a"), (16, "b"), (24, "c")]:
        items.append(
            make_item(
                t0=datetime(2020, 2, 1, hour % 24, tzinfo=UTC),
                lead_hours=0.5,
                condition_signature="era5_full",
                split="outer_train",
                block_id=block,
                stratum="event",
            )
        )
    assigned = assign_grouped_folds(items, folds=3)
    block_folds = {}
    for item in assigned:
        block_folds.setdefault(item.block_id, set()).add(item.fold_id)
    assert all(len(value) == 1 for value in block_folds.values())


def test_grouped_fold_assignment_is_order_independent_and_requires_support() -> None:
    items = [
        make_item(
            t0=datetime(2020, 2, day, tzinfo=UTC),
            lead_hours=0.5,
            condition_signature="era5_full",
            split="outer_train",
            block_id=f"event-{day}",
            stratum="event",
        )
        for day in (1, 2, 3)
    ]
    forward = assign_grouped_folds(items, folds=3, seed=7)
    reverse = assign_grouped_folds(items[::-1], folds=3, seed=7)
    assert {item.block_id: item.fold_id for item in forward} == {
        item.block_id: item.fold_id for item in reverse
    }
    with pytest.raises(ValueError, match="fewer than 3 folds"):
        assign_grouped_folds(items[:2], folds=3, seed=7)


def test_manifest_audit_catches_block_and_dependency_leakage() -> None:
    good = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=1.0,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
    )
    report = audit_manifest([good], intervals())
    assert report["unassigned_eligible_item_fraction"] is None
    covered = audit_manifest(
        [good], intervals(), eligible_sample_ids=[good.sample_id]
    )
    assert covered["unassigned_eligible_item_fraction"] == 0.0
    bad = replace(good, split="model_selection")
    with pytest.raises(ValueError, match="split mismatch"):
        audit_manifest([bad], intervals())


def test_manifest_audit_needs_independent_eligible_universe_to_prove_coverage() -> None:
    first = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
    )
    second = make_item(
        t0=datetime(2020, 2, 1, 0, 5, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
    )

    with pytest.raises(ValueError, match="eligible items are unassigned"):
        audit_manifest(
            [first],
            intervals(),
            eligible_sample_ids=[first.sample_id, second.sample_id],
        )


def test_manifest_audit_rejects_one_block_in_two_splits() -> None:
    outer = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="crossing-event",
        stratum="event",
    )
    selection = make_item(
        t0=datetime(2022, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="model_selection",
        block_id="crossing-event",
        stratum="event",
    )

    with pytest.raises(ValueError, match="blocks cross split"):
        audit_manifest([outer, selection], intervals())


def test_manifest_audit_rejects_forged_sample_identity() -> None:
    item = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
    )
    with pytest.raises(ValueError, match="non-canonical"):
        audit_manifest([replace(item, sample_id="forged")], intervals())


def test_audit_only_cli_alias_reads_a_candidate_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    item = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
    )
    path = tmp_path / "candidate-manifest.json"
    write_manifest(
        path,
        [item],
        metadata={
            "split_intervals": [
                {
                    "name": interval.name,
                    "start": interval.start.isoformat(),
                    "end": interval.end.isoformat(),
                }
                for interval in intervals()
            ]
        },
    )

    assert audit_main(["--audit", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 1


def test_audit_only_cli_enforces_declared_fold_and_probability_contracts(
    tmp_path: Path,
) -> None:
    base = make_item(
        t0=datetime(2020, 2, 1, tzinfo=UTC),
        lead_hours=0.5,
        condition_signature="era5_full",
        split="outer_train",
        block_id="event-a",
        stratum="event",
        p_target=0.5,
        p_draw=0.5,
    )
    items = [
        replace(base, fold_id=0),
        replace(
            make_item(
                t0=datetime(2020, 2, 2, tzinfo=UTC),
                lead_hours=0.5,
                condition_signature="era5_full",
                split="outer_train",
                block_id="event-b",
                stratum="event",
                p_target=0.5,
                p_draw=0.5,
            ),
            fold_id=1,
        ),
    ]
    path = tmp_path / "candidate-manifest.json"
    metadata = {
        "outer_train_folds": 2,
        "split_intervals": [
            {
                "name": interval.name,
                "start": interval.start.isoformat(),
                "end": interval.end.isoformat(),
            }
            for interval in intervals()
        ],
    }
    write_manifest(path, items, metadata=metadata)
    assert audit_main(["--audit", str(path)]) == 0

    invalid = [replace(items[0], p_draw=0.25, omega=2.0), items[1]]
    write_manifest(tmp_path / "invalid.json", invalid, metadata=metadata)
    with pytest.raises(ValueError, match="draw probabilities do not sum"):
        audit_main(["--audit", str(tmp_path / "invalid.json")])

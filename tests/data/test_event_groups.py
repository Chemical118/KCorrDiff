from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kcorrdiff.data.event_groups import (
    Interval,
    classify_background,
    context_active,
    merge_event_intervals,
    utc_day_block,
    wet_components,
)


def dt(hours: float) -> datetime:
    return datetime(2022, 1, 1, tzinfo=UTC) + timedelta(hours=hours)


def test_context_active_excludes_invalid_and_uses_inclusive_threshold() -> None:
    field = np.full((8, 8), 0.153765)
    valid = np.ones((8, 8), dtype=bool)
    assert context_active(field, valid)
    valid[0, 0] = False
    assert not context_active(field, valid)


def test_eight_connected_significant_components_and_speckle() -> None:
    field = np.zeros((5, 5), dtype=np.float32)
    field[0, 0] = field[1, 1] = field[2, 2] = field[3, 3] = 0.1
    field[4, 0] = 1.0
    components = wet_components(field, np.ones_like(field, dtype=bool))
    assert [(c.pixels, c.significant) for c in components] == [(4, True), (1, True)]
    field[4, 0] = 0.5
    components = wet_components(field, np.ones_like(field, dtype=bool))
    assert components[1].significant is False


def test_gap_merge_guard_and_recursive_overlap() -> None:
    # A/B merge by the 12 h gap. C does not, but their 7 h guards then overlap.
    seeds = [Interval(dt(0), dt(1)), Interval(dt(13), dt(14)), Interval(dt(27), dt(28))]
    assert merge_event_intervals(seeds) == (Interval(dt(-7), dt(35)),)


def test_strict_dry_requires_zero_context_and_target() -> None:
    assert classify_background(
        any_target_wet=False,
        all_dependency_context_counts=[0, 0, 0],
        dependency_timestamps_present=[True, True, True],
        target_has_valid_pixels=True,
    ) == "strict_dry"
    assert classify_background(
        any_target_wet=True,
        all_dependency_context_counts=[0, 0],
        dependency_timestamps_present=[True, True],
        target_has_valid_pixels=True,
    ) == "marginal_background"
    assert classify_background(
        any_target_wet=False,
        all_dependency_context_counts=[0, 1],
        dependency_timestamps_present=[True, True],
        target_has_valid_pixels=True,
    ) == "marginal_background"
    assert utc_day_block(dt(25), "strict_dry") == "strict_dry:2022-01-02"


def test_missing_or_unverifiable_background_cannot_be_called_strict_dry() -> None:
    with pytest.raises(ValueError, match="missing dependency timestamp"):
        classify_background(
            any_target_wet=False,
            all_dependency_context_counts=[0, 0],
            dependency_timestamps_present=[True, False],
            target_has_valid_pixels=True,
        )
    with pytest.raises(ValueError, match="no valid pixels"):
        classify_background(
            any_target_wet=False,
            all_dependency_context_counts=[0, 0],
            dependency_timestamps_present=[True, True],
            target_has_valid_pixels=False,
        )


def test_valid_nonfinite_or_negative_values_fail_seed_construction() -> None:
    valid = np.ones((8, 8), dtype=bool)
    context = np.zeros((8, 8), dtype=float)
    context[0, 0] = np.nan
    with pytest.raises(ValueError, match="valid context pixel"):
        context_active(context, valid)

    target = np.zeros((2, 2), dtype=float)
    target[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        wet_components(target, np.ones_like(target, dtype=bool))

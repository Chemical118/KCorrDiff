from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.data.normalization import (
    NormalizationArtifact,
    StreamingChannelMoments,
    normalize_channel_first,
    read_normalization_artifact,
    sha256_file,
    write_normalization_artifact,
)


def test_float64_masked_moments_are_mergeable() -> None:
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 1000.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [[True, True], [True, False]],
            [[True, False], [True, True]],
        ]
    )
    complete = StreamingChannelMoments(("a", "b"))
    complete.update(values, valid_mask=mask)

    first = StreamingChannelMoments(("a", "b"))
    second = StreamingChannelMoments(("a", "b"))
    first.update(values[:, :1], valid_mask=mask[:, :1])
    second.update(values[:, 1:], valid_mask=mask[:, 1:])
    restored = StreamingChannelMoments.from_merge_state(first.merge_state())
    restored.merge(second)

    expected = complete.finalize()
    actual = restored.finalize()
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].count == expected[name].count == 3
        assert actual[name].mean == pytest.approx(expected[name].mean, rel=1e-15)
        assert actual[name].variance == pytest.approx(
            expected[name].variance, rel=1e-15
        )
    assert actual["a"].mean == pytest.approx(2.0)
    assert actual["a"].variance == pytest.approx(2.0 / 3.0)
    assert actual["b"].mean == pytest.approx(80.0 / 3.0)


def test_nonfinite_valid_values_and_zero_variance_fail_closed() -> None:
    moments = StreamingChannelMoments(("value",))
    with pytest.raises(ValueError, match="non-finite"):
        moments.update(np.asarray([[1.0, np.nan]]))

    # A masked non-value does not enter the statistic.
    moments.update(
        np.asarray([[1.0, np.nan]]),
        valid_mask=np.asarray([[True, False]]),
    )
    moments.update(np.asarray([[1.0]]))
    with pytest.raises(ValueError, match="zero/non-finite variance"):
        moments.finalize()

    empty = StreamingChannelMoments(("value",))
    empty.update(
        np.asarray([[np.nan]]), valid_mask=np.asarray([[False]])
    )
    with pytest.raises(ValueError, match="no valid samples"):
        empty.finalize()


def _artifact() -> NormalizationArtifact:
    moments = StreamingChannelMoments(("value",))
    moments.update(np.asarray([[1.0, 3.0]], dtype=np.float32))
    return NormalizationArtifact(
        statistics={"group": moments.finalize()},
        provenance={
            "artifact_hashes": {
                "candidate_manifest": "a" * 64,
                "cache": {"manifest": "b" * 64},
            },
            "time_universe_counts": {"outer_train_unique_t0": 1},
        },
        excluded_fields={"group": ("binary",)},
    )


def test_atomic_artifact_round_trip_and_named_application(tmp_path: Path) -> None:
    output = tmp_path / "normalization.json"
    digest = write_normalization_artifact(output, _artifact())
    assert digest == sha256_file(output)
    assert not list(tmp_path.glob(".normalization.json.*.tmp"))
    restored = read_normalization_artifact(output)
    assert restored.to_json() == _artifact().to_json()
    with pytest.raises(FileExistsError):
        write_normalization_artifact(output, _artifact())

    normalized = normalize_channel_first(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        channel_names=("value",),
        statistics=restored.statistics["group"],
    )
    np.testing.assert_allclose(normalized, [[-1.0, 0.0, 1.0]])

    raw = json.loads(output.read_text())
    raw["statistics"]["group"]["value"]["variance"] = 0.0
    output.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="strictly positive"):
        read_normalization_artifact(output)


def test_artifact_without_provenance_still_loads(tmp_path: Path) -> None:
    output = tmp_path / "normalization.json"
    write_normalization_artifact(output, _artifact())
    raw = json.loads(output.read_text())
    del raw["provenance"]
    del raw["statistics_semantic_sha256"]
    stripped = tmp_path / "stripped.json"
    stripped.write_text(json.dumps(raw))
    restored = read_normalization_artifact(stripped)
    assert restored.provenance == {}
    assert restored.statistics["group"]["value"].count == 2

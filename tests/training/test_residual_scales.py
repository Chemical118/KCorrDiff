from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.training.residual_scales import (
    ResidualScaleAccumulator,
    scale_lookup,
    write_residual_scales,
)


CONDITION = "era5_oracle:era=1:tp=1:full_trajectory"


def test_scales_are_weighted_rms_with_floor_and_provenance(tmp_path: Path) -> None:
    accumulator = ResidualScaleAccumulator(
        epsilon_scale=0.25,
        minimum_independent_blocks=1,
        minimum_block_ess=1.0,
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=CONDITION,
        block_id="block-a",
        target_z=np.asarray([1.0, 2.0]),
        mu_z_oof=np.asarray([0.0, 0.0]),
        target_validity=np.asarray([True, True]),
        omega=2.0,
    )
    accumulator.update(
        lead_hours=1.0,
        condition_signature=CONDITION,
        block_id="block-b",
        target_z=np.zeros(2),
        mu_z_oof=np.zeros(2),
        target_validity=np.asarray([True, True]),
        omega=1.0,
    )
    path = tmp_path / "scales.json"
    digest = write_residual_scales(
        path,
        accumulator,
        oof_manifest_sha256="a" * 64,
        regression_checkpoint_set_sha256="b" * 64,
    )
    assert len(digest) == 64
    raw = json.loads(path.read_text())
    lookup = scale_lookup(raw)
    assert lookup[(0.5, CONDITION)] == pytest.approx(np.sqrt(2.5))
    assert lookup[(1.0, CONDITION)] == 0.25
    assert raw["records"][1]["epsilon_applied"] is True
    assert raw["pooling_order"] == [
        "full_cell",
        "lead_provider_era_present",
        "lead_provider",
        "lead_only",
    ]
    assert raw["records"][0]["pooling_level"] == "full_cell"


def test_scale_accumulator_rejects_invalid_or_empty_cells() -> None:
    accumulator = ResidualScaleAccumulator()
    with pytest.raises(ValueError, match="no residual"):
        accumulator.result()
    with pytest.raises(ValueError, match="official 12"):
        accumulator.update(
            lead_hours=0.25,
            condition_signature=CONDITION,
            block_id="block-a",
            target_z=[0.0],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
        )
    with pytest.raises(ValueError, match="canonical non-empty block_id"):
        accumulator.update(
            lead_hours=0.5,
            condition_signature=CONDITION,
            block_id=" block-a",
            target_z=[0.0],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
        )


def test_scale_multiplicity_replays_duplicate_training_draw_rows() -> None:
    accumulator = ResidualScaleAccumulator(
        minimum_independent_blocks=1, minimum_block_ess=1.0
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=CONDITION,
        block_id="block-a",
        target_z=np.asarray([1.0, 3.0]),
        mu_z_oof=np.asarray([0.0, 0.0]),
        target_validity=np.asarray([True, True]),
        omega=2.0,
        multiplicity=4,
    )
    record = accumulator.result()["records"][0]  # type: ignore[index]
    assert record["items"] == 4
    assert record["valid_pixels"] == 8
    assert record["importance_weight_sum"] == 16.0
    assert record["raw_rms"] == pytest.approx(np.sqrt(5.0))
    with pytest.raises(ValueError, match="multiplicity"):
        accumulator.update(
            lead_hours=0.5,
            condition_signature=CONDITION,
            block_id="block-a",
            target_z=[1.0],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
            multiplicity=0,
        )


def test_batched_residual_update_matches_sample_updates() -> None:
    targets = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, 3.0]]],
        dtype=np.float32,
    )
    means = np.asarray(
        [[[0.5, 1.0], [2.0, 3.0]], [[1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float32,
    )
    validity = np.asarray(
        [[[True, True], [False, True]], [[True, False], [True, True]]]
    )
    weights = np.asarray([2.0, 0.5], dtype=np.float64)
    multiplicities = np.asarray([3, 2], dtype=np.int64)
    sequential = ResidualScaleAccumulator(
        minimum_independent_blocks=1, minimum_block_ess=1.0
    )
    batched = ResidualScaleAccumulator(
        minimum_independent_blocks=1, minimum_block_ess=1.0
    )
    for index in range(2):
        sequential.update(
            lead_hours=0.5 + 0.5 * index,
            condition_signature=CONDITION,
            block_id=f"block-{index}",
            target_z=targets[index],
            mu_z_oof=means[index],
            target_validity=validity[index],
            omega=float(weights[index]),
            multiplicity=int(multiplicities[index]),
        )
    batched.update_batch(
        lead_hours=np.asarray([0.5, 1.0]),
        condition_signatures=(CONDITION, CONDITION),
        block_ids=("block-0", "block-1"),
        target_z=targets,
        mu_z_oof=means,
        target_validity=validity,
        omega=weights,
        multiplicities=multiplicities,
    )
    assert batched.merge_state() == sequential.merge_state()


def test_scale_pooling_uses_full_cell_then_fallbacks_deterministically() -> None:
    tp_absent = "era5_oracle:era=1:tp=0:full_trajectory"
    accumulator = ResidualScaleAccumulator(
        minimum_independent_blocks=2,
        minimum_block_ess=2.0,
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=CONDITION,
        block_id="block-a",
        target_z=[1.0],
        mu_z_oof=[0.0],
        target_validity=[True],
        omega=1.0,
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=tp_absent,
        block_id="block-b",
        target_z=[3.0],
        mu_z_oof=[0.0],
        target_validity=[True],
        omega=1.0,
    )
    records = accumulator.result()["records"]
    assert isinstance(records, list)
    assert {record["pooling_level"] for record in records} == {
        "lead_provider_era_present"
    }
    assert all(record["scale"] == pytest.approx(np.sqrt(5.0)) for record in records)
    assert [level["block_count"] for level in records[0]["pooling_ladder"]] == [
        1,
        2,
        2,
        2,
    ]


def test_terminal_scale_is_explicitly_unsupported_never_identity() -> None:
    accumulator = ResidualScaleAccumulator()
    accumulator.update(
        lead_hours=0.5,
        condition_signature=CONDITION,
        block_id="only-block",
        target_z=[2.0],
        mu_z_oof=[1.0],
        target_validity=[True],
        omega=1.0,
    )
    record = accumulator.result()["records"][0]  # type: ignore[index]
    assert record["terminal_fallback"] is True
    assert record["diffusion_scale_unsupported"] is True
    assert record["pooling_level"] is None
    assert record["scale"] is None and record["raw_rms"] is None
    assert scale_lookup(accumulator.result()) == {}


def test_pooling_state_round_trip_retains_block_mass_and_support() -> None:
    first = ResidualScaleAccumulator(
        minimum_independent_blocks=2, minimum_block_ess=2.0
    )
    for index in range(2):
        first.update(
            lead_hours=1.0,
            condition_signature=CONDITION,
            block_id=f"block-{index}",
            target_z=[float(index + 1)],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
        )
    restored = ResidualScaleAccumulator.from_merge_state(first.merge_state())
    assert restored.result() == first.result()


def test_pooling_state_rejects_coercible_scalar_aliases() -> None:
    accumulator = ResidualScaleAccumulator(
        minimum_independent_blocks=1, minimum_block_ess=1.0
    )
    accumulator.update(
        lead_hours=1.0,
        condition_signature=CONDITION,
        block_id="block-a",
        target_z=[1.0],
        mu_z_oof=[0.0],
        target_validity=[True],
        omega=1.0,
    )
    state = accumulator.merge_state()

    mutations = (
        (("epsilon_scale",), True),
        (("minimum_independent_blocks",), "1"),
        (("minimum_block_ess",), False),
        (("records", 0, "lead_hours"), "1.0"),
        (("records", 0, "pooling_level"), 1),
        (("records", 0, "pooling_key"), True),
        (("records", 0, "weighted_square_sum"), False),
        (("records", 0, "weight_sum"), "1.0"),
        (("records", 0, "valid_pixels"), True),
        (("records", 0, "items"), "1"),
        (("records", 0, "block_masses", 0, "block_id"), 1),
        (("records", 0, "block_masses", 0, "mass"), True),
    )
    for path, replacement in mutations:
        changed = deepcopy(state)
        target: object = changed
        for component in path[:-1]:
            target = target[component]  # type: ignore[index]
        target[path[-1]] = replacement  # type: ignore[index]
        with pytest.raises((TypeError, ValueError)):
            ResidualScaleAccumulator.from_merge_state(changed)


def test_pooling_state_rejects_noncanonical_or_ambiguous_block_ids() -> None:
    accumulator = ResidualScaleAccumulator(
        minimum_independent_blocks=1, minimum_block_ess=1.0
    )
    accumulator.update(
        lead_hours=1.0,
        condition_signature=CONDITION,
        block_id="block-a",
        target_z=[1.0],
        mu_z_oof=[0.0],
        target_validity=[True],
        omega=1.0,
    )
    state = accumulator.merge_state()
    state["records"][0]["block_masses"][0]["block_id"] = " block-a"  # type: ignore[index]
    with pytest.raises(TypeError, match="canonical non-empty string"):
        ResidualScaleAccumulator.from_merge_state(state)

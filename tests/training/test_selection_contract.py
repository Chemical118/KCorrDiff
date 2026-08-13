from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.selection_contract import load_stage2_selection_contract


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stage2-full-width.yaml"
SELECTION = ROOT / "configs" / "stage2-loader-selection.json"


def _load(path: Path = SELECTION):
    config = load_stage2_config(CONFIG)
    return load_stage2_selection_contract(
        path,
        expected_sha256=sha256_file(path),
        config=config,
        world_size=2,
        num_workers=12,
        prefetch_factor=2,
    )


def test_current_selection_is_semantically_bound_to_stage2() -> None:
    selected = _load()
    assert selected.per_rank_microbatch_size == 8
    assert selected.global_effective_batch_size == 16
    selected.validate_factory_lineage(
        {
            "candidate_manifest": selected.artifact_lineage["candidate_manifest_sha256"],
            "draw_manifest": selected.artifact_lineage["draw_manifest_sha256"],
            "bundle_metadata": selected.artifact_lineage["bundle_metadata_sha256"],
            "radar_cache_manifest": selected.artifact_lineage["radar_cache_manifest_sha256"],
            "era5_cache_manifest": selected.artifact_lineage["era5_cache_manifest_sha256"],
            "normalization": selected.artifact_lineage["normalization_sha256"],
        }
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda raw: raw["selected"].__setitem__("per_rank_microbatch_size", 4), "topology"),
        (lambda raw: raw["decision"].__setitem__("selection_evidence_uses_only_power_of_two_batch_sizes", False), "power-of-two"),
        (lambda raw: raw["runtime_contract"].__setitem__("precision", "float16"), "runtime/model"),
        (lambda raw: raw["model_benchmarks"]["power_observation_batch_16"].__setitem__("batch_size", 12), "acceptance boundary"),
    ],
)
def test_hash_valid_but_semantically_forged_selection_is_rejected(
    tmp_path: Path, mutator, message: str
) -> None:
    raw = json.loads(SELECTION.read_text())
    mutator(raw)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(raw, sort_keys=True) + "\n")
    config = load_stage2_config(CONFIG)
    with pytest.raises(ValueError, match=message):
        load_stage2_selection_contract(
            path,
            expected_sha256=sha256_file(path),
            config=config,
            world_size=2,
            num_workers=12,
            prefetch_factor=2,
        )


def test_factory_lineage_mismatch_is_rejected() -> None:
    selected = _load()
    with pytest.raises(ValueError, match="lineage mismatch"):
        selected.validate_factory_lineage(
            {
                "candidate_manifest": "0" * 64,
                "draw_manifest": selected.artifact_lineage["draw_manifest_sha256"],
                "bundle_metadata": selected.artifact_lineage["bundle_metadata_sha256"],
                "radar_cache_manifest": selected.artifact_lineage["radar_cache_manifest_sha256"],
                "era5_cache_manifest": selected.artifact_lineage["era5_cache_manifest_sha256"],
                "normalization": selected.artifact_lineage["normalization_sha256"],
            }
        )

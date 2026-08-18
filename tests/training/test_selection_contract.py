from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcorrdiff.training.selection_contract import load_stage2_selection_contract


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "configs" / "stage2-loader-selection.json"


def test_selection_artifact_reads_selected_loader_settings() -> None:
    selected = load_stage2_selection_contract(SELECTION)
    assert selected.per_rank_microbatch_size == 8
    assert selected.global_effective_batch_size == 16
    assert selected.num_workers >= 0
    assert selected.prefetch_factor >= 1


def test_selection_artifact_requires_a_selected_mapping(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"format_version": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="selected"):
        load_stage2_selection_contract(path)

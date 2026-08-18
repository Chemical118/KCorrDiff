"""Reader for the Stage 2 loader/model benchmark selection artifact.

Research code: the artifact is an informational record of a past tuning run.
This module reads the selected loader settings; nothing is hashed, pinned to
a device or world size, or compared against the live config.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


SELECTION_FORMAT = "kcorrdiff.loader-and-model-selection.v2"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Stage2SelectionContract:
    artifact_path: Path
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    global_effective_batch_size: int
    num_workers: int
    prefetch_factor: int


def load_stage2_selection_contract(path: Path) -> Stage2SelectionContract:
    """Read the selected loader settings from a selection artifact."""

    selected_path = Path(path).resolve()
    raw = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("selected"), Mapping):
        raise ValueError("selection artifact must contain a 'selected' mapping")
    selected = raw["selected"]
    workers = selected["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("selected num_workers must be a non-negative integer")
    return Stage2SelectionContract(
        artifact_path=selected_path,
        per_rank_microbatch_size=_positive_int(
            selected["per_rank_microbatch_size"], name="selected batch"
        ),
        gradient_accumulation_steps=_positive_int(
            selected["gradient_accumulation_steps"], name="selected accumulation"
        ),
        global_effective_batch_size=_positive_int(
            selected["global_effective_batch_size"], name="selected global batch"
        ),
        num_workers=workers,
        prefetch_factor=_positive_int(
            selected["prefetch_factor"], name="selected prefetch_factor"
        ),
    )


__all__ = [
    "SELECTION_FORMAT",
    "Stage2SelectionContract",
    "load_stage2_selection_contract",
]

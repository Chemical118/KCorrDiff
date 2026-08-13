"""Chronological split, dependency embargo, block, and fold contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from .availability import LEADS_HOURS, dependency_times


SplitName = Literal["outer_train", "model_selection", "calibration", "final_test"]
SPLIT_PRIORITY: tuple[SplitName, ...] = (
    "final_test",
    "calibration",
    "model_selection",
    "outer_train",
)


@dataclass(frozen=True, slots=True)
class SplitInterval:
    name: SplitName
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("split boundaries must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("split interval must be non-empty")

    def contains_dependency(self, start: datetime, end: datetime) -> bool:
        return self.start <= start and end < self.end


@dataclass(frozen=True, slots=True)
class ManifestItem:
    sample_id: str
    t0_utc: str
    lead_hours: float
    condition_signature: str
    dependency_start_utc: str
    dependency_end_utc: str
    split: SplitName
    block_id: str
    stratum: str
    fold_id: int | None = None
    p_target: float = 1.0
    p_draw: float = 1.0
    omega: float = 1.0


def parse_utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"naive manifest time: {value}")
    return result.astimezone(UTC)


def validate_split_intervals(intervals: Sequence[SplitInterval]) -> None:
    names = [item.name for item in intervals]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    chronological = sorted(intervals, key=lambda item: item.start)
    for prior, current in zip(chronological, chronological[1:], strict=False):
        if current.start < prior.end:
            raise ValueError(f"split intervals overlap: {prior.name}, {current.name}")


def assign_split(
    t0: datetime,
    intervals: Sequence[SplitInterval],
    *,
    max_lead_hours: float = 6.0,
) -> SplitName | None:
    """Assign only when the complete ``[t0-55m,t0+6h]`` lies in one split."""

    validate_split_intervals(intervals)
    dependency = dependency_times(t0, max_lead_hours)
    matches = [
        interval.name
        for interval in intervals
        if interval.contains_dependency(dependency[0], dependency[-1])
    ]
    if len(matches) > 1:
        raise AssertionError("overlapping split dependency interiors")
    return matches[0] if matches else None


def boundary_embargo_fraction(
    candidate_t0: Iterable[datetime], intervals: Sequence[SplitInterval]
) -> float:
    values = tuple(candidate_t0)
    if not values:
        return 0.0
    excluded = sum(assign_split(value, intervals) is None for value in values)
    return excluded / len(values)


def canonical_sample_id(t0: datetime, lead_hours: float, signature: str) -> str:
    if t0.tzinfo is None:
        raise ValueError("t0 must be timezone-aware")
    identity = f"{t0.astimezone(UTC).isoformat()}|{lead_hours:.1f}|{signature}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def make_item(
    *,
    t0: datetime,
    lead_hours: float,
    condition_signature: str,
    split: SplitName,
    block_id: str,
    stratum: str,
    p_target: float = 1.0,
    p_draw: float = 1.0,
) -> ManifestItem:
    if lead_hours not in LEADS_HOURS:
        raise ValueError("lead must be an official half-hour lead")
    if p_target <= 0 or p_draw <= 0:
        raise ValueError("sampling probabilities must be positive")
    # The official manifest dependency is fixed at [t0-55m, t0+6h].  Building
    # all 84 intermediate datetimes per lead item is unnecessary and costly
    # when expanding the real 928k-item CPrecNet universe.
    dependency_start = t0 - timedelta(minutes=55)
    dependency_end = t0 + timedelta(hours=6)
    return ManifestItem(
        sample_id=canonical_sample_id(t0, lead_hours, condition_signature),
        t0_utc=t0.astimezone(UTC).isoformat(),
        lead_hours=lead_hours,
        condition_signature=condition_signature,
        dependency_start_utc=dependency_start.astimezone(UTC).isoformat(),
        dependency_end_utc=dependency_end.astimezone(UTC).isoformat(),
        split=split,
        block_id=block_id,
        stratum=stratum,
        p_target=p_target,
        p_draw=p_draw,
        omega=p_target / p_draw,
    )


def assign_grouped_folds(
    items: Sequence[ManifestItem], *, folds: int = 3, seed: int = 11103
) -> tuple[ManifestItem, ...]:
    """Assign entire outer-train manifest blocks to deterministic balanced folds."""

    if folds < 2:
        raise ValueError("at least two folds are required")
    block_sizes: dict[str, int] = {}
    block_split: dict[str, SplitName] = {}
    for item in items:
        if item.block_id in block_split and block_split[item.block_id] != item.split:
            raise ValueError(f"block crosses splits: {item.block_id}")
        block_split[item.block_id] = item.split
        if item.split == "outer_train":
            block_sizes[item.block_id] = block_sizes.get(item.block_id, 0) + 1
    if block_sizes and len(block_sizes) < folds:
        raise ValueError(
            f"outer_train has {len(block_sizes)} blocks, fewer than {folds} folds"
        )
    # Largest-first load balancing. Hash is a stable seeded tie-breaker.
    ordered = sorted(
        block_sizes,
        key=lambda block: (
            -block_sizes[block],
            hashlib.sha256(f"{seed}\0{block}".encode()).hexdigest(),
        ),
    )
    loads = [0] * folds
    assignments: dict[str, int] = {}
    for block in ordered:
        fold = min(range(folds), key=lambda candidate: (loads[candidate], candidate))
        assignments[block] = fold
        loads[fold] += block_sizes[block]
    return tuple(
        replace(item, fold_id=assignments.get(item.block_id))
        for item in items
    )


def audit_manifest(
    items: Sequence[ManifestItem],
    intervals: Sequence[SplitInterval],
    *,
    eligible_sample_ids: Iterable[str] | None = None,
    expected_outer_folds: int | None = None,
    validate_split_probabilities: bool = False,
) -> dict[str, object]:
    """Fail closed on duplicate identity, split leakage, or sampling drift.

    Full-timeline coverage can only be proven when the independently derived
    eligible universe is supplied.  Without ``eligible_sample_ids`` the audit
    reports coverage as unknown instead of asserting a tautological zero
    unassigned fraction from the manifest itself.
    """

    validate_split_intervals(intervals)
    identities: set[tuple[str, float, str]] = set()
    sample_ids: set[str] = set()
    block_splits: dict[str, set[str]] = {}
    block_folds: dict[str, set[int | None]] = {}
    split_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    split_target_probabilities: dict[str, list[float]] = {}
    split_draw_probabilities: dict[str, list[float]] = {}
    populated_outer_folds: set[int] = set()
    if expected_outer_folds is not None and expected_outer_folds < 2:
        raise ValueError("expected_outer_folds must be at least two")
    for item in items:
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (item.p_target, item.p_draw, item.omega)
        ):
            raise ValueError(f"invalid sampling value: {item.sample_id}")
        identity = (item.t0_utc, item.lead_hours, item.condition_signature)
        if identity in identities:
            raise ValueError(f"duplicate manifest item: {identity}")
        identities.add(identity)
        if item.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {item.sample_id}")
        sample_ids.add(item.sample_id)
        if item.lead_hours not in LEADS_HOURS:
            raise ValueError(f"non-canonical lead: {item.lead_hours}")
        if not item.condition_signature or not item.block_id or not item.stratum:
            raise ValueError(f"empty manifest identity field: {item.sample_id}")
        if not np.isclose(item.omega, item.p_target / item.p_draw):
            raise ValueError(f"incorrect importance weight: {item.sample_id}")
        t0 = parse_utc(item.t0_utc)
        expected_id = canonical_sample_id(
            t0, item.lead_hours, item.condition_signature
        )
        if item.sample_id != expected_id:
            raise ValueError(f"non-canonical sample_id: {item.sample_id}")
        dependency_start = t0 - timedelta(minutes=55)
        dependency_end = t0 + timedelta(hours=6)
        matches = [
            interval.name
            for interval in intervals
            if interval.contains_dependency(dependency_start, dependency_end)
        ]
        if len(matches) > 1:
            raise AssertionError("overlapping split dependency interiors")
        expected_split = matches[0] if matches else None
        if expected_split != item.split:
            raise ValueError(
                f"dependency split mismatch for {item.sample_id}: "
                f"{item.split} != {expected_split}"
            )
        if parse_utc(item.dependency_start_utc) != dependency_start:
            raise ValueError(f"dependency start mismatch: {item.sample_id}")
        if parse_utc(item.dependency_end_utc) != dependency_end:
            raise ValueError(f"dependency end mismatch: {item.sample_id}")
        block_splits.setdefault(item.block_id, set()).add(item.split)
        block_folds.setdefault(item.block_id, set()).add(item.fold_id)
        split_counts[item.split] = split_counts.get(item.split, 0) + 1
        stratum_counts[item.stratum] = stratum_counts.get(item.stratum, 0) + 1
        split_target_probabilities.setdefault(item.split, []).append(item.p_target)
        split_draw_probabilities.setdefault(item.split, []).append(item.p_draw)
        if expected_outer_folds is not None:
            if item.split == "outer_train":
                if (
                    item.fold_id is None
                    or isinstance(item.fold_id, bool)
                    or not isinstance(item.fold_id, int)
                    or not 0 <= item.fold_id < expected_outer_folds
                ):
                    raise ValueError(f"invalid outer_train fold: {item.sample_id}")
                populated_outer_folds.add(item.fold_id)
            elif item.fold_id is not None:
                raise ValueError(f"holdout item has a fold: {item.sample_id}")
    leaking = sorted(block for block, splits in block_splits.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"blocks cross split boundaries: {leaking[:5]}")
    split_folds = sorted(
        block for block, values in block_folds.items() if len(values) != 1
    )
    if split_folds:
        raise ValueError(f"blocks cross fold boundaries: {split_folds[:5]}")
    if expected_outer_folds is not None and split_counts.get("outer_train", 0):
        expected = set(range(expected_outer_folds))
        if populated_outer_folds != expected:
            raise ValueError(
                f"outer_train folds are not all populated: "
                f"{sorted(populated_outer_folds)} != {sorted(expected)}"
            )
    if validate_split_probabilities:
        for split, probabilities in split_target_probabilities.items():
            if not np.isclose(
                np.asarray(probabilities, dtype=np.float64).sum(),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"target probabilities do not sum to one: {split}")
        for split, probabilities in split_draw_probabilities.items():
            if not np.isclose(
                np.asarray(probabilities, dtype=np.float64).sum(),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"draw probabilities do not sum to one: {split}")
    unassigned_fraction: float | None = None
    if eligible_sample_ids is not None:
        eligible_values = tuple(eligible_sample_ids)
        eligible = set(eligible_values)
        if len(eligible) != len(eligible_values):
            raise ValueError("eligible_sample_ids contains duplicates")
        assigned = {item.sample_id for item in items}
        unexpected = sorted(assigned - eligible)
        if unexpected:
            raise ValueError(f"manifest contains ineligible item(s): {unexpected[:5]}")
        unassigned = eligible - assigned
        unassigned_fraction = len(unassigned) / len(eligible) if eligible else 0.0
        if unassigned:
            raise ValueError(
                "eligible items are unassigned: "
                f"{len(unassigned)}/{len(eligible)}"
            )
    return {
        "items": len(items),
        "unique_blocks": len(block_splits),
        "split_counts": split_counts,
        "stratum_counts": stratum_counts,
        "unassigned_eligible_item_fraction": unassigned_fraction,
    }


def write_manifest(
    path: Path,
    items: Sequence[ManifestItem],
    *,
    metadata: Mapping[str, object],
) -> str:
    """Atomically stream the canonical JSON manifest and return its hash.

    Streaming avoids constructing a second list/dict/encoded copy of the real
    ~929k-item CPrecNet candidate universe in memory.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        def emit(payload: bytes) -> None:
            stream.write(payload)
            digest.update(payload)

        emit(b'{"items":[')
        for index, item in enumerate(items):
            if index:
                emit(b",")
            emit(
                json.dumps(
                    asdict(item), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        emit(b'],"metadata":')
        emit(
            json.dumps(
                dict(metadata), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        emit(b',"version":"kcorrdiff.split-manifest.v1"}\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args(argv)


def audit_main(argv: Sequence[str] | None = None) -> int:
    """Audit one published candidate manifest.

    Kept separate from :func:`main` so the installed
    ``kcorrdiff-audit-manifest`` alias has an audit-only help/argument surface.
    """

    arguments = _parse_args(argv)
    raw = json.loads(arguments.audit.read_text())
    intervals = tuple(
        SplitInterval(
            name=item["name"],
            start=parse_utc(item["start"]),
            end=parse_utc(item["end"]),
        )
        for item in raw["metadata"]["split_intervals"]
    )
    items = tuple(ManifestItem(**item) for item in raw["items"])
    expected_folds_raw = raw["metadata"].get("outer_train_folds")
    expected_folds = (
        int(expected_folds_raw) if expected_folds_raw is not None else None
    )
    print(
        json.dumps(
            audit_manifest(
                items,
                intervals,
                expected_outer_folds=expected_folds,
                validate_split_probabilities=True,
            ),
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--audit" in values:
        return audit_main(values)

    # Keep the installed ``kcorrdiff-build-manifest`` entry point while the
    # integrated CPrecNet event-index/candidate/draw implementation lives next
    # to the compact event-index contract.  The import is deliberately lazy to
    # avoid the module cycle at import time.
    from .event_manifest import main as build_event_training_bundle

    return build_event_training_bundle(values)


if __name__ == "__main__":
    raise SystemExit(main())

"""Compact event-conditioned CPrecNet availability and split index.

The archive is label-selected, so this is explicitly *not* the official
continuous-timeline event/dry/background manifest.  It builds strict 84-frame
windows, leakage-safe event blocks, chronological holdouts, and a compact row
per issue time.  The 12 lead items are expanded only by the bounded training
draw manifest, avoiding a large duplicated metadata table.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import yaml

from .availability import (
    LEADS_HOURS,
    AvailabilityStats,
    availability_stats,
    continuous_runs,
    dependency_times,
    intersection_timestamps,
    require_five_minute_slot,
)
from .cache import paired_archives
from .condition_augmentation import (
    ConditionAugmentationPolicy,
    canonical_draw_rows_sha256,
    expand_manifest_condition_support,
    materialize_condition_signatures,
)
from .event_groups import Interval, containing_event, merge_event_intervals
from .sampling import (
    CandidateItem,
    DrawRow,
    bounded_draw_manifest,
    enforce_byte_budget,
    full_coverage_shuffled_draw_manifest,
    oof_dense_byte_budget,
    write_draw_manifest,
)
from .split_manifest import (
    SPLIT_PRIORITY,
    ManifestItem,
    SplitInterval,
    SplitName,
    assign_grouped_folds,
    assign_split,
    audit_manifest,
    canonical_sample_id,
    make_item,
    write_manifest,
)


ALL_12_LEADS_MASK = (1 << 12) - 1
EVENT_STRATUM = "event"
OUTER_TRAIN_FOLDS = 3
BUNDLE_FORMAT_VERSION = "kcorrdiff.event-training-bundle.v1"
DrawPolicy = Literal["uniform", "block-balanced", "full-coverage-shuffled"]


@dataclass(frozen=True, slots=True)
class EventIndexRow:
    t0_utc: datetime
    block_id: str
    split: SplitName
    condition_signature: str
    available_leads_mask: int = ALL_12_LEADS_MASK


@dataclass(frozen=True, slots=True)
class EventIndexAudit:
    availability: AvailabilityStats
    strict_issue_times_before_embargo: int
    issue_times: int
    expanded_lead_items: int
    event_blocks: int
    split_counts: Mapping[str, int]
    boundary_embargo_fraction: float
    dependency_embargo_exclusions: int
    cross_split_event_exclusions: int
    total_exclusion_fraction: float
    low_support_splits: tuple[str, ...]
    estimand: str = "cprecnet_event_conditioned_pretraining"
    condition_signature: str = ""
    event_merge_gap_hours: float = 12.0
    event_guard_hours: float = 7.0
    source_timestamps_sha256: str = ""
    split_intervals_sha256: str = ""


@dataclass(frozen=True, slots=True)
class TrainingBundleResult:
    output_dir: Path
    event_rows: int
    candidates: int
    draws: int
    unique_oof_items: int
    oof_required_bytes: int
    metadata_sha256: str


def _availability_from_json(raw: Mapping[str, object]) -> AvailabilityStats:
    return AvailabilityStats(
        timestamps=int(raw["timestamps"]),
        continuous_runs=int(raw["continuous_runs"]),
        strict_six_hour_windows=int(raw["strict_six_hour_windows"]),
        supporting_runs=int(raw["supporting_runs"]),
        nonoverlapping_84_frame_blocks=int(
            raw["nonoverlapping_84_frame_blocks"]
        ),
    )


def _event_audit_from_json(raw: Mapping[str, object]) -> EventIndexAudit:
    availability = raw.get("availability")
    if not isinstance(availability, Mapping):
        raise ValueError("event-index audit has no availability mapping")
    split_counts = raw.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("event-index audit has no split_counts mapping")
    low_support = raw.get("low_support_splits", ())
    if not isinstance(low_support, Sequence) or isinstance(low_support, str):
        raise ValueError("event-index low_support_splits must be a sequence")
    return EventIndexAudit(
        availability=_availability_from_json(availability),
        strict_issue_times_before_embargo=int(
            raw["strict_issue_times_before_embargo"]
        ),
        issue_times=int(raw["issue_times"]),
        expanded_lead_items=int(raw["expanded_lead_items"]),
        event_blocks=int(raw["event_blocks"]),
        split_counts={str(key): int(value) for key, value in split_counts.items()},
        boundary_embargo_fraction=float(raw["boundary_embargo_fraction"]),
        dependency_embargo_exclusions=int(raw["dependency_embargo_exclusions"]),
        cross_split_event_exclusions=int(raw["cross_split_event_exclusions"]),
        total_exclusion_fraction=float(raw["total_exclusion_fraction"]),
        low_support_splits=tuple(str(value) for value in low_support),
        estimand=str(
            raw.get("estimand", "cprecnet_event_conditioned_pretraining")
        ),
        condition_signature=str(raw.get("condition_signature", "")),
        event_merge_gap_hours=float(raw.get("event_merge_gap_hours", 12.0)),
        event_guard_hours=float(raw.get("event_guard_hours", 7.0)),
        source_timestamps_sha256=str(raw.get("source_timestamps_sha256", "")),
        split_intervals_sha256=str(raw.get("split_intervals_sha256", "")),
    )


def _event_id(interval: Interval) -> str:
    value = f"{interval.start.astimezone(UTC).isoformat()}|{interval.end.astimezone(UTC).isoformat()}"
    return f"event-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def strict_issue_times(timestamps: Iterable[datetime]) -> tuple[datetime, ...]:
    """Issue times with all 84 frames from ``t0-55m`` through ``t0+6h``."""

    result: list[datetime] = []
    for run in continuous_runs(timestamps):
        if len(run) >= 84:
            result.extend(run[11 : len(run) - 72])
    return tuple(result)


def _canonical_event_rows(
    rows: Sequence[EventIndexRow],
) -> tuple[EventIndexRow, ...]:
    """Validate and canonically order compact event-index rows."""

    identities: set[tuple[datetime, str]] = set()
    issue_assignments: dict[datetime, tuple[str, str]] = {}
    block_splits: dict[str, set[str]] = {}
    result: list[EventIndexRow] = []
    valid_splits = set(SPLIT_PRIORITY)
    for row in rows:
        if row.t0_utc.tzinfo is None:
            raise ValueError("event-index t0 must be timezone-aware")
        t0 = row.t0_utc.astimezone(UTC)
        require_five_minute_slot(t0)
        if not row.block_id or not row.condition_signature:
            raise ValueError("event-index block and condition signature must be non-empty")
        if row.split not in valid_splits:
            raise ValueError(f"unknown event-index split: {row.split}")
        mask = row.available_leads_mask
        if (
            isinstance(mask, bool)
            or not isinstance(mask, int)
            or mask <= 0
            or mask & ~ALL_12_LEADS_MASK
        ):
            raise ValueError(f"invalid available-leads mask: {mask!r}")
        identity = (t0, row.condition_signature)
        if identity in identities:
            raise ValueError(f"duplicate compact event identity: {identity}")
        identities.add(identity)
        assignment = (row.block_id, row.split)
        prior = issue_assignments.setdefault(t0, assignment)
        if prior != assignment:
            raise ValueError(f"one issue time has inconsistent event assignment: {t0}")
        block_splits.setdefault(row.block_id, set()).add(row.split)
        result.append(replace(row, t0_utc=t0))
    leaking = sorted(
        block for block, splits in block_splits.items() if len(splits) != 1
    )
    if leaking:
        raise ValueError(f"event-index blocks cross splits: {leaking[:5]}")
    return tuple(
        sorted(
            result,
            key=lambda row: (
                row.t0_utc,
                row.condition_signature,
                row.block_id,
            ),
        )
    )


def _masked_leads(mask: int) -> tuple[float, ...]:
    return tuple(
        lead for index, lead in enumerate(LEADS_HOURS) if mask & (1 << index)
    )


def eligible_sample_ids_from_event_index(
    rows: Sequence[EventIndexRow],
) -> tuple[str, ...]:
    """Independently derive the complete semantic candidate-ID universe.

    This intentionally does not inspect a materialized candidate manifest, so
    passing the result to :func:`audit_manifest` is a real completeness check
    rather than a set derived tautologically from the artifact under audit.
    """

    identifiers: list[str] = []
    for row in _canonical_event_rows(rows):
        for lead in _masked_leads(row.available_leads_mask):
            identifiers.append(
                canonical_sample_id(row.t0_utc, lead, row.condition_signature)
            )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("compact event index expands to duplicate sample IDs")
    return tuple(identifiers)


def expand_event_candidates(
    rows: Sequence[EventIndexRow],
    *,
    draw_policy: DrawPolicy = "uniform",
    folds: int = OUTER_TRAIN_FOLDS,
    fold_seed: int = 11103,
) -> tuple[ManifestItem, ...]:
    """Expand compact event rows into all available lead-conditioned items.

    ``P_target`` is item-uniform *within each chronological split*.  ``P_draw``
    is either the same uniform distribution or gives each event block equal
    mass and each item within that block equal conditional mass.  Only the
    explicit CPrecNet ``event`` stratum is emitted; no absent dry/background
    population is manufactured.
    """

    if draw_policy not in (
        "uniform",
        "block-balanced",
        "full-coverage-shuffled",
    ):
        raise ValueError(f"unsupported draw policy: {draw_policy}")
    canonical_rows = _canonical_event_rows(rows)
    if not canonical_rows:
        raise ValueError("compact event index has no eligible lead items")
    split_sizes: dict[str, int] = {}
    block_sizes: dict[tuple[str, str], int] = {}
    split_blocks: dict[str, set[str]] = {}
    for row in canonical_rows:
        lead_count = row.available_leads_mask.bit_count()
        split_sizes[row.split] = split_sizes.get(row.split, 0) + lead_count
        key = (row.split, row.block_id)
        block_sizes[key] = block_sizes.get(key, 0) + lead_count
        split_blocks.setdefault(row.split, set()).add(row.block_id)

    items: list[ManifestItem] = []
    for row in canonical_rows:
        for lead in _masked_leads(row.available_leads_mask):
            p_target = 1.0 / split_sizes[row.split]
            if draw_policy in ("uniform", "full-coverage-shuffled"):
                p_draw = p_target
            else:
                p_draw = 1.0 / (
                    len(split_blocks[row.split])
                    * block_sizes[(row.split, row.block_id)]
                )
            items.append(
                make_item(
                    t0=row.t0_utc,
                    lead_hours=lead,
                    condition_signature=row.condition_signature,
                    split=row.split,
                    block_id=row.block_id,
                    stratum=EVENT_STRATUM,
                    p_target=p_target,
                    p_draw=p_draw,
                )
            )
    return assign_grouped_folds(items, folds=folds, seed=fold_seed)


def draw_candidates_for_split(
    items: Sequence[ManifestItem], split: SplitName
) -> tuple[CandidateItem, ...]:
    """Project one split's complete manifest support into sampler rows."""

    selected = tuple(item for item in items if item.split == split)
    if not selected:
        raise ValueError(f"candidate manifest has no items for split {split}")
    return tuple(
        CandidateItem(
            sample_id=item.sample_id,
            t0_utc=item.t0_utc,
            lead_hours=item.lead_hours,
            condition_signature=item.condition_signature,
            block_id=item.block_id,
            stratum=item.stratum,
            split=item.split,
            p_target=item.p_target,
            fold_id=item.fold_id,
        )
        for item in selected
    )


def build_event_pretrain_index(
    timestamps: Iterable[datetime],
    split_intervals: Sequence[SplitInterval],
    *,
    condition_signature: str = "era5_oracle:era=1:tp=1:full_trajectory",
    merge_gap_hours: float = 12.0,
    guard_hours: float = 7.0,
    low_support_minimum_blocks: int = 20,
) -> tuple[tuple[EventIndexRow, ...], EventIndexAudit]:
    if not condition_signature:
        raise ValueError("condition_signature must be non-empty")
    if not np.isfinite(merge_gap_hours) or merge_gap_hours < 0.0:
        raise ValueError("merge_gap_hours must be finite and non-negative")
    if not np.isfinite(guard_hours) or guard_hours < 0.0:
        raise ValueError("guard_hours must be finite and non-negative")
    values = tuple(sorted(set(timestamps)))
    runs = continuous_runs(values)
    events = merge_event_intervals(
        (Interval(run[0], run[-1]) for run in runs),
        maximum_gap=timedelta(hours=merge_gap_hours),
        guard=timedelta(hours=guard_hours),
    )
    ids = tuple(_event_id(event) for event in events)
    eligible = strict_issue_times(values)

    nominal: list[tuple[datetime, int, SplitName]] = []
    dependency_excluded = 0
    for t0 in eligible:
        split = assign_split(t0, split_intervals)
        if split is None:
            dependency_excluded += 1
            continue
        dependency = dependency_times(t0)
        event_index = containing_event(
            Interval(dependency[0], dependency[-1]), events
        )
        if event_index is None:
            raise AssertionError(f"strict CPrecNet dependency is outside every event: {t0}")
        nominal.append((t0, event_index, split))

    # If one conservatively merged event touches a chronological boundary,
    # retain it only in the higher-priority holdout. This prevents a block from
    # crossing splits; lower-priority items become explicit embargo exclusions.
    event_splits: dict[int, SplitName] = {}
    priority = {name: index for index, name in enumerate(SPLIT_PRIORITY)}
    for _, event_index, split in nominal:
        current = event_splits.get(event_index)
        if current is None or priority[split] < priority[current]:
            event_splits[event_index] = split

    rows: list[EventIndexRow] = []
    cross_split_event_excluded = 0
    for t0, event_index, nominal_split in nominal:
        selected = event_splits[event_index]
        if nominal_split != selected:
            cross_split_event_excluded += 1
            continue
        rows.append(
            EventIndexRow(
                t0_utc=t0.astimezone(UTC),
                block_id=ids[event_index],
                split=selected,
                condition_signature=condition_signature,
            )
        )
    if len({row.t0_utc for row in rows}) != len(rows):
        raise AssertionError("duplicate issue time in event index")
    block_splits: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    split_blocks: dict[str, set[str]] = {}
    for row in rows:
        block_splits.setdefault(row.block_id, set()).add(row.split)
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
        split_blocks.setdefault(row.split, set()).add(row.block_id)
    if any(len(splits) != 1 for splits in block_splits.values()):
        raise AssertionError("event block crosses chronological splits")
    audit = EventIndexAudit(
        availability=availability_stats(values),
        strict_issue_times_before_embargo=len(eligible),
        issue_times=len(rows),
        expanded_lead_items=sum(row.available_leads_mask.bit_count() for row in rows),
        event_blocks=len(block_splits),
        split_counts=split_counts,
        boundary_embargo_fraction=(
            dependency_excluded / len(eligible) if eligible else 0.0
        ),
        dependency_embargo_exclusions=dependency_excluded,
        cross_split_event_exclusions=cross_split_event_excluded,
        total_exclusion_fraction=(
            (dependency_excluded + cross_split_event_excluded) / len(eligible)
            if eligible
            else 0.0
        ),
        low_support_splits=tuple(
            sorted(
                split
                for split, blocks in split_blocks.items()
                if len(blocks) < low_support_minimum_blocks
            )
        ),
        condition_signature=condition_signature,
        event_merge_gap_hours=merge_gap_hours,
        event_guard_hours=guard_hours,
        source_timestamps_sha256=_semantic_sha256(
            value.astimezone(UTC).isoformat() for value in values
        ),
        split_intervals_sha256=_semantic_sha256(
            _split_interval_records(split_intervals)
        ),
    )
    return tuple(rows), audit


def archive_intersection(target_root: Path, condition_root: Path) -> tuple[datetime, ...]:
    """Read only ZIP central directories and verify every archive pair's keys."""

    all_timestamps: list[datetime] = []
    for target, condition in paired_archives(target_root, condition_root):
        with np.load(target, allow_pickle=False) as target_npz, np.load(
            condition, allow_pickle=False
        ) as condition_npz:
            target_keys = tuple(target_npz.files)
            condition_keys = tuple(condition_npz.files)
        if target_keys != condition_keys:
            raise ValueError(f"target/condition keys differ: {target.name}")
        all_timestamps.extend(intersection_timestamps(target_keys, condition_keys))
    if len(set(all_timestamps)) != len(all_timestamps):
        raise ValueError("timestamps overlap across source archives")
    return tuple(sorted(all_timestamps))


def write_event_index(
    output_dir: Path,
    rows: Sequence[EventIndexRow],
    audit: EventIndexAudit,
) -> None:
    """Atomically publish compact Parquet rows plus a JSON audit."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional data extra
        raise RuntimeError("pyarrow is required to write the event index") from error
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = output_dir.with_name(f".{output_dir.name}.incomplete")
    temporary.mkdir(parents=True, exist_ok=False)
    table = pa.table(
        {
            "t0_utc": pa.array([row.t0_utc for row in rows], type=pa.timestamp("ns", tz="UTC")),
            "block_id": [row.block_id for row in rows],
            "split": [row.split for row in rows],
            "condition_signature": [row.condition_signature for row in rows],
            "available_leads_mask": pa.array(
                [row.available_leads_mask for row in rows], type=pa.uint16()
            ),
        }
    )
    pq.write_table(
        table,
        temporary / "event-index.parquet",
        compression="zstd",
        use_dictionary=True,
        row_group_size=8192,
    )
    audit_dict = asdict(audit)
    (temporary / "audit.json").write_text(
        json.dumps(audit_dict, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir)


def read_event_index(path: Path) -> tuple[tuple[EventIndexRow, ...], EventIndexAudit]:
    """Read a compact Parquet index and its mandatory sibling audit artifact."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional data extra
        raise RuntimeError("pyarrow is required to read the event index") from error
    parquet_path = path / "event-index.parquet" if path.is_dir() else path
    audit_path = parquet_path.with_name("audit.json")
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Parquet event index requires its sibling audit.json: {audit_path}"
        )
    table = pq.read_table(parquet_path)
    expected_columns = {
        "t0_utc",
        "block_id",
        "split",
        "condition_signature",
        "available_leads_mask",
    }
    if set(table.column_names) != expected_columns:
        raise ValueError(
            "event-index Parquet columns differ from the compact contract: "
            f"{table.column_names}"
        )
    raw = table.to_pydict()
    rows = tuple(
        EventIndexRow(
            t0_utc=t0.astimezone(UTC),
            block_id=str(block_id),
            split=str(split),  # type: ignore[arg-type]
            condition_signature=str(signature),
            available_leads_mask=int(mask),
        )
        for t0, block_id, split, signature, mask in zip(
            raw["t0_utc"],
            raw["block_id"],
            raw["split"],
            raw["condition_signature"],
            raw["available_leads_mask"],
            strict=True,
        )
    )
    canonical = _canonical_event_rows(rows)
    audit_raw = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_raw, Mapping):
        raise ValueError("event-index audit must be a JSON object")
    audit = _event_audit_from_json(audit_raw)
    if audit.issue_times != len(canonical):
        raise ValueError(
            f"event-index audit row count mismatch: {audit.issue_times} != {len(canonical)}"
        )
    expanded = sum(row.available_leads_mask.bit_count() for row in canonical)
    if audit.expanded_lead_items != expanded:
        raise ValueError(
            "event-index audit expanded item count mismatch: "
            f"{audit.expanded_lead_items} != {expanded}"
        )
    if audit.event_blocks != len({row.block_id for row in canonical}):
        raise ValueError("event-index audit block count mismatch")
    observed_split_counts: dict[str, int] = {}
    for row in canonical:
        observed_split_counts[row.split] = observed_split_counts.get(row.split, 0) + 1
    if dict(audit.split_counts) != observed_split_counts:
        raise ValueError("event-index audit split counts mismatch")
    observed_signatures = {row.condition_signature for row in canonical}
    if audit.condition_signature and observed_signatures != {
        audit.condition_signature
    }:
        raise ValueError("event-index audit condition signature mismatch")
    return canonical, audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = json.dumps(
            value,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_metadata(path: Path, root: Path, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _split_interval_records(
    intervals: Sequence[SplitInterval],
) -> list[dict[str, str]]:
    return [
        {
            "name": interval.name,
            "start": interval.start.astimezone(UTC).isoformat(),
            "end": interval.end.astimezone(UTC).isoformat(),
        }
        for interval in intervals
    ]


def write_event_training_bundle(
    output_dir: Path,
    rows: Sequence[EventIndexRow],
    event_audit: EventIndexAudit,
    split_intervals: Sequence[SplitInterval],
    *,
    draw_count: int,
    draw_policy: DrawPolicy,
    seed: int,
    purpose_id: str,
    maximum_oof_bytes: int,
    config_sha256: str,
    protocol_version: str,
    fold_seed: int = 11103,
    oof_height: int = 256,
    oof_width: int = 256,
    oof_fields: int = 2,
    oof_bytes_per_value: int = 4,
    source_event_index: Path | None = None,
    condition_augmentation: ConditionAugmentationPolicy | None = None,
    signature_purpose_id: str | None = None,
) -> TrainingBundleResult:
    """Audit and atomically publish an end-to-end event training bundle."""

    canonical_rows = _canonical_event_rows(rows)
    base_items = expand_event_candidates(
        canonical_rows,
        draw_policy=draw_policy,
        folds=OUTER_TRAIN_FOLDS,
        fold_seed=fold_seed,
    )
    eligible_ids = eligible_sample_ids_from_event_index(canonical_rows)
    base_candidate_audit = audit_manifest(
        base_items,
        split_intervals,
        eligible_sample_ids=eligible_ids,
        expected_outer_folds=OUTER_TRAIN_FOLDS,
        validate_split_probabilities=True,
    )
    if base_candidate_audit["unassigned_eligible_item_fraction"] != 0.0:
        raise AssertionError("independent eligible-universe audit did not close")
    if any(item.stratum != EVENT_STRATUM for item in base_items):
        raise AssertionError("CPrecNet bundle contains a manufactured non-event stratum")
    if len(base_items) != event_audit.expanded_lead_items:
        raise ValueError(
            "event audit/candidate expansion mismatch: "
            f"{event_audit.expanded_lead_items} != {len(base_items)}"
        )

    candidates = draw_candidates_for_split(base_items, "outer_train")
    draw_probabilities = [
        item.p_draw
        for item in base_items
        if item.split == "outer_train"
    ]
    if draw_policy == "full-coverage-shuffled":
        if draw_count != len(candidates):
            raise ValueError(
                "full-coverage-shuffled draw_count must equal the complete "
                f"outer_train support ({len(candidates)})"
            )
        source_draw_rows = full_coverage_shuffled_draw_manifest(
            candidates,
            seed=seed,
            purpose_id=purpose_id,
        )
    else:
        source_draw_rows = bounded_draw_manifest(
            candidates,
            draws=draw_count,
            seed=seed,
            purpose_id=purpose_id,
            draw_probabilities=draw_probabilities,
        )
    condition_result = None
    if condition_augmentation is None:
        if signature_purpose_id is not None:
            raise ValueError(
                "signature_purpose_id requires a condition augmentation policy"
            )
        items = base_items
        draw_rows = source_draw_rows
        candidate_audit = base_candidate_audit
        candidate_eligible_ids = eligible_ids
        supported_signatures = tuple(
            sorted({row.condition_signature for row in canonical_rows})
        )
    else:
        if condition_augmentation.protocol_version != protocol_version:
            raise ValueError("condition augmentation protocol disagrees with bundle")
        event_signatures = {row.condition_signature for row in canonical_rows}
        if event_signatures != {
            condition_augmentation.source_signature.key
        }:
            raise ValueError(
                "event index source signature disagrees with condition policy"
            )
        if not isinstance(signature_purpose_id, str) or not signature_purpose_id:
            raise ValueError(
                "condition augmentation requires an explicit signature_purpose_id"
            )
        expanded_outer = expand_manifest_condition_support(
            tuple(item for item in base_items if item.split == "outer_train"),
            policy=condition_augmentation,
        )
        items = tuple(
            sorted(
                (
                    *expanded_outer,
                    *(item for item in base_items if item.split != "outer_train"),
                ),
                key=lambda item: (
                    item.t0_utc,
                    item.lead_hours,
                    item.condition_signature,
                ),
            )
        )
        expanded_eligible_ids = tuple(item.sample_id for item in items)
        candidate_eligible_ids = expanded_eligible_ids
        candidate_audit = audit_manifest(
            items,
            split_intervals,
            eligible_sample_ids=expanded_eligible_ids,
            expected_outer_folds=OUTER_TRAIN_FOLDS,
            validate_split_probabilities=True,
        )
        source_draw_hash = canonical_draw_rows_sha256(source_draw_rows)
        condition_result = materialize_condition_signatures(
            source_draw_rows,
            policy=condition_augmentation,
            training_seed=seed,
            source_draw_purpose_id=purpose_id,
            signature_purpose_id=signature_purpose_id,
            source_draw_manifest_sha256=source_draw_hash,
        )
        draw_rows = condition_result.rows
        supported_signatures = condition_result.provenance.supported_signatures
    required_bytes = oof_dense_byte_budget(
        draw_rows,
        height=oof_height,
        width=oof_width,
        fields=oof_fields,
        bytes_per_value=oof_bytes_per_value,
    )
    # This check deliberately precedes creation of even a temporary output
    # directory: a rejected storage plan publishes no partial artifact.
    enforce_byte_budget(required_bytes, maximum_oof_bytes)
    unique_oof_items = len(
        {
            (row.t0_utc, row.lead_hours, row.condition_signature)
            for row in draw_rows
        }
    )

    source_event_index_metadata: dict[str, object] | None = None
    if source_event_index is not None:
        parquet_path = (
            source_event_index / "event-index.parquet"
            if source_event_index.is_dir()
            else source_event_index
        )
        audit_path = parquet_path.with_name("audit.json")
        source_event_index_metadata = {
            "event_index_path": str(parquet_path.resolve()),
            "event_index_bytes": parquet_path.stat().st_size,
            "event_index_sha256": _sha256_file(parquet_path),
            "audit_path": str(audit_path.resolve()),
            "audit_bytes": audit_path.stat().st_size,
            "audit_sha256": _sha256_file(audit_path),
        }

    event_semantic_sha256 = _semantic_sha256(
        {
            "t0_utc": row.t0_utc.astimezone(UTC).isoformat(),
            "block_id": row.block_id,
            "split": row.split,
            "condition_signature": row.condition_signature,
            "available_leads_mask": row.available_leads_mask,
        }
        for row in canonical_rows
    )
    eligible_sha256 = _semantic_sha256(candidate_eligible_ids)
    candidate_semantic_sha256 = _semantic_sha256(asdict(item) for item in items)
    fold_map = sorted(
        {
            (item.block_id, item.fold_id)
            for item in items
            if item.split == "outer_train"
        }
    )
    fold_map_sha256 = _semantic_sha256(
        {"block_id": block_id, "fold_id": fold_id}
        for block_id, fold_id in fold_map
    )
    weights = np.asarray([row.omega for row in draw_rows], dtype=np.float64)
    weight_sum = float(weights.sum())
    weight_ess = float(weight_sum**2 / np.square(weights).sum())

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".incomplete",
            dir=output_dir.parent,
        )
    )
    try:
        event_index_dir = temporary / "event-index"
        write_event_index(event_index_dir, canonical_rows, event_audit)
        candidate_path = temporary / "candidate-manifest.json"
        candidate_metadata = {
            "protocol_version": protocol_version,
            "estimand": event_audit.estimand,
            "population": "cprecnet_event_conditioned_archive",
            "strata": [EVENT_STRATUM],
            "probability_scope": "within_split",
            "target_policy": "eligible_items_uniform",
            "draw_policy": draw_policy,
            "outer_train_folds": OUTER_TRAIN_FOLDS,
            "fold_seed": fold_seed,
            "split_intervals": _split_interval_records(split_intervals),
            "config_sha256": config_sha256,
            "event_index_semantic_sha256": event_semantic_sha256,
            "eligible_universe_sample_ids_sha256": eligible_sha256,
            "candidate_semantic_sha256": candidate_semantic_sha256,
            "fold_map_sha256": fold_map_sha256,
        }
        if condition_augmentation is not None:
            candidate_metadata["condition_augmentation_policy_sha256"] = (
                condition_augmentation.semantic_sha256
            )
        candidate_hash = write_manifest(
            candidate_path,
            items,
            metadata=candidate_metadata,
        )
        source_draw_path: Path | None = None
        source_draw_hash: str | None = None
        if condition_result is not None:
            source_draw_path = temporary / "source-training-draw-manifest.jsonl"
            source_draw_hash = write_draw_manifest(
                source_draw_path, source_draw_rows
            )
            if source_draw_hash != (
                condition_result.provenance.source_draw_manifest_sha256
            ):
                raise AssertionError("source draw writer/provenance hash mismatch")
        draw_path = temporary / "training-draw-manifest.jsonl"
        draw_hash = write_draw_manifest(draw_path, draw_rows)
        if candidate_hash != _sha256_file(candidate_path):
            raise AssertionError("candidate manifest writer hash mismatch")
        if draw_hash != _sha256_file(draw_path):
            raise AssertionError("draw manifest writer hash mismatch")

        artifacts = {
            "event_index": _artifact_metadata(
                event_index_dir / "event-index.parquet",
                temporary,
                rows=len(canonical_rows),
            ),
            "event_audit": _artifact_metadata(
                event_index_dir / "audit.json", temporary
            ),
            "candidate_manifest": _artifact_metadata(
                candidate_path, temporary, rows=len(items)
            ),
            "training_draw_manifest": _artifact_metadata(
                draw_path, temporary, rows=len(draw_rows)
            ),
        }
        if source_draw_path is not None:
            source_record = _artifact_metadata(
                source_draw_path, temporary, rows=len(source_draw_rows)
            )
            source_record["purpose_id"] = purpose_id
            artifacts["source_training_draw_manifest"] = source_record
        sampling_metadata = {
            "draw_split": "outer_train",
            "target_policy": "eligible_items_uniform",
            "probability_scope": "within_split",
            "draw_policy": draw_policy,
            "draw_count": draw_count,
            "seed": seed,
            "purpose_id": purpose_id,
            "weight_clipping": None,
            "omega_min": float(np.min(weights)),
            "omega_median": float(np.median(weights)),
            "omega_p95": float(np.percentile(weights, 95)),
            "omega_max": float(np.max(weights)),
            "weighted_sample_ess": weight_ess,
        }
        if draw_policy == "full-coverage-shuffled":
            sampling_metadata.update(
                {
                    "coverage": "all_outer_train_base_candidates_once",
                    "replacement": False,
                    "base_candidate_count": len(candidates),
                    "shuffle_key": "sha256(seed,purpose_id,sample_id)",
                }
            )
        condition_metadata: dict[str, object] = {}
        if condition_result is not None:
            sampling_metadata.update(
                {
                    "condition_selection_order": (
                        "base_sample_then_one_condition_signature"
                    ),
                    "signature_purpose_id": signature_purpose_id,
                }
            )
            condition_metadata = {
                "condition_augmentation_policy_sha256": (
                    condition_augmentation.semantic_sha256
                ),
                "condition_augmentation_provenance_sha256": (
                    condition_result.provenance.semantic_sha256
                ),
                "condition_augmentation": (
                    condition_result.provenance.to_json()
                ),
            }
        metadata = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "protocol_version": protocol_version,
            "estimand": event_audit.estimand,
            "population": "cprecnet_event_conditioned_archive",
            "strata": [EVENT_STRATUM],
            "condition_signatures": list(supported_signatures),
            "config_sha256": config_sha256,
            "event_index_semantic_sha256": event_semantic_sha256,
            "eligible_universe_sample_ids_sha256": eligible_sha256,
            "candidate_semantic_sha256": candidate_semantic_sha256,
            "fold_map_sha256": fold_map_sha256,
            "source_event_index": source_event_index_metadata,
            "candidate_audit": candidate_audit,
            "sampling": sampling_metadata,
            "folds": {
                "split": "outer_train",
                "count": OUTER_TRAIN_FOLDS,
                "seed": fold_seed,
                "block_counts": {
                    str(fold): sum(fold_id == fold for _, fold_id in fold_map)
                    for fold in range(OUTER_TRAIN_FOLDS)
                },
            },
            "oof_preflight": {
                "unique_semantic_items": unique_oof_items,
                "height": oof_height,
                "width": oof_width,
                "fields": oof_fields,
                "bytes_per_value": oof_bytes_per_value,
                "required_bytes": required_bytes,
                "maximum_bytes": maximum_oof_bytes,
            },
            "artifacts": artifacts,
            **condition_metadata,
        }
        metadata_path = temporary / "bundle-metadata.json"
        metadata_payload = (
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _write_fsynced(metadata_path, metadata_payload)
        metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
        _write_fsynced(
            temporary / "bundle-metadata.sha256",
            f"{metadata_sha256}  bundle-metadata.json\n".encode("ascii"),
        )
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(temporary, output_dir)
        directory_descriptor = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return TrainingBundleResult(
        output_dir=output_dir,
        event_rows=len(canonical_rows),
        candidates=len(items),
        draws=len(draw_rows),
        unique_oof_items=unique_oof_items,
        oof_required_bytes=required_bytes,
        metadata_sha256=metadata_sha256,
    )


def split_intervals_from_config(config: Mapping[str, object]) -> tuple[SplitInterval, ...]:
    manifest = config["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest config must be a mapping")
    raw_intervals = manifest["split_intervals"]
    if not isinstance(raw_intervals, Sequence):
        raise TypeError("split_intervals must be a sequence")
    result: list[SplitInterval] = []
    for raw in raw_intervals:
        if not isinstance(raw, Mapping):
            raise TypeError("split interval must be a mapping")
        start = datetime.fromisoformat(str(raw["start"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(raw["end"]).replace("Z", "+00:00"))
        result.append(SplitInterval(name=str(raw["name"]), start=start, end=end))  # type: ignore[arg-type]
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--event-index",
        type=Path,
        help="existing event-index.parquet or directory containing it and audit.json",
    )
    source.add_argument("--target-root", type=Path)
    parser.add_argument(
        "--condition-root",
        type=Path,
        help="required with --target-root; ignored for no other source mode",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--draws",
        type=int,
        help="planned global training draws (or set sampling.draws in config)",
    )
    parser.add_argument(
        "--sampler-policy",
        "--draw-policy",
        dest="draw_policy",
        choices=("uniform", "block-balanced", "full-coverage-shuffled"),
        help="override config sampling.draw_policy",
    )
    parser.add_argument("--seed", type=int, help="override config sampling.seed")
    parser.add_argument(
        "--purpose-id", help="override config sampling.purpose_id counter namespace"
    )
    parser.add_argument(
        "--fold-seed", type=int, help="override config sampling.fold_seed"
    )
    parser.add_argument(
        "--maximum-oof-gib",
        type=float,
        help="hard dense OOF budget in GiB (1024^3 bytes)",
    )
    parser.add_argument(
        "--maximum-oof-bytes",
        "--oof-max-bytes",
        dest="maximum_oof_bytes",
        type=int,
        help=(
            "hard dense OOF byte budget "
            "(or set sampling.maximum_oof_bytes in config)"
        ),
    )
    return parser.parse_args(argv)


def _parse_event_index_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build only the compact CPrecNet event index and audit."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sampling_config(config: Mapping[str, object]) -> Mapping[str, object]:
    raw = config.get("sampling", {})
    if not isinstance(raw, Mapping):
        raise TypeError("sampling config must be a mapping")
    return raw


def _validate_event_manifest_config(manifest: Mapping[str, object]) -> None:
    raw_leads = manifest.get("leads_hours")
    if not isinstance(raw_leads, Sequence) or isinstance(raw_leads, str):
        raise TypeError("manifest.leads_hours must be a sequence")
    declared_leads = tuple(float(value) for value in raw_leads)
    if declared_leads != LEADS_HOURS:
        raise ValueError(f"CPrecNet bundle requires the official 12 leads: {LEADS_HOURS}")
    if manifest.get("target_distribution") != "event_conditioned_archive":
        raise ValueError(
            "this CLI only builds the CPrecNet event-conditioned archive population"
        )
    boundary_hours = float(manifest.get("boundary_embargo_hours", 7.0))
    if not np.isfinite(boundary_hours) or boundary_hours != 7.0:
        raise ValueError(
            "the fixed [t0-55m,t0+6h] dependency requires a 7-hour "
            "issue-time boundary embargo"
        )


def _validate_sampling_config(sampling: Mapping[str, object]) -> None:
    if sampling.get("target_policy", "eligible_items_uniform") != (
        "eligible_items_uniform"
    ):
        raise ValueError("only eligible_items_uniform target sampling is supported")
    if sampling.get("weight_clipping") is not None:
        raise ValueError("v1.1.3b forbids importance-weight clipping")


def _resolved_setting(
    override: object | None,
    config: Mapping[str, object],
    key: str,
    *,
    default: object | None = None,
) -> object:
    if override is not None:
        return override
    return config.get(key, default)


def event_index_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the compact-index-only CLI alias."""

    arguments = _parse_event_index_args(argv)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    manifest_config = config.get("manifest")
    era5_config = config.get("era5")
    if not isinstance(manifest_config, Mapping):
        raise TypeError("manifest config must be a mapping")
    if not isinstance(era5_config, Mapping):
        raise TypeError("era5 config must be a mapping")
    _validate_event_manifest_config(manifest_config)
    timestamps = archive_intersection(
        arguments.target_root, arguments.condition_root
    )
    rows, audit = build_event_pretrain_index(
        timestamps,
        split_intervals_from_config(config),
        condition_signature=str(era5_config["condition_signature"]),
        merge_gap_hours=float(manifest_config["event_merge_gap_hours"]),
        guard_hours=float(manifest_config["event_guard_hours"]),
    )
    write_event_index(arguments.output_dir, rows, audit)
    print(json.dumps(asdict(audit), default=str, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    manifest_config = config.get("manifest")
    era5_config = config.get("era5")
    if not isinstance(manifest_config, Mapping):
        raise TypeError("manifest config must be a mapping")
    if not isinstance(era5_config, Mapping):
        raise TypeError("era5 config must be a mapping")
    _validate_event_manifest_config(manifest_config)
    sampling_config = _sampling_config(config)
    _validate_sampling_config(sampling_config)
    draw_count_raw = _resolved_setting(
        arguments.draws, sampling_config, "draws"
    )
    if arguments.maximum_oof_gib is not None:
        if not np.isfinite(arguments.maximum_oof_gib) or arguments.maximum_oof_gib <= 0:
            raise ValueError("--maximum-oof-gib must be finite and positive")
        if arguments.maximum_oof_bytes is not None:
            raise ValueError(
                "declare only one of --maximum-oof-gib and --maximum-oof-bytes"
            )
        maximum_oof_bytes_raw: object | None = int(
            arguments.maximum_oof_gib * 1024**3
        )
    else:
        maximum_oof_bytes_raw = _resolved_setting(
            arguments.maximum_oof_bytes,
            sampling_config,
            "maximum_oof_bytes",
        )
    if draw_count_raw is None:
        raise ValueError("declare --draws or sampling.draws")
    if maximum_oof_bytes_raw is None:
        raise ValueError(
            "declare --maximum-oof-bytes or sampling.maximum_oof_bytes"
        )
    draw_policy_value = str(
        _resolved_setting(
            arguments.draw_policy,
            sampling_config,
            "draw_policy",
            default="uniform",
        )
    ).replace("_", "-")
    if draw_policy_value not in (
        "uniform",
        "block-balanced",
        "full-coverage-shuffled",
    ):
        raise ValueError(f"unsupported draw policy: {draw_policy_value}")
    seed = int(
        _resolved_setting(arguments.seed, sampling_config, "seed", default=11103)
    )
    purpose_id = str(
        _resolved_setting(
            arguments.purpose_id,
            sampling_config,
            "purpose_id",
            default="cprecnet-event-pretraining",
        )
    )
    fold_seed = int(
        _resolved_setting(
            arguments.fold_seed,
            sampling_config,
            "fold_seed",
            default=11103,
        )
    )
    raw_condition_augmentation = era5_config.get("condition_augmentation")
    condition_augmentation = (
        ConditionAugmentationPolicy.from_mapping(raw_condition_augmentation)
        if isinstance(raw_condition_augmentation, Mapping)
        else None
    )
    if raw_condition_augmentation is not None and (
        condition_augmentation is None
    ):
        raise TypeError("era5.condition_augmentation must be a mapping")
    signature_purpose_raw = sampling_config.get("signature_purpose_id")
    signature_purpose_id = (
        str(signature_purpose_raw)
        if signature_purpose_raw is not None
        else None
    )
    intervals = split_intervals_from_config(config)
    if arguments.event_index is not None:
        if arguments.condition_root is not None:
            raise ValueError("--condition-root cannot be used with --event-index")
        rows, audit = read_event_index(arguments.event_index)
        expected_signature = str(era5_config["condition_signature"])
        observed_signatures = {row.condition_signature for row in rows}
        if observed_signatures != {expected_signature}:
            raise ValueError(
                "event-index/config condition signature mismatch: "
                f"{sorted(observed_signatures)} != {[expected_signature]}"
            )
        expected_merge_gap = float(manifest_config["event_merge_gap_hours"])
        expected_guard = float(manifest_config["event_guard_hours"])
        expected_split_hash = _semantic_sha256(_split_interval_records(intervals))
        if not audit.source_timestamps_sha256 or not audit.split_intervals_sha256:
            raise ValueError(
                "event-index audit lacks construction provenance; rebuild it"
            )
        if audit.condition_signature != expected_signature:
            raise ValueError("event-index audit/config condition signature mismatch")
        if not np.isclose(
            audit.event_merge_gap_hours,
            expected_merge_gap,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("event-index/config merge-gap mismatch")
        if not np.isclose(
            audit.event_guard_hours,
            expected_guard,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("event-index/config guard mismatch")
        if audit.split_intervals_sha256 != expected_split_hash:
            raise ValueError("event-index/config split-interval mismatch")
        source_event_index = arguments.event_index
    else:
        if arguments.target_root is None or arguments.condition_root is None:
            raise ValueError("--target-root requires --condition-root")
        timestamps = archive_intersection(
            arguments.target_root, arguments.condition_root
        )
        rows, audit = build_event_pretrain_index(
            timestamps,
            intervals,
            condition_signature=str(era5_config["condition_signature"]),
            merge_gap_hours=float(manifest_config["event_merge_gap_hours"]),
            guard_hours=float(manifest_config["event_guard_hours"]),
        )
        source_event_index = None
    result = write_event_training_bundle(
        arguments.output_dir,
        rows,
        audit,
        intervals,
        draw_count=int(draw_count_raw),
        draw_policy=draw_policy_value,  # type: ignore[arg-type]
        seed=seed,
        purpose_id=purpose_id,
        maximum_oof_bytes=int(maximum_oof_bytes_raw),
        config_sha256=_sha256_file(arguments.config),
        protocol_version=str(config.get("protocol_version", "unknown")),
        fold_seed=fold_seed,
        source_event_index=source_event_index,
        condition_augmentation=condition_augmentation,
        signature_purpose_id=signature_purpose_id,
    )
    print(
        json.dumps(
            {
                "event_audit": asdict(audit),
                "bundle": {
                    **asdict(result),
                    "output_dir": str(result.output_dir),
                },
            },
            default=str,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

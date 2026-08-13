"""Leakage-safe event grouping and background stratification primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy import ndimage


CONTEXT_ACTIVE_THRESHOLD_MM_H = 0.153765


@dataclass(frozen=True, slots=True, order=True)
class Interval:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("interval endpoints must be timezone-aware")
        if self.end < self.start:
            raise ValueError("interval end precedes start")

    def expand(self, delta: timedelta) -> "Interval":
        if delta < timedelta(0):
            raise ValueError("expansion must be non-negative")
        return Interval(self.start - delta, self.end + delta)


def context_threshold_count(
    localmax_r_mm_h: np.ndarray,
    valid: np.ndarray,
    *,
    threshold_mm_h: float = CONTEXT_ACTIVE_THRESHOLD_MM_H,
) -> int:
    values = np.asarray(localmax_r_mm_h)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("context field and validity mask must have equal shape")
    if not np.isfinite(threshold_mm_h) or threshold_mm_h < 0.0:
        raise ValueError("context threshold must be finite and non-negative")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("a valid context pixel contains a non-finite value")
    if np.any(mask & (values < 0.0)):
        raise ValueError("a valid context rain rate is negative")
    return int(np.count_nonzero(mask & (values >= threshold_mm_h)))


def context_active(
    localmax_r_mm_h: np.ndarray,
    valid: np.ndarray,
    *,
    threshold_mm_h: float = CONTEXT_ACTIVE_THRESHOLD_MM_H,
    minimum_pixels: int = 64,
) -> bool:
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    return context_threshold_count(
        localmax_r_mm_h, valid, threshold_mm_h=threshold_mm_h
    ) >= minimum_pixels


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    pixels: int
    maximum_mm: float
    significant: bool


def wet_components(
    accumulation_mm: np.ndarray,
    valid: np.ndarray,
    *,
    a_wet_mm: float = 0.1,
    minimum_pixels: int = 4,
    maximum_threshold_mm: float = 1.0,
) -> tuple[ComponentSummary, ...]:
    values = np.asarray(accumulation_mm)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("target and validity must be equal 2-D fields")
    if not np.isfinite(a_wet_mm) or a_wet_mm <= 0.0:
        raise ValueError("a_wet_mm must be finite and positive")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    if not np.isfinite(maximum_threshold_mm) or maximum_threshold_mm <= 0.0:
        raise ValueError("maximum_threshold_mm must be finite and positive")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("a valid target pixel contains a non-finite value")
    if np.any(mask & (values < 0.0)):
        raise ValueError("a valid target accumulation is negative")
    wet = mask & (values >= a_wet_mm)
    labels, count = ndimage.label(wet, structure=np.ones((3, 3), dtype=np.uint8))
    summaries: list[ComponentSummary] = []
    for label in range(1, count + 1):
        member = labels == label
        pixels = int(np.count_nonzero(member))
        maximum = float(np.max(values[member]))
        summaries.append(
            ComponentSummary(
                pixels=pixels,
                maximum_mm=maximum,
                significant=pixels >= minimum_pixels
                or maximum >= maximum_threshold_mm,
            )
        )
    return tuple(summaries)


def active_runs(
    times: Iterable[datetime], *, step: timedelta = timedelta(minutes=5)
) -> tuple[Interval, ...]:
    ordered = sorted(set(times))
    if not ordered:
        return ()
    runs: list[Interval] = []
    start = prior = ordered[0]
    for value in ordered[1:]:
        if value - prior != step:
            runs.append(Interval(start, prior))
            start = value
        prior = value
    runs.append(Interval(start, prior))
    return tuple(runs)


def merge_event_intervals(
    seeds: Iterable[Interval],
    *,
    maximum_gap: timedelta = timedelta(hours=12),
    guard: timedelta = timedelta(hours=7),
) -> tuple[Interval, ...]:
    """Merge seed gaps, expand guards, and recursively merge guard overlaps."""

    if maximum_gap < timedelta(0) or guard < timedelta(0):
        raise ValueError("gap and guard must be non-negative")
    ordered = sorted(seeds)
    if not ordered:
        return ()
    initially_merged: list[Interval] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        if candidate.start - current.end <= maximum_gap:
            current = Interval(current.start, max(current.end, candidate.end))
        else:
            initially_merged.append(current)
            current = candidate
    initially_merged.append(current)

    expanded = [interval.expand(guard) for interval in initially_merged]
    final: list[Interval] = []
    current = expanded[0]
    for candidate in expanded[1:]:
        if candidate.start <= current.end:
            current = Interval(current.start, max(current.end, candidate.end))
        else:
            final.append(current)
            current = candidate
    final.append(current)
    return tuple(final)


def containing_event(interval: Interval, events: Sequence[Interval]) -> int | None:
    matches = [
        index
        for index, event in enumerate(events)
        if event.start <= interval.start and interval.end <= event.end
    ]
    if len(matches) > 1:
        raise AssertionError("event intervals overlap")
    return matches[0] if matches else None


def classify_background(
    *,
    any_target_wet: bool,
    all_dependency_context_counts: Sequence[int],
    dependency_timestamps_present: Sequence[bool],
    target_has_valid_pixels: bool,
) -> Literal["strict_dry", "marginal_background"]:
    if not all_dependency_context_counts:
        raise ValueError("context counts are required for the dependency interval")
    if len(dependency_timestamps_present) != len(all_dependency_context_counts):
        raise ValueError(
            "dependency timestamp validity and context counts must have equal length"
        )
    if not all(dependency_timestamps_present):
        raise ValueError("a missing dependency timestamp is not a dry observation")
    if not target_has_valid_pixels:
        raise ValueError("a target with no valid pixels cannot be classified as dry")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
        for value in all_dependency_context_counts
    ):
        raise ValueError("context threshold counts must be non-negative integers")
    if not any_target_wet and all(value == 0 for value in all_dependency_context_counts):
        return "strict_dry"
    return "marginal_background"


def utc_day_block(t0: datetime, stratum: str) -> str:
    if t0.tzinfo is None:
        raise ValueError("t0 must be timezone-aware")
    return f"{stratum}:{t0.astimezone(UTC).date().isoformat()}"

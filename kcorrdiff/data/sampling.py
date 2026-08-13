"""Deterministic, bounded training-draw manifests and importance weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CandidateItem:
    sample_id: str
    t0_utc: str
    lead_hours: float
    condition_signature: str
    block_id: str
    stratum: str
    split: str
    p_target: float
    fold_id: int | None = None


@dataclass(frozen=True, slots=True)
class DrawRow:
    global_example_index: int
    sample_id: str
    t0_utc: str
    lead_hours: float
    condition_signature: str
    block_id: str
    stratum: str
    split: str
    p_target: float
    p_draw: float
    omega: float
    fold_id: int | None = None


def canonical_counter(seed: int, purpose_id: str, global_example_index: int) -> int:
    """Stable 64-bit counter seed independent of batch/GPU/worker topology."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not purpose_id:
        raise ValueError("purpose_id must be non-empty")
    if (
        isinstance(global_example_index, bool)
        or not isinstance(global_example_index, int)
        or global_example_index < 0
    ):
        raise ValueError("global_example_index must be a non-negative integer")
    encoded = f"{seed}\0{purpose_id}\0{global_example_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def block_balanced_draw_probabilities(
    candidates: Sequence[CandidateItem],
) -> NDArray[np.float64]:
    """Give every independent manifest block equal draw mass.

    Within a block, its complete ``(t0, tau, signature)`` items are uniform.
    The returned flat probabilities therefore preserve the exact item-level
    ``P_draw`` needed for ``omega=P_target/P_draw`` while implementing the
    storm-balanced CPrecNet pretraining policy.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    block_sizes: dict[str, int] = {}
    for item in candidates:
        if not item.block_id:
            raise ValueError("candidate block_id must be non-empty")
        block_sizes[item.block_id] = block_sizes.get(item.block_id, 0) + 1
    block_mass = 1.0 / len(block_sizes)
    return np.asarray(
        [block_mass / block_sizes[item.block_id] for item in candidates],
        dtype=np.float64,
    )


def bounded_draw_manifest(
    candidates: Sequence[CandidateItem],
    *,
    draws: int,
    seed: int,
    purpose_id: str,
    draw_probabilities: Sequence[float] | None = None,
) -> tuple[DrawRow, ...]:
    """Materialize every stochastic draw before training starts.

    Changing accumulation, microbatch, worker, or GPU count cannot change the
    returned sequence because each row is keyed only by its global index.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be positive")
    # Validate the counter namespace before entering the potentially large
    # materialization loop.  Index zero is sufficient because every generated
    # index is a non-negative integer below ``draws``.
    canonical_counter(seed, purpose_id, 0)
    sample_ids = [item.sample_id for item in candidates]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("candidate sample_id values must be unique")
    identities = [
        (item.t0_utc, item.lead_hours, item.condition_signature)
        for item in candidates
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate (t0, lead, signature) identities must be unique")
    if any(
        item.fold_id is not None
        and (
            isinstance(item.fold_id, bool)
            or not isinstance(item.fold_id, int)
            or item.fold_id < 0
        )
        for item in candidates
    ):
        raise ValueError("candidate fold IDs must be non-negative integers or null")
    target_probabilities = np.asarray(
        [item.p_target for item in candidates], dtype=np.float64
    )
    if np.any(~np.isfinite(target_probabilities)) or np.any(
        target_probabilities <= 0.0
    ):
        raise ValueError("target probabilities must be finite and strictly positive")
    if not np.isclose(target_probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("target probabilities must sum to one over candidates")
    if draw_probabilities is None:
        probabilities = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    else:
        probabilities = np.asarray(draw_probabilities, dtype=np.float64)
        if probabilities.shape != (len(candidates),):
            raise ValueError("draw_probabilities must match candidates")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
            raise ValueError("draw probabilities must be finite and strictly positive")
        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("draw probabilities must sum to one over candidates")
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    # Avoid an out-of-range search result from the last-bit floating sum while
    # preserving every preceding probability boundary exactly.
    cumulative[-1] = 1.0
    rows: list[DrawRow] = []
    for index in range(draws):
        counter = canonical_counter(seed, purpose_id, index)
        # The high 53 bits form an exact binary uniform in [0, 1).  A single
        # precomputed CDF plus binary search keeps million-row plans O(N+DlogN)
        # instead of rebuilding a categorical distribution for every draw.
        uniform = (counter >> 11) * (1.0 / (1 << 53))
        selected = int(np.searchsorted(cumulative, uniform, side="right"))
        item = candidates[selected]
        p_draw = float(probabilities[selected])
        rows.append(
            DrawRow(
                global_example_index=index,
                **asdict(item),
                p_draw=p_draw,
                omega=item.p_target / p_draw,
            )
        )
    return tuple(rows)


def oof_dense_byte_budget(
    rows: Sequence[DrawRow],
    *,
    height: int = 256,
    width: int = 256,
    fields: int = 2,
    bytes_per_value: int = 4,
) -> int:
    """Upper bound for unique materialized OOF (logit/p and mu_z) fields."""

    if min(height, width, fields, bytes_per_value) <= 0:
        raise ValueError("shape, field count and precision must be positive")
    # Count the semantic OOF identity rather than trusting ``sample_id``.  A
    # forged or colliding ID must never make the storage preflight undercount.
    unique = {
        (row.t0_utc, row.lead_hours, row.condition_signature)
        for row in rows
    }
    return len(unique) * height * width * fields * bytes_per_value


def enforce_byte_budget(required_bytes: int, maximum_bytes: int) -> None:
    if required_bytes < 0 or maximum_bytes <= 0:
        raise ValueError("invalid byte budget")
    if required_bytes > maximum_bytes:
        raise OSError(
            f"OOF materialization requires {required_bytes} bytes, "
            f"exceeding declared budget {maximum_bytes} bytes"
        )


def write_draw_manifest(path: Path, rows: Sequence[DrawRow]) -> str:
    """Atomically write JSONL and return its SHA-256 content hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in rows:
                payload = (
                    json.dumps(asdict(row), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                stream.write(payload)
                digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def read_draw_manifest(path: Path) -> tuple[DrawRow, ...]:
    rows: list[DrawRow] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = DrawRow(**json.loads(line))
            if row.global_example_index != len(rows):
                raise ValueError(f"non-canonical row index at line {line_number}")
            if row.fold_id is not None and (
                isinstance(row.fold_id, bool)
                or not isinstance(row.fold_id, int)
                or row.fold_id < 0
            ):
                raise ValueError(f"invalid fold ID at line {line_number}")
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in (row.p_target, row.p_draw, row.omega)
            ):
                raise ValueError(f"invalid probability/weight at line {line_number}")
            if not math.isclose(
                row.omega,
                row.p_target / row.p_draw,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"incorrect importance weight at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("draw manifest is empty")
    return tuple(rows)


__all__ = [
    "CandidateItem",
    "DrawRow",
    "block_balanced_draw_probabilities",
    "bounded_draw_manifest",
    "canonical_counter",
    "enforce_byte_budget",
    "oof_dense_byte_budget",
    "read_draw_manifest",
    "write_draw_manifest",
]

"""Train-only, mergeable normalization statistics and atomic artifacts.

The model-facing statistics in this module use float64 parallel/Welford
moments even though the immutable input caches are float32.  A validity mask
selects observations before they enter a moment state; missing values are
never interpreted as physical zero.  Finalization is deliberately strict:
empty, constant, or non-finite channels cannot silently become identity
normalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


NORMALIZATION_FORMAT_VERSION = "kcorrdiff.train-only-normalization.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: object) -> str:
    """Hash a JSON-compatible value using a path/order-independent encoding."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ChannelStatistics:
    """Population statistics for one named continuous input channel."""

    count: int
    mean: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("normalization count must be an integer")
        if self.count <= 0:
            raise ValueError("normalization count must be positive")
        values = (
            self.mean,
            self.variance,
            self.standard_deviation,
            self.minimum,
            self.maximum,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("normalization statistics must be finite")
        if self.variance <= 0.0 or self.standard_deviation <= 0.0:
            raise ValueError("normalization variance must be strictly positive")
        if self.minimum > self.maximum:
            raise ValueError("normalization minimum exceeds maximum")
        if not math.isclose(
            self.standard_deviation * self.standard_deviation,
            self.variance,
            rel_tol=2.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("standard deviation does not match variance")

    def to_json(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class StreamingChannelMoments:
    """Float64 per-channel parallel moments with deterministic merge support.

    Values are channel-first.  ``valid_mask`` may match the complete values
    shape or any shape NumPy can broadcast to it, including a shared spatial
    mask.  Only selected elements are required to be finite.
    """

    def __init__(self, channel_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in channel_names)
        if not names or any(not name for name in names):
            raise ValueError("channel names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("channel names must be unique")
        self.channel_names = names
        channels = len(names)
        self.count = np.zeros(channels, dtype=np.int64)
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)
        self.minimum = np.full(channels, np.inf, dtype=np.float64)
        self.maximum = np.full(channels, -np.inf, dtype=np.float64)

    def update(
        self,
        values: ArrayLike,
        *,
        valid_mask: ArrayLike | None = None,
    ) -> None:
        raw = np.asarray(values)
        if raw.ndim < 1 or raw.shape[0] != len(self.channel_names):
            raise ValueError(
                "values must be channel-first with leading size "
                f"{len(self.channel_names)}"
            )
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise TypeError("normalization values must be real numeric data")
        values64 = np.asarray(raw, dtype=np.float64)
        if valid_mask is None:
            mask = np.ones(values64.shape, dtype=np.bool_)
        else:
            raw_mask = np.asarray(valid_mask)
            if raw_mask.dtype != np.bool_:
                raise TypeError("valid_mask must have boolean dtype")
            try:
                mask = np.broadcast_to(raw_mask, values64.shape)
            except ValueError as error:
                raise ValueError(
                    f"valid_mask shape {raw_mask.shape} cannot broadcast to "
                    f"values shape {values64.shape}"
                ) from error

        for channel in range(len(self.channel_names)):
            selected = values64[channel][mask[channel]]
            if selected.size == 0:
                continue
            if not np.all(np.isfinite(selected)):
                raise ValueError(
                    f"valid values for {self.channel_names[channel]} are non-finite"
                )
            batch_count = int(selected.size)
            batch_mean = float(np.mean(selected, dtype=np.float64))
            difference = selected - batch_mean
            batch_m2 = float(np.sum(difference * difference, dtype=np.float64))
            self._merge_channel(
                channel,
                count=batch_count,
                mean=batch_mean,
                m2=batch_m2,
                minimum=float(np.min(selected)),
                maximum=float(np.max(selected)),
            )

    def _merge_channel(
        self,
        channel: int,
        *,
        count: int,
        mean: float,
        m2: float,
        minimum: float,
        maximum: float,
    ) -> None:
        if count == 0:
            return
        prior_count = int(self.count[channel])
        if prior_count == 0:
            self.count[channel] = count
            self.mean[channel] = mean
            self.m2[channel] = m2
            self.minimum[channel] = minimum
            self.maximum[channel] = maximum
            return
        total = prior_count + count
        delta = mean - float(self.mean[channel])
        self.mean[channel] += delta * (count / total)
        self.m2[channel] += m2 + delta * delta * prior_count * count / total
        self.count[channel] = total
        self.minimum[channel] = min(float(self.minimum[channel]), minimum)
        self.maximum[channel] = max(float(self.maximum[channel]), maximum)

    def merge(self, other: "StreamingChannelMoments") -> None:
        """Merge another independently accumulated state without raw samples."""

        if not isinstance(other, StreamingChannelMoments):
            raise TypeError("can only merge StreamingChannelMoments")
        if other.channel_names != self.channel_names:
            raise ValueError("cannot merge different channel schemas")
        for channel in range(len(self.channel_names)):
            self._merge_channel(
                channel,
                count=int(other.count[channel]),
                mean=float(other.mean[channel]),
                m2=float(other.m2[channel]),
                minimum=float(other.minimum[channel]),
                maximum=float(other.maximum[channel]),
            )

    def merge_state(self) -> dict[str, object]:
        """Return the sufficient state for distributed/streaming reduction."""

        return {
            "channel_names": list(self.channel_names),
            "count": [int(value) for value in self.count],
            "mean": [float(value) for value in self.mean],
            "m2": [float(value) for value in self.m2],
            "minimum": [float(value) for value in self.minimum],
            "maximum": [float(value) for value in self.maximum],
        }

    @classmethod
    def from_merge_state(cls, raw: Mapping[str, object]) -> "StreamingChannelMoments":
        names = tuple(str(value) for value in raw["channel_names"])  # type: ignore[arg-type]
        result = cls(names)
        arrays = {
            "count": np.asarray(raw["count"], dtype=np.int64),
            "mean": np.asarray(raw["mean"], dtype=np.float64),
            "m2": np.asarray(raw["m2"], dtype=np.float64),
            "minimum": np.asarray(raw["minimum"], dtype=np.float64),
            "maximum": np.asarray(raw["maximum"], dtype=np.float64),
        }
        expected = (len(names),)
        if any(array.shape != expected for array in arrays.values()):
            raise ValueError("merge-state arrays do not match channel schema")
        if np.any(arrays["count"] < 0):
            raise ValueError("merge-state counts cannot be negative")
        populated = arrays["count"] > 0
        finite_fields = ("mean", "m2", "minimum", "maximum")
        if any(
            not np.all(np.isfinite(arrays[name][populated]))
            for name in finite_fields
        ):
            raise ValueError("populated merge-state values must be finite")
        if np.any(arrays["m2"][populated] < 0.0):
            raise ValueError("merge-state m2 cannot be negative")
        result.count = arrays["count"].copy()
        result.mean = arrays["mean"].copy()
        result.m2 = arrays["m2"].copy()
        result.minimum = arrays["minimum"].copy()
        result.maximum = arrays["maximum"].copy()
        return result

    def finalize(self) -> dict[str, ChannelStatistics]:
        """Finalize population moments, rejecting empty/constant channels."""

        result: dict[str, ChannelStatistics] = {}
        for channel, name in enumerate(self.channel_names):
            count = int(self.count[channel])
            if count == 0:
                raise ValueError(f"normalization channel has no valid samples: {name}")
            variance = float(self.m2[channel] / count)
            if not math.isfinite(variance) or variance <= 0.0:
                raise ValueError(
                    f"normalization channel has zero/non-finite variance: {name}"
                )
            result[name] = ChannelStatistics(
                count=count,
                mean=float(self.mean[channel]),
                variance=variance,
                standard_deviation=math.sqrt(variance),
                minimum=float(self.minimum[channel]),
                maximum=float(self.maximum[channel]),
            )
        return result


@dataclass(frozen=True, slots=True)
class NormalizationArtifact:
    """Validated immutable train-only normalization publication."""

    statistics: Mapping[str, Mapping[str, ChannelStatistics]]
    provenance: Mapping[str, object]
    excluded_fields: Mapping[str, Sequence[str]]
    fit_split: str = "outer_train"
    format_version: str = NORMALIZATION_FORMAT_VERSION
    variance_convention: str = "population_m2_div_count"
    accumulation_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.format_version != NORMALIZATION_FORMAT_VERSION:
            raise ValueError("unsupported normalization artifact format")
        if self.fit_split != "outer_train":
            raise ValueError("normalization must be fit on outer_train only")
        if self.variance_convention != "population_m2_div_count":
            raise ValueError("normalization variance convention changed")
        if self.accumulation_dtype != "float64":
            raise ValueError("normalization moments must use float64")
        if not self.statistics:
            raise ValueError("normalization statistics cannot be empty")
        for group, channels in self.statistics.items():
            if not group or not channels:
                raise ValueError("normalization groups/channels must be non-empty")
            if any(not isinstance(value, ChannelStatistics) for value in channels.values()):
                raise TypeError("statistics must contain ChannelStatistics")
        hashes = self.provenance.get("artifact_hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError("artifact hash provenance is required")

        def validate_hash_tree(value: object) -> None:
            if isinstance(value, str):
                if not _SHA256.fullmatch(value):
                    raise ValueError(f"invalid SHA-256 provenance value: {value!r}")
            elif isinstance(value, Mapping):
                if not value:
                    raise ValueError("artifact hash mapping cannot be empty")
                for nested in value.values():
                    validate_hash_tree(nested)
            else:
                raise TypeError("artifact hash provenance must contain strings/maps")

        validate_hash_tree(hashes)
        universe = self.provenance.get("time_universe_counts")
        if not isinstance(universe, Mapping) or not universe:
            raise ValueError("time-universe provenance is required")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in universe.values()
        ):
            raise ValueError("time-universe counts must be non-negative integers")

    def to_json(self) -> dict[str, object]:
        statistics = {
            group: {
                name: channels[name].to_json()
                for name in sorted(channels)
            }
            for group, channels in sorted(self.statistics.items())
        }
        return {
            "format_version": self.format_version,
            "fit_split": self.fit_split,
            "variance_convention": self.variance_convention,
            "accumulation_dtype": self.accumulation_dtype,
            "statistics": statistics,
            "channel_order": {
                group: list(channels)
                for group, channels in sorted(self.statistics.items())
            },
            "excluded_fields": {
                group: sorted(map(str, fields))
                for group, fields in sorted(self.excluded_fields.items())
            },
            "provenance": dict(self.provenance),
            "statistics_semantic_sha256": semantic_sha256(statistics),
        }


def artifact_from_json(raw: Mapping[str, object]) -> NormalizationArtifact:
    statistics_raw = raw.get("statistics")
    channel_order_raw = raw.get("channel_order")
    if not isinstance(statistics_raw, Mapping) or not isinstance(
        channel_order_raw, Mapping
    ):
        raise ValueError("normalization JSON has no statistics mapping")
    statistics: dict[str, dict[str, ChannelStatistics]] = {}
    for group, channels_raw in statistics_raw.items():
        if not isinstance(group, str) or not isinstance(channels_raw, Mapping):
            raise ValueError("invalid normalization statistics group")
        order_raw = channel_order_raw.get(group)
        if not isinstance(order_raw, list) or any(
            not isinstance(value, str) for value in order_raw
        ):
            raise ValueError(f"normalization channel order is missing for {group}")
        order = tuple(order_raw)
        if len(order) != len(set(order)) or set(order) != set(channels_raw):
            raise ValueError(f"normalization channel order/schema mismatch for {group}")
        statistics[group] = {}
        for name in order:
            values_raw = channels_raw[name]
            if not isinstance(values_raw, Mapping):
                raise ValueError("invalid normalization channel record")
            statistics[group][name] = ChannelStatistics(
                count=int(values_raw["count"]),
                mean=float(values_raw["mean"]),
                variance=float(values_raw["variance"]),
                standard_deviation=float(values_raw["standard_deviation"]),
                minimum=float(values_raw["minimum"]),
                maximum=float(values_raw["maximum"]),
            )
    provenance = raw.get("provenance")
    excluded = raw.get("excluded_fields")
    if not isinstance(provenance, Mapping) or not isinstance(excluded, Mapping):
        raise ValueError("normalization provenance/excluded-fields are required")
    artifact = NormalizationArtifact(
        statistics=statistics,
        provenance=provenance,
        excluded_fields={
            str(group): tuple(str(value) for value in values)  # type: ignore[arg-type]
            for group, values in excluded.items()
        },
        fit_split=str(raw.get("fit_split", "")),
        format_version=str(raw.get("format_version", "")),
        variance_convention=str(raw.get("variance_convention", "")),
        accumulation_dtype=str(raw.get("accumulation_dtype", "")),
    )
    expected = raw.get("statistics_semantic_sha256")
    if expected != artifact.to_json()["statistics_semantic_sha256"]:
        raise ValueError("normalization statistics semantic hash mismatch")
    return artifact


def write_normalization_artifact(path: Path, artifact: NormalizationArtifact) -> str:
    """Atomically publish one immutable JSON artifact and return its hash."""

    if not isinstance(artifact, NormalizationArtifact):
        raise TypeError("artifact must be a NormalizationArtifact")
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            artifact.to_json(), sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publication is atomic and, unlike os.replace(), cannot
        # overwrite a file another producer published between our two checks.
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def read_normalization_artifact(path: Path) -> NormalizationArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("normalization artifact root must be a mapping")
    return artifact_from_json(raw)


def normalize_channel_first(
    values: ArrayLike,
    *,
    channel_names: Sequence[str],
    statistics: Mapping[str, ChannelStatistics],
) -> NDArray[np.float32]:
    """Apply a frozen named schema without implicit dtype/schema fallback."""

    raw = np.asarray(values)
    names = tuple(channel_names)
    if raw.ndim < 1 or raw.shape[0] != len(names):
        raise ValueError("values/channel_names shape mismatch")
    if set(names) != set(statistics) or len(names) != len(set(names)):
        raise ValueError("normalization channel schema mismatch")
    if not np.all(np.isfinite(raw)):
        raise ValueError("values to normalize must be finite")
    output = np.asarray(raw, dtype=np.float32).copy()
    for index, name in enumerate(names):
        record = statistics[name]
        output[index] = (
            output[index] - np.float32(record.mean)
        ) / np.float32(record.standard_deviation)
    if not np.all(np.isfinite(output)):
        raise ValueError("normalization produced a non-finite result")
    return output


__all__ = [
    "NORMALIZATION_FORMAT_VERSION",
    "ChannelStatistics",
    "NormalizationArtifact",
    "StreamingChannelMoments",
    "artifact_from_json",
    "normalize_channel_first",
    "read_normalization_artifact",
    "semantic_sha256",
    "sha256_file",
    "write_normalization_artifact",
]

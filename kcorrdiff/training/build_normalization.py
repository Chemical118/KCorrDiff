"""Build the immutable outer-train input-normalization artifact.

This CLI consumes the published candidate manifest, not the bounded training
draw: repeated sampler rows must not change data statistics.  It derives a
deduplicated issue-time history universe and a signature/access-aware ERA
valid-time universe, then scans the existing radar-v1 and era5-v1 mmap caches.

Example::

    python -m kcorrdiff.training.build_normalization \
      --candidate-manifest /workspace/data/manifests/event-pretrain-initial-v1/candidate-manifest.json \
      --radar-cache-root /workspace/data/cache/radar-v1 \
      --era5-cache-root /workspace/data/cache/era5-v1 \
      --target-static /workspace/data/static/target_static.npz \
      --context-coordinates /workspace/data/static/condition-grid-coordinates.npz \
      --source-manifest configs/cprecnet-archive-manifest.json \
      --output-json /workspace/data/normalization/outer-train-v1.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from kcorrdiff.data.availability import format_radar_timestamp, history_times
from kcorrdiff.data.cache import RadarCache
from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.data.era5_reader import (
    Era5MmapCache,
    native_hour_times,
    temporal_access_mask,
    trajectory_window_mask,
)
from kcorrdiff.data.normalization import (
    NormalizationArtifact,
    StreamingChannelMoments,
    semantic_sha256,
    sha256_file,
    write_normalization_artifact,
)
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.radar_values import cprecnet_normalized_to_rain_rate
from kcorrdiff.data.split_manifest import canonical_sample_id, parse_utc


_STATIC_CONTINUOUS = ("elevation", "slope_x", "slope_y", "local_relief")
_STATIC_EXCLUDED = ("land_mask", "lcc_x_normalized", "lcc_y_normalized")


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    radar_issue_times: tuple[datetime, ...]
    era_instantaneous_times: tuple[datetime, ...]
    era_tp_times: tuple[datetime, ...]
    total_candidate_rows: int
    outer_train_candidate_rows: int
    semantic_sha256: str
    split_definition_sha256: str
    candidate_metadata_hashes: Mapping[str, str]


def _canonical_json_hash(value: object) -> str:
    return semantic_sha256(value)


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} root must be a JSON mapping")
    return raw


def _outer_train_interval(metadata: Mapping[str, object]) -> tuple[datetime, datetime]:
    raw_intervals = metadata.get("split_intervals")
    if not isinstance(raw_intervals, list):
        raise ValueError("candidate metadata must contain split_intervals")
    outer: list[tuple[datetime, datetime]] = []
    seen: set[str] = set()
    previous_end: datetime | None = None
    chronological: list[tuple[datetime, datetime, str]] = []
    for raw in raw_intervals:
        if not isinstance(raw, Mapping):
            raise ValueError("split interval must be a mapping")
        name = str(raw.get("name", ""))
        if not name or name in seen:
            raise ValueError("split interval names must be unique and non-empty")
        seen.add(name)
        start = parse_utc(str(raw.get("start", "")))
        end = parse_utc(str(raw.get("end", "")))
        if end <= start:
            raise ValueError(f"empty split interval: {name}")
        chronological.append((start, end, name))
        if name == "outer_train":
            outer.append((start, end))
    for start, end, name in sorted(chronological):
        if previous_end is not None and start < previous_end:
            raise ValueError(f"overlapping split interval at {name}")
        previous_end = end
    if len(outer) != 1:
        raise ValueError("candidate metadata must define exactly one outer_train")
    return outer[0]


def derive_candidate_universe(path: Path) -> CandidateUniverse:
    """Derive unique train-only input times from a candidate manifest."""

    raw = _read_json_mapping(path, name="candidate manifest")
    if raw.get("version") != "kcorrdiff.split-manifest.v1":
        raise ValueError("unsupported candidate manifest version")
    metadata = raw.get("metadata")
    items = raw.get("items")
    if not isinstance(metadata, Mapping) or not isinstance(items, list):
        raise ValueError("candidate manifest requires metadata and items")
    outer_start, outer_end = _outer_train_interval(metadata)
    split_definition = metadata["split_intervals"]

    issue_times: set[datetime] = set()
    instant_times: set[datetime] = set()
    tp_times: set[datetime] = set()
    identities: set[tuple[str, float, str]] = set()
    semantic_rows: list[tuple[str, float, str, str, int]] = []
    outer_count = 0
    for line, item in enumerate(items, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidate item {line} is not a mapping")
        if item.get("split") != "outer_train":
            continue
        outer_count += 1
        t0 = parse_utc(str(item.get("t0_utc", "")))
        lead = float(item.get("lead_hours", math.nan))
        signature_text = str(item.get("condition_signature", ""))
        signature = parse_condition_signature(signature_text)
        sample_id = str(item.get("sample_id", ""))
        if sample_id != canonical_sample_id(t0, lead, signature_text):
            raise ValueError(f"non-canonical outer_train sample ID: {sample_id}")
        identity = (t0.isoformat(), lead, signature_text)
        if identity in identities:
            raise ValueError(f"duplicate outer_train candidate identity: {identity}")
        identities.add(identity)
        fold_id = item.get("fold_id")
        if (
            isinstance(fold_id, bool)
            or not isinstance(fold_id, int)
            or fold_id < 0
        ):
            raise ValueError(f"outer_train candidate has invalid fold: {sample_id}")
        dependency_start = t0 - timedelta(minutes=55)
        dependency_end = t0 + timedelta(hours=6)
        if not (outer_start <= dependency_start and dependency_end < outer_end):
            raise ValueError(f"outer_train dependency crosses split: {sample_id}")
        if "dependency_start_utc" in item and parse_utc(
            str(item["dependency_start_utc"])
        ) != dependency_start:
            raise ValueError(f"candidate dependency start mismatch: {sample_id}")
        if "dependency_end_utc" in item and parse_utc(
            str(item["dependency_end_utc"])
        ) != dependency_end:
            raise ValueError(f"candidate dependency end mismatch: {sample_id}")
        issue_times.add(t0)
        semantic_rows.append(
            (t0.isoformat(), lead, signature_text, str(item.get("block_id", "")), fold_id)
        )

        if not signature.era_present:
            continue
        valid_times = native_hour_times(t0)
        trajectory = trajectory_window_mask(t0)
        access = temporal_access_mask(
            t0,
            lead_hours=lead,
            access_mode=signature.temporal_access_mode,  # type: ignore[arg-type]
        )
        allowed = trajectory & access
        instant_times.update(
            valid_time
            for valid_time, selected in zip(valid_times, allowed, strict=True)
            if selected
        )
        if signature.tp_present:
            tp_times.update(
                valid_time
                for valid_time, selected in zip(valid_times, allowed, strict=True)
                if selected
            )

    if outer_count == 0 or not issue_times:
        raise ValueError("candidate manifest has no outer_train input universe")
    metadata_hashes: dict[str, str] = {}
    for key in (
        "config_sha256",
        "event_index_semantic_sha256",
        "eligible_universe_sample_ids_sha256",
        "candidate_semantic_sha256",
        "fold_map_sha256",
        "condition_augmentation_policy_sha256",
    ):
        value = metadata.get(key)
        if value is not None:
            text = str(value)
            if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
                raise ValueError(f"candidate metadata has invalid {key}")
            metadata_hashes[key] = text
    return CandidateUniverse(
        radar_issue_times=tuple(sorted(issue_times)),
        era_instantaneous_times=tuple(sorted(instant_times)),
        era_tp_times=tuple(sorted(tp_times)),
        total_candidate_rows=len(items),
        outer_train_candidate_rows=outer_count,
        semantic_sha256=_canonical_json_hash(sorted(semantic_rows)),
        split_definition_sha256=_canonical_json_hash(split_definition),
        candidate_metadata_hashes=metadata_hashes,
    )


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_static(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {*_STATIC_CONTINUOUS, *_STATIC_EXCLUDED, "coverage"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"target static artifact is incomplete: {missing}")
        coverage_raw = np.asarray(archive["coverage"])
        if coverage_raw.dtype != np.bool_ and not (
            np.issubdtype(coverage_raw.dtype, np.number)
            and np.all((coverage_raw == 0) | (coverage_raw == 1))
        ):
            raise ValueError("target static coverage must be boolean/0-1")
        coverage = np.asarray(coverage_raw, dtype=np.bool_)
        fields = np.stack(
            [np.asarray(archive[name]) for name in _STATIC_CONTINUOUS], axis=0
        )
        excluded = [np.asarray(archive[name]) for name in _STATIC_EXCLUDED]
    if coverage.ndim != 2 or any(
        field.shape != coverage.shape for field in (*fields, *excluded)
    ):
        raise ValueError("target static model fields must share one [H,W] grid")
    if not coverage.any():
        raise ValueError("target static coverage contains no valid pixels")
    if not np.all(np.isfinite(fields[:, coverage])):
        raise ValueError("covered target static continuous fields are non-finite")
    if not np.all(np.isfinite(np.stack(excluded, axis=0))):
        raise ValueError("target static excluded model fields are non-finite")
    land_mask = excluded[0]
    if not np.all((land_mask == 0) | (land_mask == 1)):
        raise ValueError("target static land_mask must be binary")
    return fields, coverage, tuple(map(int, coverage.shape))


def _context_geometry_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "latitude" not in archive.files or "longitude" not in archive.files:
            raise ValueError("context coordinates require latitude and longitude")
        latitude = np.asarray(archive["latitude"])
        longitude = np.asarray(archive["longitude"])
    if latitude.shape != expected_shape or longitude.shape != expected_shape:
        raise ValueError(
            "context coordinate grid does not match condition radar cache shape"
        )
    mask = np.isfinite(latitude) & np.isfinite(longitude)
    if not mask.any():
        raise ValueError("context coordinates contain no valid geometry")
    return mask


def _verify_bundle_metadata(
    path: Path | None, *, candidate_path: Path, candidate_sha256: str
) -> tuple[str | None, str | None]:
    if path is None:
        sibling = candidate_path.with_name("bundle-metadata.json")
        path = sibling if sibling.is_file() else None
    if path is None:
        return None, None
    path = path.resolve()
    raw = _read_json_mapping(path, name="bundle metadata")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("bundle metadata has no artifacts mapping")
    candidate = artifacts.get("candidate_manifest")
    if not isinstance(candidate, Mapping):
        raise ValueError("bundle metadata has no candidate_manifest artifact")
    if candidate.get("sha256") != candidate_sha256:
        raise ValueError("candidate manifest hash disagrees with bundle metadata")
    sidecar = path.with_suffix(".sha256")
    actual = sha256_file(path)
    if sidecar.is_file():
        parts = sidecar.read_text(encoding="ascii").split()
        if len(parts) < 1 or parts[0] != actual:
            raise ValueError("bundle metadata SHA-256 sidecar mismatch")
    return str(path), actual


def _source_manifest_hashes(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name
        if name in hashes:
            raise ValueError(f"duplicate source-manifest basename: {name}")
        hashes[name] = sha256_file(path)
        resolved[name] = str(path)
    if not hashes:
        raise ValueError("at least one source manifest is required")
    return hashes, resolved


def _radar_provenance(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = root / "manifest.json"
    raw = _read_json_mapping(manifest_path, name="radar cache manifest")
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("radar cache manifest has no shards")
    source_records: list[dict[str, object]] = []
    indexes: dict[str, str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("invalid radar cache shard")
        source_records.append(
            {
                "source": shard.get("source"),
                "archive_name": shard.get("archive_name"),
                "archive_size": shard.get("archive_size"),
                "archive_sha256": shard.get("archive_sha256"),
                "values_file": shard.get("values_file"),
                "values_sha256": shard.get("values_sha256"),
            }
        )
        index_name = str(shard.get("timestamps_file", ""))
        index_path = root / index_name
        indexes[index_name] = sha256_file(index_path)
    hashes = {
        "cache_manifest": sha256_file(manifest_path),
        "source_archives_semantic": semantic_sha256(
            sorted(source_records, key=lambda value: (str(value["source"]), str(value["archive_name"])))
        ),
        "timestamp_indexes_semantic": semantic_sha256(indexes),
    }
    return hashes, {
        "cache_manifest": str(manifest_path.resolve()),
        "cache_root": str(root.resolve()),
    }


def _era_provenance(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = root / "manifest.json"
    raw = _read_json_mapping(manifest_path, name="ERA5 cache manifest")
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("ERA5 cache manifest has no shards")
    validity: dict[str, str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("invalid ERA5 cache shard")
        for key in ("data_valid_inst_file", "tp_valid_file"):
            name = str(shard.get(key, ""))
            validity[name] = sha256_file(root / name)
    coordinates = {
        str(raw.get("latitude_file", "")): sha256_file(
            root / str(raw.get("latitude_file", ""))
        ),
        str(raw.get("longitude_file", "")): sha256_file(
            root / str(raw.get("longitude_file", ""))
        ),
    }
    source_contract = {
        "channels": raw.get("channels"),
        "instantaneous_channel_names": raw.get("instantaneous_channel_names"),
        "tp_channel_name": raw.get("tp_channel_name"),
        "tp_units": raw.get("tp_units"),
        "tp_interval": raw.get("tp_interval"),
        "provenance": raw.get("provenance"),
    }
    hashes = {
        "cache_manifest": sha256_file(manifest_path),
        "source_contract_semantic": semantic_sha256(source_contract),
        "validity_masks_semantic": semantic_sha256(validity),
        "coordinates_semantic": semantic_sha256(coordinates),
    }
    return hashes, {
        "cache_manifest": str(manifest_path.resolve()),
        "cache_root": str(root.resolve()),
    }


def build_normalization_artifact(
    *,
    candidate_manifest: Path,
    radar_cache_root: Path,
    era5_cache_root: Path,
    target_static: Path,
    context_coordinates: Path,
    source_manifests: Sequence[Path],
    bundle_metadata: Path | None = None,
    radar_timezone: str = "Asia/Seoul",
    chunk_size: int = 32,
    verify_cache_hashes: bool = True,
) -> NormalizationArtifact:
    """Scan exact outer-train input universes and build a strict artifact."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    candidate_manifest = candidate_manifest.resolve()
    radar_cache_root = radar_cache_root.resolve()
    era5_cache_root = era5_cache_root.resolve()
    target_static = target_static.resolve()
    context_coordinates = context_coordinates.resolve()
    candidate_hash = sha256_file(candidate_manifest)
    universe = derive_candidate_universe(candidate_manifest)
    bundle_path, bundle_hash = _verify_bundle_metadata(
        bundle_metadata,
        candidate_path=candidate_manifest,
        candidate_sha256=candidate_hash,
    )

    static_values, target_coverage, target_shape = _load_static(target_static)
    static_moments = StreamingChannelMoments(_STATIC_CONTINUOUS)
    static_moments.update(static_values, valid_mask=target_coverage)

    source_hashes, source_paths = _source_manifest_hashes(source_manifests)
    radar_hashes, radar_paths = _radar_provenance(radar_cache_root)
    era_hashes, era_paths = _era_provenance(era5_cache_root)

    radar_cache = RadarCache(
        radar_cache_root, verify_hashes=bool(verify_cache_hashes)
    )
    target_moments = StreamingChannelMoments(("cprecnet_normalized",))
    context_moments = StreamingChannelMoments(("rain_rate_mm_per_hour",))
    radar_keys = tuple(
        sorted(
            {
                format_radar_timestamp(value, timezone=radar_timezone)
                for t0 in universe.radar_issue_times
                for value in history_times(t0)
            }
        )
    )
    try:
        target_available = set(radar_cache.timestamps("target"))
        context_available = set(radar_cache.timestamps("condition"))
        missing_target = sorted(set(radar_keys) - target_available)
        missing_context = sorted(set(radar_keys) - context_available)
        if missing_target or missing_context:
            raise KeyError(
                "outer_train radar history is absent from cache: "
                f"target={missing_target[:3]}, condition={missing_context[:3]}"
            )
        context_shape: tuple[int, int] | None = None
        context_mask: np.ndarray | None = None
        for keys in _chunks(radar_keys, chunk_size):
            target = radar_cache.read_many("target", keys)
            context_normalized = radar_cache.read_many("condition", keys)
            if target.ndim != 3 or tuple(target.shape[1:]) != target_shape:
                raise ValueError("target radar cache grid does not match target static")
            valid_target = target[:, target_coverage]
            if not np.all(np.isfinite(valid_target)) or np.any(
                (valid_target < 0.0) | (valid_target > 1.0)
            ):
                raise ValueError(
                    "valid target CPrecNet cache values must be finite in [0,1]"
                )
            target_moments.update(target[None], valid_mask=target_coverage)

            if context_normalized.ndim != 3:
                raise ValueError("condition radar cache fields must have shape [T,H,W]")
            current_context_shape = tuple(map(int, context_normalized.shape[1:]))
            if context_shape is None:
                context_shape = current_context_shape
                context_mask = _context_geometry_mask(
                    context_coordinates, context_shape
                )
            elif current_context_shape != context_shape:
                raise ValueError("condition radar cache grid changed between shards")
            assert context_mask is not None
            context_rate = np.zeros(context_normalized.shape, dtype=np.float64)
            context_rate[:, context_mask] = cprecnet_normalized_to_rain_rate(
                context_normalized[:, context_mask]
            )
            context_moments.update(context_rate[None], valid_mask=context_mask)
    finally:
        radar_cache.close()

    era_cache = Era5MmapCache(
        era5_cache_root, verify_hashes=bool(verify_cache_hashes)
    )
    inst_moments = StreamingChannelMoments(ERA5_CHANNEL_NAMES[:-1])
    tp_moments = StreamingChannelMoments((ERA5_CHANNEL_NAMES[-1],))
    inst_requested = set(universe.era_instantaneous_times)
    tp_requested = set(universe.era_tp_times)
    valid_inst_tokens = 0
    valid_tp_tokens = 0
    try:
        for valid_time in sorted(inst_requested | tp_requested):
            # The cache's absolute-hour reader is the canonical indexed access
            # used by read_window; scanning the deduplicated universe avoids
            # repeating each native hour for every lead/signature candidate.
            instantaneous, tp, inst_valid, tp_valid = era_cache._read_hour(  # noqa: SLF001
                valid_time
            )
            if instantaneous.shape[0] != 23 or tp.shape[0] != 1:
                raise ValueError("ERA5 cache does not preserve 23+1 channel split")
            if valid_time in inst_requested and inst_valid:
                inst_moments.update(instantaneous)
                valid_inst_tokens += 1
            if valid_time in tp_requested and tp_valid:
                tp_moments.update(tp)
                valid_tp_tokens += 1
    finally:
        era_cache.close()

    statistics = {
        "target_radar": target_moments.finalize(),
        "context_radar": context_moments.finalize(),
        "era5_instantaneous": inst_moments.finalize(),
        "era5_tp": tp_moments.finalize(),
        "target_static": static_moments.finalize(),
    }
    time_counts = {
        "candidate_rows_total": universe.total_candidate_rows,
        "outer_train_candidate_rows": universe.outer_train_candidate_rows,
        "outer_train_unique_t0": len(universe.radar_issue_times),
        "radar_history_unique_timestamps": len(radar_keys),
        "era_instantaneous_requested_unique_hours": len(inst_requested),
        "era_instantaneous_valid_unique_hours": valid_inst_tokens,
        "era_tp_requested_unique_hours": len(tp_requested),
        "era_tp_valid_unique_hours": valid_tp_tokens,
        "target_static_valid_pixels": int(target_coverage.sum()),
        "context_geometry_valid_pixels": int(context_mask.sum()) if context_mask is not None else 0,
    }
    artifact_hashes: dict[str, object] = {
        "candidate_manifest": candidate_hash,
        "candidate_outer_train_semantic": universe.semantic_sha256,
        "split_definition_semantic": universe.split_definition_sha256,
        "radar": radar_hashes,
        "era5": era_hashes,
        "static": {
            "target_static": sha256_file(target_static),
            "context_coordinates": sha256_file(context_coordinates),
        },
        "source_manifests": source_hashes,
    }
    artifact_paths: dict[str, object] = {
        "candidate_manifest": str(candidate_manifest),
        "radar": radar_paths,
        "era5": era_paths,
        "target_static": str(target_static),
        "context_coordinates": str(context_coordinates),
        "source_manifests": source_paths,
    }
    if bundle_hash is not None and bundle_path is not None:
        artifact_hashes["bundle_metadata"] = bundle_hash
        artifact_paths["bundle_metadata"] = bundle_path
    return NormalizationArtifact(
        statistics=statistics,
        excluded_fields={
            "target_static": _STATIC_EXCLUDED,
            "context_dynamic": (
                "wet_mask",
                "valid_fraction",
                "interpolation_confidence",
            ),
            "era5_state": (
                "data_valid_inst",
                "tp_valid",
                "trajectory_window_mask",
                "temporal_access_mask",
                "era_present",
                "tp_present",
            ),
        },
        provenance={
            "artifact_hashes": artifact_hashes,
            "artifact_paths": artifact_paths,
            "candidate_metadata_hashes": dict(universe.candidate_metadata_hashes),
            "time_universe_counts": time_counts,
            "radar_history_policy": "12_issue_time_frames_only_no_future_label",
            "target_radar_value_space": "cprecnet_normalized_[0,1]",
            "context_radar_value_space": "linear_mm_per_hour_before_regrid",
            "context_validity_policy": "finite_source_latitude_and_longitude",
            "era_time_policy": "signature_trajectory_and_temporal_access_deduplicated",
            "era_validity_policy": "instantaneous_and_tp_masks_separate",
            "era_tp_value_space": "mm_per_1_hour",
            "static_validity_policy": "target_static_coverage",
            "cache_hash_verification": bool(verify_cache_hashes),
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--bundle-metadata",
        type=Path,
        help="optional bundle-metadata.json; auto-detected next to candidate manifest",
    )
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--era5-cache-root", type=Path, required=True)
    parser.add_argument("--target-static", type=Path, required=True)
    parser.add_argument("--context-coordinates", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        required=True,
        help="source provenance manifest; repeat for multiple providers",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--radar-timezone", default="Asia/Seoul")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--verify-cache-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify every cache value-shard hash before fitting (default: true)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    artifact = build_normalization_artifact(
        candidate_manifest=arguments.candidate_manifest,
        bundle_metadata=arguments.bundle_metadata,
        radar_cache_root=arguments.radar_cache_root,
        era5_cache_root=arguments.era5_cache_root,
        target_static=arguments.target_static,
        context_coordinates=arguments.context_coordinates,
        source_manifests=tuple(arguments.source_manifest),
        radar_timezone=arguments.radar_timezone,
        chunk_size=arguments.chunk_size,
        verify_cache_hashes=arguments.verify_cache_hashes,
    )
    digest = write_normalization_artifact(arguments.output_json, artifact)
    print(
        json.dumps(
            {
                "output_json": str(arguments.output_json.resolve()),
                "sha256": digest,
                "time_universe_counts": artifact.provenance[
                    "time_universe_counts"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateUniverse",
    "build_normalization_artifact",
    "derive_candidate_universe",
    "main",
]

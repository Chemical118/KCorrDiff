"""Fail-closed Stage 2 data construction and rank-local loading.

The factory is the production boundary between immutable Stage 1 artifacts
and :class:`~kcorrdiff.data.dataset.KCorrDiffDataset`.  It validates every
content-addressed input before constructing model geometry, while deferring
the two mmap cache objects until a DataLoader worker requests its first real
sample.  Distributed tail padding stays an explicit zero-loss sentinel; a
missing slot is never replaced by a training row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from kcorrdiff.data.cache import FORMAT_VERSION as RADAR_CACHE_FORMAT
from kcorrdiff.data.cache import RadarCache
from kcorrdiff.data.condition_augmentation import (
    ConditionAugmentationProvenance,
    materialize_condition_signatures,
)
from kcorrdiff.data.condition_schema import StaticConditions, parse_condition_signature
from kcorrdiff.data.context_regrid import (
    ContextRegridOperator,
    build_context_regrid_operator,
)
from kcorrdiff.data.coordinates import (
    DEFAULT_COORDINATE_SCALE_KM,
    normalize_lcc_coordinates,
    target_center_from_axes,
)
from kcorrdiff.data.dataset import KCorrDiffDataset, TrainingSample
from kcorrdiff.data.era5_reader import ERA5_CACHE_FORMAT, Era5MmapCache
from kcorrdiff.data.normalization import (
    ChannelStatistics,
    NormalizationArtifact,
    read_normalization_artifact,
    semantic_sha256,
    sha256_file,
)
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.sampling import DrawRow, read_draw_manifest
from kcorrdiff.data.split_manifest import canonical_sample_id
from kcorrdiff.data.static_loader import (
    StaticNormalization,
    TargetStaticArtifact,
    load_target_static,
)
from kcorrdiff.models.context_encoder import CONTEXT_STATIC_CHANNEL_NAMES
from kcorrdiff.training.batch import (
    PhysicalGridSpec,
    TrainingBatch,
    TrainingBatchCollator,
)
from kcorrdiff.training.config import Stage2Config
from kcorrdiff.training.plan import (
    DistributedDrawPlan,
    DrawSlot,
    build_distributed_draw_plan,
)


PRODUCTION_RADAR_SIZE = 256
PRODUCTION_ERA_SIZE = 33
PRODUCTION_RESEARCH_TRACK = "cprecnet_event_conditioned_era5_full_trajectory_oracle"
EVENT_POPULATION = "cprecnet_event_conditioned_archive"
EVENT_ESTIMAND = "cprecnet_event_conditioned_pretraining"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATIC_CONTINUOUS = ("elevation", "slope_x", "slope_y", "local_relief")
_T = TypeVar("_T")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(raw, name=name)


def _same_hash_mapping(
    actual: Mapping[str, str], expected: object, *, name: str
) -> None:
    expected_mapping = _mapping(expected, name=name)
    parsed = {
        str(key): _sha256(value, name=f"{name}.{key}")
        for key, value in expected_mapping.items()
    }
    if parsed != dict(actual):
        raise ValueError(f"{name} does not match the immutable cache artifacts")


@dataclass(frozen=True, slots=True)
class CoordinateArtifact:
    """Hashed coordinate axes and their physical latitude/longitude grid."""

    path: Path
    sha256: str
    x_lcc_km: NDArray[np.float64]
    y_lcc_km: NDArray[np.float64]
    latitude_degrees: NDArray[np.float64]
    longitude_degrees: NDArray[np.float64]


def load_coordinate_artifact(
    path: Path,
    *,
    expected_sha256: str,
    kind: str,
) -> CoordinateArtifact:
    """Load one authoritative 256-square LCC artifact after hashing it."""

    if kind not in {"target", "context"}:
        raise ValueError("coordinate kind must be 'target' or 'context'")
    selected = path.resolve()
    expected = _sha256(expected_sha256, name=f"expected {kind} coordinate hash")
    actual = sha256_file(selected)
    if actual != expected:
        raise ValueError(f"{kind} coordinate artifact SHA-256 mismatch")
    with np.load(selected, allow_pickle=False) as archive:
        required = {"lcc_x_m", "lcc_y_m", "latitude", "longitude"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{kind} coordinate artifact is incomplete: {missing}")
        x = np.asarray(archive["lcc_x_m"], dtype=np.float64) / 1000.0
        y = np.asarray(archive["lcc_y_m"], dtype=np.float64) / 1000.0
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
    expected_axis = (PRODUCTION_RADAR_SIZE,)
    expected_grid = (PRODUCTION_RADAR_SIZE, PRODUCTION_RADAR_SIZE)
    if x.shape != expected_axis or y.shape != expected_axis:
        raise ValueError(f"{kind} LCC axes must each contain exactly 256 centres")
    if latitude.shape != expected_grid or longitude.shape != expected_grid:
        raise ValueError(f"{kind} latitude/longitude must be exactly 256 x 256")
    arrays = (x, y, latitude, longitude)
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{kind} coordinates contain a non-finite value")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ValueError(f"{kind} LCC axes must be strictly increasing")
    if kind == "target" and (
        not np.allclose(np.diff(x), 0.5, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), 0.5, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("target LCC axes must retain exact 0.5 km spacing")
    for values in arrays:
        values.setflags(write=False)
    return CoordinateArtifact(
        path=selected,
        sha256=actual,
        x_lcc_km=x,
        y_lcc_km=y,
        latitude_degrees=latitude,
        longitude_degrees=longitude,
    )


def _radar_cache_hashes(root: Path) -> dict[str, str]:
    manifest_path = root.resolve() / "manifest.json"
    raw = _read_json_mapping(manifest_path, name="radar cache manifest")
    if raw.get("format_version") != RADAR_CACHE_FORMAT or raw.get("dtype") != "float32":
        raise ValueError("radar cache format/dtype contract mismatch")
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("radar cache manifest has no shards")
    source_records: list[dict[str, object]] = []
    indexes: dict[str, str] = {}
    sources: set[str] = set()
    timestamps_by_source: dict[str, set[str]] = {}
    for raw_shard in shards:
        shard = _mapping(raw_shard, name="radar cache shard")
        source = str(shard.get("source", ""))
        if source not in {"target", "condition"}:
            raise ValueError("radar cache contains an unsupported source")
        sources.add(source)
        if tuple(shard.get("shape", ())) != (256, 256) or shard.get("dtype") != "float32":
            raise ValueError("radar cache shards must retain float32 256 x 256 fields")
        values_name = str(shard.get("values_file", ""))
        if not values_name or not (root / values_name).is_file():
            raise FileNotFoundError("radar cache shard payload is missing")
        _sha256(shard.get("values_sha256"), name="radar values_sha256")
        source_records.append(
            {
                "source": source,
                "archive_name": shard.get("archive_name"),
                "archive_size": shard.get("archive_size"),
                "archive_sha256": shard.get("archive_sha256"),
                "values_file": shard.get("values_file"),
                "values_sha256": shard.get("values_sha256"),
            }
        )
        index_name = str(shard.get("timestamps_file", ""))
        if not index_name or index_name in indexes:
            raise ValueError("radar timestamp-index names must be non-empty and unique")
        index_path = root / index_name
        indexes[index_name] = sha256_file(index_path)
        timestamp_values = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(timestamp_values, list) or any(
            not isinstance(value, str) or not value for value in timestamp_values
        ):
            raise ValueError("radar timestamp index must be a string list")
        if len(timestamp_values) != shard.get("samples"):
            raise ValueError("radar timestamp count disagrees with cache manifest")
        source_timestamps = timestamps_by_source.setdefault(source, set())
        if source_timestamps.intersection(timestamp_values):
            raise ValueError("radar cache repeats a source timestamp")
        source_timestamps.update(timestamp_values)
    if sources != {"target", "condition"}:
        raise ValueError("radar cache must contain target and condition shards")
    if timestamps_by_source["target"] != timestamps_by_source["condition"]:
        raise ValueError("target and condition radar cache timestamps disagree")
    return {
        "cache_manifest": sha256_file(manifest_path),
        "source_archives_semantic": semantic_sha256(
            sorted(
                source_records,
                key=lambda item: (str(item["source"]), str(item["archive_name"])),
            )
        ),
        "timestamp_indexes_semantic": semantic_sha256(indexes),
    }


def _era_cache_hashes_and_geometry(
    root: Path,
) -> tuple[dict[str, str], NDArray[np.float64], NDArray[np.float64], Mapping[str, object]]:
    selected = root.resolve()
    manifest_path = selected / "manifest.json"
    raw = _read_json_mapping(manifest_path, name="ERA5 cache manifest")
    if raw.get("format_version") != ERA5_CACHE_FORMAT or raw.get("dtype") != "float32":
        raise ValueError("ERA5 cache format/dtype contract mismatch")
    if tuple(raw.get("grid_shape", ())) != (PRODUCTION_ERA_SIZE, PRODUCTION_ERA_SIZE):
        raise ValueError("ERA5 cache must retain its native 33 x 33 grid")
    if tuple(raw.get("instantaneous_channel_names", ())) != ERA5_CHANNEL_NAMES[:-1]:
        raise ValueError("ERA5 instantaneous channel schema/order mismatch")
    if raw.get("tp_channel_name") != "tp" or raw.get("tp_units") != "mm per 1 hour":
        raise ValueError("ERA5 tp semantic contract mismatch")
    if (
        raw.get("latitude_orientation") != "south_to_north"
        or raw.get("longitude_orientation") != "west_to_east"
        or raw.get("time_zone") != "UTC"
        or raw.get("tp_interval") != "(valid_time - 1 hour, valid_time]"
    ):
        raise ValueError("ERA5 orientation/time contract mismatch")
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("ERA5 cache manifest has no shards")
    validity: dict[str, str] = {}
    years: set[int] = set()
    for raw_shard in shards:
        shard = _mapping(raw_shard, name="ERA5 cache shard")
        try:
            year = int(shard["year"])
            hours = int(shard["hours"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("ERA5 shard year/hour metadata is invalid") from error
        expected_hours = 8784 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 8760
        if year in years or hours != expected_hours:
            raise ValueError("ERA5 cache has a duplicate year or incorrect hour count")
        years.add(year)
        for key in ("instantaneous_file", "tp_file"):
            payload_name = str(shard.get(key, ""))
            if not payload_name or not (selected / payload_name).is_file():
                raise FileNotFoundError("ERA5 cache shard payload is missing")
        for key in ("data_valid_inst_file", "tp_valid_file"):
            name = str(shard.get(key, ""))
            if not name or name in validity:
                raise ValueError("ERA5 validity-mask filenames must be unique")
            validity[name] = sha256_file(selected / name)
    latitude_name = str(raw.get("latitude_file", ""))
    longitude_name = str(raw.get("longitude_file", ""))
    if not latitude_name or not longitude_name:
        raise ValueError("ERA5 cache manifest has no coordinate files")
    coordinate_hashes = {
        latitude_name: sha256_file(selected / latitude_name),
        longitude_name: sha256_file(selected / longitude_name),
    }
    latitude = np.asarray(
        np.load(selected / latitude_name, allow_pickle=False), dtype=np.float64
    )
    longitude = np.asarray(
        np.load(selected / longitude_name, allow_pickle=False), dtype=np.float64
    )
    if latitude.ndim == longitude.ndim == 1:
        if latitude.shape != (33,) or longitude.shape != (33,):
            raise ValueError("ERA5 coordinate axes must each contain 33 centres")
        if not np.all(np.diff(latitude) > 0.0) or not np.all(np.diff(longitude) > 0.0):
            raise ValueError("ERA5 axes must be south-to-north/west-to-east")
    elif latitude.ndim == longitude.ndim == 2:
        if latitude.shape != (33, 33) or longitude.shape != (33, 33):
            raise ValueError("ERA5 coordinate grids must be exactly 33 x 33")
        if not np.all(np.diff(latitude, axis=0) > 0.0) or not np.all(
            np.diff(longitude, axis=1) > 0.0
        ):
            raise ValueError("ERA5 grids must be south-to-north/west-to-east")
    else:
        raise ValueError("ERA5 latitude and longitude ranks disagree")
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("ERA5 coordinates contain a non-finite value")
    provenance = _mapping(raw.get("provenance"), name="ERA5 cache provenance")
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
        "coordinates_semantic": semantic_sha256(coordinate_hashes),
    }
    latitude.setflags(write=False)
    longitude.setflags(write=False)
    return hashes, latitude, longitude, provenance


@dataclass(frozen=True, slots=True)
class FactoryInputs:
    """External immutable paths/hashes not carried by ``Stage2Config``."""

    radar_cache_root: Path
    era5_cache_root: Path
    target_static: Path
    candidate_manifest: Path
    draw_manifest: Path
    bundle_metadata: Path
    expected_target_coordinates_sha256: str
    expected_context_coordinates_sha256: str
    verify_cache_hashes: bool = False

    def __post_init__(self) -> None:
        for name in (
            "radar_cache_root",
            "era5_cache_root",
            "target_static",
            "candidate_manifest",
            "draw_manifest",
            "bundle_metadata",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        _sha256(
            self.expected_target_coordinates_sha256,
            name="expected_target_coordinates_sha256",
        )
        _sha256(
            self.expected_context_coordinates_sha256,
            name="expected_context_coordinates_sha256",
        )
        if not isinstance(self.verify_cache_hashes, bool):
            raise TypeError("verify_cache_hashes must be boolean")


def _artifact_record(
    bundle: Mapping[str, object], name: str
) -> Mapping[str, object]:
    artifacts = _mapping(bundle.get("artifacts"), name="bundle artifacts")
    return _mapping(artifacts.get(name), name=f"bundle artifact {name}")


def _stream_candidate_items(
    path: Path,
) -> tuple[Iterable[Mapping[str, object]], Callable[[], Mapping[str, object]]]:
    """Incrementally decode the canonical large candidate JSON.

    The returned iterator owns the file until exhaustion.  ``read_tail`` is
    valid only afterwards and returns the small metadata/version suffix.
    This avoids materializing the ~929k item list a second time in RAM.
    """

    stream = path.open("r", encoding="utf-8")
    prefix = stream.read(len('{"items":['))
    if prefix != '{"items":[':
        stream.close()
        raise ValueError("candidate manifest is not canonical streamed JSON")
    decoder = json.JSONDecoder()
    buffer = ""
    offset = 0
    tail: dict[str, Mapping[str, object]] = {}

    def fill() -> bool:
        nonlocal buffer, offset
        if offset:
            buffer = buffer[offset:]
            offset = 0
        chunk = stream.read(1024 * 1024)
        buffer += chunk
        return bool(chunk)

    def items() -> Iterable[Mapping[str, object]]:
        nonlocal buffer, offset
        try:
            fill()
            first = True
            while True:
                while True:
                    while offset < len(buffer) and buffer[offset].isspace():
                        offset += 1
                    if offset < len(buffer):
                        break
                    if not fill():
                        raise ValueError("candidate manifest ended inside items")
                if buffer[offset] == "]":
                    offset += 1
                    suffix = buffer[offset:] + stream.read()
                    raw_tail = json.loads('{"items":[]' + suffix)
                    tail["value"] = _mapping(raw_tail, name="candidate manifest")
                    return
                if first:
                    first = False
                else:
                    if buffer[offset] != ",":
                        raise ValueError("candidate manifest item separator is invalid")
                    offset += 1
                while True:
                    try:
                        value, next_offset = decoder.raw_decode(buffer, offset)
                    except json.JSONDecodeError:
                        if not fill():
                            raise ValueError("candidate manifest has a truncated item") from None
                        continue
                    offset = next_offset
                    yield _mapping(value, name="candidate item")
                    break
        finally:
            stream.close()

    iterator = items()

    def read_tail() -> Mapping[str, object]:
        if "value" not in tail:
            raise RuntimeError("candidate iterator must be exhausted before metadata")
        return tail["value"]

    return iterator, read_tail


def _audit_draw_membership(
    *,
    path: Path,
    rows: Sequence[DrawRow],
    bundle: Mapping[str, object],
    config: Stage2Config,
) -> None:
    """Independently prove every decoded draw belongs to the hashed universe."""

    wanted = {row.sample_id: row for row in rows}
    found: set[str] = set()
    count = 0
    items, read_tail = _stream_candidate_items(path)
    for item in items:
        count += 1
        sample_id = str(item.get("sample_id", ""))
        row = wanted.get(sample_id)
        if row is None:
            continue
        if sample_id in found:
            raise ValueError("candidate manifest repeats a drawn sample_id")
        invariant = {
            "t0_utc": row.t0_utc,
            "lead_hours": row.lead_hours,
            "condition_signature": row.condition_signature,
            "block_id": row.block_id,
            "stratum": row.stratum,
            "split": row.split,
            "fold_id": row.fold_id,
        }
        if any(item.get(name) != value for name, value in invariant.items()):
            raise ValueError("draw row metadata disagrees with candidate universe")
        required_sampling = ("p_target",)
        if config.data.condition_augmentation is not None:
            required_sampling = ("p_target", "p_draw", "omega")
        sampling_values: list[tuple[str, float, float]] = []
        for name, row_value in (
            ("p_target", row.p_target),
            ("p_draw", row.p_draw),
            ("omega", row.omega),
        ):
            if name not in item:
                if name in required_sampling:
                    raise ValueError(
                        f"candidate full-item {name} is missing"
                    )
                continue
            try:
                candidate_value = float(item[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"candidate full-item {name} is invalid"
                ) from error
            sampling_values.append((name, candidate_value, row_value))
        for name, candidate_value, row_value in sampling_values:
            if not math.isclose(
                candidate_value,
                row_value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"draw-row {name} disagrees with candidate universe"
                )
        found.add(sample_id)
    missing = sorted(set(wanted) - found)
    if missing:
        raise ValueError(f"draw manifest contains non-candidate rows: {missing[:3]}")
    tail = read_tail()
    if tail.get("version") != "kcorrdiff.split-manifest.v1":
        raise ValueError("candidate manifest version mismatch")
    metadata = _mapping(tail.get("metadata"), name="candidate metadata")
    if metadata.get("protocol_version") != config.protocol_version:
        raise ValueError("candidate protocol version disagrees with config")
    if metadata.get("outer_train_folds") != config.crossfit.folds:
        raise ValueError("candidate fold count disagrees with config")
    if (
        metadata.get("population") != EVENT_POPULATION
        or tuple(metadata.get("strata", ())) != ("event",)
        or metadata.get("probability_scope") != "within_split"
        or metadata.get("target_policy") != "eligible_items_uniform"
        or metadata.get("draw_policy")
        not in {"uniform", "block-balanced", "full-coverage-shuffled"}
    ):
        raise ValueError("candidate population/sampling contract mismatch")
    for key in (
        "config_sha256",
        "event_index_semantic_sha256",
        "eligible_universe_sample_ids_sha256",
        "candidate_semantic_sha256",
        "fold_map_sha256",
    ):
        if metadata.get(key) != bundle.get(key):
            raise ValueError(f"candidate and bundle {key} disagree")
    policy_sha256 = config.data.condition_augmentation_sha256
    if policy_sha256 is None:
        if "condition_augmentation_policy_sha256" in metadata:
            raise ValueError(
                "single-signature candidate unexpectedly declares augmentation"
            )
    elif metadata.get("condition_augmentation_policy_sha256") != policy_sha256:
        raise ValueError(
            "candidate condition augmentation policy hash disagrees with config"
        )
    record = _artifact_record(bundle, "candidate_manifest")
    declared_rows = record.get("rows")
    if declared_rows != count:
        raise ValueError("candidate row count disagrees with bundle metadata")


def _validate_bundle_and_rows(
    *, config: Stage2Config, inputs: FactoryInputs, normalization: NormalizationArtifact
) -> tuple[tuple[DrawRow, ...], dict[str, str], Mapping[str, object]]:
    candidate_hash = sha256_file(inputs.candidate_manifest)
    draw_hash = sha256_file(inputs.draw_manifest)
    bundle_hash = sha256_file(inputs.bundle_metadata)
    bundle = _read_json_mapping(inputs.bundle_metadata, name="bundle metadata")
    sidecar = inputs.bundle_metadata.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError("bundle metadata SHA-256 sidecar is required")
    tokens = sidecar.read_text(encoding="ascii").split()
    if not tokens or tokens[0] != bundle_hash:
        raise ValueError("bundle metadata SHA-256 sidecar mismatch")

    expected_artifacts = {
        "candidate_manifest": candidate_hash,
        "training_draw_manifest": draw_hash,
    }
    for name, actual in expected_artifacts.items():
        record = _artifact_record(bundle, name)
        if record.get("sha256") != actual:
            raise ValueError(f"{name} hash disagrees with bundle metadata")
        declared_path = record.get("path")
        if not isinstance(declared_path, str) or not declared_path:
            raise ValueError(f"{name} bundle path is missing")
        explicit_path = (
            inputs.candidate_manifest
            if name == "candidate_manifest"
            else inputs.draw_manifest
        )
        if (inputs.bundle_metadata.parent / declared_path).resolve() != explicit_path:
            raise ValueError(f"{name} path disagrees with bundle metadata")
        declared_bytes = record.get("bytes")
        if declared_bytes is not None and declared_bytes != explicit_path.stat().st_size:
            raise ValueError(f"{name} byte count disagrees with bundle metadata")
    if bundle.get("protocol_version") != config.protocol_version:
        raise ValueError("bundle and Stage 2 protocol versions disagree")
    if config.research_track != PRODUCTION_RESEARCH_TRACK:
        raise ValueError("Stage 2 factory only accepts the declared oracle research track")
    if bundle.get("population") != EVENT_POPULATION or bundle.get("estimand") != EVENT_ESTIMAND:
        raise ValueError("bundle is not the event-conditioned CPrecNet population")
    if tuple(bundle.get("strata", ())) != ("event",):
        raise ValueError("event-conditioned bundle must contain only the event stratum")
    expected_signatures = config.data.condition_signatures
    if tuple(bundle.get("condition_signatures", ())) != expected_signatures:
        raise ValueError("bundle condition signatures disagree with Stage 2 config")
    policy = config.data.condition_augmentation
    condition_provenance: ConditionAugmentationProvenance | None = None
    source_draw_rows: tuple[DrawRow, ...] | None = None
    condition_hashes: dict[str, str] = {}
    raw_condition_provenance = bundle.get("condition_augmentation")
    if policy is None:
        if raw_condition_provenance is not None or any(
            key in bundle
            for key in (
                "condition_augmentation_policy_sha256",
                "condition_augmentation_provenance_sha256",
            )
        ):
            raise ValueError(
                "single-signature bundle unexpectedly declares augmentation"
            )
    else:
        condition_provenance = ConditionAugmentationProvenance.from_mapping(
            _mapping(
                raw_condition_provenance,
                name="bundle condition augmentation provenance",
            )
        )
        policy_sha256 = policy.semantic_sha256
        if bundle.get("condition_augmentation_policy_sha256") != policy_sha256:
            raise ValueError("bundle condition policy SHA-256 disagrees with config")
        if condition_provenance.policy_sha256 != policy_sha256:
            raise ValueError("bundle condition provenance uses another policy")
        if condition_provenance.training_seed != config.seed:
            raise ValueError("condition augmentation seed disagrees with config")
        if condition_provenance.supported_signatures != expected_signatures:
            raise ValueError("condition provenance signature support disagrees")
        if bundle.get("condition_augmentation_provenance_sha256") != (
            condition_provenance.semantic_sha256
        ):
            raise ValueError("bundle condition provenance SHA-256 mismatch")

        source_record = _artifact_record(bundle, "source_training_draw_manifest")
        source_name = source_record.get("path")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source training draw bundle path is missing")
        relative_source = Path(source_name)
        if relative_source.is_absolute():
            raise ValueError("source training draw path must be bundle-relative")
        bundle_root = inputs.bundle_metadata.parent.resolve()
        source_path = (bundle_root / relative_source).resolve()
        try:
            source_path.relative_to(bundle_root)
        except ValueError as error:
            raise ValueError(
                "source training draw path escapes the bundle directory"
            ) from error
        if not source_path.is_file():
            raise FileNotFoundError("source training draw manifest is missing")
        source_hash = sha256_file(source_path)
        if source_hash != condition_provenance.source_draw_manifest_sha256 or (
            source_record.get("sha256") != source_hash
        ):
            raise ValueError("source training draw manifest SHA-256 mismatch")
        if source_record.get("bytes") != source_path.stat().st_size:
            raise ValueError("source training draw manifest byte count mismatch")
        if source_record.get("purpose_id") != (
            condition_provenance.source_draw_purpose_id
        ):
            raise ValueError("source training draw purpose disagrees with provenance")
        source_draw_rows = read_draw_manifest(source_path)
        if source_record.get("rows") != len(source_draw_rows):
            raise ValueError("source training draw row count mismatch")
        condition_hashes = {
            "condition_augmentation_policy": policy_sha256,
            "condition_augmentation_provenance": (
                condition_provenance.semantic_sha256
            ),
            "source_draw_manifest": source_hash,
        }
    folds = _mapping(bundle.get("folds"), name="bundle folds")
    if folds.get("split") != config.crossfit.train_split or folds.get("count") != config.crossfit.folds:
        raise ValueError("bundle fold contract disagrees with Stage 2 config")
    sampling = _mapping(bundle.get("sampling"), name="bundle sampling")
    if sampling.get("draw_split") != config.crossfit.train_split:
        raise ValueError("bundle draw split disagrees with Stage 2 config")
    if (
        sampling.get("target_policy") != "eligible_items_uniform"
        or sampling.get("probability_scope") != "within_split"
        or sampling.get("draw_policy")
        not in {"uniform", "block-balanced", "full-coverage-shuffled"}
        or sampling.get("weight_clipping") is not None
    ):
        raise ValueError("bundle sampling policy disagrees with v1.1.3b")
    if policy is not None:
        assert condition_provenance is not None
        if sampling.get("condition_selection_order") != (
            "base_sample_then_one_condition_signature"
        ):
            raise ValueError("bundle condition selection order is not sample-first")
        if sampling.get("signature_purpose_id") != (
            condition_provenance.signature_purpose_id
        ):
            raise ValueError("bundle signature purpose disagrees with provenance")
    if sampling.get("draw_policy") == "full-coverage-shuffled":
        if sampling.get("coverage") != "all_outer_train_base_candidates_once":
            raise ValueError("production full-coverage sampling contract is missing")
        if sampling.get("replacement") is not False:
            raise ValueError("production full-coverage sampling must be no-replacement")
        if sampling.get("draw_count") != sampling.get("base_candidate_count"):
            raise ValueError("full-coverage draw count/base support mismatch")

    hashes = _mapping(
        normalization.provenance.get("artifact_hashes"),
        name="normalization artifact hashes",
    )
    if hashes.get("candidate_manifest") != candidate_hash:
        raise ValueError("normalization and candidate manifest hashes disagree")
    if hashes.get("bundle_metadata") != bundle_hash:
        raise ValueError(
            "normalization must bind the bundle metadata containing the draw hash"
        )
    candidate_metadata = _mapping(
        normalization.provenance.get("candidate_metadata_hashes"),
        name="normalization candidate metadata hashes",
    )
    for key in (
        "config_sha256",
        "event_index_semantic_sha256",
        "eligible_universe_sample_ids_sha256",
        "candidate_semantic_sha256",
        "fold_map_sha256",
    ):
        if key in candidate_metadata and bundle.get(key) != candidate_metadata[key]:
            raise ValueError(f"bundle {key} disagrees with normalization provenance")
    policy_sha256 = config.data.condition_augmentation_sha256
    if policy_sha256 is not None and candidate_metadata.get(
        "condition_augmentation_policy_sha256"
    ) != policy_sha256:
        raise ValueError(
            "normalization candidate provenance lacks the condition policy hash"
        )

    rows = read_draw_manifest(inputs.draw_manifest)
    draw_record = _artifact_record(bundle, "training_draw_manifest")
    if draw_record.get("rows") != len(rows):
        raise ValueError("draw row count disagrees with bundle metadata")
    if policy is not None:
        assert condition_provenance is not None
        assert source_draw_rows is not None
        reproduced = materialize_condition_signatures(
            source_draw_rows,
            policy=policy,
            training_seed=config.seed,
            source_draw_purpose_id=(
                condition_provenance.source_draw_purpose_id
            ),
            signature_purpose_id=condition_provenance.signature_purpose_id,
            source_draw_manifest_sha256=(
                condition_provenance.source_draw_manifest_sha256
            ),
        )
        if reproduced.rows != rows:
            raise ValueError(
                "augmented draw rows are not the deterministic policy expansion"
            )
        if reproduced.provenance != condition_provenance:
            raise ValueError(
                "condition provenance does not match reproduced augmentation"
            )
    block_folds: dict[str, int] = {}
    seen_folds: set[int] = set()
    supported_signatures = set(expected_signatures)
    for position, row in enumerate(rows):
        if row.global_example_index != position:
            raise ValueError("draw rows are not in canonical order")
        if row.split != config.crossfit.train_split:
            raise ValueError("Stage 2 draw rows must be outer_train")
        if row.stratum != "event" or not row.block_id:
            raise ValueError("Stage 2 accepts only grouped event draw rows")
        signature = parse_condition_signature(row.condition_signature)
        if signature.key not in supported_signatures:
            raise ValueError("draw-row condition signature disagrees with config")
        if row.fold_id is None or not 0 <= row.fold_id < config.crossfit.folds:
            raise ValueError("draw-row fold_id is outside the configured folds")
        expected_id = canonical_sample_id(
            datetime.fromisoformat(row.t0_utc), row.lead_hours, row.condition_signature
        )
        if row.sample_id != expected_id:
            raise ValueError("draw row has a non-canonical sample_id")
        if not math.isclose(row.lead_hours * 2.0, round(row.lead_hours * 2.0)) or not (
            0.5 <= row.lead_hours <= 6.0
        ):
            raise ValueError("draw row has a non-canonical lead")
        prior = block_folds.setdefault(row.block_id, row.fold_id)
        if prior != row.fold_id:
            raise ValueError("one manifest block appears in multiple folds")
        seen_folds.add(row.fold_id)
    if seen_folds != set(range(config.crossfit.folds)):
        raise ValueError("draw manifest does not cover every configured fold")
    _audit_draw_membership(
        path=inputs.candidate_manifest,
        rows=rows,
        bundle=bundle,
        config=config,
    )
    return rows, {
        "candidate_manifest": candidate_hash,
        "draw_manifest": draw_hash,
        "bundle_metadata": bundle_hash,
        **condition_hashes,
    }, bundle


@dataclass(frozen=True, slots=True)
class EraArtifactNormalization:
    """Named 23+1 ERA transform with validity/presence kept independent."""

    instantaneous: Mapping[str, ChannelStatistics]
    tp: Mapping[str, ChannelStatistics]
    artifact_id: str
    era_manifest_path: Path
    condition_signature: str
    provider_id: str
    provider_version: str
    dataset: str
    condition_signatures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.instantaneous) != ERA5_CHANNEL_NAMES[:-1]:
            raise ValueError("ERA instantaneous normalization schema/order mismatch")
        if tuple(self.tp) != ERA5_CHANNEL_NAMES[-1:]:
            raise ValueError("ERA tp normalization schema/order mismatch")
        _sha256(self.artifact_id, name="normalization artifact_id")
        source_signature = parse_condition_signature(self.condition_signature).key
        configured = self.condition_signatures or (source_signature,)
        signatures = tuple(
            sorted(parse_condition_signature(value).key for value in configured)
        )
        if len(signatures) != len(set(signatures)):
            raise ValueError("ERA normalization condition signatures must be unique")
        if source_signature not in signatures:
            raise ValueError(
                "ERA normalization support must include the source signature"
            )
        object.__setattr__(self, "condition_signature", source_signature)
        object.__setattr__(self, "condition_signatures", signatures)
        if not self.provider_id or not self.provider_version or not self.dataset:
            raise ValueError("ERA normalization requires complete provider provenance")
        if not self.era_manifest_path.resolve().is_file():
            raise FileNotFoundError("ERA normalization cache manifest is missing")

    def normalize(
        self,
        values: np.ndarray,
        *,
        channel_names: tuple[str, ...],
        data_valid_inst: np.ndarray,
        tp_valid: np.ndarray,
        era_present: np.ndarray,
        tp_present: np.ndarray,
        provenance: tuple[object, ...],
    ) -> NDArray[np.float32]:
        raw = np.asarray(values)
        if raw.ndim != 5 or raw.shape[0] == 0 or raw.shape[1:] != (8, 24, 33, 33):
            raise ValueError("ERA values must have shape [B,8,24,33,33]")
        if tuple(channel_names) != ERA5_CHANNEL_NAMES:
            raise ValueError("ERA values use the wrong named channel order")
        batch = raw.shape[0]
        masks = {
            "data_valid_inst": np.asarray(data_valid_inst),
            "tp_valid": np.asarray(tp_valid),
            "era_present": np.asarray(era_present),
            "tp_present": np.asarray(tp_present),
        }
        if masks["data_valid_inst"].shape != (batch, 8) or masks["tp_valid"].shape != (
            batch,
            8,
        ):
            raise ValueError("ERA validity masks must have shape [B,8]")
        if masks["era_present"].shape != (batch,) or masks["tp_present"].shape != (
            batch,
        ):
            raise ValueError("ERA presence masks must have shape [B]")
        if any(mask.dtype != np.bool_ for mask in masks.values()):
            raise TypeError("ERA validity/presence masks must be boolean")
        if np.any(masks["tp_present"] & ~masks["era_present"]):
            raise ValueError("tp cannot be present when the ERA source is absent")
        if len(provenance) != batch:
            raise ValueError("ERA provenance length must equal batch size")
        expected_manifest = self.era_manifest_path.resolve()
        for item in provenance:
            if Path(str(getattr(item, "cache_manifest", ""))).resolve() != expected_manifest:
                raise ValueError("ERA sample provenance references another cache")
            provenance_signature = parse_condition_signature(
                getattr(item, "condition_signature", None)
            ).key
            if provenance_signature not in self.condition_signatures:
                raise ValueError("ERA sample provenance signature disagrees with config")
            for name, expected in (
                ("provider_id", self.provider_id),
                ("provider_version", self.provider_version),
                ("dataset", self.dataset),
            ):
                if getattr(item, name, None) != expected:
                    raise ValueError(f"ERA sample provenance {name} mismatch")

        output = np.zeros(raw.shape, dtype=np.float32)
        inst_token_valid = masks["data_valid_inst"] & masks["era_present"][:, None]
        tp_token_valid = (
            masks["tp_valid"]
            & masks["era_present"][:, None]
            & masks["tp_present"][:, None]
        )
        for channel, name in enumerate(ERA5_CHANNEL_NAMES[:-1]):
            selected = np.asarray(raw[:, :, channel], dtype=np.float32)
            valid = np.broadcast_to(inst_token_valid[:, :, None, None], selected.shape)
            if not np.all(np.isfinite(selected[valid])):
                raise ValueError(f"valid ERA channel {name} contains non-finite values")
            stats = self.instantaneous[name]
            normalized = (selected[valid] - np.float32(stats.mean)) / np.float32(
                stats.standard_deviation
            )
            output[:, :, channel][valid] = normalized
        selected_tp = np.asarray(raw[:, :, -1], dtype=np.float32)
        valid_tp = np.broadcast_to(tp_token_valid[:, :, None, None], selected_tp.shape)
        if not np.all(np.isfinite(selected_tp[valid_tp])):
            raise ValueError("valid ERA tp contains non-finite values")
        tp_stats = self.tp[ERA5_CHANNEL_NAMES[-1]]
        output[:, :, -1][valid_tp] = (
            selected_tp[valid_tp] - np.float32(tp_stats.mean)
        ) / np.float32(tp_stats.standard_deviation)
        if not np.all(np.isfinite(output)):
            raise ValueError("ERA normalization produced a non-finite value")
        return output


@dataclass(frozen=True, slots=True)
class FactoryArtifacts:
    rows: tuple[DrawRow, ...]
    context_regrid_operator: ContextRegridOperator
    target_static: TargetStaticArtifact
    static_conditions: StaticConditions
    geometry: PhysicalGridSpec
    era_normalization: EraArtifactNormalization
    condition_signatures: tuple[str, ...]
    context_static_fields: NDArray[np.float32]
    artifact_hashes: Mapping[str, str]

    @property
    def context_static_channel_names(self) -> tuple[str, ...]:
        return CONTEXT_STATIC_CHANNEL_NAMES


def _target_static_normalization(
    artifact: NormalizationArtifact, *, candidate_sha256: str
) -> StaticNormalization:
    raw = artifact.statistics.get("target_static")
    if raw is None or tuple(raw) != _STATIC_CONTINUOUS:
        raise ValueError("target-static normalization schema/order mismatch")
    return StaticNormalization(
        means={name: raw[name].mean for name in _STATIC_CONTINUOUS},
        standard_deviations={
            name: raw[name].standard_deviation for name in _STATIC_CONTINUOUS
        },
        fit_split=artifact.fit_split,
        fit_manifest_sha256=candidate_sha256,
    )


def build_factory_artifacts(
    config: Stage2Config,
    inputs: FactoryInputs,
    *,
    operator_builder: Callable[..., ContextRegridOperator] = build_context_regrid_operator,
) -> FactoryArtifacts:
    """Validate all immutable artifacts and build shared geometry exactly once."""

    if not isinstance(config, Stage2Config):
        raise TypeError("config must be a validated Stage2Config")
    normalization_path = config.data.normalization.resolve()
    normalization = read_normalization_artifact(normalization_path)
    if normalization.provenance.get("cache_hash_verification") is not True:
        raise ValueError("normalization must record verified cache payload hashes")
    normalization_hash = sha256_file(normalization_path)
    rows, manifest_hashes, _ = _validate_bundle_and_rows(
        config=config, inputs=inputs, normalization=normalization
    )
    hashes = _mapping(
        normalization.provenance.get("artifact_hashes"),
        name="normalization artifact hashes",
    )

    target_coordinates = load_coordinate_artifact(
        config.data.target_coordinates,
        expected_sha256=inputs.expected_target_coordinates_sha256,
        kind="target",
    )
    context_coordinates = load_coordinate_artifact(
        config.data.context_coordinates,
        expected_sha256=inputs.expected_context_coordinates_sha256,
        kind="context",
    )
    static_hashes = _mapping(hashes.get("static"), name="normalization static hashes")
    target_static_hash = sha256_file(inputs.target_static)
    if static_hashes.get("target_static") != target_static_hash:
        raise ValueError("normalization and target-static hashes disagree")
    if static_hashes.get("context_coordinates") != context_coordinates.sha256:
        raise ValueError("normalization and context-coordinate hashes disagree")

    radar_hashes = _radar_cache_hashes(inputs.radar_cache_root)
    _same_hash_mapping(radar_hashes, hashes.get("radar"), name="normalization radar hashes")
    era_hashes, era_latitude, era_longitude, era_provenance = (
        _era_cache_hashes_and_geometry(inputs.era5_cache_root)
    )
    _same_hash_mapping(era_hashes, hashes.get("era5"), name="normalization ERA5 hashes")

    static_normalization = _target_static_normalization(
        normalization, candidate_sha256=manifest_hashes["candidate_manifest"]
    )
    target_static = load_target_static(
        inputs.target_static,
        normalization=static_normalization,
        expected_sha256=target_static_hash,
    )
    for values in (
        target_static.values,
        target_static.coverage,
        target_static.x_lcc_km,
        target_static.y_lcc_km,
    ):
        values.setflags(write=False)
    if not np.all(
        (target_static.values[4] == 0.0) | (target_static.values[4] == 1.0)
    ):
        raise ValueError("target static land/sea mask must remain binary")
    if not np.allclose(
        target_static.x_lcc_km,
        target_coordinates.x_lcc_km,
        rtol=0.0,
        atol=1.0e-9,
    ) or not np.allclose(
        target_static.y_lcc_km,
        target_coordinates.y_lcc_km,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("target coordinate NPZ and static LCC axes disagree")

    context_operator = operator_builder(
        context_coordinates.x_lcc_km,
        context_coordinates.y_lcc_km,
        destination_shape=(PRODUCTION_RADAR_SIZE, PRODUCTION_RADAR_SIZE),
    )
    static_conditions = StaticConditions.from_arrays(
        target_values=target_static.values,
        target_static_coverage=target_static.coverage,
        target_channel_names=target_static.channel_names,
        context_regrid_operator=context_operator,
    )
    center_x, center_y = target_center_from_axes(
        target_coordinates.x_lcc_km, target_coordinates.y_lcc_km
    )
    target_x, target_y = np.meshgrid(
        target_coordinates.x_lcc_km,
        target_coordinates.y_lcc_km,
        indexing="xy",
    )
    expected_target_x, expected_target_y = normalize_lcc_coordinates(
        target_x,
        target_y,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        scale_km=DEFAULT_COORDINATE_SCALE_KM,
    )
    if not np.allclose(
        target_static.values[-2], expected_target_x, rtol=0.0, atol=2.0e-5
    ) or not np.allclose(
        target_static.values[-1], expected_target_y, rtol=0.0, atol=2.0e-5
    ):
        raise ValueError("target static shared LCC fields disagree with coordinate axes")

    context_x, context_y = np.meshgrid(
        context_operator.destination_x_km,
        context_operator.destination_y_km,
        indexing="xy",
    )
    context_x_shared, context_y_shared = normalize_lcc_coordinates(
        context_x,
        context_y,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        scale_km=DEFAULT_COORDINATE_SCALE_KM,
    )
    context_static_fields = np.stack(
        (
            context_x_shared,
            context_y_shared,
            static_conditions.context.source_dx_km,
            static_conditions.context.source_dy_km,
            static_conditions.context.nearest_source_distance_km,
            static_conditions.context.geometry_confidence,
        ),
        axis=0,
    ).astype(np.float32, copy=False)
    if context_static_fields.shape != (6, 256, 256) or not np.all(
        np.isfinite(context_static_fields)
    ):
        raise ValueError("named context static fields must be finite [6,256,256]")
    context_static_fields.setflags(write=False)

    geometry = PhysicalGridSpec(
        target_x_lcc_km=target_coordinates.x_lcc_km,
        target_y_lcc_km=target_coordinates.y_lcc_km,
        context_x_lcc_km=context_operator.destination_x_km,
        context_y_lcc_km=context_operator.destination_y_km,
        era_latitude_degrees=era_latitude,
        era_longitude_degrees=era_longitude,
    )
    instantaneous = normalization.statistics.get("era5_instantaneous")
    tp = normalization.statistics.get("era5_tp")
    if instantaneous is None or tp is None:
        raise ValueError("normalization artifact has no ERA 23+1 statistics")
    era_normalization = EraArtifactNormalization(
        instantaneous=instantaneous,
        tp=tp,
        artifact_id=normalization_hash,
        era_manifest_path=inputs.era5_cache_root / "manifest.json",
        condition_signature=config.data.condition_signature,
        provider_id=str(era_provenance.get("provider_id", "")),
        provider_version=str(era_provenance.get("provider_version", "")),
        dataset=str(era_provenance.get("dataset", "")),
        condition_signatures=config.data.condition_signatures,
    )
    all_hashes = {
        **manifest_hashes,
        "normalization": normalization_hash,
        "target_coordinates": target_coordinates.sha256,
        "context_coordinates": context_coordinates.sha256,
        "radar_cache_manifest": radar_hashes["cache_manifest"],
        "era5_cache_manifest": era_hashes["cache_manifest"],
        "target_static": target_static_hash,
    }
    return FactoryArtifacts(
        rows=rows,
        context_regrid_operator=context_operator,
        target_static=target_static,
        static_conditions=static_conditions,
        geometry=geometry,
        era_normalization=era_normalization,
        condition_signatures=config.data.condition_signatures,
        context_static_fields=context_static_fields,
        artifact_hashes=all_hashes,
    )


def _open_radar_cache(path: Path, verify_hashes: bool) -> RadarCache:
    return RadarCache(path, verify_hashes=verify_hashes)


def _open_era_cache(path: Path, verify_hashes: bool) -> Era5MmapCache:
    return Era5MmapCache(path, verify_hashes=verify_hashes)


def _build_dataset(**kwargs: object) -> KCorrDiffDataset:
    return KCorrDiffDataset(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DatasetDependencies:
    """Pickle-safe cache/dataset constructors; injectable for protocol tests."""

    radar_cache: Callable[[Path, bool], object] = _open_radar_cache
    era5_cache: Callable[[Path, bool], object] = _open_era_cache
    dataset: Callable[..., object] = _build_dataset


@dataclass(frozen=True, slots=True)
class PlannedSample:
    """One exact plan slot and, only for a real slot, its decoded sample."""

    slot: DrawSlot
    sample: TrainingSample | object | None

    def __post_init__(self) -> None:
        if self.slot.is_padding != (self.sample is None):
            raise ValueError("padding slots must contain no sample")

    @property
    def is_padding(self) -> bool:
        return self.slot.is_padding


class LazyRankSlotDataset(Dataset[PlannedSample]):
    """Rank-local plan view whose mmap handles are opened in the using PID."""

    def __init__(
        self,
        *,
        rows: Sequence[DrawRow],
        slots: Sequence[DrawSlot],
        radar_cache_root: Path,
        era5_cache_root: Path,
        context_regrid_operator: ContextRegridOperator,
        static_conditions: StaticConditions,
        condition_signature: str | Sequence[str],
        radar_timezone: str,
        context_detail_mode: str,
        context_wet_threshold_mm_per_hour: float,
        verify_cache_hashes: bool,
        dependencies: DatasetDependencies = DatasetDependencies(),
    ) -> None:
        self.rows = tuple(rows)
        self.slots = tuple(slots)
        self.radar_cache_root = Path(radar_cache_root).resolve()
        self.era5_cache_root = Path(era5_cache_root).resolve()
        self.context_regrid_operator = context_regrid_operator
        self.static_conditions = static_conditions
        configured_signatures = (
            (condition_signature,)
            if isinstance(condition_signature, str)
            else tuple(condition_signature)
        )
        if not configured_signatures:
            raise ValueError("planned condition signature support cannot be empty")
        self.condition_signatures = tuple(
            sorted(
                parse_condition_signature(value).key
                for value in configured_signatures
            )
        )
        if len(self.condition_signatures) != len(set(self.condition_signatures)):
            raise ValueError("planned condition signatures must be unique")
        self.radar_timezone = radar_timezone
        self.context_detail_mode = context_detail_mode
        self.context_wet_threshold_mm_per_hour = float(
            context_wet_threshold_mm_per_hour
        )
        self.verify_cache_hashes = bool(verify_cache_hashes)
        self.dependencies = dependencies
        self._pid: int | None = None
        self._radar_cache: object | None = None
        self._era5_cache: object | None = None
        self._dataset: object | None = None
        for slot in self.slots:
            if slot.row_index is not None and not 0 <= slot.row_index < len(self.rows):
                raise IndexError("draw plan row_index is outside the draw manifest")

    def __len__(self) -> int:
        return len(self.slots)

    @property
    def cache_open(self) -> bool:
        return self._dataset is not None

    def _close(self) -> None:
        for cache in (self._radar_cache, self._era5_cache):
            close = getattr(cache, "close", None)
            if callable(close):
                close()
        self._pid = None
        self._radar_cache = None
        self._era5_cache = None
        self._dataset = None

    def _ensure_dataset(self) -> object:
        pid = os.getpid()
        if self._dataset is not None and self._pid == pid:
            return self._dataset
        if self._dataset is not None or self._pid is not None:
            self._close()
        radar = self.dependencies.radar_cache(
            self.radar_cache_root, self.verify_cache_hashes
        )
        try:
            era5 = self.dependencies.era5_cache(
                self.era5_cache_root, self.verify_cache_hashes
            )
            try:
                dataset = self.dependencies.dataset(
                    cache=radar,
                    rows=self.rows,
                    context_regrid_operator=self.context_regrid_operator,
                    era5_cache=era5,
                    static_conditions=self.static_conditions,
                    context_detail_mode=self.context_detail_mode,
                    context_wet_threshold_mm_per_hour=(
                        self.context_wet_threshold_mm_per_hour
                    ),
                    strict_era5_instantaneous=True,
                    radar_timezone=self.radar_timezone,
                )
            except BaseException:
                close = getattr(era5, "close", None)
                if callable(close):
                    close()
                raise
        except BaseException:
            close = getattr(radar, "close", None)
            if callable(close):
                close()
            raise
        self._pid = pid
        self._radar_cache = radar
        self._era5_cache = era5
        self._dataset = dataset
        return dataset

    def __getitem__(self, index: int) -> PlannedSample:
        slot = self.slots[index]
        if slot.is_padding:
            return PlannedSample(slot=slot, sample=None)
        assert slot.row_index is not None
        row = self.rows[slot.row_index]
        signature = parse_condition_signature(row.condition_signature).key
        if signature not in self.condition_signatures:
            raise ValueError("planned row signature changed after factory preflight")
        dataset = self._ensure_dataset()
        sample = dataset[slot.row_index]  # type: ignore[index]
        return PlannedSample(slot=slot, sample=sample)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.update(_pid=None, _radar_cache=None, _era5_cache=None, _dataset=None)
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter timing is nondeterministic
        try:
            self._close()
        except Exception:
            pass


def _pin_tree(value: _T) -> _T:
    if isinstance(value, Tensor):
        return value.pin_memory()  # type: ignore[return-value]
    custom_pin = getattr(value, "pin_memory", None)
    if callable(custom_pin):
        # Delegate once and return the result directly.  A custom implementation
        # owns its internal aliasing and must not be recursively re-entered.
        return custom_pin()  # type: ignore[no-any-return]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _pin_tree(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
        return replace(value, **updates)
    if isinstance(value, tuple):
        return tuple(_pin_tree(item) for item in value)  # type: ignore[return-value]
    if isinstance(value, list):
        return [_pin_tree(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, Mapping):
        return type(value)((key, _pin_tree(item)) for key, item in value.items())
    return value


@dataclass(frozen=True, slots=True)
class RankLocalBatch:
    """Compressed real batch plus explicit plan-width padding weights."""

    training: TrainingBatch | object | None
    slots: tuple[DrawSlot, ...]
    active_slot_indices: tuple[int, ...]
    loss_weights: Tensor

    def __post_init__(self) -> None:
        if self.loss_weights.dtype != torch.float32 or self.loss_weights.ndim != 1:
            raise TypeError("loss_weights must be a float32 vector")
        if self.loss_weights.shape[0] != len(self.slots):
            raise ValueError("loss_weights must retain the complete plan width")
        expected = tuple(index for index, slot in enumerate(self.slots) if not slot.is_padding)
        if self.active_slot_indices != expected:
            raise ValueError("active_slot_indices disagree with explicit padding")
        expected_weights = torch.tensor(
            [0.0 if slot.is_padding else 1.0 for slot in self.slots],
            dtype=torch.float32,
            device=self.loss_weights.device,
        )
        if not torch.equal(self.loss_weights, expected_weights):
            raise ValueError("padding must have exactly zero loss weight")
        if (self.training is None) != (not self.active_slot_indices):
            raise ValueError("only an all-padding batch may omit TrainingBatch")

    @property
    def is_zero_loss_sentinel(self) -> bool:
        return self.training is None

    @property
    def padding_count(self) -> int:
        return sum(slot.is_padding for slot in self.slots)

    def connected_zero_loss(self, parameters: Iterable[Tensor]) -> Tensor:
        """Return graph-connected scalar zero for an all-padding DDP step."""

        if not self.is_zero_loss_sentinel:
            raise ValueError("connected_zero_loss is only for an all-padding batch")
        result: Tensor | None = None
        for parameter in parameters:
            term = parameter.sum() * 0.0
            result = term if result is None else result + term
        if result is None:
            raise ValueError("at least one model parameter is required")
        return result

    def pin_memory(self) -> "RankLocalBatch":
        return RankLocalBatch(
            training=_pin_tree(self.training),
            slots=self.slots,
            active_slot_indices=self.active_slot_indices,
            loss_weights=self.loss_weights.pin_memory(),
        )


class RankSlotCollator:
    """Collate real slots only while retaining a zero-weight padding mask."""

    def __init__(self, collator: Callable[[Sequence[object]], object]) -> None:
        self.collator = collator

    def __call__(self, values: Sequence[PlannedSample]) -> RankLocalBatch:
        planned = tuple(values)
        if not planned:
            raise ValueError("cannot collate an empty rank-local batch")
        slots = tuple(value.slot for value in planned)
        active = tuple(index for index, value in enumerate(planned) if not value.is_padding)
        samples = tuple(planned[index].sample for index in active)
        training = self.collator(samples) if samples else None
        return RankLocalBatch(
            training=training,
            slots=slots,
            active_slot_indices=active,
            loss_weights=torch.tensor(
                [0.0 if slot.is_padding else 1.0 for slot in slots],
                dtype=torch.float32,
            ),
        )


def dataloader_options(
    *,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> dict[str, object]:
    """Return valid deterministic worker options, including workers=0."""

    if isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if isinstance(prefetch_factor, bool) or prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer")
    if not isinstance(pin_memory, bool) or not isinstance(persistent_workers, bool):
        raise TypeError("pin_memory and persistent_workers must be boolean")
    result: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        result["persistent_workers"] = persistent_workers
        result["prefetch_factor"] = prefetch_factor
    return result


def _loader_defaults(config: Stage2Config) -> tuple[int, int, bool, bool]:
    tuning = _mapping(config.raw.get("loader_tuning"), name="loader_tuning")
    data = _mapping(config.raw.get("data"), name="data config")
    workers = tuning.get("selected_num_workers")
    prefetch = tuning.get("selected_prefetch_factor")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("selected_num_workers must be an integer")
    if isinstance(prefetch, bool) or not isinstance(prefetch, int):
        raise TypeError("selected_prefetch_factor must be an integer")
    pin = data.get("pin_memory")
    persistent = data.get("persistent_workers")
    if not isinstance(pin, bool) or not isinstance(persistent, bool):
        raise TypeError("loader boolean config is invalid")
    if tuning.get("selected_batch_size_per_rank") != (
        config.optimization.per_rank_microbatch_size
    ):
        raise ValueError("loader tuning batch size disagrees with optimization config")
    return workers, prefetch, pin, persistent


class KCorrDiffDataFactory:
    """Validated artifacts plus deterministic plan/dataset/loader builders."""

    def __init__(
        self,
        config: Stage2Config,
        inputs: FactoryInputs,
        artifacts: FactoryArtifacts,
        *,
        dependencies: DatasetDependencies = DatasetDependencies(),
    ) -> None:
        self.config = config
        self.inputs = inputs
        self.artifacts = artifacts
        self.dependencies = dependencies

    def plan(
        self,
        *,
        role: str,
        fold_id: int | None,
        world_size: int,
    ) -> DistributedDrawPlan:
        self.config.validate_topology(world_size=world_size)
        return build_distributed_draw_plan(
            self.artifacts.rows,
            role=role,
            fold_id=fold_id,
            world_size=world_size,
            per_rank_microbatch_size=(
                self.config.optimization.per_rank_microbatch_size
            ),
            gradient_accumulation_steps=(
                self.config.optimization.gradient_accumulation_steps
            ),
        )

    def rank_dataset(
        self, plan: DistributedDrawPlan, *, rank: int
    ) -> LazyRankSlotDataset:
        if not 0 <= rank < plan.world_size:
            raise ValueError("rank is outside plan world_size")
        expected = build_distributed_draw_plan(
            self.artifacts.rows,
            role=plan.role,
            fold_id=plan.fold_id,
            world_size=plan.world_size,
            per_rank_microbatch_size=(
                self.config.optimization.per_rank_microbatch_size
            ),
            gradient_accumulation_steps=(
                self.config.optimization.gradient_accumulation_steps
            ),
        )
        if plan != expected:
            raise ValueError("distributed draw plan does not match factory rows/config")
        return LazyRankSlotDataset(
            rows=self.artifacts.rows,
            slots=plan.rank_slots[rank],
            radar_cache_root=self.inputs.radar_cache_root,
            era5_cache_root=self.inputs.era5_cache_root,
            context_regrid_operator=self.artifacts.context_regrid_operator,
            static_conditions=self.artifacts.static_conditions,
            condition_signature=self.artifacts.condition_signatures,
            radar_timezone=self.config.data.radar_timezone,
            context_detail_mode=self.config.data.context_detail_mode,
            context_wet_threshold_mm_per_hour=(
                self.config.data.context_wet_threshold_mm_per_hour
            ),
            verify_cache_hashes=self.inputs.verify_cache_hashes,
            dependencies=self.dependencies,
        )

    def rank_dataloader(
        self,
        plan: DistributedDrawPlan,
        *,
        rank: int,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
    ) -> DataLoader[RankLocalBatch]:
        dataset = self.rank_dataset(plan, rank=rank)
        default_workers, default_prefetch, pin, persistent = _loader_defaults(
            self.config
        )
        workers = default_workers if num_workers is None else num_workers
        prefetch = default_prefetch if prefetch_factor is None else prefetch_factor
        options = dataloader_options(
            num_workers=workers,
            prefetch_factor=prefetch,
            pin_memory=pin,
            persistent_workers=persistent,
        )
        sample_collator = TrainingBatchCollator(
            self.artifacts.geometry,
            era_normalization=self.artifacts.era_normalization,
            device="cpu",
        )
        return DataLoader(
            dataset,
            batch_size=plan.per_rank_microbatch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=RankSlotCollator(sample_collator),
            **options,
        )


def build_data_factory(
    config: Stage2Config,
    inputs: FactoryInputs,
    *,
    dependencies: DatasetDependencies = DatasetDependencies(),
) -> KCorrDiffDataFactory:
    """Build the production factory after complete content/geometry preflight."""

    artifacts = build_factory_artifacts(config, inputs)
    return KCorrDiffDataFactory(
        config, inputs, artifacts, dependencies=dependencies
    )


__all__ = [
    "CoordinateArtifact",
    "DatasetDependencies",
    "EraArtifactNormalization",
    "FactoryArtifacts",
    "FactoryInputs",
    "KCorrDiffDataFactory",
    "LazyRankSlotDataset",
    "PlannedSample",
    "RankLocalBatch",
    "RankSlotCollator",
    "build_data_factory",
    "build_factory_artifacts",
    "dataloader_options",
    "load_coordinate_artifact",
]

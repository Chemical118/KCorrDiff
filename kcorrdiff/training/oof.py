"""Bounded, sharded float32 OOF regression prediction artifacts."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterator, Mapping, Sequence
import zipfile

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kcorrdiff.data.sampling import enforce_byte_budget
from kcorrdiff.training.oof_remote import (
    DEFAULT_PVC_RESERVE_BYTES,
    PVCRemoteSpillController,
    RemoteShardReceipt,
    RemoteShardStore,
)


OOF_FORMAT_VERSION = "kcorrdiff.regression-oof.v1"
OOF_COMPRESSED_FORMAT_VERSION = "kcorrdiff.regression-oof.v2"
OOF_REMOTE_FORMAT_VERSION = "kcorrdiff.regression-oof.v3"
OOF_FIELDS = ("occurrence_probability", "mu_z")
OOF_DIRECT_BUILD_FORMAT = "kcorrdiff.regression-oof-direct-build.v1"
OOF_DIRECT_PARTITION_FORMAT = "kcorrdiff.regression-oof-direct-partition.v1"
OOF_COMPRESSED_BUILD_FORMAT = "kcorrdiff.regression-oof-compressed-build.v1"
OOF_COMPRESSED_PARTITION_FORMAT = "kcorrdiff.regression-oof-compressed-partition.v1"
OOF_COMPRESSED_ENCODING = "npz-deflate-byte-shuffle-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIRECT_FIXED_METADATA_RESERVE = 8 * 1024 * 1024
_DIRECT_PARTITION_METADATA_RESERVE = 64 * 1024
_LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _mapping_from_json(path: Path, *, name: str) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_build_layout(
    path: Path,
    expected: Mapping[str, object],
    *,
    keys: Sequence[str],
    name: str,
) -> None:
    """Validate only the tensor/storage layout needed to resume a build.

    Budgets, compression probes, hashes and launch provenance deliberately do
    not participate. They remain useful metadata without pinning a research
    run to the choices made when the build directory was initialized.
    """

    actual = _mapping_from_json(path, name=name)
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{name} scientific layout mismatch: {key}")


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_sha256(value: str, *, name: str) -> str:
    del name
    return value


def _identity_line(index: int, identity: "OOFIdentity") -> bytes:
    return _canonical_json_bytes({"row": index, **asdict(identity)})


def _receipt_line(index: int, digest: str) -> bytes:
    return _canonical_json_bytes({"row": index, "sha256": digest})


def _partition_storage_id(partition_id: str) -> str:
    """Map arbitrary experiment labels to a stable local storage name."""

    return hashlib.sha256(partition_id.encode("utf-8")).hexdigest()[:24]


def _npy_header_bytes(shape: tuple[int, ...]) -> int:
    """Return the exact NumPy-v1 header length used by ``open_memmap``."""

    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return len(stream.getvalue())


@dataclass(frozen=True, slots=True)
class OOFIdentity:
    sample_id: str
    t0_utc: str
    lead_hours: float
    condition_signature: str
    block_id: str
    fold_id: int
    regression_checkpoint_sha256: str

    @property
    def semantic_key(self) -> tuple[str, float, str]:
        return (self.t0_utc, self.lead_hours, self.condition_signature)

    def validate(self) -> None:
        if not all(
            (
                self.sample_id,
                self.t0_utc,
                self.condition_signature,
                self.block_id,
                self.regression_checkpoint_sha256,
            )
        ):
            raise ValueError("OOF identity fields must be non-empty")
        if isinstance(self.fold_id, bool) or self.fold_id < 0:
            raise ValueError("OOF fold_id must be a non-negative integer")
        if self.lead_hours not in tuple(index / 2 for index in range(1, 13)):
            raise ValueError("OOF lead must be one of the official 12 leads")


def _scientific_identity(identity: OOFIdentity) -> tuple[object, ...]:
    """Identity fields that affect fold/sample semantics, excluding hash metadata."""

    return (
        identity.sample_id,
        identity.t0_utc,
        identity.lead_hours,
        identity.condition_signature,
        identity.block_id,
        identity.fold_id,
    )


def _same_identity_sequence(
    left: Sequence[OOFIdentity], right: Sequence[OOFIdentity]
) -> bool:
    return tuple(map(_scientific_identity, left)) == tuple(
        map(_scientific_identity, right)
    )


@dataclass(frozen=True, slots=True)
class OOFEntry:
    identity: OOFIdentity
    occurrence_probability: NDArray[np.float32]
    mu_z: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class OOFPartitionSpec:
    """One immutable, rank-disjoint slice of the canonical OOF row space."""

    partition_id: str
    fold_id: int
    rank: int
    world_size: int
    row_indices: tuple[int, ...]
    regression_checkpoint_sha256: str

    def validate(
        self, *, items: int, identities: Sequence[OOFIdentity]
    ) -> None:
        if not isinstance(self.partition_id, str) or not self.partition_id:
            raise ValueError("OOF partition_id must be non-empty")
        if (
            isinstance(self.fold_id, bool)
            or self.fold_id < 0
            or isinstance(self.rank, bool)
            or self.rank < 0
            or isinstance(self.world_size, bool)
            or self.world_size <= 0
            or self.rank >= self.world_size
        ):
            raise ValueError("invalid OOF fold/rank partition identity")
        _valid_sha256(
            self.regression_checkpoint_sha256,
            name="OOF partition regression checkpoint SHA-256",
        )
        if tuple(sorted(self.row_indices)) != self.row_indices or len(
            set(self.row_indices)
        ) != len(self.row_indices):
            raise ValueError("OOF partition row indices must be strictly increasing")
        for index in self.row_indices:
            if isinstance(index, bool) or index < 0 or index >= items:
                raise ValueError("OOF partition row index is outside the artifact")
            identity = identities[index]
            if identity.fold_id != self.fold_id:
                raise ValueError(
                    "OOF partition row/checkpoint provenance does not match its identity"
                )

    @property
    def row_indices_sha256(self) -> str:
        digest = hashlib.sha256()
        for index in self.row_indices:
            digest.update(index.to_bytes(8, "little", signed=False))
        return digest.hexdigest()

    def record(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "fold_id": self.fold_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "items": len(self.row_indices),
            "row_indices_sha256": self.row_indices_sha256,
            "regression_checkpoint_sha256": self.regression_checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class OOFDirectStoragePlan:
    """Exact dense size and a conservative bound for all direct-build metadata."""

    items: int
    dense_payload_bytes: int
    shard_header_bytes: int
    index_bytes: int
    receipt_bytes: int
    metadata_budget_bytes: int

    @property
    def peak_persistent_bytes(self) -> int:
        # There is deliberately no second dense term here.  Shards are the
        # canonical final payload from the moment they are preallocated.
        return self.dense_payload_bytes + self.metadata_budget_bytes


def direct_oof_storage_plan(
    identities: Sequence[OOFIdentity],
    partitions: Sequence[OOFPartitionSpec],
    *,
    height: int = 256,
    width: int = 256,
    shard_items: int = 64,
) -> OOFDirectStoragePlan:
    """Calculate ``D + bounded metadata`` for a direct canonical build.

    ``D = N * 2 * H * W * 4`` is present exactly once.  The returned metadata
    budget includes exact NumPy headers, exact canonical index/receipt lengths,
    and fixed per-build/per-partition headroom for intents, manifests, hashes,
    and a torn final journal record.
    """

    if min(height, width, shard_items) <= 0:
        raise ValueError("OOF direct storage dimensions must be positive")
    items = len(identities)
    _validate_direct_universe(identities, partitions)
    dense = items * len(OOF_FIELDS) * height * width * np.dtype(np.float32).itemsize
    header_bytes = sum(
        _npy_header_bytes((min(shard_items, items - start), 2, height, width))
        for start in range(0, items, shard_items)
    )
    index_bytes = sum(
        len(_identity_line(index, identity))
        for index, identity in enumerate(identities)
    )
    receipt_bytes = sum(
        len(_receipt_line(index, "0" * 64)) for index in range(items)
    )
    budget = (
        header_bytes
        + index_bytes
        + receipt_bytes
        + _DIRECT_FIXED_METADATA_RESERVE
        + len(partitions) * _DIRECT_PARTITION_METADATA_RESERVE
    )
    return OOFDirectStoragePlan(
        items=items,
        dense_payload_bytes=dense,
        shard_header_bytes=header_bytes,
        index_bytes=index_bytes,
        receipt_bytes=receipt_bytes,
        metadata_budget_bytes=budget,
    )


def _validate_direct_universe(
    identities: Sequence[OOFIdentity], partitions: Sequence[OOFPartitionSpec]
) -> None:
    keys: set[tuple[str, float, str]] = set()
    for identity in identities:
        identity.validate()
        if identity.semantic_key in keys:
            raise ValueError(f"duplicate OOF semantic item: {identity.semantic_key}")
        keys.add(identity.semantic_key)
    partition_ids: set[str] = set()
    coverage: list[int] = []
    for partition in partitions:
        partition.validate(items=len(identities), identities=identities)
        if partition.partition_id in partition_ids:
            raise ValueError(f"duplicate OOF partition: {partition.partition_id}")
        partition_ids.add(partition.partition_id)
        coverage.extend(partition.row_indices)
    ordered = sorted(coverage)
    expected = list(range(len(identities)))
    if ordered != expected:
        observed: set[int] = set()
        duplicate: int | None = None
        for index in ordered:
            if index in observed:
                duplicate = index
                break
            observed.add(index)
        if duplicate is not None:
            raise ValueError(f"duplicate canonical OOF row partition: {duplicate}")
        missing = next(index for index in expected if index not in set(ordered))
        raise ValueError(f"missing canonical OOF row partition: {missing}")


class OOFDirectBuild:
    """Crash-safe direct writer for one canonical, sharded FP32 artifact.

    All ranks write disjoint rows into the *final* shard files.  Completion
    receipts are durable only after the corresponding shard data has been
    flushed and fsynced.  Rank zero later hashes and atomically renames this
    build directory; no dense partial-to-final copy is ever made.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        identities: Sequence[OOFIdentity],
        partitions: Sequence[OOFPartitionSpec],
        maximum_bytes: int,
        height: int = 256,
        width: int = 256,
        shard_items: int = 64,
        protocol_version: str = "v1.1.3b",
        draw_manifest_sha256: str,
        target_builder_version: str,
        config_sha256: str,
        launch_identity_sha256: str,
        initialize: bool = False,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.build_dir = self.output_dir.parent / f".{self.output_dir.name}.direct-build"
        self.identities = tuple(identities)
        self.partitions = tuple(partitions)
        self.height = int(height)
        self.width = int(width)
        self.shard_items = int(shard_items)
        self.maximum_bytes = int(maximum_bytes)
        self.protocol_version = protocol_version
        self.draw_manifest_sha256 = _valid_sha256(
            draw_manifest_sha256, name="OOF draw-manifest SHA-256"
        )
        self.target_builder_version = target_builder_version
        self.config_sha256 = _valid_sha256(
            config_sha256, name="OOF config SHA-256"
        )
        self.launch_identity_sha256 = _valid_sha256(
            launch_identity_sha256, name="OOF launch identity SHA-256"
        )
        if not target_builder_version or not protocol_version:
            raise ValueError("OOF protocol/target-builder versions must be non-empty")
        if min(self.height, self.width, self.shard_items) <= 0:
            raise ValueError("OOF direct build dimensions must be positive")
        _validate_direct_universe(self.identities, self.partitions)
        self.storage_plan = direct_oof_storage_plan(
            self.identities,
            self.partitions,
            height=self.height,
            width=self.width,
            shard_items=self.shard_items,
        )
        enforce_byte_budget(
            self.storage_plan.dense_payload_bytes, self.maximum_bytes
        )
        self._intent = self._intent_record()
        self._intent_payload = _canonical_json_bytes(self._intent)
        self.intent_sha256 = hashlib.sha256(self._intent_payload).hexdigest()
        if initialize:
            self._initialize_or_resume()
        else:
            self._validate_ready_build()

    def _shard_layout(self) -> tuple[dict[str, object], ...]:
        result: list[dict[str, object]] = []
        for shard, start in enumerate(
            range(0, len(self.identities), self.shard_items)
        ):
            count = min(self.shard_items, len(self.identities) - start)
            shape = (count, 2, self.height, self.width)
            result.append(
                {
                    "file": f"fields-{shard:05d}.npy",
                    "start": start,
                    "items": count,
                    "shape": list(shape),
                    "bytes": count * 2 * self.height * self.width * 4
                    + _npy_header_bytes(shape),
                }
            )
        return tuple(result)

    def _intent_record(self) -> dict[str, object]:
        identity_digest = hashlib.sha256()
        index_bytes = 0
        for index, identity in enumerate(self.identities):
            payload = _identity_line(index, identity)
            identity_digest.update(payload)
            index_bytes += len(payload)
        return {
            "format_version": OOF_DIRECT_BUILD_FORMAT,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "shape": [len(self.identities), 2, self.height, self.width],
            "items": len(self.identities),
            "required_dense_bytes": self.storage_plan.dense_payload_bytes,
            "maximum_dense_bytes": self.maximum_bytes,
            "metadata_budget_bytes": self.storage_plan.metadata_budget_bytes,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "target_builder_version": self.target_builder_version,
            "config_sha256": self.config_sha256,
            "launch_identity_sha256": self.launch_identity_sha256,
            "shard_items": self.shard_items,
            "index": {
                "file": "index.jsonl",
                "bytes": index_bytes,
                "sha256": identity_digest.hexdigest(),
            },
            "shards": list(self._shard_layout()),
            "partitions": [partition.record() for partition in self.partitions],
        }

    def _initialize_or_resume(self) -> None:
        if self.output_dir.exists():
            artifact = OOFArtifact(self.output_dir, verify_hashes=True)
            if not _same_identity_sequence(artifact.identities, self.identities):
                raise ValueError("existing OOF artifact identity universe mismatch")
            return
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=False, exist_ok=True)
        intent_path = self.build_dir / "build-intent.json"
        if intent_path.exists():
            _validate_build_layout(
                intent_path,
                self._intent,
                keys=("format_version", "dtype", "fields", "shape", "items", "shards"),
                name="existing OOF direct build",
            )
        else:
            _atomic_bytes(intent_path, self._intent_payload, replace=False)
        self._write_or_verify_index()
        self._create_or_verify_shards()
        (self.build_dir / ".partitions").mkdir(exist_ok=True)
        _fsync_directory(self.build_dir)

    def _validate_ready_build(self) -> None:
        if self.output_dir.exists():
            artifact = OOFArtifact(self.output_dir, verify_hashes=True)
            if not _same_identity_sequence(artifact.identities, self.identities):
                raise ValueError("existing OOF artifact identity universe mismatch")
            return
        if not self.build_dir.is_dir():
            raise FileNotFoundError("OOF direct build has not been initialized")
        _validate_build_layout(
            self.build_dir / "build-intent.json",
            self._intent,
            keys=("format_version", "dtype", "fields", "shape", "items", "shards"),
            name="OOF direct build",
        )
        self._write_or_verify_index(allow_write=False)
        self._create_or_verify_shards(allow_create=False)

    def _write_or_verify_index(self, *, allow_write: bool = True) -> None:
        path = self.build_dir / "index.jsonl"
        if path.exists():
            return
        if not allow_write:
            raise FileNotFoundError("OOF direct-build canonical index is missing")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".index.jsonl.", suffix=".tmp", dir=self.build_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                for index, identity in enumerate(self.identities):
                    stream.write(_identity_line(index, identity))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(self.build_dir)
        finally:
            temporary.unlink(missing_ok=True)

    def _create_or_verify_shards(self, *, allow_create: bool = True) -> None:
        for raw in self._shard_layout():
            path = self.build_dir / str(raw["file"])
            shape = tuple(int(value) for value in raw["shape"])  # type: ignore[arg-type]
            if not path.exists():
                if not allow_create:
                    raise FileNotFoundError(f"OOF direct shard is missing: {path.name}")
                values = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=shape
                )
                values.flush()
                mmap = getattr(values, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _fsync_directory(self.build_dir)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if values.dtype != np.float32 or tuple(values.shape) != shape:
                    raise ValueError(f"OOF direct shard schema mismatch: {path.name}")
            finally:
                mmap = getattr(values, "_mmap", None)
                if mmap is not None:
                    mmap.close()

    def partition(self, partition_id: str, *, commit_items: int | None = None) -> "OOFDirectPartitionWriter":
        if self.output_dir.exists():
            raise RuntimeError("sealed OOF artifacts are immutable")
        matches = [
            partition for partition in self.partitions
            if partition.partition_id == partition_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown OOF direct partition: {partition_id}")
        return OOFDirectPartitionWriter(
            self,
            matches[0],
            commit_items=self.shard_items if commit_items is None else commit_items,
        )

    def seal(self) -> str:
        """Verify exact coverage/hashes and atomically publish the build."""

        if self.output_dir.exists():
            artifact = OOFArtifact(self.output_dir, verify_hashes=True)
            if not _same_identity_sequence(artifact.identities, self.identities):
                raise ValueError("sealed OOF artifact identity mismatch")
            return _sha256(self.output_dir / "manifest.json")
        self._validate_ready_build()
        covered: list[int] = []
        for partition in self.partitions:
            writer = OOFDirectPartitionWriter(self, partition, commit_items=1)
            try:
                if not writer.complete:
                    raise ValueError(
                        f"OOF partition is incomplete: {partition.partition_id}"
                    )
                covered.extend(writer.completed_row_indices)
            finally:
                writer.abort()
        if sorted(covered) != list(range(len(self.identities))):
            raise ValueError("OOF direct receipts have missing/duplicate canonical rows")
        shards: list[dict[str, object]] = []
        for raw in self._shard_layout():
            path = self.build_dir / str(raw["file"])
            shards.append({**raw, "sha256": _sha256(path)})
        index = self._intent["index"]
        assert isinstance(index, Mapping)
        manifest = {
            "format_version": OOF_FORMAT_VERSION,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "shape": [len(self.identities), 2, self.height, self.width],
            "items": len(self.identities),
            "required_dense_bytes": self.storage_plan.dense_payload_bytes,
            "maximum_dense_bytes": self.maximum_bytes,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "target_builder_version": self.target_builder_version,
            "index": dict(index),
            "shards": shards,
        }
        payload = _canonical_json_bytes(manifest)
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_bytes(self.build_dir / "manifest.json", payload, replace=True)
        _atomic_bytes(
            self.build_dir / "manifest.sha256",
            f"{digest}  manifest.json\n".encode("ascii"),
            replace=True,
        )
        _fsync_directory(self.build_dir)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        os.replace(self.build_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        OOFArtifact(self.output_dir, verify_hashes=True)
        return digest


class OOFDirectPartitionWriter:
    """Append/replay one partition with data-before-receipt durability."""

    def __init__(
        self,
        build: OOFDirectBuild,
        spec: OOFPartitionSpec,
        *,
        commit_items: int,
    ) -> None:
        if commit_items <= 0:
            raise ValueError("OOF direct commit_items must be positive")
        self.build = build
        self.spec = spec
        self.commit_items = int(commit_items)
        self.partition_dir = build.build_dir / ".partitions"
        storage_id = _partition_storage_id(spec.partition_id)
        self.intent_path = self.partition_dir / f"{storage_id}.json"
        self.journal_path = self.partition_dir / f"{storage_id}.receipts.jsonl"
        self._intent = {
            "format_version": OOF_DIRECT_PARTITION_FORMAT,
            "build_intent_sha256": build.intent_sha256,
            **spec.record(),
        }
        self._intent_payload = _canonical_json_bytes(self._intent)
        if self.intent_path.exists():
            _validate_build_layout(
                self.intent_path,
                self._intent,
                keys=(
                    "format_version",
                    "partition_id",
                    "fold_id",
                    "items",
                    "row_indices_sha256",
                ),
                name="OOF partition",
            )
        else:
            _atomic_bytes(self.intent_path, self._intent_payload, replace=False)
        self._receipts = self._read_receipts(repair_torn_tail=True, verify_dense=True)
        self._seen = 0
        self._pending: list[tuple[int, str]] = []
        self._maps: dict[int, NDArray[np.float32]] = {}

    @property
    def completed_row_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self._receipts)

    @property
    def complete(self) -> bool:
        return len(self._receipts) == len(self.spec.row_indices) and not self._pending

    def _read_receipts(
        self, *, repair_torn_tail: bool, verify_dense: bool
    ) -> list[tuple[int, str]]:
        if not self.journal_path.exists():
            return []
        payload = self.journal_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            if not repair_torn_tail:
                raise ValueError("OOF partition journal has a torn final record")
            end = payload.rfind(b"\n") + 1
            with self.journal_path.open("r+b") as stream:
                stream.truncate(end)
                stream.flush()
                os.fsync(stream.fileno())
            payload = payload[:end]
        receipts: list[tuple[int, str]] = []
        for position, line in enumerate(payload.splitlines(keepends=True)):
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("OOF partition journal record is invalid") from error
            if not isinstance(record, Mapping) or "row" not in record:
                raise ValueError("OOF partition journal record is invalid")
            if position >= len(self.spec.row_indices):
                raise ValueError("OOF partition journal contains a duplicate/extra row")
            expected_index = self.spec.row_indices[position]
            if record["row"] != expected_index:
                raise ValueError("OOF partition journal row sequence mismatch")
            digest = str(record.get("sha256", ""))
            receipts.append((expected_index, digest))
        return receipts

    def _location(self, canonical_index: int) -> tuple[int, int, Path]:
        shard = canonical_index // self.build.shard_items
        local = canonical_index % self.build.shard_items
        return (
            shard,
            local,
            self.build.build_dir / f"fields-{shard:05d}.npy",
        )

    def _dense_row_sha256(self, canonical_index: int) -> str:
        _, local, path = self._location(canonical_index)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            row = np.ascontiguousarray(values[local], dtype=np.float32)
            return hashlib.sha256(row.tobytes(order="C")).hexdigest()
        finally:
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def _field(self, value: ArrayLike, *, name: str) -> NDArray[np.float32]:
        result = np.asarray(value)
        if result.shape == (1, self.build.height, self.build.width):
            result = result[0]
        expected = (self.build.height, self.build.width)
        if result.shape != expected:
            raise ValueError(f"OOF {name} shape {result.shape}, expected {expected}")
        if result.dtype != np.float32:
            raise TypeError(f"OOF {name} must be stored as float32")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"OOF {name} contains a non-finite value")
        return np.asarray(result, dtype=np.float32, order="C")

    def append(
        self,
        canonical_index: int,
        identity: OOFIdentity,
        occurrence_probability: ArrayLike,
        mu_z: ArrayLike,
    ) -> None:
        if self._seen >= len(self.spec.row_indices):
            raise ValueError("OOF partition received more than its declared rows")
        expected_index = self.spec.row_indices[self._seen]
        if canonical_index != expected_index:
            raise ValueError(
                f"OOF partition row order mismatch: {canonical_index} != {expected_index}"
            )
        if _scientific_identity(identity) != _scientific_identity(
            self.build.identities[canonical_index]
        ):
            raise ValueError("OOF partition scientific identity mismatch")
        probability = self._field(
            occurrence_probability, name="occurrence_probability"
        )
        mean = self._field(mu_z, name="mu_z")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("OOF occurrence probability must lie in [0, 1]")
        if np.any(mean < 0.0):
            raise ValueError("OOF transformed mean cannot be negative")
        values = np.stack((probability, mean), axis=0).astype(
            np.float32, copy=False
        )
        digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        if self._seen < len(self._receipts):
            receipt_index, _ = self._receipts[self._seen]
            _, local, path = self._location(canonical_index)
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            matches = np.array_equal(existing[local], values)
            mmap = getattr(existing, "_mmap", None)
            if mmap is not None:
                mmap.close()
            if receipt_index != canonical_index or not matches:
                raise ValueError(
                    "replayed OOF row differs from its durable crash receipt"
                )
        else:
            shard, local, path = self._location(canonical_index)
            mapped = self._maps.get(shard)
            if mapped is None:
                mapped = np.load(path, mmap_mode="r+", allow_pickle=False)
                self._maps[shard] = mapped
            mapped[local] = values
            self._pending.append((canonical_index, digest))
            if len(self._pending) >= self.commit_items:
                self.commit()
        self._seen += 1

    def commit(self) -> None:
        if not self._pending:
            return
        paths: list[Path] = []
        for shard, values in self._maps.items():
            values.flush()  # type: ignore[union-attr]
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            paths.append(self.build.build_dir / f"fields-{shard:05d}.npy")
        self._maps.clear()
        # Receipt durability is strictly ordered after durable dense bytes.
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        with self.journal_path.open("ab") as stream:
            for index, digest in self._pending:
                stream.write(_receipt_line(index, digest))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.partition_dir)
        self._receipts.extend(self._pending)
        self._pending.clear()

    def finish(self, *, manifest_path: Path | None = None) -> str:
        self.commit()
        if self._seen != len(self.spec.row_indices):
            raise ValueError(
                f"OOF partition item count {self._seen} != {len(self.spec.row_indices)}"
            )
        verified = self._read_receipts(repair_torn_tail=False, verify_dense=True)
        if len(verified) != len(self.spec.row_indices):
            raise ValueError("OOF partition journal is incomplete")
        journal_sha = _sha256(self.journal_path) if self.journal_path.exists() else hashlib.sha256(b"").hexdigest()
        record = {
            **self._intent,
            "journal": {
                "file": self.journal_path.name,
                "bytes": self.journal_path.stat().st_size if self.journal_path.exists() else 0,
                "sha256": journal_sha,
            },
            "complete": True,
        }
        payload = _canonical_json_bytes(record)
        digest = hashlib.sha256(payload).hexdigest()
        destination = (
            self.partition_dir
            / f"{_partition_storage_id(self.spec.partition_id)}.manifest.json"
            if manifest_path is None
            else manifest_path.resolve()
        )
        _atomic_bytes(destination, payload, replace=True)
        return digest

    def abort(self) -> None:
        """Drop unreceipted mappings; resume will deterministically rewrite them."""

        for values in self._maps.values():
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._maps.clear()
        self._pending.clear()

    def __enter__(self) -> "OOFDirectPartitionWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()


@dataclass(frozen=True, slots=True)
class OOFCompressedStoragePlan:
    """PVC reservation for streamed, lossless FP32 shard compression."""

    items: int
    logical_dense_bytes: int
    maximum_compressed_bytes: int
    maximum_active_shard_bytes: int
    maximum_archive_temporary_bytes: int
    concurrent_writers: int
    metadata_budget_bytes: int

    @property
    def peak_persistent_bytes(self) -> int:
        return (
            self.maximum_compressed_bytes
            + self.concurrent_writers
            * (
                self.maximum_active_shard_bytes
                + self.maximum_archive_temporary_bytes
            )
            + self.metadata_budget_bytes
        )


def _deflate_archive_bound(uncompressed_bytes: int) -> int:
    # zlib's compressBound formula plus ample ZIP64/local/central-directory
    # and deterministic .npy-header allowance.  This is a reservation bound,
    # not an assumed compression ratio.
    return (
        uncompressed_bytes
        + (uncompressed_bytes >> 12)
        + (uncompressed_bytes >> 14)
        + (uncompressed_bytes >> 25)
        + 13
        + 64 * 1024
    )


def compressed_oof_storage_plan(
    identities: Sequence[OOFIdentity],
    partitions: Sequence[OOFPartitionSpec],
    *,
    maximum_compressed_bytes: int,
    concurrent_writers: int,
    height: int = 256,
    width: int = 256,
    shard_items: int = 32,
) -> OOFCompressedStoragePlan:
    """Return an advisory PVC estimate without assuming a compression ratio."""

    if maximum_compressed_bytes <= 0 or concurrent_writers <= 0:
        raise ValueError("compressed OOF cap/writer count must be positive")
    if min(height, width, shard_items) <= 0:
        raise ValueError("compressed OOF dimensions must be positive")
    _validate_direct_universe(identities, partitions)
    items = len(identities)
    logical = items * 2 * height * width * 4
    active_items = min(items, shard_items)
    active_shape = (active_items, 2, height, width)
    active_dense = active_items * 2 * height * width * 4
    active_file = active_dense + (_npy_header_bytes(active_shape) if items else 0)
    archive_bound = _deflate_archive_bound(active_file) if items else 0
    index_bytes = sum(
        len(_identity_line(index, identity))
        for index, identity in enumerate(identities)
    )
    # Active receipts are deleted after each shard is sealed, so only one
    # shard of receipt lines per writer is live.  Final shard/partition
    # sidecars are bounded separately by fixed per-record headroom.
    active_receipts = sum(
        len(_receipt_line(index, "0" * 64))
        for index in range(min(items, shard_items))
    )
    shard_count = sum(
        math.ceil(len(partition.row_indices) / shard_items)
        for partition in partitions
    )
    metadata = (
        index_bytes
        + concurrent_writers * active_receipts
        + _DIRECT_FIXED_METADATA_RESERVE
        + len(partitions) * _DIRECT_PARTITION_METADATA_RESERVE
        + shard_count * 16 * 1024
    )
    return OOFCompressedStoragePlan(
        items=items,
        logical_dense_bytes=logical,
        maximum_compressed_bytes=int(maximum_compressed_bytes),
        maximum_active_shard_bytes=active_file,
        maximum_archive_temporary_bytes=archive_bound,
        concurrent_writers=int(concurrent_writers),
        metadata_budget_bytes=metadata,
    )


def _write_deterministic_npz(path: Path, values: NDArray[np.float32]) -> None:
    """Write deterministic byte-shuffled, ZIP_DEFLATED FP32 bit patterns."""

    logical = np.asarray(values, dtype=np.dtype("<f4"), order="C")
    byte_view = logical.view(np.uint8).reshape((*logical.shape, 4))
    # Group byte significance planes before DEFLATE.  Smooth FP32 fields share
    # exponent/high-mantissa bytes spatially, so this is materially more
    # compressible while remaining an exactly reversible permutation.
    shuffled = np.ascontiguousarray(np.moveaxis(byte_view, -1, 0))

    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            info = zipfile.ZipInfo("fields.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, shuffled, allow_pickle=False)
        raw.flush()
        os.fsync(raw.fileno())


class OOFCompressedBuild:
    """Lossless FP32 OOF publication with only bounded uncompressed staging.

    Artifact rows are ordered by immutable partition, then by that partition's
    original draw order.  A partition owns its shard files exclusively, so
    ranks never write the same path.  Completed shards are immediately
    compressed and their staging files removed.  The final rename is metadata
    only and therefore never creates a second dense payload.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        identities: Sequence[OOFIdentity],
        partitions: Sequence[OOFPartitionSpec],
        maximum_compressed_bytes: int,
        maximum_compression_ratio: float,
        compression_probe_bytes: int,
        concurrent_writers: int,
        height: int = 256,
        width: int = 256,
        shard_items: int = 32,
        protocol_version: str = "v1.1.3b",
        draw_manifest_sha256: str,
        target_builder_version: str,
        config_sha256: str,
        launch_identity_sha256: str,
        remote_store: RemoteShardStore | None = None,
        local_reserve_bytes: int = DEFAULT_PVC_RESERVE_BYTES,
        initialize: bool = False,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.build_dir = self.output_dir.parent / (
            f".{self.output_dir.name}.compressed-build"
        )
        self.identities = tuple(identities)
        self.partitions = tuple(partitions)
        self.maximum_compressed_bytes = int(maximum_compressed_bytes)
        self.maximum_compression_ratio = float(maximum_compression_ratio)
        self.compression_probe_bytes = int(compression_probe_bytes)
        self.concurrent_writers = int(concurrent_writers)
        self.height = int(height)
        self.width = int(width)
        self.shard_items = int(shard_items)
        self.protocol_version = protocol_version
        self.draw_manifest_sha256 = _valid_sha256(
            draw_manifest_sha256, name="OOF draw-manifest SHA-256"
        )
        self.target_builder_version = target_builder_version
        self.config_sha256 = _valid_sha256(config_sha256, name="OOF config SHA-256")
        self.launch_identity_sha256 = _valid_sha256(
            launch_identity_sha256, name="OOF launch identity SHA-256"
        )
        self.remote_store = remote_store
        self.local_reserve_bytes = int(local_reserve_bytes)
        if self.local_reserve_bytes <= 0:
            raise ValueError("OOF local PVC reserve must be positive")
        self.artifact_format_version = (
            OOF_REMOTE_FORMAT_VERSION
            if self.remote_store is not None
            else OOF_COMPRESSED_FORMAT_VERSION
        )
        if (
            not math.isfinite(self.maximum_compression_ratio)
            or self.maximum_compression_ratio <= 0.0
            or self.compression_probe_bytes <= 0
        ):
            raise ValueError("OOF compression ratio/probe must be positive")
        if min(self.height, self.width, self.shard_items) <= 0:
            raise ValueError("OOF compressed build dimensions must be positive")
        _validate_direct_universe(self.identities, self.partitions)
        cursor = 0
        for partition in self.partitions:
            expected = tuple(range(cursor, cursor + len(partition.row_indices)))
            if partition.row_indices != expected:
                raise ValueError(
                    "compressed OOF partitions must be contiguous in artifact order"
                )
            cursor += len(partition.row_indices)
        self.storage_plan = compressed_oof_storage_plan(
            self.identities,
            self.partitions,
            maximum_compressed_bytes=self.maximum_compressed_bytes,
            concurrent_writers=self.concurrent_writers,
            height=self.height,
            width=self.width,
            shard_items=self.shard_items,
        )
        self._layout = self._shard_layout()
        self._layout_by_file = {str(item["file"]): item for item in self._layout}
        self._intent = self._intent_record()
        self._intent_payload = _canonical_json_bytes(self._intent)
        self.intent_sha256 = hashlib.sha256(self._intent_payload).hexdigest()
        if initialize:
            self._initialize_or_resume()
        else:
            self._validate_ready_build()
        self.remote_spill = (
            PVCRemoteSpillController(
                self.build_dir,
                build_intent_sha256=self.intent_sha256,
                store=self.remote_store,
                local_reserve_bytes=self.local_reserve_bytes,
            )
            if self.remote_store is not None and self.build_dir.is_dir()
            else None
        )

    def _shard_layout(self) -> tuple[dict[str, object], ...]:
        result: list[dict[str, object]] = []
        shard_index = 0
        for partition in self.partitions:
            rows = partition.row_indices
            for offset in range(0, len(rows), self.shard_items):
                selected = rows[offset : offset + self.shard_items]
                count = len(selected)
                shape = (count, 2, self.height, self.width)
                result.append(
                    {
                        "file": f"fields-{shard_index:05d}.npz",
                        "partition_id": partition.partition_id,
                        "start": selected[0],
                        "items": count,
                        "shape": list(shape),
                        "logical_bytes": count * 2 * self.height * self.width * 4,
                    }
                )
                shard_index += 1
        return tuple(result)

    def _intent_record(self) -> dict[str, object]:
        identity_digest = hashlib.sha256()
        index_bytes = 0
        for index, identity in enumerate(self.identities):
            payload = _identity_line(index, identity)
            identity_digest.update(payload)
            index_bytes += len(payload)
        return {
            "format_version": OOF_COMPRESSED_BUILD_FORMAT,
            "artifact_format_version": self.artifact_format_version,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "encoding": OOF_COMPRESSED_ENCODING,
            "shape": [len(self.identities), 2, self.height, self.width],
            "items": len(self.identities),
            "logical_dense_bytes": self.storage_plan.logical_dense_bytes,
            "maximum_compressed_bytes": self.maximum_compressed_bytes,
            "maximum_compression_ratio": self.maximum_compression_ratio,
            "compression_probe_bytes": self.compression_probe_bytes,
            "metadata_budget_bytes": self.storage_plan.metadata_budget_bytes,
            "concurrent_writers": self.concurrent_writers,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "target_builder_version": self.target_builder_version,
            "config_sha256": self.config_sha256,
            "launch_identity_sha256": self.launch_identity_sha256,
            "shard_items": self.shard_items,
            "index": {
                "file": "index.jsonl",
                "bytes": index_bytes,
                "sha256": identity_digest.hexdigest(),
            },
            "shards": list(self._layout),
            "partitions": [partition.record() for partition in self.partitions],
            **(
                {
                    "remote_store": self.remote_store.identity.record(),
                    "remote_store_sha256": self.remote_store.identity.semantic_sha256,
                    "local_reserve_bytes": self.local_reserve_bytes,
                }
                if self.remote_store is not None
                else {}
            ),
        }

    def _initialize_or_resume(self) -> None:
        if self.output_dir.exists():
            self._validate_sealed_artifact()
            return
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        intent = self.build_dir / "build-intent.json"
        if intent.exists():
            _validate_build_layout(
                intent,
                self._intent,
                keys=(
                    "format_version",
                    "artifact_format_version",
                    "dtype",
                    "fields",
                    "encoding",
                    "shape",
                    "items",
                    "shards",
                ),
                name="compressed OOF build",
            )
        else:
            _atomic_bytes(intent, self._intent_payload, replace=False)
        self._write_or_verify_index()
        (self.build_dir / ".partitions").mkdir(exist_ok=True)
        (self.build_dir / ".compression.lock").touch(exist_ok=True)
        _fsync_directory(self.build_dir)

    def _validate_ready_build(self) -> None:
        if self.output_dir.exists():
            self._validate_sealed_artifact()
            return
        if not self.build_dir.is_dir():
            raise FileNotFoundError("compressed OOF build has not been initialized")
        _validate_build_layout(
            self.build_dir / "build-intent.json",
            self._intent,
            keys=(
                "format_version",
                "artifact_format_version",
                "dtype",
                "fields",
                "encoding",
                "shape",
                "items",
                "shards",
            ),
            name="compressed OOF build",
        )
        self._write_or_verify_index(allow_write=False)

    def _validate_sealed_artifact(self) -> OOFArtifact:
        artifact = OOFArtifact(
            self.output_dir,
            verify_hashes=True,
            remote_store=self.remote_store,
        )
        manifest = artifact.manifest
        expected = {
            "format_version": self.artifact_format_version,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "encoding": OOF_COMPRESSED_ENCODING,
            "shape": [len(self.identities), 2, self.height, self.width],
            "items": len(self.identities),
            "target_builder_version": self.target_builder_version,
        }
        if not _same_identity_sequence(artifact.identities, self.identities):
            raise ValueError("existing compressed OOF identity universe mismatch")
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"existing compressed OOF provenance mismatch: {name}"
                )
        return artifact

    def _write_or_verify_index(self, *, allow_write: bool = True) -> None:
        path = self.build_dir / "index.jsonl"
        if path.exists():
            return
        if not allow_write:
            raise FileNotFoundError("compressed OOF canonical index is missing")
        descriptor, name = tempfile.mkstemp(
            prefix=".index.jsonl.", suffix=".tmp", dir=self.build_dir
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                for index, identity in enumerate(self.identities):
                    stream.write(_identity_line(index, identity))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(self.build_dir)
        finally:
            temporary.unlink(missing_ok=True)

    def partition(
        self, partition_id: str, *, commit_items: int | None = None
    ) -> "OOFCompressedPartitionWriter":
        if self.output_dir.exists():
            raise RuntimeError("sealed compressed OOF artifacts are immutable")
        matches = [item for item in self.partitions if item.partition_id == partition_id]
        if len(matches) != 1:
            raise KeyError(f"unknown compressed OOF partition: {partition_id}")
        return OOFCompressedPartitionWriter(
            self,
            matches[0],
            commit_items=self.shard_items if commit_items is None else commit_items,
        )

    def partition_layout(self, partition_id: str) -> tuple[dict[str, object], ...]:
        return tuple(
            item for item in self._layout if item["partition_id"] == partition_id
        )

    def _shard_sidecar_record(
        self, layout: Mapping[str, object]
    ) -> dict[str, object]:
        filename = str(layout["file"])
        sidecar = self.build_dir / ".partitions" / f"{filename}.json"
        if not sidecar.is_file():
            raise ValueError(f"compressed OOF shard sidecar is missing: {filename}")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError(f"compressed OOF shard metadata is invalid: {filename}")
        expected = {
            "file": filename,
            "start": layout["start"],
            "items": layout["items"],
            "shape": layout["shape"],
            "logical_bytes": layout["logical_bytes"],
            "encoding": OOF_COMPRESSED_ENCODING,
        }
        for name, value in expected.items():
            if record.get(name) != value:
                raise ValueError(f"compressed OOF shard metadata mismatch: {filename}/{name}")
        return dict(record)

    def _verified_shard_record(
        self,
        layout: Mapping[str, object],
        *,
        verify_remote: bool = False,
    ) -> dict[str, object]:
        filename = str(layout["file"])
        record = self._shard_sidecar_record(layout)
        archive = self.build_dir / filename
        receipt: RemoteShardReceipt | None = None
        values: NDArray[np.float32] | None = None
        if archive.is_file():
            values = _load_npz_fields(archive)
            if tuple(values.shape) != tuple(layout["shape"]) or values.dtype != np.float32:
                raise ValueError(f"compressed OOF shard logical schema mismatch: {filename}")
        else:
            if self.remote_spill is None:
                raise FileNotFoundError(f"compressed OOF shard is missing: {filename}")
            receipt = self.remote_spill.receipt_for(
                filename, verify_remote=verify_remote
            )
            if receipt is None:
                raise FileNotFoundError(
                    f"compressed OOF shard has no local or remote copy: {filename}"
                )
            if receipt.bytes != record.get("bytes"):
                raise ValueError(f"remote OOF shard byte count mismatch: {filename}")
        return {
            **record,
            "storage": "remote_rsync" if receipt is not None else "local",
            **({"remote": receipt.record()} if receipt is not None else {}),
        }

    def seal(self) -> str:
        if self.output_dir.exists():
            self._validate_sealed_artifact()
            return _sha256(self.output_dir / "manifest.json")
        self._validate_ready_build()
        shard_records = [
            self._verified_shard_record(item, verify_remote=True)
            for item in self._layout
        ]
        actual = sum(int(record["bytes"]) for record in shard_records)
        logical = self.storage_plan.logical_dense_bytes
        ratio = actual / logical if logical else 0.0
        if actual > self.maximum_compressed_bytes:
            _LOGGER.warning(
                "compressed OOF artifact uses %d bytes, above the configured "
                "reference budget of %d bytes; continuing",
                actual,
                self.maximum_compressed_bytes,
            )
        if logical >= self.compression_probe_bytes and ratio > self.maximum_compression_ratio:
            _LOGGER.warning(
                "lossless OOF compression ratio %.6f exceeds the configured "
                "reference %.6f; continuing",
                ratio,
                self.maximum_compression_ratio,
            )
        index = self._intent["index"]
        assert isinstance(index, Mapping)
        manifest = {
            "format_version": self.artifact_format_version,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "encoding": OOF_COMPRESSED_ENCODING,
            "shape": [len(self.identities), 2, self.height, self.width],
            "items": len(self.identities),
            "logical_dense_bytes": logical,
            "maximum_compressed_bytes": self.maximum_compressed_bytes,
            "actual_compressed_bytes": actual,
            "compression_ratio": ratio,
            "maximum_compression_ratio": self.maximum_compression_ratio,
            "compression_probe_bytes": self.compression_probe_bytes,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "target_builder_version": self.target_builder_version,
            "config_sha256": self.config_sha256,
            "launch_identity_sha256": self.launch_identity_sha256,
            **(
                {
                    "remote_store": self.remote_store.identity.record(),
                    "remote_store_sha256": self.remote_store.identity.semantic_sha256,
                    "local_reserve_bytes": self.local_reserve_bytes,
                }
                if self.remote_store is not None
                else {}
            ),
            "index": dict(index),
            "shards": [
                {
                    key: record[key]
                    for key in (
                        "file",
                        "start",
                        "items",
                        "shape",
                        "logical_bytes",
                        "bytes",
                        "sha256",
                        "array_sha256",
                        "encoding",
                    )
                }
                | (
                    {
                        "storage": record["storage"],
                        **(
                            {"remote": record["remote"]}
                            if "remote" in record
                            else {}
                        ),
                    }
                    if self.remote_store is not None
                    else {}
                )
                for record in shard_records
            ],
        }
        payload = _canonical_json_bytes(manifest)
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_bytes(self.build_dir / "manifest.json", payload, replace=True)
        _atomic_bytes(
            self.build_dir / "manifest.sha256",
            f"{digest}  manifest.json\n".encode("ascii"),
            replace=True,
        )
        _fsync_directory(self.build_dir)
        os.replace(self.build_dir, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        OOFArtifact(
            self.output_dir,
            verify_hashes=True,
            remote_store=self.remote_store,
        )
        return digest


def load_compressed_oof_fields(path: Path) -> NDArray[np.float32]:
    """Losslessly invert the v2 byte shuffle into a C-contiguous FP32 shard."""

    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"compressed OOF shard is not NPZ: {path.name}")
    try:
        if loaded.files != ["fields"]:
            raise ValueError(f"compressed OOF shard member schema mismatch: {path.name}")
        shuffled = loaded["fields"]
        if shuffled.dtype != np.uint8 or shuffled.ndim != 5 or shuffled.shape[0] != 4:
            raise ValueError(f"compressed OOF byte-shuffle schema mismatch: {path.name}")
        byte_view = np.ascontiguousarray(np.moveaxis(shuffled, 0, -1))
        logical_shape = shuffled.shape[1:]
        restored = byte_view.view(np.dtype("<f4")).reshape(logical_shape)
        return np.array(restored, dtype=np.float32, copy=True, order="C")
    finally:
        loaded.close()


# Internal alias retained for the artifact reader and older focused tests.
_load_npz_fields = load_compressed_oof_fields


class OOFCompressedPartitionWriter:
    """One rank-exclusive streamed compressed partition."""

    def __init__(
        self,
        build: OOFCompressedBuild,
        spec: OOFPartitionSpec,
        *,
        commit_items: int,
    ) -> None:
        if commit_items <= 0:
            raise ValueError("compressed OOF commit_items must be positive")
        self.build = build
        self.spec = spec
        self.commit_items = int(commit_items)
        self.partition_dir = build.build_dir / ".partitions"
        self.layout = build.partition_layout(spec.partition_id)
        self._seen = 0
        self._active_layout: Mapping[str, object] | None = None
        self._active_values: NDArray[np.float32] | None = None
        self._active_receipts: list[tuple[int, str]] = []
        self._pending: list[tuple[int, str]] = []
        self._sealed_row_hashes: dict[int, str] = {}
        self._sealed_cache_file: str | None = None
        self._sealed_cache_values: NDArray[np.float32] | None = None
        self._postprocess_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="oof-shard-postprocess",
        )
        self._postprocess_futures: list[Future[None]] = []
        self._maximum_pending_postprocess = 50
        self._postprocess_closed = False
        self._load_sealed_prefix()
        self._recover_ready_shards()

    def _sidecar_path(self, layout: Mapping[str, object]) -> Path:
        return self.partition_dir / f"{layout['file']}.json"

    def _stage_path(self, layout: Mapping[str, object]) -> Path:
        return self.partition_dir / f"{layout['file']}.active.npy"

    def _ready_path(self, layout: Mapping[str, object]) -> Path:
        return self.partition_dir / f"{layout['file']}.ready.npy"

    def _journal_path(self, layout: Mapping[str, object]) -> Path:
        return self.partition_dir / f"{layout['file']}.receipts.jsonl"

    def _load_sealed_prefix(self) -> None:
        saw_gap = False
        for layout in self.layout:
            sidecar = self._sidecar_path(layout)
            if not sidecar.exists():
                saw_gap = True
                continue
            if saw_gap:
                raise ValueError("compressed OOF partition has a non-prefix sealed shard")
            record = self.build._shard_sidecar_record(layout)
            filename = str(layout["file"])
            archive = self.build.build_dir / filename
            expected_bytes = record.get("bytes")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes <= 0
            ):
                raise ValueError("compressed OOF shard byte metadata is invalid")
            if archive.is_file():
                if archive.stat().st_size != expected_bytes:
                    raise ValueError("compressed OOF shard byte count changed")
            else:
                receipt = (
                    self.build.remote_spill.receipt_for(
                        filename, verify_remote=False
                    )
                    if self.build.remote_spill is not None
                    else None
                )
                if receipt is None or receipt.bytes != expected_bytes:
                    raise FileNotFoundError(
                        f"compressed OOF shard has no local or remote copy: {filename}"
                    )
            for offset in range(int(layout["items"])):
                self._sealed_row_hashes[int(layout["start"]) + offset] = ""
            self._stage_path(layout).unlink(missing_ok=True)
            self._ready_path(layout).unlink(missing_ok=True)
            self._journal_path(layout).unlink(missing_ok=True)

    def _recover_ready_shards(self) -> None:
        """Finish durable raw shards left by an interrupted queue worker."""

        for layout in self.layout:
            if self._sidecar_path(layout).is_file():
                continue
            ready = self._ready_path(layout)
            if not ready.is_file():
                continue
            receipts = self._read_receipts_for(layout)
            if len(receipts) != int(layout["items"]):
                raise ValueError("queued compressed OOF shard has incomplete receipts")
            self._seal_ready_shard(
                layout,
                ready,
                tuple(digest for _, digest in receipts),
            )

    @property
    def complete(self) -> bool:
        return len(self._sealed_row_hashes) == len(self.spec.row_indices)

    @property
    def sealed_items(self) -> int:
        """Number of leading partition rows already stored in sealed shards."""

        return len(self._sealed_row_hashes)

    def sealed_fields(
        self, artifact_index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Read a previously sealed row without recomputing model inference."""

        if artifact_index not in self._sealed_row_hashes:
            raise KeyError(f"row {artifact_index} is not sealed")
        layout = self._layout_for_row(artifact_index)
        filename = str(layout["file"])
        if self._sealed_cache_file != filename:
            archive = self.build.build_dir / filename
            temporary_directory: tempfile.TemporaryDirectory[str] | None = None
            try:
                if not archive.is_file():
                    if self.build.remote_spill is None or self.build.remote_store is None:
                        raise FileNotFoundError(f"compressed OOF shard is missing: {filename}")
                    receipt = self.build.remote_spill.receipt_for(
                        filename, verify_remote=False
                    )
                    if receipt is None:
                        raise FileNotFoundError(
                            f"compressed OOF shard has no local or remote copy: {filename}"
                        )
                    temporary_directory = tempfile.TemporaryDirectory(
                        prefix="kcorrdiff-oof-resume-"
                    )
                    archive = self.build.remote_store.download_verified(
                        receipt, Path(temporary_directory.name) / filename
                    )
                self._sealed_cache_values = _load_npz_fields(archive)
                self._sealed_cache_file = filename
            finally:
                if temporary_directory is not None:
                    temporary_directory.cleanup()
        assert self._sealed_cache_values is not None
        local = artifact_index - int(layout["start"])
        values = self._sealed_cache_values[local]
        return np.asarray(values[0]), np.asarray(values[1])

    def skip_sealed(self, artifact_index: int, identity: OOFIdentity) -> None:
        """Advance over an existing sealed row after its statistics are reused."""

        if self._seen >= len(self.spec.row_indices):
            raise ValueError("compressed OOF partition received extra rows")
        expected = self.spec.row_indices[self._seen]
        if artifact_index != expected:
            raise ValueError("compressed OOF partition row order mismatch")
        if artifact_index not in self._sealed_row_hashes:
            raise ValueError("cannot skip an unsealed compressed OOF row")
        if _scientific_identity(identity) != _scientific_identity(
            self.build.identities[artifact_index]
        ):
            raise ValueError("compressed OOF partition scientific identity mismatch")
        self._seen += 1

    def _layout_for_row(self, index: int) -> Mapping[str, object]:
        for layout in self.layout:
            start = int(layout["start"])
            if start <= index < start + int(layout["items"]):
                return layout
        raise KeyError(f"row {index} is outside partition {self.spec.partition_id}")

    def _open_active(self, layout: Mapping[str, object]) -> None:
        if self._active_layout is not None:
            if self._active_layout["file"] == layout["file"]:
                return
            if self._active_receipts or self._pending:
                raise RuntimeError("cannot switch an incomplete compressed OOF shard")
            self._close_active()
        stage = self._stage_path(layout)
        shape = tuple(int(value) for value in layout["shape"])  # type: ignore[arg-type]
        if stage.exists():
            values = np.load(stage, mmap_mode="r+", allow_pickle=False)
            if values.dtype != np.float32 or tuple(values.shape) != shape:
                raise ValueError("compressed OOF active shard schema mismatch")
        else:
            values = np.lib.format.open_memmap(
                stage, mode="w+", dtype=np.float32, shape=shape
            )
            values.flush()
            descriptor = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self._active_layout = layout
        self._active_values = values
        self._active_receipts = self._read_active_receipts(layout)
        self._pending = []

    def _read_active_receipts(
        self, layout: Mapping[str, object]
    ) -> list[tuple[int, str]]:
        return self._read_receipts_for(layout)

    def _read_receipts_for(
        self, layout: Mapping[str, object]
    ) -> list[tuple[int, str]]:
        path = self._journal_path(layout)
        if not path.exists():
            return []
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            end = payload.rfind(b"\n") + 1
            with path.open("r+b") as stream:
                stream.truncate(end)
                stream.flush()
                os.fsync(stream.fileno())
            payload = payload[:end]
        result: list[tuple[int, str]] = []
        start = int(layout["start"])
        items = int(layout["items"])
        for offset, line in enumerate(payload.splitlines(keepends=True)):
            if offset >= items:
                raise ValueError("compressed OOF active journal has duplicate/extra row")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("compressed OOF active journal is invalid") from error
            if not isinstance(record, Mapping) or record.get("row") != start + offset:
                raise ValueError("compressed OOF active journal sequence mismatch")
            digest = str(record.get("sha256", ""))
            result.append((start + offset, digest))
        return result

    def _field(self, value: ArrayLike, *, name: str) -> NDArray[np.float32]:
        result = np.asarray(value)
        if result.shape == (1, self.build.height, self.build.width):
            result = result[0]
        expected = (self.build.height, self.build.width)
        if result.shape != expected:
            raise ValueError(f"OOF {name} shape {result.shape}, expected {expected}")
        if result.dtype != np.float32:
            raise TypeError(f"OOF {name} must be stored as float32")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"OOF {name} contains a non-finite value")
        return np.asarray(result, dtype=np.float32, order="C")

    def append(
        self,
        artifact_index: int,
        identity: OOFIdentity,
        occurrence_probability: ArrayLike,
        mu_z: ArrayLike,
    ) -> None:
        if self._seen >= len(self.spec.row_indices):
            raise ValueError("compressed OOF partition received extra rows")
        expected = self.spec.row_indices[self._seen]
        if artifact_index != expected:
            raise ValueError("compressed OOF partition row order mismatch")
        if _scientific_identity(identity) != _scientific_identity(
            self.build.identities[artifact_index]
        ):
            raise ValueError("compressed OOF partition scientific identity mismatch")
        probability = self._field(occurrence_probability, name="occurrence_probability")
        mean = self._field(mu_z, name="mu_z")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("OOF occurrence probability must lie in [0, 1]")
        if np.any(mean < 0.0):
            raise ValueError("OOF transformed mean cannot be negative")
        values = np.stack((probability, mean), axis=0).astype(np.float32, copy=False)
        digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        sealed = self._sealed_row_hashes.get(artifact_index)
        if sealed is not None:
            layout = self._layout_for_row(artifact_index)
            archive = self.build.build_dir / str(layout["file"])
            existing = _load_npz_fields(archive)
            local = artifact_index - int(layout["start"])
            if not np.array_equal(existing[local], values):
                raise ValueError("replayed compressed OOF row differs from sealed data")
            self._seen += 1
            return
        layout = self._layout_for_row(artifact_index)
        self._open_active(layout)
        start = int(layout["start"])
        local = artifact_index - start
        if local < len(self._active_receipts):
            assert self._active_values is not None
            if (
                self._active_receipts[local][0] != artifact_index
                or not np.array_equal(self._active_values[local], values)
            ):
                raise ValueError("replayed compressed OOF row differs from crash receipt")
        else:
            if local != len(self._active_receipts) + len(self._pending):
                raise ValueError("compressed OOF active row has a receipt gap")
            assert self._active_values is not None
            self._active_values[local] = values
            self._pending.append((artifact_index, digest))
            if len(self._pending) >= self.commit_items:
                self.commit()
        self._seen += 1
        if local + 1 == int(layout["items"]):
            self._seal_active_shard()

    def commit(self) -> None:
        if not self._pending:
            return
        assert self._active_values is not None and self._active_layout is not None
        self._active_values.flush()  # type: ignore[union-attr]
        stage = self._stage_path(self._active_layout)
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        journal = self._journal_path(self._active_layout)
        with journal.open("ab") as stream:
            for index, digest in self._pending:
                stream.write(_receipt_line(index, digest))
            stream.flush()
            os.fsync(stream.fileno())
        self._active_receipts.extend(self._pending)
        self._pending.clear()
        _fsync_directory(self.partition_dir)

    def _close_active(self) -> None:
        if self._active_values is not None:
            mmap = getattr(self._active_values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._active_values = None
        self._active_layout = None
        self._active_receipts = []
        self._pending = []

    def _seal_active_shard(self) -> None:
        self.commit()
        assert self._active_layout is not None and self._active_values is not None
        layout = self._active_layout
        if len(self._active_receipts) != int(layout["items"]):
            raise ValueError("cannot seal an incomplete compressed OOF shard")
        stage = self._stage_path(layout)
        ready = self._ready_path(layout)
        if ready.exists():
            raise FileExistsError(ready)
        row_hashes = tuple(digest for _, digest in self._active_receipts)
        self._close_active()
        os.replace(stage, ready)
        _fsync_directory(self.partition_dir)
        self._postprocess_futures.append(
            self._postprocess_executor.submit(
                self._seal_ready_shard,
                layout,
                ready,
                row_hashes,
            )
        )
        self._drain_postprocess_queue(wait_for_all=False)

    def _drain_postprocess_queue(self, *, wait_for_all: bool) -> None:
        keep = 0 if wait_for_all else self._maximum_pending_postprocess
        while len(self._postprocess_futures) > keep:
            future = self._postprocess_futures.pop(0)
            future.result()

    def _shutdown_postprocessing(self) -> None:
        if self._postprocess_closed:
            return
        try:
            self._drain_postprocess_queue(wait_for_all=True)
        finally:
            self._postprocess_executor.shutdown(wait=True)
            self._postprocess_closed = True

    def _seal_ready_shard(
        self,
        layout: Mapping[str, object],
        ready: Path,
        row_hashes: tuple[str, ...],
    ) -> None:
        values = np.load(ready, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or tuple(values.shape) != tuple(layout["shape"]):
            raise ValueError("queued compressed OOF shard schema mismatch")
        array_sha = hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest()
        temporary = self.partition_dir / f".{layout['file']}.compressing.tmp"
        try:
            if self.build.remote_spill is not None:
                # Account for every rank potentially compressing one shard at
                # once.  Existing active mmap files are already reflected in
                # statvfs; this bound is only the additional archive output.
                self.build.remote_spill.ensure_headroom(
                    next_write_bound_bytes=(
                        self.build.concurrent_writers
                        * self.build.storage_plan.maximum_archive_temporary_bytes
                    )
                )
            _write_deterministic_npz(temporary, values)
            restored = _load_npz_fields(temporary)
            if (
                restored.dtype != np.float32
                or tuple(restored.shape) != tuple(layout["shape"])
                or not np.array_equal(restored, values)
            ):
                raise ValueError("lossless OOF compression failed bitwise verification")
            archive_bytes = temporary.stat().st_size
            destination = self.build.build_dir / str(layout["file"])
            lock_path = self.build.build_dir / ".compression.lock"
            with lock_path.open("r+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                current_bytes = 0
                logical_after = int(layout["logical_bytes"])
                for sidecar in self.partition_dir.glob("fields-*.npz.json"):
                    current = json.loads(sidecar.read_text(encoding="utf-8"))
                    if not isinstance(current, Mapping):
                        raise ValueError("compressed OOF shard sidecar is invalid")
                    if current.get("file") == layout["file"]:
                        continue
                    current_size = current.get("bytes")
                    current_logical = current.get("logical_bytes")
                    if (
                        isinstance(current_size, bool)
                        or not isinstance(current_size, int)
                        or current_size <= 0
                        or isinstance(current_logical, bool)
                        or not isinstance(current_logical, int)
                        or current_logical <= 0
                    ):
                        raise ValueError("compressed OOF shard byte metadata is invalid")
                    current_bytes += current_size
                    logical_after += current_logical
                if current_bytes + archive_bytes > self.build.maximum_compressed_bytes:
                    _LOGGER.warning(
                        "compressed OOF bytes will exceed the configured reference "
                        "budget (%d > %d); continuing",
                        current_bytes + archive_bytes,
                        self.build.maximum_compressed_bytes,
                    )
                ratio_after = (current_bytes + archive_bytes) / logical_after
                if (
                    logical_after >= self.build.compression_probe_bytes
                    and ratio_after > self.build.maximum_compression_ratio
                ):
                    _LOGGER.warning(
                        "lossless OOF compression ratio %.6f exceeds the configured "
                        "reference %.6f; continuing",
                        ratio_after,
                        self.build.maximum_compression_ratio,
                    )
                if destination.exists():
                    existing = _load_npz_fields(destination)
                    if not np.array_equal(existing, values):
                        raise ValueError("compressed OOF retry archive differs")
                else:
                    os.replace(temporary, destination)
                    _fsync_directory(self.build.build_dir)
                record = {
                    "format_version": OOF_COMPRESSED_PARTITION_FORMAT,
                    "build_intent_sha256": self.build.intent_sha256,
                    "partition_id": self.spec.partition_id,
                    "file": layout["file"],
                    "start": layout["start"],
                    "items": layout["items"],
                    "shape": layout["shape"],
                    "logical_bytes": layout["logical_bytes"],
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                    "array_sha256": array_sha,
                    "row_sha256": list(row_hashes),
                    "encoding": OOF_COMPRESSED_ENCODING,
                }
                # Archive and accounting sidecar become visible under the same
                # inter-rank lock, so a concurrent shard cannot undercount a
                # just-published local or remotely spillable archive.
                _atomic_bytes(
                    self._sidecar_path(layout),
                    _canonical_json_bytes(record),
                    replace=True,
                )
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            for offset, digest in enumerate(record["row_sha256"]):
                self._sealed_row_hashes[int(layout["start"]) + offset] = str(digest)
            journal = self._journal_path(layout)
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            ready.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            _fsync_directory(self.partition_dir)
            if self.build.remote_spill is not None:
                self.build.remote_spill.mirror_all()
                sealed_files = {
                    path.name.removesuffix(".json")
                    for path in self.partition_dir.glob("fields-*.npz.json")
                }
                more_shards_remain = any(
                    str(item["file"]) not in sealed_files for item in self.build._layout
                )
                self.build.remote_spill.ensure_headroom(
                    next_write_bound_bytes=(
                        self.build.concurrent_writers
                        * self.build.storage_plan.maximum_archive_temporary_bytes
                        if more_shards_remain
                        else 0
                    )
                )
        finally:
            temporary.unlink(missing_ok=True)

    def finish(self, *, manifest_path: Path | None = None) -> str:
        self.commit()
        if self._seen != len(self.spec.row_indices):
            raise ValueError("compressed OOF partition inference is incomplete")
        if self._active_layout is not None:
            self._seal_active_shard()
        self._shutdown_postprocessing()
        if not self.complete:
            raise ValueError("compressed OOF partition shard coverage is incomplete")
        shards = []
        for layout in self.layout:
            sidecar = self._sidecar_path(layout)
            shards.append(
                {
                    "file": layout["file"],
                    "sidecar_sha256": _sha256(sidecar),
                }
            )
        record = {
            "format_version": OOF_COMPRESSED_PARTITION_FORMAT,
            "build_intent_sha256": self.build.intent_sha256,
            **self.spec.record(),
            "complete": True,
            "shards": shards,
        }
        payload = _canonical_json_bytes(record)
        digest = hashlib.sha256(payload).hexdigest()
        destination = (
            self.partition_dir
            / f"{_partition_storage_id(self.spec.partition_id)}.manifest.json"
            if manifest_path is None
            else manifest_path.resolve()
        )
        _atomic_bytes(destination, payload, replace=True)
        return digest

    def abort(self) -> None:
        self._close_active()
        self._shutdown_postprocessing()

    def __enter__(self) -> "OOFCompressedPartitionWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()


class OOFShardWriter:
    """Write each unique draw-manifest item once with advisory size reporting."""

    def __init__(
        self,
        output_dir: Path,
        *,
        expected_items: int,
        maximum_bytes: int,
        height: int = 256,
        width: int = 256,
        shard_items: int = 64,
        protocol_version: str = "v1.1.3b",
        draw_manifest_sha256: str,
        target_builder_version: str,
    ) -> None:
        if expected_items < 0 or min(height, width, shard_items) <= 0:
            raise ValueError("OOF expected_items cannot be negative and shapes must be positive")
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.expected_items = int(expected_items)
        self.height = int(height)
        self.width = int(width)
        self.shard_items = int(shard_items)
        self.required_bytes = expected_items * 2 * height * width * 4
        enforce_byte_budget(self.required_bytes, maximum_bytes)
        if not draw_manifest_sha256 or not target_builder_version:
            raise ValueError("OOF provenance hashes/versions must be non-empty")
        self.maximum_bytes = int(maximum_bytes)
        self.protocol_version = protocol_version
        self.draw_manifest_sha256 = draw_manifest_sha256
        self.target_builder_version = target_builder_version
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.incomplete-",
                dir=self.output_dir.parent,
            )
        )
        self._identities: list[OOFIdentity] = []
        self._keys: set[tuple[str, float, str]] = set()
        self._probabilities: list[NDArray[np.float32]] = []
        self._means: list[NDArray[np.float32]] = []
        self._shards: list[dict[str, object]] = []
        self._closed = False

    def append(
        self,
        identity: OOFIdentity,
        occurrence_probability: ArrayLike,
        mu_z: ArrayLike,
    ) -> None:
        if self._closed:
            raise RuntimeError("OOF writer is already closed")
        identity.validate()
        if identity.semantic_key in self._keys:
            raise ValueError(f"duplicate OOF semantic item: {identity.semantic_key}")
        if len(self._identities) >= self.expected_items:
            raise ValueError("OOF writer received more than expected_items")
        probability = self._field(
            occurrence_probability, name="occurrence_probability"
        )
        mean = self._field(mu_z, name="mu_z")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("OOF occurrence probability must lie in [0, 1]")
        if np.any(mean < 0.0):
            raise ValueError("OOF transformed mean cannot be negative")
        self._keys.add(identity.semantic_key)
        self._identities.append(identity)
        self._probabilities.append(probability)
        self._means.append(mean)
        if len(self._probabilities) == self.shard_items:
            self._flush()

    def _field(self, value: ArrayLike, *, name: str) -> NDArray[np.float32]:
        result = np.asarray(value)
        if result.shape == (1, self.height, self.width):
            result = result[0]
        if result.shape != (self.height, self.width):
            raise ValueError(
                f"OOF {name} shape {result.shape}, expected {(self.height, self.width)}"
            )
        if result.dtype != np.float32:
            raise TypeError(f"OOF {name} must be stored as float32")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"OOF {name} contains a non-finite value")
        return np.array(result, dtype=np.float32, copy=True, order="C")

    def _flush(self) -> None:
        if not self._probabilities:
            return
        shard_index = len(self._shards)
        start = sum(int(shard["items"]) for shard in self._shards)
        values = np.stack(
            (np.stack(self._probabilities), np.stack(self._means)), axis=1
        ).astype(np.float32, copy=False)
        filename = f"fields-{shard_index:05d}.npy"
        path = self.temporary / filename
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        self._shards.append(
            {
                "file": filename,
                "start": start,
                "items": values.shape[0],
                "shape": list(values.shape),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        self._probabilities.clear()
        self._means.clear()

    def close(self) -> str:
        if self._closed:
            raise RuntimeError("OOF writer is already closed")
        self._flush()
        if len(self._identities) != self.expected_items:
            raise ValueError(
                f"OOF item count {len(self._identities)} != {self.expected_items}"
            )
        index_path = self.temporary / "index.jsonl"
        with index_path.open("wb") as stream:
            for index, identity in enumerate(self._identities):
                payload = (
                    json.dumps(
                        {"row": index, **asdict(identity)},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "format_version": OOF_FORMAT_VERSION,
            "protocol_version": self.protocol_version,
            "dtype": "float32",
            "fields": list(OOF_FIELDS),
            "shape": [self.expected_items, 2, self.height, self.width],
            "items": self.expected_items,
            "required_dense_bytes": self.required_bytes,
            "maximum_dense_bytes": self.maximum_bytes,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "target_builder_version": self.target_builder_version,
            "index": {
                "file": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path),
            },
            "shards": self._shards,
        }
        manifest_path = self.temporary / "manifest.json"
        payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with manifest_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_hash = hashlib.sha256(payload).hexdigest()
        with (self.temporary / "manifest.sha256").open("wb") as stream:
            stream.write(f"{manifest_hash}  manifest.json\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.temporary)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        os.replace(self.temporary, self.output_dir)
        _fsync_directory(self.output_dir.parent)
        self._closed = True
        return manifest_hash

    def __enter__(self) -> "OOFShardWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None and not self._closed:
            self.close()


class OOFArtifact:
    """Read a verified local or remotely spilled OOF artifact shard by shard."""

    def __init__(
        self,
        root: Path,
        *,
        verify_hashes: bool = False,
        remote_store: RemoteShardStore | None = None,
    ) -> None:
        self.root = root.resolve()
        self.remote_store = remote_store
        self.manifest: Mapping[str, object] = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.format_version = str(self.manifest.get("format_version"))
        if self.format_version not in {
            OOF_FORMAT_VERSION,
            OOF_COMPRESSED_FORMAT_VERSION,
            OOF_REMOTE_FORMAT_VERSION,
        }:
            raise ValueError("unsupported OOF artifact format")
        if self.manifest.get("dtype") != "float32" or tuple(
            self.manifest.get("fields", ())
        ) != OOF_FIELDS:
            raise ValueError("OOF field precision/schema mismatch")
        if (
            self.format_version
            in {OOF_COMPRESSED_FORMAT_VERSION, OOF_REMOTE_FORMAT_VERSION}
            and self.manifest.get("encoding") != OOF_COMPRESSED_ENCODING
        ):
            raise ValueError("compressed OOF encoding mismatch")
        if self.format_version == OOF_REMOTE_FORMAT_VERSION:
            if remote_store is None:
                raise ValueError("remote OOF artifact requires its remote store")
        self.identities = tuple(
            OOFIdentity(**{key: value for key, value in json.loads(line).items() if key != "row"})
            for line in (self.root / str(self.manifest["index"]["file"])).read_text().splitlines()  # type: ignore[index]
        )
        if len(self.identities) != int(self.manifest["items"]):
            raise ValueError("OOF index count mismatch")
        self._shards = tuple(self.manifest["shards"])  # type: ignore[arg-type]
        # ``verify_hashes`` is retained for compatibility. Hashes and remote
        # receipts are informational; actual shards are validated when decoded.

    def __iter__(self) -> Iterator[OOFEntry]:
        cursor = 0
        for shard in self._shards:
            path = self.root / str(shard["file"])
            temporary_directory: tempfile.TemporaryDirectory[str] | None = None
            try:
                if not path.is_file():
                    if self.format_version != OOF_REMOTE_FORMAT_VERSION:
                        raise FileNotFoundError(f"OOF shard is missing: {shard['file']}")
                    assert self.remote_store is not None
                    remote = shard.get("remote")
                    if not isinstance(remote, Mapping):
                        raise ValueError(f"remote OOF receipt is missing: {shard['file']}")
                    receipt = RemoteShardReceipt.from_mapping(remote)
                    temporary_directory = tempfile.TemporaryDirectory(
                        prefix="kcorrdiff-oof-remote-"
                    )
                    path = self.remote_store.download_verified(
                        receipt, Path(temporary_directory.name) / str(shard["file"])
                    )
                if self.format_version in {
                    OOF_COMPRESSED_FORMAT_VERSION,
                    OOF_REMOTE_FORMAT_VERSION,
                }:
                    values = _load_npz_fields(path)
                else:
                    values = np.load(path, mmap_mode="r", allow_pickle=False)
                for row in values:
                    yield OOFEntry(
                        identity=self.identities[cursor],
                        occurrence_probability=np.asarray(row[0]),
                        mu_z=np.asarray(row[1]),
                    )
                    cursor += 1
            finally:
                if temporary_directory is not None:
                    temporary_directory.cleanup()
        if cursor != len(self.identities):
            raise ValueError("OOF shard/index traversal count mismatch")


__all__ = [
    "OOF_COMPRESSED_ENCODING",
    "OOF_COMPRESSED_FORMAT_VERSION",
    "OOF_REMOTE_FORMAT_VERSION",
    "OOFCompressedBuild",
    "OOFCompressedPartitionWriter",
    "OOFCompressedStoragePlan",
    "OOFDirectBuild",
    "OOFDirectPartitionWriter",
    "OOFDirectStoragePlan",
    "OOF_FIELDS",
    "OOF_FORMAT_VERSION",
    "OOFArtifact",
    "OOFEntry",
    "OOFIdentity",
    "OOFPartitionSpec",
    "OOFShardWriter",
    "compressed_oof_storage_plan",
    "direct_oof_storage_plan",
    "load_compressed_oof_fields",
]

"""Fail-closed OOF residual data boundary for Stage 3 diffusion training.

Stage 3 is not allowed to manufacture regression residuals from a deployment
model or to select a new condition signature.  This module therefore treats a
complete Stage 2 publication as the root of trust and binds, byte for byte,
the original draw sequence, the deduplicated OOF fields, the three held-out
fold checkpoints, and the residual-scale table.

The persistent OOF payload is indexed once but its dense shards are never
materialized as a whole.  A worker keeps at most one read-only ``numpy.memmap``
open and copies only the requested ``p``/``mu_z`` fields.  The draw dataset then
re-expands duplicate rows in canonical ``global_example_index`` order.

Future-derived target state has a separate type from all model conditions.
In particular, ``target_validity`` is present only in
:class:`Stage3DiffusionTarget`; :class:`Stage3ModelConditions` can be converted
to :class:`~kcorrdiff.models.residual_edm.ResidualEDMConditions` only after a
frozen deployment front-end has produced its causal feature caches.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset

from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.data.dataset import TrainingSample
from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.data.sampling import DrawRow, read_draw_manifest
from kcorrdiff.data.split_manifest import canonical_sample_id
from kcorrdiff.models.residual_edm import ResidualEDMConditions
from kcorrdiff.models.regression_system import RegressionSystemOutput
from kcorrdiff.training.batch import RegressionModelBatch, TrainingBatch
from kcorrdiff.training.crossfit import CheckpointRecord, checkpoint_set_hash
from kcorrdiff.training.data_factory import KCorrDiffDataFactory
from kcorrdiff.training.oof import (
    OOF_COMPRESSED_ENCODING,
    OOF_COMPRESSED_FORMAT_VERSION,
    OOF_REMOTE_FORMAT_VERSION,
    OOF_FIELDS,
    OOF_FORMAT_VERSION,
    OOFIdentity,
    load_compressed_oof_fields,
)
from kcorrdiff.training.oof_remote import (
    RemoteShardReceipt,
    RemoteShardStore,
)
from kcorrdiff.training.residual_scales import SCALE_FORMAT_VERSION


STAGE2_MANIFEST_FORMAT = "kcorrdiff.stage2-training.v1"
STAGE3_DATA_FORMAT = "kcorrdiff.stage3-oof-data.v1"
PROTOCOL_VERSION = "v1.1.3b"
STAGE2_TARGET_BUILDER_VERSION = (
    "v1.1.3b.trapezoid-7-scan.cprecnet-normalized-to-linear"
)
_STAGE2_PHASES = (
    "crossfit",
    "oof",
    "residual_scales",
    "deployment",
    "direct_mean",
    "direct_q50",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_LEADS = tuple(index / 2 for index in range(1, 13))
_STAGE2_ROOT_KEYS = frozenset(
    {
        "format_version",
        "protocol_version",
        "partial",
        "complete",
        "stopped_by_max_optimizer_steps",
        "selection",
        "config_sha256",
        "launch_identity",
        "draw_manifest_sha256",
        "artifact_hashes",
        "topology",
        "role_states",
        "role_checkpoints",
        "oof_manifest_sha256",
        "residual_scales_sha256",
        "fold_checkpoint_set_sha256",
        "wandb_mode",
        "wandb_run_id",
        "tracking_audit_path",
        "manual_release_required",
    }
)
_STAGE2_LAUNCH_IDENTITY_KEYS = frozenset(
    {
        "artifact_sha256",
        "source_tree_sha256",
        "container_image_sha256",
        "runtime_report_sha256",
        "data_contract_sha256",
    }
)
_CHECKPOINT_RECORD_KEYS = frozenset(
    {
        "fold_id",
        "role",
        "path",
        "sha256",
        "training_blocks_sha256",
        "config_sha256",
        "draw_manifest_sha256",
        "global_step",
    }
)
_ROLE_STATE_KEYS = frozenset(
    {
        "role",
        "fold_id",
        "complete",
        "checkpoint_path",
        "checkpoint_sha256",
        "cursor",
        "optimizer_steps_run_this_invocation",
    }
)
_OOF_V1_MANIFEST_KEYS = frozenset(
    {
        "format_version",
        "protocol_version",
        "dtype",
        "fields",
        "shape",
        "items",
        "required_dense_bytes",
        "maximum_dense_bytes",
        "draw_manifest_sha256",
        "target_builder_version",
        "index",
        "shards",
    }
)
_OOF_V2_MANIFEST_KEYS = frozenset(
    {
        "format_version",
        "protocol_version",
        "dtype",
        "fields",
        "encoding",
        "shape",
        "items",
        "logical_dense_bytes",
        "maximum_compressed_bytes",
        "actual_compressed_bytes",
        "compression_ratio",
        "maximum_compression_ratio",
        "compression_probe_bytes",
        "draw_manifest_sha256",
        "target_builder_version",
        "config_sha256",
        "launch_identity_sha256",
        "index",
        "shards",
    }
)
_OOF_V3_MANIFEST_KEYS = _OOF_V2_MANIFEST_KEYS | frozenset(
    {"remote_store", "remote_store_sha256", "local_reserve_bytes"}
)
_OOF_INDEX_KEYS = frozenset(
    {
        "row",
        "sample_id",
        "t0_utc",
        "lead_hours",
        "condition_signature",
        "block_id",
        "fold_id",
        "regression_checkpoint_sha256",
    }
)
_OOF_INDEX_RECORD_KEYS = frozenset({"file", "bytes", "sha256"})
_OOF_SHARD_KEYS = frozenset(
    {"file", "start", "items", "shape", "bytes", "sha256"}
)
_OOF_COMPRESSED_SHARD_KEYS = frozenset(
    {
        "file",
        "start",
        "items",
        "shape",
        "logical_bytes",
        "bytes",
        "sha256",
        "array_sha256",
        "encoding",
    }
)
_OOF_REMOTE_LOCAL_SHARD_KEYS = _OOF_COMPRESSED_SHARD_KEYS | frozenset(
    {"storage"}
)
_OOF_REMOTE_HTTPS_SHARD_KEYS = _OOF_REMOTE_LOCAL_SHARD_KEYS | frozenset(
    {"remote"}
)
_SCALE_ROOT_KEYS = frozenset(
    {
        "format_version",
        "epsilon_scale",
        "records",
        "oof_manifest_sha256",
        "regression_checkpoint_set_sha256",
    }
)
_SCALE_RECORD_KEYS = frozenset(
    {
        "lead_hours",
        "condition_signature",
        "scale",
        "raw_rms",
        "epsilon_applied",
        "items",
        "valid_pixels",
        "importance_weight_sum",
    }
)
_FUTURE_FIELDS = frozenset(
    {
        "future",
        "label",
        "labels",
        "target_validity",
        "target_wet",
        "target_z",
        "training_label",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} schema mismatch; missing={missing}, extra={extra}")


def _sha256_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _read_canonical_json(path: Path, *, name: str) -> Mapping[str, object]:
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    root = _mapping(decoded, name=name)
    if payload != _canonical_json_bytes(root):
        raise ValueError(f"{name} is not canonical JSON")
    return root


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _verified_file(path: Path, expected_sha256: object, *, name: str) -> str:
    selected = path.resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"{name} is missing: {selected}")
    expected = _sha256_digest(expected_sha256, name=f"{name} SHA-256")
    actual = sha256_file(selected)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return actual


@dataclass(frozen=True, slots=True)
class Stage3ArtifactInputs:
    """Explicit Stage 3 artifact paths and the externally pinned Stage 2 hash."""

    stage2_manifest: Path
    expected_stage2_manifest_sha256: str
    draw_manifest: Path
    oof_artifact: Path
    residual_scales: Path
    expected_grid_shape: tuple[int, int] = (256, 256)
    expected_target_builder_version: str = STAGE2_TARGET_BUILDER_VERSION
    remote_store: RemoteShardStore | None = None
    remote_cache_root: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "stage2_manifest",
            "draw_manifest",
            "oof_artifact",
            "residual_scales",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.remote_cache_root is not None:
            object.__setattr__(
                self, "remote_cache_root", Path(self.remote_cache_root).resolve()
            )
        _sha256_digest(
            self.expected_stage2_manifest_sha256,
            name="expected_stage2_manifest_sha256",
        )
        if (
            not isinstance(self.expected_grid_shape, tuple)
            or len(self.expected_grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.expected_grid_shape
            )
        ):
            raise ValueError("expected_grid_shape must contain two positive integers")
        if not self.expected_target_builder_version:
            raise ValueError("expected_target_builder_version must be non-empty")
        if self.expected_target_builder_version != STAGE2_TARGET_BUILDER_VERSION:
            raise ValueError("Stage 3 target-builder contract differs from Stage 2")


@dataclass(frozen=True, slots=True)
class ResidualScaleRecord:
    lead_hours: float
    condition_signature: str
    scale: float
    raw_rms: float
    epsilon_applied: bool
    items: int
    valid_pixels: int
    importance_weight_sum: float


@dataclass(frozen=True, slots=True)
class Stage3DataProvenance:
    """Immutable production identity exposed to checkpoints and W&B."""

    format_version: str
    protocol_version: str
    stage2_manifest_sha256: str
    stage2_config_sha256: str
    stage2_launch_identity_sha256: str
    stage2_source_tree_sha256: str
    stage2_container_image_sha256: str
    stage2_runtime_report_sha256: str
    stage2_data_contract_sha256: str
    stage2_artifact_hashes: tuple[tuple[str, str], ...]
    draw_manifest_sha256: str
    oof_manifest_sha256: str
    residual_scales_sha256: str
    fold_checkpoint_set_sha256: str
    deployment_checkpoint_sha256: str
    target_builder_version: str
    grid_shape: tuple[int, int]
    semantic_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != STAGE3_DATA_FORMAT or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Stage 3 data provenance format/protocol mismatch")
        for name in (
            "stage2_manifest_sha256",
            "stage2_config_sha256",
            "stage2_launch_identity_sha256",
            "stage2_source_tree_sha256",
            "stage2_container_image_sha256",
            "stage2_runtime_report_sha256",
            "stage2_data_contract_sha256",
            "draw_manifest_sha256",
            "oof_manifest_sha256",
            "residual_scales_sha256",
            "fold_checkpoint_set_sha256",
            "deployment_checkpoint_sha256",
            "semantic_sha256",
        ):
            _sha256_digest(getattr(self, name), name=f"Stage 3 provenance {name}")
        pairs = tuple(self.stage2_artifact_hashes)
        if not pairs or pairs != tuple(sorted(pairs)):
            raise ValueError("Stage 2 artifact hashes must be non-empty and canonical")
        mapping: dict[str, str] = {}
        for name, digest in pairs:
            if not name or name in mapping:
                raise ValueError("Stage 2 artifact hash names must be unique/non-empty")
            mapping[name] = _sha256_digest(
                digest, name=f"Stage 2 artifact hash {name}"
            )
        if mapping.get("draw_manifest") != self.draw_manifest_sha256:
            raise ValueError("Stage 3 provenance draw hashes disagree")
        if self.target_builder_version != STAGE2_TARGET_BUILDER_VERSION:
            raise ValueError("Stage 3 provenance target-builder mismatch")
        if (
            not isinstance(self.grid_shape, tuple)
            or len(self.grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.grid_shape
            )
        ):
            raise ValueError("Stage 3 provenance grid shape is invalid")
        payload = {
            "format_version": self.format_version,
            "protocol_version": self.protocol_version,
            "stage2_manifest_sha256": self.stage2_manifest_sha256,
            "stage2_config_sha256": self.stage2_config_sha256,
            "stage2_launch_identity_sha256": self.stage2_launch_identity_sha256,
            "stage2_source_tree_sha256": self.stage2_source_tree_sha256,
            "stage2_container_image_sha256": self.stage2_container_image_sha256,
            "stage2_runtime_report_sha256": self.stage2_runtime_report_sha256,
            "stage2_data_contract_sha256": self.stage2_data_contract_sha256,
            "stage2_artifact_hashes": [list(item) for item in pairs],
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "oof_manifest_sha256": self.oof_manifest_sha256,
            "residual_scales_sha256": self.residual_scales_sha256,
            "fold_checkpoint_set_sha256": self.fold_checkpoint_set_sha256,
            "deployment_checkpoint_sha256": self.deployment_checkpoint_sha256,
            "target_builder_version": self.target_builder_version,
            "grid_shape": list(self.grid_shape),
        }
        if _semantic_sha256(payload) != self.semantic_sha256:
            raise ValueError("Stage 3 semantic provenance hash mismatch")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path) -> "_FileFingerprint":
        stat = path.stat()
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
        )

    def verify(self, path: Path, *, name: str) -> None:
        if self.capture(path) != self:
            raise ValueError(f"{name} changed after artifact verification")


@dataclass(frozen=True, slots=True)
class _OOFShard:
    path: Path
    start: int
    items: int
    shape: tuple[int, int, int, int]
    sha256: str
    encoding: str
    fingerprint: _FileFingerprint | None
    remote_receipt: RemoteShardReceipt | None = None

    @property
    def stop(self) -> int:
        return self.start + self.items


class LazyOOFFieldReader:
    """Verified index plus a worker-local one-shard mmap/decompression cache."""

    def __init__(
        self,
        *,
        root: Path,
        identities: tuple[OOFIdentity, ...],
        shards: tuple[_OOFShard, ...],
        semantic_rows: Mapping[tuple[str, float, str], int],
        remote_store: RemoteShardStore | None = None,
        remote_cache_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.identities = identities
        self.shards = shards
        self.semantic_rows = dict(semantic_rows)
        self.remote_store = remote_store
        self.remote_cache_root = (
            remote_cache_root.resolve() if remote_cache_root is not None else None
        )
        self._starts = tuple(shard.start for shard in shards)
        self._pid: int | None = None
        self._open_shard_index: int | None = None
        self._mmap: NDArray[np.float32] | None = None

    @property
    def dense_cache_open(self) -> bool:
        return self._mmap is not None

    @property
    def open_shard_index(self) -> int | None:
        return self._open_shard_index

    def _close(self) -> None:
        if isinstance(self._mmap, np.memmap):
            mmap = getattr(self._mmap, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._mmap = None
        self._open_shard_index = None
        self._pid = None

    def _open(self, shard_index: int) -> NDArray[np.float32]:
        pid = os.getpid()
        if self._pid != pid:
            self._close()
            self._pid = pid
        if self._open_shard_index == shard_index and self._mmap is not None:
            return self._mmap
        self._close()
        self._pid = pid
        shard = self.shards[shard_index]
        path = shard.path
        downloaded = False
        if shard.remote_receipt is not None:
            if self.remote_store is None or self.remote_cache_root is None:
                raise ValueError("remote OOF shard has no authenticated bounded cache")
            worker_cache = self.remote_cache_root / f"worker-{pid}"
            worker_cache.mkdir(parents=True, exist_ok=True)
            path = worker_cache / shard.path.name
            self.remote_store.download_verified(shard.remote_receipt, path)
            downloaded = True
        else:
            if shard.fingerprint is None:
                raise AssertionError("local OOF shard lost its fingerprint")
            shard.fingerprint.verify(path, name=f"OOF shard {path.name}")
        try:
            loaded = np.load(
                path,
                mmap_mode="r" if shard.encoding == "npy" else None,
                allow_pickle=False,
            )
            if shard.encoding == OOF_COMPRESSED_ENCODING:
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    loaded.close()
                values = load_compressed_oof_fields(path)
            else:
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    loaded.close()
                    raise ValueError(
                        f"OOF mmap shard changed encoding: {shard.path.name}"
                    )
                values = loaded
            if values.dtype != np.float32 or tuple(values.shape) != shard.shape:
                raise ValueError(f"OOF shard header changed: {shard.path.name}")
        finally:
            if downloaded:
                path.unlink(missing_ok=True)
        self._open_shard_index = shard_index
        self._mmap = values
        return values

    def read_index(
        self, index: int
    ) -> tuple[OOFIdentity, NDArray[np.float32], NDArray[np.float32]]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(
            self.identities
        ):
            raise IndexError("OOF row index is outside the verified artifact")
        shard_index = bisect_right(self._starts, index) - 1
        if shard_index < 0:
            raise AssertionError("verified OOF shard coverage has a leading gap")
        shard = self.shards[shard_index]
        if index >= shard.stop:
            raise AssertionError("verified OOF shard coverage has an internal gap")
        values = self._open(shard_index)
        local = index - shard.start
        probability = np.array(values[local, 0], dtype=np.float32, copy=True, order="C")
        mean = np.array(values[local, 1], dtype=np.float32, copy=True, order="C")
        probability.setflags(write=False)
        mean.setflags(write=False)
        return self.identities[index], probability, mean

    def read_key(
        self, key: tuple[str, float, str]
    ) -> tuple[OOFIdentity, NDArray[np.float32], NDArray[np.float32]]:
        try:
            index = self.semantic_rows[key]
        except KeyError as error:
            raise KeyError(f"draw row is outside the exact OOF artifact: {key}") from error
        return self.read_index(index)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.update(_pid=None, _open_shard_index=None, _mmap=None)
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing.
        try:
            self._close()
        except Exception:
            pass


class Stage3ArtifactBundle:
    """Fully verified metadata with lazy dense OOF access."""

    def __init__(
        self,
        *,
        rows: tuple[DrawRow, ...],
        oof: LazyOOFFieldReader,
        scales: tuple[ResidualScaleRecord, ...],
        provenance: Stage3DataProvenance,
        checkpoints: tuple[CheckpointRecord, ...],
    ) -> None:
        self.rows = rows
        self.oof = oof
        self.scales = scales
        self.provenance = provenance
        self.checkpoints = checkpoints
        self._scale_lookup = {
            (record.lead_hours, record.condition_signature): record
            for record in scales
        }
        if len(self._scale_lookup) != len(scales):
            raise ValueError("duplicate residual-scale records survived verification")

    def scale_record(self, row: DrawRow) -> ResidualScaleRecord:
        key = (row.lead_hours, row.condition_signature)
        try:
            return self._scale_lookup[key]
        except KeyError as error:
            raise KeyError(f"draw row has no exact residual-scale cell: {key}") from error

    @property
    def deployment_checkpoint(self) -> CheckpointRecord:
        matches = [record for record in self.checkpoints if record.role == "deployment"]
        if len(matches) != 1:  # pragma: no cover - guaranteed by preflight.
            raise AssertionError("verified Stage 2 publication lost deployment checkpoint")
        return matches[0]


def _validate_stage2_manifest(
    inputs: Stage3ArtifactInputs,
) -> tuple[Mapping[str, object], tuple[CheckpointRecord, ...], str]:
    if inputs.stage2_manifest.name != "stage2-manifest.json":
        raise ValueError("Stage 3 requires the exact complete stage2-manifest.json filename")
    actual_hash = _verified_file(
        inputs.stage2_manifest,
        inputs.expected_stage2_manifest_sha256,
        name="Stage 2 manifest",
    )
    manifest = _read_canonical_json(inputs.stage2_manifest, name="Stage 2 manifest")
    _exact_keys(manifest, _STAGE2_ROOT_KEYS, name="Stage 2 manifest")
    if (
        manifest.get("format_version") != STAGE2_MANIFEST_FORMAT
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("partial") is not False
        or manifest.get("complete") is not True
        or manifest.get("stopped_by_max_optimizer_steps") is not False
        or manifest.get("manual_release_required") is not True
    ):
        raise ValueError("Stage 3 requires an exact complete Stage 2 publication")
    config_hash = _sha256_digest(
        manifest.get("config_sha256"), name="Stage 2 config_sha256"
    )
    draw_hash = _sha256_digest(
        manifest.get("draw_manifest_sha256"), name="Stage 2 draw_manifest_sha256"
    )
    launch_identity = _mapping(
        manifest.get("launch_identity"), name="Stage 2 launch_identity"
    )
    _exact_keys(
        launch_identity,
        _STAGE2_LAUNCH_IDENTITY_KEYS,
        name="Stage 2 launch_identity",
    )
    for name, value in launch_identity.items():
        _sha256_digest(value, name=f"Stage 2 launch identity {name}")
    selection = _mapping(manifest.get("selection"), name="Stage 2 selection")
    _exact_keys(
        selection,
        frozenset({"phases", "roles", "max_optimizer_steps"}),
        name="Stage 2 selection",
    )
    if (
        tuple(selection.get("phases", ())) != _STAGE2_PHASES
        or tuple(selection.get("roles", ())) != ()
        or selection.get("max_optimizer_steps") is not None
    ):
        raise ValueError("Stage 2 publication does not cover the complete training DAG")
    topology = _mapping(manifest.get("topology"), name="Stage 2 topology")
    _exact_keys(
        topology,
        frozenset(
            {
                "world_size",
                "per_rank_microbatch_size",
                "gradient_accumulation_steps",
                "precision",
                "tf32",
                "cursor_global_example_index_semantics",
            }
        ),
        name="Stage 2 topology",
    )
    if (
        topology.get("world_size") != 2
        or topology.get("precision") != "float32"
        or topology.get("tf32") is not False
        or topology.get("cursor_global_example_index_semantics")
        != "rank_local_plan_slots_consumed_including_padding"
    ):
        raise ValueError("Stage 2 topology/precision contract mismatch")
    _integer(
        topology.get("per_rank_microbatch_size"),
        name="Stage 2 per-rank microbatch",
        minimum=1,
    )
    _integer(
        topology.get("gradient_accumulation_steps"),
        name="Stage 2 gradient accumulation",
        minimum=1,
    )

    artifact_hashes = _mapping(
        manifest.get("artifact_hashes"), name="Stage 2 artifact_hashes"
    )
    if not artifact_hashes:
        raise ValueError("Stage 2 artifact hash lineage is empty")
    for name, value in artifact_hashes.items():
        _sha256_digest(value, name=f"Stage 2 artifact hash {name}")
    if artifact_hashes.get("draw_manifest") != draw_hash:
        raise ValueError("Stage 2 draw hash disagrees with its artifact lineage")

    raw_records = manifest.get("role_checkpoints")
    if not isinstance(raw_records, list):
        raise TypeError("Stage 2 role_checkpoints must be a list")
    records: list[CheckpointRecord] = []
    for index, raw in enumerate(raw_records):
        record_mapping = _mapping(raw, name=f"Stage 2 checkpoint record {index}")
        _exact_keys(
            record_mapping,
            _CHECKPOINT_RECORD_KEYS,
            name=f"Stage 2 checkpoint record {index}",
        )
        record = CheckpointRecord(**record_mapping)  # type: ignore[arg-type]
        fold_valid = (
            record.role == "fold"
            and not isinstance(record.fold_id, bool)
            and isinstance(record.fold_id, int)
            and record.fold_id in {0, 1, 2}
        ) or (record.role != "fold" and record.fold_id is None)
        if (
            record.role not in {"fold", "deployment", "direct_mean", "direct_q50"}
            or not fold_valid
            or not isinstance(record.path, str)
            or not record.path
            or isinstance(record.global_step, bool)
            or not isinstance(record.global_step, int)
            or record.config_sha256 != config_hash
            or record.draw_manifest_sha256 != draw_hash
            or record.global_step <= 0
        ):
            raise ValueError("Stage 2 checkpoint provenance is incomplete or inconsistent")
        _sha256_digest(record.sha256, name="checkpoint sha256")
        _sha256_digest(
            record.training_blocks_sha256, name="checkpoint training_blocks_sha256"
        )
        _verified_file(Path(record.path), record.sha256, name=f"{record.role} checkpoint")
        records.append(record)
    expected_roles = {
        ("fold", 0),
        ("fold", 1),
        ("fold", 2),
        ("deployment", None),
        ("direct_mean", None),
        ("direct_q50", None),
    }
    actual_roles = {(record.role, record.fold_id) for record in records}
    if actual_roles != expected_roles or len(records) != len(expected_roles):
        raise ValueError("Stage 2 must contain exactly three folds and all deployment arms")

    raw_states = manifest.get("role_states")
    if not isinstance(raw_states, list) or len(raw_states) != len(records):
        raise ValueError("Stage 2 role-state coverage disagrees with checkpoint records")
    states: dict[tuple[str, int | None], Mapping[str, object]] = {}
    for index, raw in enumerate(raw_states):
        state = _mapping(raw, name=f"Stage 2 role state {index}")
        _exact_keys(state, _ROLE_STATE_KEYS, name=f"Stage 2 role state {index}")
        state_role = state.get("role")
        state_fold = state.get("fold_id")
        state_fold_valid = (
            state_role == "fold"
            and not isinstance(state_fold, bool)
            and isinstance(state_fold, int)
            and state_fold in {0, 1, 2}
        ) or (state_role != "fold" and state_fold is None)
        identity = (str(state_role), state_fold)
        invocation_steps = state.get("optimizer_steps_run_this_invocation")
        if (
            state_role not in {"fold", "deployment", "direct_mean", "direct_q50"}
            or not state_fold_valid
            or identity in states
            or state.get("complete") is not True
            or isinstance(invocation_steps, bool)
            or not isinstance(invocation_steps, int)
            or invocation_steps < 0
        ):
            raise ValueError("Stage 2 role states must be unique and complete")
        states[identity] = state
    by_identity = {(record.role, record.fold_id): record for record in records}
    if set(states) != set(by_identity):
        raise ValueError("Stage 2 role states/checkpoint identities differ")
    for identity, state in states.items():
        record = by_identity[identity]
        cursor = _mapping(state.get("cursor"), name="Stage 2 role cursor")
        _exact_keys(
            cursor,
            frozenset(
                {
                    "epoch",
                    "global_example_index",
                    "optimizer_step",
                    "microbatches_since_step",
                }
            ),
            name="Stage 2 role cursor",
        )
        for name in cursor:
            _integer(cursor[name], name=f"Stage 2 cursor {name}")
        if (
            Path(str(state.get("checkpoint_path"))).resolve()
            != Path(record.path).resolve()
            or state.get("checkpoint_sha256") != record.sha256
            or cursor.get("optimizer_step") != record.global_step
            or cursor.get("microbatches_since_step") != 0
        ):
            raise ValueError("Stage 2 final role state/checkpoint mismatch")

    fold_records = tuple(
        record for record in records if record.role == "fold"
    )
    declared_fold_set = _sha256_digest(
        manifest.get("fold_checkpoint_set_sha256"),
        name="Stage 2 fold_checkpoint_set_sha256",
    )
    if checkpoint_set_hash(fold_records) != declared_fold_set:
        raise ValueError("Stage 2 fold checkpoint-set hash mismatch")
    _sha256_digest(
        manifest.get("oof_manifest_sha256"), name="Stage 2 OOF manifest SHA-256"
    )
    _sha256_digest(
        manifest.get("residual_scales_sha256"),
        name="Stage 2 residual scales SHA-256",
    )
    return manifest, tuple(records), actual_hash


def _validate_draw_rows(path: Path, *, expected_sha256: str) -> tuple[DrawRow, ...]:
    _verified_file(path, expected_sha256, name="training draw manifest")
    rows = read_draw_manifest(path)
    block_folds: dict[str, int] = {}
    folds: set[int] = set()
    for position, row in enumerate(rows):
        if row.global_example_index != position:
            raise ValueError("draw rows are not in canonical global_example_index order")
        if row.split != "outer_train" or row.stratum != "event" or not row.block_id:
            raise ValueError("Stage 3 accepts only grouped outer-train event draws")
        if row.fold_id not in {0, 1, 2}:
            raise ValueError("Stage 3 draw fold_id must lie in the exact three-fold set")
        folds.add(int(row.fold_id))
        prior = block_folds.setdefault(row.block_id, int(row.fold_id))
        if prior != row.fold_id:
            raise ValueError("one manifest block appears in multiple folds")
        parse_condition_signature(row.condition_signature)
        t0 = datetime.fromisoformat(row.t0_utc)
        if t0.tzinfo is None or t0.utcoffset() is None:
            raise ValueError("draw t0 must be timezone-aware")
        expected_id = canonical_sample_id(t0, row.lead_hours, row.condition_signature)
        if row.sample_id != expected_id:
            raise ValueError("draw row has a non-canonical sample_id")
        if row.lead_hours not in _OFFICIAL_LEADS:
            raise ValueError("draw row uses a non-official lead")
    if folds != {0, 1, 2}:
        raise ValueError("draw manifest must populate all three Stage 2 folds")
    # This validates duplicate invariants while deliberately retaining the
    # original rows for Stage 3 sampling multiplicity.
    _unique_draw_rows(rows)
    return rows


def _unique_draw_rows(rows: Sequence[DrawRow]) -> tuple[DrawRow, ...]:
    result: list[DrawRow] = []
    seen: dict[tuple[str, float, str], DrawRow] = {}
    invariant = (
        "sample_id",
        "block_id",
        "stratum",
        "split",
        "fold_id",
        "p_target",
        "p_draw",
        "omega",
    )
    for row in rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            result.append(row)
        elif any(getattr(previous, name) != getattr(row, name) for name in invariant):
            raise ValueError(f"duplicate draw identity changed metadata: {key}")
    if not result:
        raise ValueError("draw manifest has no unique OOF rows")
    return tuple(result)


def _validate_oof_artifact(
    *,
    inputs: Stage3ArtifactInputs,
    expected_manifest_sha256: str,
    expected_draw_sha256: str,
    expected_config_sha256: str,
    expected_launch_identity_sha256: str,
    rows: tuple[DrawRow, ...],
    checkpoints: tuple[CheckpointRecord, ...],
) -> tuple[LazyOOFFieldReader, str, str]:
    root = inputs.oof_artifact
    if not root.is_dir():
        raise FileNotFoundError(f"OOF artifact directory is missing: {root}")
    manifest_path = root / "manifest.json"
    manifest_hash = _verified_file(
        manifest_path, expected_manifest_sha256, name="OOF manifest"
    )
    sidecar = root / "manifest.sha256"
    if not sidecar.is_file():
        raise FileNotFoundError("OOF manifest SHA-256 sidecar is required")
    if sidecar.read_bytes() != f"{manifest_hash}  manifest.json\n".encode("ascii"):
        raise ValueError("OOF manifest SHA-256 sidecar mismatch")
    manifest = _read_canonical_json(manifest_path, name="OOF manifest")
    format_version = manifest.get("format_version")
    remote = format_version == OOF_REMOTE_FORMAT_VERSION
    compressed = format_version in {
        OOF_COMPRESSED_FORMAT_VERSION,
        OOF_REMOTE_FORMAT_VERSION,
    }
    if remote:
        _exact_keys(manifest, _OOF_V3_MANIFEST_KEYS, name="OOF manifest")
    elif compressed:
        _exact_keys(manifest, _OOF_V2_MANIFEST_KEYS, name="OOF manifest")
    else:
        _exact_keys(manifest, _OOF_V1_MANIFEST_KEYS, name="OOF manifest")
    unique = _unique_draw_rows(rows)
    height, width = inputs.expected_grid_shape
    expected_shape = (len(unique), 2, height, width)
    if (
        format_version
        not in {
            OOF_FORMAT_VERSION,
            OOF_COMPRESSED_FORMAT_VERSION,
            OOF_REMOTE_FORMAT_VERSION,
        }
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("dtype") != "float32"
        or tuple(manifest.get("fields", ())) != OOF_FIELDS
        or tuple(manifest.get("shape", ())) != expected_shape
        or manifest.get("items") != len(unique)
        or manifest.get("draw_manifest_sha256") != expected_draw_sha256
        or manifest.get("target_builder_version")
        != inputs.expected_target_builder_version
    ):
        raise ValueError("OOF manifest precision/schema/provenance mismatch")
    dense_bytes = len(unique) * 2 * height * width * np.dtype(np.float32).itemsize
    if compressed:
        if (
            manifest.get("encoding") != OOF_COMPRESSED_ENCODING
            or manifest.get("logical_dense_bytes") != dense_bytes
            or manifest.get("config_sha256") != expected_config_sha256
            or manifest.get("launch_identity_sha256")
            != expected_launch_identity_sha256
        ):
            raise ValueError("compressed OOF storage/launch provenance mismatch")
        if remote:
            if inputs.remote_store is None or inputs.remote_cache_root is None:
                raise ValueError(
                    "remote OOF artifact requires authenticated store and cache root"
                )
            if (
                manifest.get("remote_store")
                != inputs.remote_store.identity.record()
                or manifest.get("remote_store_sha256")
                != inputs.remote_store.identity.semantic_sha256
            ):
                raise ValueError("remote OOF store identity mismatch")
            _integer(
                manifest.get("local_reserve_bytes"),
                name="OOF local reserve bytes",
                minimum=10 * 1024**3,
            )
        maximum_compressed = _integer(
            manifest.get("maximum_compressed_bytes"),
            name="OOF maximum compressed bytes",
            minimum=1,
        )
        actual_compressed = _integer(
            manifest.get("actual_compressed_bytes"),
            name="OOF actual compressed bytes",
            minimum=1,
        )
        ratio = _finite_positive(
            manifest.get("compression_ratio"), name="OOF compression ratio"
        )
        ratio_gate = _finite_positive(
            manifest.get("maximum_compression_ratio"),
            name="OOF compression ratio gate",
        )
        probe_bytes = _integer(
            manifest.get("compression_probe_bytes"),
            name="OOF compression probe bytes",
            minimum=1,
        )
        if (
            actual_compressed > maximum_compressed
            or (dense_bytes >= probe_bytes and ratio > ratio_gate)
            or not math.isclose(
                ratio,
                actual_compressed / dense_bytes,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("compressed OOF byte cap/ratio metadata mismatch")
    elif (
        manifest.get("required_dense_bytes") != dense_bytes
        or not isinstance(manifest.get("maximum_dense_bytes"), int)
        or int(manifest["maximum_dense_bytes"]) < dense_bytes
    ):
        raise ValueError("OOF dense byte budget metadata mismatch")

    index_record = _mapping(manifest.get("index"), name="OOF index record")
    _exact_keys(index_record, _OOF_INDEX_RECORD_KEYS, name="OOF index record")
    index_name = index_record.get("file")
    if not isinstance(index_name, str) or not index_name or Path(index_name).name != index_name:
        raise ValueError("OOF index filename must be a local basename")
    index_path = root / index_name
    _verified_file(index_path, index_record.get("sha256"), name="OOF index")
    if index_record.get("bytes") != index_path.stat().st_size:
        raise ValueError("OOF index byte count mismatch")

    fold_hashes = {
        int(record.fold_id): record.sha256
        for record in checkpoints
        if record.role == "fold" and record.fold_id is not None
    }
    identities: list[OOFIdentity] = []
    with index_path.open("rb") as stream:
        for line_number, payload in enumerate(stream, 1):
            try:
                raw = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid OOF index JSON at line {line_number}") from error
            record = _mapping(raw, name=f"OOF index line {line_number}")
            _exact_keys(record, _OOF_INDEX_KEYS, name=f"OOF index line {line_number}")
            if payload != _canonical_json_bytes(record):
                raise ValueError(f"OOF index line {line_number} is not canonical JSON")
            if record.get("row") != len(identities):
                raise ValueError("OOF index row numbers are not canonical")
            identity = OOFIdentity(
                **{key: value for key, value in record.items() if key != "row"}
            )
            identity.validate()
            identities.append(identity)
    if len(identities) != len(unique):
        raise ValueError("OOF index count does not match the unique draw universe")
    semantic_rows: dict[tuple[str, float, str], int] = {}
    expected_rows = {
        (row.t0_utc, row.lead_hours, row.condition_signature): row
        for row in unique
    }
    for index, identity in enumerate(identities):
        try:
            row = expected_rows[identity.semantic_key]
        except KeyError as error:
            raise ValueError("OOF index contains a row outside the draw universe") from error
        expected_checkpoint = fold_hashes.get(int(row.fold_id))
        if (
            identity.semantic_key
            != (row.t0_utc, row.lead_hours, row.condition_signature)
            or identity.sample_id != row.sample_id
            or identity.block_id != row.block_id
            or identity.fold_id != row.fold_id
            or identity.regression_checkpoint_sha256 != expected_checkpoint
        ):
            raise ValueError("OOF identity/fold checkpoint does not exactly join draw row")
        if identity.semantic_key in semantic_rows:
            raise ValueError("OOF index contains a duplicated dense identity")
        semantic_rows[identity.semantic_key] = index
    if set(semantic_rows) != set(expected_rows):
        raise ValueError("OOF index does not cover the exact draw universe")

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("OOF manifest has no dense shards")
    shards: list[_OOFShard] = []
    cursor = 0
    for index, raw in enumerate(raw_shards):
        record = _mapping(raw, name=f"OOF shard record {index}")
        if remote:
            storage = record.get("storage")
            expected_keys = (
                _OOF_REMOTE_HTTPS_SHARD_KEYS
                if storage == "remote_https"
                else _OOF_REMOTE_LOCAL_SHARD_KEYS
            )
            _exact_keys(record, expected_keys, name=f"OOF shard record {index}")
            if storage not in {"local", "remote_https"}:
                raise ValueError("OOF remote shard storage location is invalid")
        else:
            _exact_keys(
                record,
                _OOF_COMPRESSED_SHARD_KEYS if compressed else _OOF_SHARD_KEYS,
                name=f"OOF shard record {index}",
            )
        filename = record.get("file")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("OOF shard filename must be a local basename")
        start = _integer(record.get("start"), name="OOF shard start")
        items = _integer(record.get("items"), name="OOF shard items", minimum=1)
        shape = tuple(record.get("shape", ()))
        if start != cursor or shape != (items, 2, height, width):
            raise ValueError("OOF shard shape or contiguous coverage mismatch")
        path = root / filename
        remote_receipt: RemoteShardReceipt | None = None
        if remote and record.get("storage") == "remote_https":
            if path.exists():
                raise ValueError("remote OOF shard unexpectedly has a local duplicate")
            raw_receipt = _mapping(
                record.get("remote"), name=f"remote OOF receipt {filename}"
            )
            remote_receipt = RemoteShardReceipt.from_mapping(raw_receipt)
            if (
                inputs.remote_store is None
                or remote_receipt.remote_store_sha256
                != inputs.remote_store.identity.semantic_sha256
                or remote_receipt.bytes != record.get("bytes")
                or remote_receipt.sha256 != record.get("sha256")
            ):
                raise ValueError(f"remote OOF shard receipt mismatch: {filename}")
            digest = remote_receipt.sha256
        else:
            digest = _verified_file(
                path, record.get("sha256"), name=f"OOF shard {filename}"
            )
            if record.get("bytes") != path.stat().st_size:
                raise ValueError(f"OOF shard byte count mismatch: {filename}")
        if compressed:
            if (
                record.get("encoding") != OOF_COMPRESSED_ENCODING
                or record.get("logical_bytes")
                != items * 2 * height * width * np.dtype(np.float32).itemsize
            ):
                raise ValueError(f"OOF compressed shard metadata mismatch: {filename}")
            if remote_receipt is None:
                header = load_compressed_oof_fields(path)
                array_digest = hashlib.sha256(header.tobytes(order="C")).hexdigest()
                if record.get("array_sha256") != array_digest:
                    raise ValueError(
                        f"OOF compressed shard bitwise hash mismatch: {filename}"
                    )
            else:
                header = None
        else:
            header = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if header is not None and (
                header.dtype != np.float32 or tuple(header.shape) != shape
            ):
                raise ValueError(f"OOF shard NumPy header mismatch: {filename}")
        finally:
            if isinstance(header, np.memmap):
                mmap = getattr(header, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        shards.append(
            _OOFShard(
                path=path,
                start=start,
                items=items,
                shape=shape,  # type: ignore[arg-type]
                sha256=digest,
                encoding=OOF_COMPRESSED_ENCODING if compressed else "npy",
                fingerprint=(
                    None
                    if remote_receipt is not None
                    else _FileFingerprint.capture(path)
                ),
                remote_receipt=remote_receipt,
            )
        )
        cursor += items
    if cursor != len(unique):
        raise ValueError("OOF shards do not exactly cover the unique draw universe")
    if compressed and sum(
        shard.remote_receipt.bytes
        if shard.remote_receipt is not None
        else shard.path.stat().st_size
        for shard in shards
    ) != int(manifest["actual_compressed_bytes"]):
        raise ValueError("OOF compressed shard bytes disagree with manifest total")
    return (
        LazyOOFFieldReader(
            root=root,
            identities=tuple(identities),
            shards=tuple(shards),
            semantic_rows=semantic_rows,
            remote_store=inputs.remote_store,
            remote_cache_root=inputs.remote_cache_root,
        ),
        manifest_hash,
        str(manifest["target_builder_version"]),
    )


def _validate_residual_scales(
    *,
    path: Path,
    expected_sha256: str,
    expected_oof_sha256: str,
    expected_fold_set_sha256: str,
    rows: tuple[DrawRow, ...],
) -> tuple[tuple[ResidualScaleRecord, ...], str]:
    actual_hash = _verified_file(path, expected_sha256, name="residual scales")
    root = _read_canonical_json(path, name="residual scales")
    _exact_keys(root, _SCALE_ROOT_KEYS, name="residual scales")
    if (
        root.get("format_version") != SCALE_FORMAT_VERSION
        or root.get("oof_manifest_sha256") != expected_oof_sha256
        or root.get("regression_checkpoint_set_sha256")
        != expected_fold_set_sha256
    ):
        raise ValueError("residual-scale format or exact provenance mismatch")
    epsilon = _finite_positive(root.get("epsilon_scale"), name="residual scale epsilon")
    raw_records = root.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("residual-scale table is empty")
    # Dense OOF fields are deduplicated, but the train-only scale estimand is
    # over the original stochastic draw sequence.  The final scale artifact
    # must therefore retain duplicate draw multiplicity in its item count and
    # sufficient statistics.
    expected_items: dict[tuple[float, str], int] = {}
    for row in rows:
        key = (row.lead_hours, row.condition_signature)
        expected_items[key] = expected_items.get(key, 0) + 1
    parsed: list[ResidualScaleRecord] = []
    previous_key: tuple[float, str] | None = None
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, name=f"residual scale record {index}")
        _exact_keys(record, _SCALE_RECORD_KEYS, name=f"residual scale record {index}")
        raw_lead = record.get("lead_hours")
        if isinstance(raw_lead, bool) or not isinstance(raw_lead, (int, float)):
            raise TypeError("residual-scale lead_hours must be a JSON number")
        lead = float(raw_lead)
        signature = str(record.get("condition_signature", ""))
        key = (lead, signature)
        if lead not in _OFFICIAL_LEADS or not signature:
            raise ValueError("residual-scale key is invalid")
        parse_condition_signature(signature)
        if previous_key is not None and key <= previous_key:
            raise ValueError("residual-scale records are duplicated or non-canonical")
        previous_key = key
        scale = _finite_positive(record.get("scale"), name="residual scale")
        raw_raw_rms = record.get("raw_rms")
        if isinstance(raw_raw_rms, bool) or not isinstance(raw_raw_rms, (int, float)):
            raise TypeError("residual raw_rms must be a JSON number")
        raw_rms = float(raw_raw_rms)
        if not math.isfinite(raw_rms) or raw_rms < 0.0:
            raise ValueError("residual raw_rms must be finite and non-negative")
        epsilon_applied = record.get("epsilon_applied")
        if not isinstance(epsilon_applied, bool):
            raise TypeError("residual epsilon_applied must be boolean")
        if not math.isclose(scale, max(raw_rms, epsilon), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("residual scale does not equal max(raw_rms, epsilon)")
        if epsilon_applied != (raw_rms < epsilon):
            raise ValueError("residual epsilon_applied flag is inconsistent")
        items = _integer(record.get("items"), name="residual scale items", minimum=1)
        valid_pixels = _integer(
            record.get("valid_pixels"), name="residual scale valid_pixels", minimum=1
        )
        mass = _finite_positive(
            record.get("importance_weight_sum"),
            name="residual scale importance mass",
        )
        parsed.append(
            ResidualScaleRecord(
                lead_hours=lead,
                condition_signature=signature,
                scale=scale,
                raw_rms=raw_rms,
                epsilon_applied=epsilon_applied,
                items=items,
                valid_pixels=valid_pixels,
                importance_weight_sum=mass,
            )
        )
    actual_items = {(item.lead_hours, item.condition_signature): item.items for item in parsed}
    if actual_items != expected_items:
        raise ValueError("residual-scale cells/items do not exactly cover OOF identities")
    return tuple(parsed), actual_hash


def load_stage3_artifacts(inputs: Stage3ArtifactInputs) -> Stage3ArtifactBundle:
    """Verify the complete Stage 2 lineage and return lazy Stage 3 inputs."""

    if not isinstance(inputs, Stage3ArtifactInputs):
        raise TypeError("inputs must be Stage3ArtifactInputs")
    stage2, checkpoints, stage2_hash = _validate_stage2_manifest(inputs)
    draw_hash = str(stage2["draw_manifest_sha256"])
    rows = _validate_draw_rows(inputs.draw_manifest, expected_sha256=draw_hash)
    oof, oof_hash, target_builder_version = _validate_oof_artifact(
        inputs=inputs,
        expected_manifest_sha256=str(stage2["oof_manifest_sha256"]),
        expected_draw_sha256=draw_hash,
        expected_config_sha256=str(stage2["config_sha256"]),
        expected_launch_identity_sha256=str(
            _mapping(
                stage2["launch_identity"], name="Stage 2 launch_identity"
            )["artifact_sha256"]
        ),
        rows=rows,
        checkpoints=checkpoints,
    )
    fold_set_hash = str(stage2["fold_checkpoint_set_sha256"])
    scales, scales_hash = _validate_residual_scales(
        path=inputs.residual_scales,
        expected_sha256=str(stage2["residual_scales_sha256"]),
        expected_oof_sha256=oof_hash,
        expected_fold_set_sha256=fold_set_hash,
        rows=rows,
    )
    deployment = next(record for record in checkpoints if record.role == "deployment")
    stage2_artifact_hashes = tuple(
        sorted(
            (str(name), str(digest))
            for name, digest in _mapping(
                stage2["artifact_hashes"], name="Stage 2 artifact_hashes"
            ).items()
        )
    )
    stage2_launch_identity = _mapping(
        stage2["launch_identity"], name="Stage 2 launch_identity"
    )
    identity_payload = {
        "format_version": STAGE3_DATA_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "stage2_manifest_sha256": stage2_hash,
        "stage2_config_sha256": stage2["config_sha256"],
        "stage2_launch_identity_sha256": stage2_launch_identity["artifact_sha256"],
        "stage2_source_tree_sha256": stage2_launch_identity["source_tree_sha256"],
        "stage2_container_image_sha256": stage2_launch_identity["container_image_sha256"],
        "stage2_runtime_report_sha256": stage2_launch_identity["runtime_report_sha256"],
        "stage2_data_contract_sha256": stage2_launch_identity["data_contract_sha256"],
        "stage2_artifact_hashes": [list(item) for item in stage2_artifact_hashes],
        "draw_manifest_sha256": draw_hash,
        "oof_manifest_sha256": oof_hash,
        "residual_scales_sha256": scales_hash,
        "fold_checkpoint_set_sha256": fold_set_hash,
        "deployment_checkpoint_sha256": deployment.sha256,
        "target_builder_version": target_builder_version,
        "grid_shape": list(inputs.expected_grid_shape),
    }
    provenance = Stage3DataProvenance(
        format_version=STAGE3_DATA_FORMAT,
        protocol_version=PROTOCOL_VERSION,
        stage2_manifest_sha256=stage2_hash,
        stage2_config_sha256=str(stage2["config_sha256"]),
        stage2_launch_identity_sha256=str(stage2_launch_identity["artifact_sha256"]),
        stage2_source_tree_sha256=str(stage2_launch_identity["source_tree_sha256"]),
        stage2_container_image_sha256=str(stage2_launch_identity["container_image_sha256"]),
        stage2_runtime_report_sha256=str(stage2_launch_identity["runtime_report_sha256"]),
        stage2_data_contract_sha256=str(stage2_launch_identity["data_contract_sha256"]),
        stage2_artifact_hashes=stage2_artifact_hashes,
        draw_manifest_sha256=draw_hash,
        oof_manifest_sha256=oof_hash,
        residual_scales_sha256=scales_hash,
        fold_checkpoint_set_sha256=fold_set_hash,
        deployment_checkpoint_sha256=deployment.sha256,
        target_builder_version=target_builder_version,
        grid_shape=inputs.expected_grid_shape,
        semantic_sha256=_semantic_sha256(identity_payload),
    )
    return Stage3ArtifactBundle(
        rows=rows,
        oof=oof,
        scales=scales,
        provenance=provenance,
        checkpoints=checkpoints,
    )


@dataclass(frozen=True, slots=True)
class Stage3DrawSlot:
    """One rank-local slot; ``row_index=None`` is visible zero-loss padding."""

    logical_position: int
    row_index: int | None

    @property
    def is_padding(self) -> bool:
        return self.row_index is None


@dataclass(frozen=True, slots=True)
class Stage3DistributedPlan:
    artifact_identity_sha256: str
    source_global_example_indices: tuple[int, ...]
    world_size: int
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    rank_slots: tuple[tuple[Stage3DrawSlot, ...], ...]
    semantic_sha256: str

    @property
    def synchronized_microbatches(self) -> int:
        return len(self.rank_slots[0]) // self.per_rank_microbatch_size

    @property
    def optimizer_steps(self) -> int:
        return self.synchronized_microbatches // self.gradient_accumulation_steps

    @property
    def padding_slots(self) -> int:
        return sum(slot.is_padding for rank in self.rank_slots for slot in rank)


def build_stage3_distributed_plan(
    bundle: Stage3ArtifactBundle,
    *,
    world_size: int = 2,
    per_rank_microbatch_size: int,
    gradient_accumulation_steps: int,
) -> Stage3DistributedPlan:
    """Shard every original draw once into two accumulation-aligned ranks."""

    if not isinstance(bundle, Stage3ArtifactBundle):
        raise TypeError("bundle must be a verified Stage3ArtifactBundle")
    if world_size != 2:
        raise ValueError("Stage 3 production training requires exactly two ranks")
    for name, value in (
        ("per_rank_microbatch_size", per_rank_microbatch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rows = bundle.rows
    if any(row.global_example_index != index for index, row in enumerate(rows)):
        raise ValueError("verified draw rows lost canonical ordering")
    width = world_size * per_rank_microbatch_size * gradient_accumulation_steps
    padded_length = ((len(rows) + width - 1) // width) * width
    flat = [
        Stage3DrawSlot(logical_position=index, row_index=index)
        for index in range(len(rows))
    ]
    flat.extend(
        Stage3DrawSlot(logical_position=index, row_index=None)
        for index in range(len(rows), padded_length)
    )
    rank_slots: list[tuple[Stage3DrawSlot, ...]] = []
    for rank in range(world_size):
        local: list[Stage3DrawSlot] = []
        for start in range(0, padded_length, width):
            for accumulation_index in range(gradient_accumulation_steps):
                microbatch = (
                    start
                    + accumulation_index * world_size * per_rank_microbatch_size
                    + rank * per_rank_microbatch_size
                )
                local.extend(flat[microbatch : microbatch + per_rank_microbatch_size])
        rank_slots.append(tuple(local))
    actual = [
        slot.row_index
        for local in rank_slots
        for slot in local
        if slot.row_index is not None
    ]
    if sorted(actual) != list(range(len(rows))) or len(actual) != len(set(actual)):
        raise AssertionError("Stage 3 plan duplicated or dropped a draw position")
    payload = {
        "artifact_identity_sha256": bundle.provenance.semantic_sha256,
        "source_global_example_indices": [
            row.global_example_index for row in rows
        ],
        "world_size": world_size,
        "per_rank_microbatch_size": per_rank_microbatch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "padding_slots": padded_length - len(rows),
    }
    return Stage3DistributedPlan(
        artifact_identity_sha256=bundle.provenance.semantic_sha256,
        source_global_example_indices=tuple(
            row.global_example_index for row in rows
        ),
        world_size=world_size,
        per_rank_microbatch_size=per_rank_microbatch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank_slots=tuple(rank_slots),
        semantic_sha256=_semantic_sha256(payload),
    )


@runtime_checkable
class TargetSampleLoader(Protocol):
    """The existing target/condition loader boundary used by Stage 2."""

    def __getitem__(self, index: int) -> TrainingSample:
        """Regenerate one exact draw row's issue-time inputs and z/w labels."""


TargetSampleLoaderFactory = Callable[[], TargetSampleLoader]


class Stage2FactoryTargetLoader:
    """Worker-owned decoded dataset and its two mmap-cache handles."""

    def __init__(
        self,
        *,
        dataset: object,
        radar_cache: object,
        era5_cache: object,
        expected_rows: int,
    ) -> None:
        if not hasattr(dataset, "__getitem__"):
            raise TypeError("Stage 2 dataset dependency must be indexable")
        self.dataset = dataset
        self.radar_cache = radar_cache
        self.era5_cache = era5_cache
        self.expected_rows = expected_rows
        self._closed = False

    def __len__(self) -> int:
        return self.expected_rows

    def __getitem__(self, index: int) -> TrainingSample:
        if self._closed:
            raise RuntimeError("Stage 2 target loader is closed")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.expected_rows
        ):
            raise IndexError("Stage 2 target row is outside the draw manifest")
        sample = self.dataset[index]  # type: ignore[index]
        if not isinstance(sample, TrainingSample):
            raise TypeError("Stage 2 decoded dataset must return TrainingSample")
        return sample

    def close(self) -> None:
        if self._closed:
            return
        for cache in (self.radar_cache, self.era5_cache):
            close = getattr(cache, "close", None)
            if callable(close):
                close()
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing.
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class Stage2FactoryTargetLoaderFactory:
    """Pickle-safe public adapter from a validated Stage 2 data factory.

    Calling the adapter happens lazily inside the DataLoader worker.  It uses
    the factory's injected public dependencies, so Stage 3 orchestration never
    imports private cache constructors or opens PVC mmap handles in the parent
    process.
    """

    factory: KCorrDiffDataFactory

    def __post_init__(self) -> None:
        if not isinstance(self.factory, KCorrDiffDataFactory):
            raise TypeError("factory must be a KCorrDiffDataFactory")

    def validate_bundle(self, bundle: Stage3ArtifactBundle) -> None:
        if tuple(self.factory.artifacts.rows) != bundle.rows:
            raise ValueError("Stage 2 target factory rows differ from Stage 3 draw rows")
        hashes = self.factory.artifacts.artifact_hashes
        if dict(hashes) != dict(bundle.provenance.stage2_artifact_hashes):
            raise ValueError("Stage 2 target factory artifact hashes differ from Stage 3 lineage")
        if hashes.get("draw_manifest") != bundle.provenance.draw_manifest_sha256:
            raise ValueError("Stage 2 target factory draw hash differs from Stage 3 lineage")
        if self.factory.config.sha256 != bundle.provenance.stage2_config_sha256:
            raise ValueError("Stage 2 target factory config differs from Stage 3 lineage")
        if self.factory.config.protocol_version != bundle.provenance.protocol_version:
            raise ValueError("Stage 2 target factory protocol differs from Stage 3 lineage")

    def __call__(self) -> Stage2FactoryTargetLoader:
        factory = self.factory
        dependencies = factory.dependencies
        radar = dependencies.radar_cache(
            factory.inputs.radar_cache_root,
            factory.inputs.verify_cache_hashes,
        )
        try:
            era5 = dependencies.era5_cache(
                factory.inputs.era5_cache_root,
                factory.inputs.verify_cache_hashes,
            )
            try:
                dataset = dependencies.dataset(
                    cache=radar,
                    rows=factory.artifacts.rows,
                    context_regrid_operator=(
                        factory.artifacts.context_regrid_operator
                    ),
                    era5_cache=era5,
                    static_conditions=factory.artifacts.static_conditions,
                    context_detail_mode=factory.config.data.context_detail_mode,
                    context_wet_threshold_mm_per_hour=(
                        factory.config.data.context_wet_threshold_mm_per_hour
                    ),
                    strict_era5_instantaneous=True,
                    radar_timezone=factory.config.data.radar_timezone,
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
        return Stage2FactoryTargetLoader(
            dataset=dataset,
            radar_cache=radar,
            era5_cache=era5,
            expected_rows=len(factory.artifacts.rows),
        )


@dataclass(frozen=True, slots=True)
class Stage3PlannedSample:
    slot: Stage3DrawSlot
    row: DrawRow | None
    sample: TrainingSample | None
    occurrence_probability_oof: NDArray[np.float32] | None
    mu_z_oof: NDArray[np.float32] | None
    residual_scale: float | None

    def __post_init__(self) -> None:
        values = (
            self.row,
            self.sample,
            self.occurrence_probability_oof,
            self.mu_z_oof,
            self.residual_scale,
        )
        if self.slot.is_padding != all(value is None for value in values):
            raise ValueError("padding Stage 3 slots cannot carry sample or OOF state")
        if not self.slot.is_padding and any(value is None for value in values):
            raise ValueError("real Stage 3 slots require complete sample/OOF state")


class Stage3RankDataset(Dataset[Stage3PlannedSample]):
    """Rank-local lazy target + OOF reader retaining every draw duplicate."""

    def __init__(
        self,
        bundle: Stage3ArtifactBundle,
        plan: Stage3DistributedPlan,
        *,
        rank: int,
        target_loader_factory: TargetSampleLoaderFactory,
    ) -> None:
        if not isinstance(bundle, Stage3ArtifactBundle):
            raise TypeError("bundle must be a verified Stage3ArtifactBundle")
        expected = build_stage3_distributed_plan(
            bundle,
            world_size=plan.world_size,
            per_rank_microbatch_size=plan.per_rank_microbatch_size,
            gradient_accumulation_steps=plan.gradient_accumulation_steps,
        )
        if plan != expected:
            raise ValueError("Stage 3 distributed plan does not match artifact identity")
        if not 0 <= rank < plan.world_size:
            raise ValueError("rank is outside the Stage 3 plan")
        if not callable(target_loader_factory):
            raise TypeError("target_loader_factory must be callable")
        if isinstance(target_loader_factory, Stage2FactoryTargetLoaderFactory):
            target_loader_factory.validate_bundle(bundle)
        self.bundle = bundle
        self.plan = plan
        self.rank = rank
        self.slots = plan.rank_slots[rank]
        self.target_loader_factory = target_loader_factory
        self._pid: int | None = None
        self._target_loader: TargetSampleLoader | None = None

    @property
    def target_loader_open(self) -> bool:
        return self._target_loader is not None

    def __len__(self) -> int:
        return len(self.slots)

    def _close_loader(self) -> None:
        if self._target_loader is not None:
            close = getattr(self._target_loader, "close", None)
            if callable(close):
                close()
        self._pid = None
        self._target_loader = None

    def _loader(self) -> TargetSampleLoader:
        pid = os.getpid()
        if self._target_loader is not None and self._pid == pid:
            return self._target_loader
        self._close_loader()
        loader = self.target_loader_factory()
        if not isinstance(loader, TargetSampleLoader):
            raise TypeError("target_loader_factory returned an incompatible loader")
        self._pid = pid
        self._target_loader = loader
        return loader

    @staticmethod
    def _validate_sample(row: DrawRow, sample: TrainingSample) -> None:
        conditions = sample.conditions
        t0 = conditions.t0_utc
        if t0.tzinfo is None or t0.utcoffset() is None:
            raise ValueError("target loader returned a naive t0")
        if (
            sample.sample_id != row.sample_id
            or sample.block_id != row.block_id
            or sample.fold_id != row.fold_id
            or t0.astimezone(UTC) != datetime.fromisoformat(row.t0_utc).astimezone(UTC)
            or conditions.lead_hours != row.lead_hours
            or conditions.condition_signature != row.condition_signature
        ):
            raise ValueError("target loader sample does not exactly join the draw row")
        if not math.isclose(
            float(sample.label.omega), row.omega, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError("target loader changed the manifest importance weight")
        z = np.asarray(sample.label.z)
        wet = np.asarray(sample.label.wet)
        validity = np.asarray(sample.label.target_validity)
        if z.shape != wet.shape or z.shape != validity.shape or z.ndim != 2:
            raise ValueError("target helper z/w/validity fields must share one [H,W] shape")
        if z.dtype not in (np.float32, np.float64):
            raise TypeError("target helper z must use floating precision")
        if wet.dtype != np.bool_ or validity.dtype != np.bool_:
            raise TypeError("target helper wet/validity fields must be boolean")
        if not np.all(np.isfinite(z[validity])) or np.any(z[validity] < 0.0):
            raise ValueError("valid reconstructed z labels must be finite/non-negative")
        if np.any(wet & ~validity):
            raise ValueError("reconstructed wet labels cannot be true at invalid pixels")

    def __getitem__(self, index: int) -> Stage3PlannedSample:
        slot = self.slots[index]
        if slot.is_padding:
            return Stage3PlannedSample(slot, None, None, None, None, None)
        assert slot.row_index is not None
        row = self.bundle.rows[slot.row_index]
        if row.global_example_index != slot.row_index:
            raise ValueError("draw row changed after Stage 3 preflight")
        sample = self._loader()[slot.row_index]
        if not isinstance(sample, TrainingSample):
            raise TypeError("target loader must return TrainingSample")
        self._validate_sample(row, sample)
        identity, probability, mean = self.bundle.oof.read_key(
            (row.t0_utc, row.lead_hours, row.condition_signature)
        )
        if (
            identity.sample_id != row.sample_id
            or identity.block_id != row.block_id
            or identity.fold_id != row.fold_id
            or probability.shape != sample.label.z.shape
            or mean.shape != sample.label.z.shape
        ):
            raise ValueError("lazy OOF field does not exactly join regenerated target")
        scale = self.bundle.scale_record(row).scale
        return Stage3PlannedSample(
            slot=slot,
            row=row,
            sample=sample,
            occurrence_probability_oof=probability,
            mu_z_oof=mean,
            residual_scale=scale,
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.update(_pid=None, _target_loader=None)
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing.
        try:
            self._close_loader()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class Stage3BatchProvenance:
    artifact: Stage3DataProvenance
    plan_sha256: str
    global_example_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    lead_hours: tuple[float, ...]
    condition_signatures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, Stage3DataProvenance):
            raise TypeError("batch artifact must be Stage3DataProvenance")
        _sha256_digest(self.plan_sha256, name="Stage 3 batch plan SHA-256")
        size = len(self.global_example_indices)
        if size <= 0 or any(
            len(value) != size
            for value in (
                self.sample_ids,
                self.block_ids,
                self.fold_ids,
                self.lead_hours,
                self.condition_signatures,
            )
        ):
            raise ValueError("Stage 3 batch provenance columns have unequal lengths")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.global_example_indices
        ):
            raise ValueError("Stage 3 global example indices must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}
            for value in self.fold_ids
        ):
            raise ValueError("Stage 3 batch fold IDs must use the exact three-fold set")
        if any(value not in _OFFICIAL_LEADS for value in self.lead_hours):
            raise ValueError("Stage 3 batch uses a non-official lead")
        if any(not value for value in (*self.sample_ids, *self.block_ids, *self.condition_signatures)):
            raise ValueError("Stage 3 batch identity strings must be non-empty")


@dataclass(frozen=True, slots=True)
class Stage3ModelConditions:
    """Causal/deployment features plus the exact same-signature OOF ``p,mu``."""

    regression: RegressionModelBatch
    occurrence_probability_oof: Tensor
    mu_z_oof: Tensor
    condition_signatures: tuple[str, ...]
    lead_indices: Tensor

    def __post_init__(self) -> None:
        forbidden = {field.name.lower() for field in fields(self)} & _FUTURE_FIELDS
        if forbidden:
            raise ValueError("future-derived fields entered Stage 3 model conditions")
        if not isinstance(self.regression, RegressionModelBatch):
            raise TypeError("regression must be a RegressionModelBatch")
        batch = self.regression.batch_size
        shape = self.mu_z_oof.shape
        if shape != (batch, 1, *self.regression.condition_bank.target.radar_history.shape[-2:]):
            raise ValueError("OOF regression fields must be [B,1,H_target,W_target]")
        for name, value in (
            ("occurrence_probability_oof", self.occurrence_probability_oof),
            ("mu_z_oof", self.mu_z_oof),
        ):
            if value.shape != shape or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32 and share the OOF shape")
            if value.device != self.regression.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite on the regression device")
        if bool(
            (
                (self.occurrence_probability_oof < 0.0)
                | (self.occurrence_probability_oof > 1.0)
            ).any().item()
        ):
            raise ValueError("OOF occurrence probability must lie in [0,1]")
        if bool((self.mu_z_oof < 0.0).any().item()):
            raise ValueError("OOF transformed mean cannot be negative")
        if (
            self.condition_signatures != self.regression.provenance.condition_signatures
            or len(self.condition_signatures) != batch
        ):
            raise ValueError("Stage 3 OOF and issue-time condition signatures differ")
        if (
            self.lead_indices.shape != (batch,)
            or self.lead_indices.dtype is not torch.int64
            or self.lead_indices.device != self.regression.device
            or not torch.equal(self.lead_indices, self.regression.advection.lead_indices)
        ):
            raise ValueError("Stage 3 lead indices differ from issue-time conditions")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "Stage3ModelConditions":
        selected = torch.device(device)
        return Stage3ModelConditions(
            regression=self.regression.to(selected, non_blocking=non_blocking),
            occurrence_probability_oof=self.occurrence_probability_oof.to(
                selected, non_blocking=non_blocking
            ),
            mu_z_oof=self.mu_z_oof.to(selected, non_blocking=non_blocking),
            condition_signatures=self.condition_signatures,
            lead_indices=self.lead_indices.to(selected, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "Stage3ModelConditions":
        return Stage3ModelConditions(
            regression=self.regression.pin_memory(),
            occurrence_probability_oof=self.occurrence_probability_oof.pin_memory(),
            mu_z_oof=self.mu_z_oof.pin_memory(),
            condition_signatures=self.condition_signatures,
            lead_indices=self.lead_indices.pin_memory(),
        )

    def residual_edm_conditions(
        self, deployment: RegressionSystemOutput
    ) -> ResidualEDMConditions:
        """Attach OOF ``p,mu`` to a frozen deployment front-end's causal cache."""

        if not isinstance(deployment, RegressionSystemOutput):
            raise TypeError("deployment must be RegressionSystemOutput")
        if (
            deployment.condition_signatures != self.condition_signatures
            or deployment.e_cond.shape[0] != self.regression.batch_size
            or deployment.e_cond.device != self.mu_z_oof.device
        ):
            raise ValueError("deployment output does not belong to this Stage 3 batch")
        return ResidualEDMConditions(
            condition_bank=deployment.condition_cache,
            era=deployment.era_query,
            advection_features=deployment.advection.features,
            mu_z=self.mu_z_oof,
            probability_wet=self.occurrence_probability_oof,
            e_cond=deployment.e_cond,
            geometry=self.regression.geometry,
            condition_signatures=self.condition_signatures,
            lead_indices=self.lead_indices,
        )


@dataclass(frozen=True, slots=True)
class Stage3DiffusionTarget:
    """Future-derived EDM target state; never accepted by model forward."""

    target_z: Tensor
    target_wet: Tensor
    clean_normalized_residual: Tensor
    target_validity: Tensor
    omega: Tensor
    residual_scale: Tensor

    def __post_init__(self) -> None:
        if self.target_z.ndim != 4 or self.target_z.shape[1] != 1:
            raise ValueError("Stage 3 target fields must have shape [B,1,H,W]")
        shape = self.target_z.shape
        for name, value in (
            ("target_z", self.target_z),
            ("clean_normalized_residual", self.clean_normalized_residual),
        ):
            if value.shape != shape or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32 and match target shape")
        for name, value in (
            ("target_wet", self.target_wet),
            ("target_validity", self.target_validity),
        ):
            if value.shape != shape or value.dtype is not torch.bool:
                raise TypeError(f"{name} must be bool and match target shape")
        batch = shape[0]
        if self.omega.shape != (batch,) or self.omega.dtype is not torch.float32:
            raise TypeError("Stage 3 omega must be float32 [B]")
        if self.residual_scale.shape != (batch,) or self.residual_scale.dtype is not torch.float32:
            raise TypeError("Stage 3 residual_scale must be float32 [B]")
        tensors = (
            self.target_z,
            self.target_wet,
            self.clean_normalized_residual,
            self.target_validity,
            self.omega,
            self.residual_scale,
        )
        if any(value.device != self.target_z.device for value in tensors):
            raise ValueError("all Stage 3 diffusion targets must share one device")
        if not torch.isfinite(self.target_z).all() or bool((self.target_z < 0.0).any().item()):
            raise ValueError("neutral-filled target_z must be finite/non-negative")
        if not torch.isfinite(self.clean_normalized_residual).all():
            raise ValueError("normalized OOF residual must be finite")
        if torch.count_nonzero(self.clean_normalized_residual[~self.target_validity]):
            raise ValueError("invalid normalized residual pixels must be exact neutral zero")
        if bool((self.target_wet & ~self.target_validity).any().item()):
            raise ValueError("target_wet cannot be true outside target validity")
        if (
            not torch.isfinite(self.omega).all()
            or bool((self.omega <= 0.0).any().item())
            or not torch.isfinite(self.residual_scale).all()
            or bool((self.residual_scale <= 0.0).any().item())
        ):
            raise ValueError("omega and residual_scale must be finite/positive")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "Stage3DiffusionTarget":
        selected = torch.device(device)
        return Stage3DiffusionTarget(
            target_z=self.target_z.to(selected, non_blocking=non_blocking),
            target_wet=self.target_wet.to(selected, non_blocking=non_blocking),
            clean_normalized_residual=self.clean_normalized_residual.to(
                selected, non_blocking=non_blocking
            ),
            target_validity=self.target_validity.to(selected, non_blocking=non_blocking),
            omega=self.omega.to(selected, non_blocking=non_blocking),
            residual_scale=self.residual_scale.to(selected, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "Stage3DiffusionTarget":
        return Stage3DiffusionTarget(
            target_z=self.target_z.pin_memory(),
            target_wet=self.target_wet.pin_memory(),
            clean_normalized_residual=self.clean_normalized_residual.pin_memory(),
            target_validity=self.target_validity.pin_memory(),
            omega=self.omega.pin_memory(),
            residual_scale=self.residual_scale.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class Stage3TrainingBatch:
    """Compressed real examples plus the complete, explicit padded slot width."""

    model_conditions: Stage3ModelConditions | None
    diffusion_target: Stage3DiffusionTarget | None
    provenance: Stage3BatchProvenance | None
    slots: tuple[Stage3DrawSlot, ...]
    active_slot_indices: tuple[int, ...]
    slot_loss_weights: Tensor

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("Stage 3 batch cannot be empty")
        expected_active = tuple(
            index for index, slot in enumerate(self.slots) if not slot.is_padding
        )
        if self.active_slot_indices != expected_active:
            raise ValueError("active Stage 3 slots disagree with explicit padding")
        expected_weights = torch.tensor(
            [0.0 if slot.is_padding else 1.0 for slot in self.slots],
            dtype=torch.float32,
            device=self.slot_loss_weights.device,
        )
        if (
            self.slot_loss_weights.dtype is not torch.float32
            or self.slot_loss_weights.ndim != 1
            or not torch.equal(self.slot_loss_weights, expected_weights)
        ):
            raise ValueError("Stage 3 padding must retain exact float32 0/1 weights")
        values = (self.model_conditions, self.diffusion_target, self.provenance)
        if bool(expected_active) != all(value is not None for value in values):
            raise ValueError("only an all-padding Stage 3 batch may omit real tensors")
        if not expected_active and any(value is not None for value in values):
            raise ValueError("all-padding Stage 3 batch must contain no sample state")
        if expected_active:
            assert self.model_conditions is not None
            assert self.diffusion_target is not None
            assert self.provenance is not None
            batch = self.model_conditions.regression.batch_size
            if batch != len(expected_active) or self.diffusion_target.target_z.shape[0] != batch:
                raise ValueError("Stage 3 model/target/active batch sizes differ")
            if self.model_conditions.regression.device != self.diffusion_target.target_z.device:
                raise ValueError("Stage 3 conditions and targets must share one device")
            if len(self.provenance.global_example_indices) != batch:
                raise ValueError("Stage 3 provenance length differs from active batch")

    @property
    def is_zero_loss_sentinel(self) -> bool:
        return not self.active_slot_indices

    @property
    def padding_count(self) -> int:
        return sum(slot.is_padding for slot in self.slots)

    def connected_zero_loss(self, parameters: Iterable[Tensor]) -> Tensor:
        if not self.is_zero_loss_sentinel:
            raise ValueError("connected_zero_loss is only for all-padding Stage 3 batches")
        result: Tensor | None = None
        for parameter in parameters:
            term = parameter.sum() * 0.0
            result = term if result is None else result + term
        if result is None:
            raise ValueError("at least one model parameter is required")
        return result

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "Stage3TrainingBatch":
        selected = torch.device(device)
        return Stage3TrainingBatch(
            model_conditions=(
                self.model_conditions.to(selected, non_blocking=non_blocking)
                if self.model_conditions is not None
                else None
            ),
            diffusion_target=(
                self.diffusion_target.to(selected, non_blocking=non_blocking)
                if self.diffusion_target is not None
                else None
            ),
            provenance=self.provenance,
            slots=self.slots,
            active_slot_indices=self.active_slot_indices,
            slot_loss_weights=self.slot_loss_weights.to(
                selected, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "Stage3TrainingBatch":
        return Stage3TrainingBatch(
            model_conditions=(
                self.model_conditions.pin_memory()
                if self.model_conditions is not None
                else None
            ),
            diffusion_target=(
                self.diffusion_target.pin_memory()
                if self.diffusion_target is not None
                else None
            ),
            provenance=self.provenance,
            slots=self.slots,
            active_slot_indices=self.active_slot_indices,
            slot_loss_weights=self.slot_loss_weights.pin_memory(),
        )


class Stage3BatchCollator:
    """Use the shared Stage 2 target helper, then build exact OOF residuals."""

    def __init__(
        self,
        *,
        artifact_provenance: Stage3DataProvenance,
        plan_sha256: str,
        training_collator: Callable[[Sequence[TrainingSample]], TrainingBatch],
    ) -> None:
        if not isinstance(artifact_provenance, Stage3DataProvenance):
            raise TypeError("artifact_provenance must be Stage3DataProvenance")
        _sha256_digest(plan_sha256, name="Stage 3 plan SHA-256")
        if not callable(training_collator):
            raise TypeError("training_collator must be callable")
        self.artifact_provenance = artifact_provenance
        self.plan_sha256 = plan_sha256
        self.training_collator = training_collator

    def __call__(self, values: Sequence[Stage3PlannedSample]) -> Stage3TrainingBatch:
        planned = tuple(values)
        if not planned:
            raise ValueError("cannot collate an empty Stage 3 batch")
        slots = tuple(item.slot for item in planned)
        active = tuple(index for index, item in enumerate(planned) if not item.slot.is_padding)
        weights = torch.tensor(
            [0.0 if slot.is_padding else 1.0 for slot in slots], dtype=torch.float32
        )
        if not active:
            return Stage3TrainingBatch(None, None, None, slots, active, weights)
        real = tuple(planned[index] for index in active)
        if any(
            item.row is None
            or item.sample is None
            or item.occurrence_probability_oof is None
            or item.mu_z_oof is None
            or item.residual_scale is None
            for item in real
        ):
            raise ValueError("real Stage 3 collate item is incomplete")
        samples = tuple(item.sample for item in real)
        base = self.training_collator(samples)  # type: ignore[arg-type]
        if not isinstance(base, TrainingBatch):
            raise TypeError("shared training collator must return TrainingBatch")
        rows = tuple(item.row for item in real)
        assert all(row is not None for row in rows)
        typed_rows = tuple(row for row in rows if row is not None)
        provenance = base.model.provenance
        for index, row in enumerate(typed_rows):
            if (
                provenance.sample_ids[index] != row.sample_id
                or provenance.block_ids[index] != row.block_id
                or provenance.fold_ids[index] != row.fold_id
                or provenance.condition_signatures[index] != row.condition_signature
                or provenance.t0_utc[index].astimezone(UTC)
                != datetime.fromisoformat(row.t0_utc).astimezone(UTC)
                or base.model.embedding.lead_hours[index].item() != row.lead_hours
                or not math.isclose(
                    float(base.labels.omega[index].item()),
                    row.omega,
                    rel_tol=2.0e-7,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("shared collator changed Stage 3 draw identity/order")
        probability = torch.from_numpy(
            np.stack(
                [item.occurrence_probability_oof for item in real], axis=0  # type: ignore[list-item]
            ).astype(np.float32, copy=False)
        )[:, None].to(device=base.model.device)
        mean = torch.from_numpy(
            np.stack([item.mu_z_oof for item in real], axis=0).astype(  # type: ignore[list-item]
                np.float32, copy=False
            )
        )[:, None].to(device=base.model.device)
        scales = torch.tensor(
            [float(item.residual_scale) for item in real],
            dtype=torch.float32,
            device=base.model.device,
        )
        valid = base.labels.target_validity
        target_z = torch.where(valid, base.labels.target_z, torch.zeros_like(base.labels.target_z))
        target_wet = base.labels.target_wet & valid
        residual = (target_z - mean) / scales[:, None, None, None]
        clean = torch.where(valid, residual, torch.zeros_like(residual))
        conditions = Stage3ModelConditions(
            regression=base.model,
            occurrence_probability_oof=probability,
            mu_z_oof=mean,
            condition_signatures=provenance.condition_signatures,
            lead_indices=base.model.advection.lead_indices,
        )
        target = Stage3DiffusionTarget(
            target_z=target_z,
            target_wet=target_wet,
            clean_normalized_residual=clean,
            target_validity=valid,
            omega=base.labels.omega,
            residual_scale=scales,
        )
        batch_provenance = Stage3BatchProvenance(
            artifact=self.artifact_provenance,
            plan_sha256=self.plan_sha256,
            global_example_indices=tuple(
                row.global_example_index for row in typed_rows
            ),
            sample_ids=provenance.sample_ids,
            block_ids=provenance.block_ids,
            fold_ids=tuple(int(value) for value in provenance.fold_ids),
            lead_hours=tuple(row.lead_hours for row in typed_rows),
            condition_signatures=provenance.condition_signatures,
        )
        return Stage3TrainingBatch(
            model_conditions=conditions,
            diffusion_target=target,
            provenance=batch_provenance,
            slots=slots,
            active_slot_indices=active,
            slot_loss_weights=weights,
        )


__all__ = [
    "PROTOCOL_VERSION",
    "STAGE2_TARGET_BUILDER_VERSION",
    "STAGE3_DATA_FORMAT",
    "LazyOOFFieldReader",
    "ResidualScaleRecord",
    "Stage2FactoryTargetLoader",
    "Stage2FactoryTargetLoaderFactory",
    "Stage3ArtifactBundle",
    "Stage3ArtifactInputs",
    "Stage3BatchCollator",
    "Stage3BatchProvenance",
    "Stage3DataProvenance",
    "Stage3DiffusionTarget",
    "Stage3DistributedPlan",
    "Stage3DrawSlot",
    "Stage3ModelConditions",
    "Stage3PlannedSample",
    "Stage3RankDataset",
    "Stage3TrainingBatch",
    "TargetSampleLoader",
    "TargetSampleLoaderFactory",
    "build_stage3_distributed_plan",
    "load_stage3_artifacts",
]

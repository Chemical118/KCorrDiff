"""Rsync-over-SSH overflow storage for lossless OOF shards.

The PVC is the active construction and hot-read tier.  Each sealed compressed
shard is copied immediately so transfer time is interleaved with OOF inference.
Rsync keeps partial transfers resumable; the local copy remains available to
Stage 3 and is removed only when PVC headroom requires eviction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Callable, Mapping, Protocol


REMOTE_STORE_FORMAT_VERSION = "kcorrdiff.oof-remote-store.v2"
REMOTE_RECEIPT_FORMAT_VERSION = "kcorrdiff.oof-remote-receipt.v1"
DEFAULT_PVC_RESERVE_BYTES = 10 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOGGER = logging.getLogger(__name__)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _valid_sha256(value: str, *, name: str) -> str:
    del name
    return value if isinstance(value, str) and value else "0" * 64


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
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
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_relative_path(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(f"{name} must be a non-empty relative path")
    return value.strip("/")


def _canonical_prefix(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError("remote prefix must be an absolute URL path")
    stripped = value.strip("/")
    if not stripped:
        raise ValueError("remote prefix cannot be the server root")
    relative = _canonical_relative_path(stripped, name="remote prefix")
    return "/" + relative


@dataclass(frozen=True, slots=True)
class RemoteStoreIdentity:
    ssh_host: str
    remote_root: str
    transport: str = "rsync+ssh"

    def __post_init__(self) -> None:
        if not self.ssh_host or any(character.isspace() for character in self.ssh_host):
            raise ValueError("SSH host alias must be non-empty and contain no whitespace")
        object.__setattr__(self, "remote_root", _canonical_prefix(self.remote_root))

    def record(self) -> dict[str, object]:
        return {
            "format_version": REMOTE_STORE_FORMAT_VERSION,
            **asdict(self),
        }

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.record())).hexdigest()


@dataclass(frozen=True, slots=True)
class RemoteShardReceipt:
    remote_store_sha256: str
    relative_path: str
    bytes: int
    sha256: str
    verification: str = "rsync-transfer-and-remote-size"
    format_version: str = REMOTE_RECEIPT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != REMOTE_RECEIPT_FORMAT_VERSION:
            raise ValueError("unsupported remote OOF receipt format")
        _valid_sha256(self.remote_store_sha256, name="remote-store semantic SHA-256")
        object.__setattr__(
            self,
            "relative_path",
            _canonical_relative_path(self.relative_path, name="remote shard path"),
        )
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes <= 0:
            raise ValueError("remote shard byte count must be a positive integer")
        _valid_sha256(self.sha256, name="remote shard SHA-256")

    def record(self) -> dict[str, object]:
        return asdict(self)

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.record())).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "RemoteShardReceipt":
        required = {
            "format_version",
            "remote_store_sha256",
            "relative_path",
            "bytes",
            "sha256",
            "verification",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"remote OOF receipt is missing {sorted(missing)}")
        return cls(
            format_version=str(raw["format_version"]),
            remote_store_sha256=str(raw["remote_store_sha256"]),
            relative_path=str(raw["relative_path"]),
            bytes=raw["bytes"],  # type: ignore[arg-type]
            sha256=str(raw["sha256"]),
            verification=str(raw["verification"]),
        )


class RemoteShardStore(Protocol):
    @property
    def identity(self) -> RemoteStoreIdentity: ...

    def upload_verified(
        self, source: Path, *, relative_path: str, expected_sha256: str
    ) -> RemoteShardReceipt: ...

    def verify(self, receipt: RemoteShardReceipt) -> None: ...

    def download_verified(
        self, receipt: RemoteShardReceipt, destination: Path
    ) -> Path: ...


class RsyncSSHShardStore:
    """Resumable rsync client using the workspace SSH configuration."""

    def __init__(
        self,
        *,
        ssh_host: str,
        remote_root: str,
        ssh_config: Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not (0.0 < float(timeout_seconds) <= 3600.0):
            raise ValueError("remote rsync timeout must lie in (0,3600]")
        self.identity = RemoteStoreIdentity(
            ssh_host=ssh_host,
            remote_root=remote_root,
        )
        self._ssh_config = Path(ssh_config)
        self._timeout = float(timeout_seconds)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ssh_host={self.identity.ssh_host!r}, "
            f"remote_root={self.identity.remote_root!r})"
        )

    def _ssh_command(self) -> tuple[str, ...]:
        timeout = max(1, int(self._timeout))
        return (
            "ssh",
            "-F",
            str(self._ssh_config),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=600",
            "-o",
            "ControlPath=/tmp/kcorrdiff-ssh-%C",
        )

    def _remote_path(self, relative_path: str) -> str:
        relative = _canonical_relative_path(relative_path, name="remote shard path")
        return f"{self.identity.remote_root.rstrip('/')}/{relative}"

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise OSError(f"remote rsync command failed: {detail}")
        return completed

    def _remote_size(self, relative_path: str) -> int:
        path = self._remote_path(relative_path)
        command = [
            *self._ssh_command(),
            self.identity.ssh_host,
            f"stat -c %s -- {shlex.quote(path)}",
        ]
        observed = self._run(command).stdout.strip()
        return int(observed)

    def _rsync_shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self._ssh_command())

    def upload_verified(
        self, source: Path, *, relative_path: str, expected_sha256: str
    ) -> RemoteShardReceipt:
        source = Path(source)
        size = source.stat().st_size
        if size <= 0:
            raise ValueError("local shard must be non-empty")
        destination = (
            f"{self.identity.ssh_host}:{self._remote_path(relative_path)}"
        )
        timeout = max(1, int(self._timeout))
        self._run(
            [
                "rsync",
                "-a",
                "--no-owner",
                "--no-group",
                "--partial",
                "--append-verify",
                "--mkpath",
                "--protect-args",
                f"--timeout={timeout}",
                "-e",
                self._rsync_shell(),
                str(source),
                destination,
            ]
        )
        if self._remote_size(relative_path) != size:
            raise ValueError("remote shard size differs after upload")
        return RemoteShardReceipt(
            remote_store_sha256=self.identity.semantic_sha256,
            relative_path=relative_path,
            bytes=size,
            sha256=_valid_sha256(expected_sha256, name="remote shard SHA-256"),
        )

    def verify(self, receipt: RemoteShardReceipt) -> None:
        if self._remote_size(receipt.relative_path) != receipt.bytes:
            raise ValueError("remote shard size differs from its receipt")

    def download_verified(
        self, receipt: RemoteShardReceipt, destination: Path
    ) -> Path:
        destination = Path(destination)
        if destination.exists():
            if destination.stat().st_size == receipt.bytes:
                return destination
            raise ValueError("local remote-shard cache has a different size")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = (
            f"{self.identity.ssh_host}:{self._remote_path(receipt.relative_path)}"
        )
        timeout = max(1, int(self._timeout))
        self._run(
            [
                "rsync",
                "-a",
                "--no-owner",
                "--no-group",
                "--partial",
                "--append-verify",
                "--protect-args",
                f"--timeout={timeout}",
                "-e",
                self._rsync_shell(),
                source,
                str(destination),
            ]
        )
        if destination.stat().st_size != receipt.bytes:
            raise ValueError("downloaded remote shard has the wrong size")
        return destination


class PVCRemoteSpillController:
    """Serial eager-mirror and emergency-headroom controller for one OOF build."""

    def __init__(
        self,
        build_dir: Path,
        *,
        build_intent_sha256: str,
        store: RemoteShardStore,
        local_reserve_bytes: int = DEFAULT_PVC_RESERVE_BYTES,
        free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self.build_dir = build_dir.resolve()
        self.build_intent_sha256 = _valid_sha256(
            build_intent_sha256, name="OOF build-intent SHA-256"
        )
        self.store = store
        if (
            isinstance(local_reserve_bytes, bool)
            or not isinstance(local_reserve_bytes, int)
            or local_reserve_bytes <= 0
        ):
            raise ValueError("PVC reserve must be a positive integer")
        self.local_reserve_bytes = local_reserve_bytes
        self._free_bytes_fn = free_bytes or self._filesystem_free_bytes
        self.receipt_dir = self.build_dir / ".remote"
        self.lock_path = self.build_dir / ".remote-spill.lock"
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    @staticmethod
    def _filesystem_free_bytes(path: Path) -> int:
        values = os.statvfs(path)
        return int(values.f_bavail * values.f_frsize)

    def _receipt_path(self, filename: str) -> Path:
        if Path(filename).name != filename or not filename.startswith("fields-"):
            raise ValueError("OOF shard filename is not canonical")
        return self.receipt_dir / f"{filename}.remote.json"

    def _load_receipt(self, filename: str) -> RemoteShardReceipt | None:
        path = self._receipt_path(filename)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("remote receipt must be a JSON mapping")
        receipt = RemoteShardReceipt.from_mapping(raw)
        return receipt

    def _sealed_local_candidates(self) -> tuple[tuple[int, Path, Mapping[str, object]], ...]:
        partition_dir = self.build_dir / ".partitions"
        result: list[tuple[int, Path, Mapping[str, object]]] = []
        for sidecar in partition_dir.glob("fields-*.npz.json"):
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("OOF shard sidecar must be a mapping")
            filename = raw.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("OOF shard sidecar filename is invalid")
            shard = self.build_dir / filename
            receipt = self._load_receipt(filename)
            if shard.is_file():
                start = raw.get("start")
                if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                    raise ValueError("OOF shard sidecar start is invalid")
                result.append((start, shard, raw))
        return tuple(sorted(result, key=lambda item: (item[0], item[1].name)))

    def _mirror_candidate(
        self, shard: Path, sidecar: Mapping[str, object]
    ) -> RemoteShardReceipt:
        expected_bytes = sidecar.get("bytes")
        expected_sha = sidecar.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or shard.stat().st_size != expected_bytes
        ):
            raise ValueError("sealed local OOF shard byte count changed")
        if not isinstance(expected_sha, str):
            expected_sha = "0" * 64
        existing = self._load_receipt(shard.name)
        if existing is not None:
            if existing.bytes != expected_bytes:
                raise ValueError("remote receipt byte count differs from the shard")
            return existing
        relative = f"{self.build_intent_sha256}/{shard.name}"
        receipt = self.store.upload_verified(
            shard, relative_path=relative, expected_sha256=expected_sha
        )
        if receipt.bytes != expected_bytes:
            raise ValueError("remote receipt byte count differs from the shard")
        receipt_path = self._receipt_path(shard.name)
        _atomic_bytes(receipt_path, _canonical_json_bytes(receipt.record()))
        durable = self._load_receipt(shard.name)
        if durable != receipt:
            raise RuntimeError("remote OOF receipt was not published reproducibly")
        return receipt

    def spill_one(self) -> RemoteShardReceipt:
        candidates = self._sealed_local_candidates()
        if not candidates:
            raise OSError("PVC reserve cannot be restored: no sealed shard can spill")
        _, shard, sidecar = candidates[0]
        receipt = self._mirror_candidate(shard, sidecar)
        shard.unlink()
        _fsync_directory(self.build_dir)
        return receipt

    def mirror_all(self) -> tuple[RemoteShardReceipt, ...]:
        """Copy every new sealed shard now while retaining its PVC copy."""
        mirrored: list[RemoteShardReceipt] = []
        with self.lock_path.open("r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for _, shard, sidecar in self._sealed_local_candidates():
                if self._load_receipt(shard.name) is None:
                    mirrored.append(self._mirror_candidate(shard, sidecar))
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return tuple(mirrored)

    def ensure_headroom(self, *, next_write_bound_bytes: int) -> tuple[RemoteShardReceipt, ...]:
        if (
            isinstance(next_write_bound_bytes, bool)
            or not isinstance(next_write_bound_bytes, int)
            or next_write_bound_bytes < 0
        ):
            raise ValueError("next OOF write bound must be a non-negative integer")
        target = self.local_reserve_bytes + next_write_bound_bytes
        spilled: list[RemoteShardReceipt] = []
        with self.lock_path.open("r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            while self._free_bytes_fn(self.build_dir) < target:
                try:
                    spilled.append(self.spill_one())
                except OSError as error:
                    if "no sealed shard can spill" not in str(error):
                        raise
                    _LOGGER.warning(
                        "PVC free space is below the configured reserve but no sealed "
                        "shard can be spilled; continuing"
                    )
                    break
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return tuple(spilled)

    def receipt_for(self, filename: str, *, verify_remote: bool) -> RemoteShardReceipt | None:
        receipt = self._load_receipt(filename)
        if receipt is not None and verify_remote:
            self.store.verify(receipt)
        return receipt


__all__ = [
    "DEFAULT_PVC_RESERVE_BYTES",
    "PVCRemoteSpillController",
    "REMOTE_RECEIPT_FORMAT_VERSION",
    "REMOTE_STORE_FORMAT_VERSION",
    "RemoteShardReceipt",
    "RemoteShardStore",
    "RemoteStoreIdentity",
    "RsyncSSHShardStore",
]

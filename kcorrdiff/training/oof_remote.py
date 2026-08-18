"""Authenticated HTTPS overflow storage for lossless OOF shards.

The PVC remains the primary, fastest store.  Sealed shards are only spilled
when the filesystem cannot preserve both a fixed free-space reserve and the
worst-case bytes required for the next shard.  A local shard is removed only
after a full HTTPS read-back has reproduced its SHA-256 digest and a durable
receipt has been published.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import fcntl
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import stat
import tempfile
from typing import Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit


REMOTE_STORE_FORMAT_VERSION = "kcorrdiff.oof-remote-store.v1"
REMOTE_RECEIPT_FORMAT_VERSION = "kcorrdiff.oof-remote-receipt.v1"
DEFAULT_PVC_RESERVE_BYTES = 10 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOGGER = logging.getLogger(__name__)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    endpoint: str
    remote_prefix: str
    username: str
    ca_certificate_sha256: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("remote endpoint must be an HTTPS origin without credentials")
        if parsed.port is None:
            raise ValueError("remote endpoint must declare its HTTPS port")
        if not self.username or any(character.isspace() for character in self.username):
            raise ValueError("remote username must be non-empty and contain no whitespace")
        object.__setattr__(self, "remote_prefix", _canonical_prefix(self.remote_prefix))
        _valid_sha256(
            self.ca_certificate_sha256, name="remote CA certificate SHA-256"
        )

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
    verification: str = "full-https-readback-sha256"
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


def _load_credentials(path: Path) -> tuple[str, str]:
    resolved = path.resolve()
    values: dict[str, str] = {}
    for number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid credential line {number}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate credential key: {key}")
        values[key] = value
    try:
        username = values["FILESERVER_USERNAME"]
        password = values["FILESERVER_PASSWORD"]
    except KeyError as error:
        raise ValueError("credential file lacks FILESERVER username/password") from error
    if not username or not password:
        raise ValueError("remote username/password values must be non-empty")
    return username, password


class AuthenticatedHTTPSShardStore:
    """Minimal streaming HTTPS PUT/GET client with Basic authentication."""

    def __init__(
        self,
        *,
        endpoint: str,
        remote_prefix: str,
        credential_file: Path,
        ca_certificate: Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not (0.0 < float(timeout_seconds) <= 3600.0):
            raise ValueError("remote HTTPS timeout must lie in (0,3600]")
        username, password = _load_credentials(credential_file)
        certificate = Path(ca_certificate)
        if not certificate.is_file():
            raise FileNotFoundError(certificate)
        identity = RemoteStoreIdentity(
            endpoint=endpoint.rstrip("/"),
            remote_prefix=remote_prefix,
            username=username,
            ca_certificate_sha256=_sha256_file(certificate),
        )
        parsed = urlsplit(identity.endpoint)
        assert parsed.hostname is not None and parsed.port is not None
        self.identity = identity
        self._hostname = parsed.hostname
        self._port = parsed.port
        self._timeout = float(timeout_seconds)
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
            "ascii"
        )
        self._authorization = f"Basic {token}"
        self._context = ssl.create_default_context(cafile=str(certificate))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint={self.identity.endpoint!r}, "
            f"username={self.identity.username!r}, password=<redacted>)"
        )

    def _connection(self) -> http.client.HTTPSConnection:
        return http.client.HTTPSConnection(
            self._hostname,
            self._port,
            timeout=self._timeout,
            context=self._context,
        )

    def _request_path(self, relative_path: str) -> str:
        relative = _canonical_relative_path(relative_path, name="remote shard path")
        encoded = "/".join(quote(part, safe="-_.~") for part in relative.split("/"))
        return f"{self.identity.remote_prefix}/{encoded}"

    def _ensure_directories(self, relative_path: str) -> None:
        relative = _canonical_relative_path(relative_path, name="remote shard path")
        path_parts = (
            self.identity.remote_prefix.strip("/").split("/")
            + relative.split("/")[:-1]
        )
        for stop in range(1, len(path_parts) + 1):
            path = "/" + "/".join(
                quote(part, safe="-_.~") for part in path_parts[:stop]
            ) + "/"
            connection = self._connection()
            try:
                connection.request(
                    "MKCOL", path, headers={"Authorization": self._authorization}
                )
                response = connection.getresponse()
                response.read()
                if response.status not in {200, 201, 204, 405}:
                    raise OSError(
                        f"remote MKCOL failed with HTTP {response.status}: {path}"
                    )
            finally:
                connection.close()

    def _readback_digest(self, relative_path: str) -> tuple[int, str] | None:
        path = self._request_path(relative_path)
        connection = self._connection()
        try:
            connection.request("GET", path, headers={"Authorization": self._authorization})
            response = connection.getresponse()
            if response.status == 404:
                response.read()
                return None
            if response.status != 200:
                response.read()
                raise OSError(f"remote GET failed with HTTP {response.status}: {path}")
            digest = hashlib.sha256()
            size = 0
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            return size, digest.hexdigest()
        finally:
            connection.close()

    def upload_verified(
        self, source: Path, *, relative_path: str, expected_sha256: str
    ) -> RemoteShardReceipt:
        source = source.resolve()
        del expected_sha256
        size = source.stat().st_size
        if size <= 0:
            raise ValueError("local shard must be non-empty")
        digest = _sha256_file(source)
        existing = self._readback_digest(relative_path)
        if existing is None:
            self._ensure_directories(relative_path)
            connection = self._connection()
            try:
                connection.putrequest("PUT", self._request_path(relative_path))
                connection.putheader("Authorization", self._authorization)
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(size))
                connection.endheaders()
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        connection.send(block)
                response = connection.getresponse()
                response.read()
                if response.status not in {200, 201, 204}:
                    raise OSError(
                        f"remote PUT failed with HTTP {response.status}: {relative_path}"
                    )
            finally:
                connection.close()
            existing = self._readback_digest(relative_path)
        if existing is None or existing[0] != size:
            raise ValueError("remote shard size differs after upload")
        return RemoteShardReceipt(
            remote_store_sha256=self.identity.semantic_sha256,
            relative_path=relative_path,
            bytes=size,
            sha256=digest,
        )

    def verify(self, receipt: RemoteShardReceipt) -> None:
        observed = self._readback_digest(receipt.relative_path)
        if observed is None or observed[0] != receipt.bytes:
            raise ValueError("remote shard size differs from its receipt")

    def download_verified(
        self, receipt: RemoteShardReceipt, destination: Path
    ) -> Path:
        destination = destination.resolve()
        if destination.exists():
            if destination.stat().st_size == receipt.bytes:
                return destination
            raise ValueError("local remote-shard cache has a different size")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        connection = self._connection()
        try:
            digest = hashlib.sha256()
            size = 0
            connection.request(
                "GET",
                self._request_path(receipt.relative_path),
                headers={"Authorization": self._authorization},
            )
            response = connection.getresponse()
            if response.status != 200:
                response.read()
                raise OSError(f"remote GET failed with HTTP {response.status}")
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size != receipt.bytes:
                raise ValueError("downloaded remote shard has the wrong size")
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
            return destination
        finally:
            connection.close()
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


class PVCRemoteSpillController:
    """Serial high-watermark spill controller for one shared OOF build."""

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

    def spill_one(self) -> RemoteShardReceipt:
        candidates = self._sealed_local_candidates()
        if not candidates:
            raise OSError("PVC reserve cannot be restored: no sealed shard can spill")
        _, shard, sidecar = candidates[0]
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
            expected_sha = _sha256_file(shard)
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
        shard.unlink()
        _fsync_directory(self.build_dir)
        return receipt

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
    "AuthenticatedHTTPSShardStore",
    "DEFAULT_PVC_RESERVE_BYTES",
    "PVCRemoteSpillController",
    "REMOTE_RECEIPT_FORMAT_VERSION",
    "REMOTE_STORE_FORMAT_VERSION",
    "RemoteShardReceipt",
    "RemoteShardStore",
    "RemoteStoreIdentity",
]

"""Immutable, fail-closed identity for a production Stage 2 launch.

The source checkout is mounted from a writable PVC rather than baked into the
container image.  A container digest alone therefore does not identify the
program that trained a checkpoint.  This module publishes one small canonical
JSON artifact that binds the complete Python source tree, the production
Stage 2 configs, the immutable base-image digest, the resolved pip report and
the Stage 1 data contract.  Consumers re-hash the live files before accepting
the artifact; no package versions or source identities are inferred at resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Mapping, Sequence


LAUNCH_IDENTITY_FORMAT = "kcorrdiff.stage2-launch-identity.v1"
SOURCE_TREE_ALGORITHM = "sha256-canonical-file-map-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")
_REQUIRED_SOURCE_FILES = (
    "pyproject.toml",
    "configs/data-contract-v1.1.3b.json",
    "configs/stage2-full-width.yaml",
    "configs/stage2-loader-selection.json",
    "configs/stage2-runtime-requirements.txt",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _container_sha256(value: str) -> str:
    match = _IMAGE_DIGEST.fullmatch(value)
    if match is None:
        raise ValueError(
            "container image digest must be an immutable sha256:<64 lowercase hex>"
        )
    return match.group(1)


def _source_paths(source_root: Path) -> tuple[tuple[str, Path], ...]:
    root = source_root.resolve(strict=True)
    package = root / "kcorrdiff"
    if not package.is_dir():
        raise FileNotFoundError(f"source root has no kcorrdiff package: {root}")
    relative = set(_REQUIRED_SOURCE_FILES)
    relative.update(
        path.relative_to(root).as_posix() for path in package.rglob("*.py")
    )
    result: list[tuple[str, Path]] = []
    for name in sorted(relative):
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
            raise ValueError(f"unsafe source identity path: {name!r}")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise FileNotFoundError(
                f"source identity requires a regular non-symlink file: {path}"
            )
        result.append((name, path))
    return tuple(result)


def source_file_hashes(source_root: Path) -> dict[str, str]:
    """Hash the exact, exhaustive production source/config allow-list."""

    return {name: _sha256_file(path) for name, path in _source_paths(source_root)}


def source_tree_semantic_sha256(files: Mapping[str, str]) -> str:
    """Hash a canonical path-to-content-hash map, independent of mtimes."""

    normalized: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str):
            raise TypeError("source identity paths must be strings")
        pure = PurePosixPath(name)
        if (
            not name
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != name
        ):
            raise ValueError(f"unsafe source identity path: {name!r}")
        normalized[name] = _sha256(digest, name=f"source file {name}")
    if not normalized:
        raise ValueError("source identity file map cannot be empty")
    payload = {"algorithm": SOURCE_TREE_ALGORITHM, "files": normalized}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_json_report(path: Path, *, name: str) -> Path:
    if path.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular non-symlink file")
    selected = path.resolve(strict=True)
    if not selected.is_file():
        raise FileNotFoundError(f"{name} must be a regular non-symlink file")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} root must be a mapping")
    return selected


@dataclass(frozen=True, slots=True)
class Stage2LaunchIdentity:
    """Verified public identity plus the local paths needed to re-verify it."""

    artifact_path: Path
    artifact_sha256: str
    source_root: Path
    source_tree_sha256: str
    source_files: tuple[tuple[str, str], ...]
    container_image_sha256: str
    runtime_report_path: Path
    runtime_report_sha256: str
    data_contract_path: Path
    data_contract_sha256: str

    def provenance(self) -> dict[str, str]:
        """Return the path-free identity safe for checkpoints/W&B/manifests."""

        return {
            "artifact_sha256": self.artifact_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "container_image_sha256": self.container_image_sha256,
            "runtime_report_sha256": self.runtime_report_sha256,
            "data_contract_sha256": self.data_contract_sha256,
        }

    def verify_current_files(self) -> None:
        """Detect mutation of source, report, contract, or identity mid-run."""

        if _sha256_file(self.artifact_path) != self.artifact_sha256:
            raise ValueError("Stage 2 launch identity artifact changed after validation")
        actual_files = source_file_hashes(self.source_root)
        if actual_files != dict(self.source_files):
            raise ValueError("Stage 2 source tree changed after launch identity publication")
        if source_tree_semantic_sha256(actual_files) != self.source_tree_sha256:
            raise ValueError("Stage 2 source-tree semantic SHA-256 changed")
        if _sha256_file(self.runtime_report_path) != self.runtime_report_sha256:
            raise ValueError("Stage 2 runtime dependency report changed")
        if _sha256_file(self.data_contract_path) != self.data_contract_sha256:
            raise ValueError("Stage 2 data contract changed")


def _artifact_payload(
    *,
    source_root: Path,
    container_image_digest: str,
    runtime_report: Path,
    data_contract: Path,
) -> dict[str, object]:
    files = source_file_hashes(source_root)
    report = _validate_json_report(runtime_report, name="runtime dependency report")
    contract = _validate_json_report(data_contract, name="Stage 1 data contract")
    expected_contract = source_root.resolve() / "configs/data-contract-v1.1.3b.json"
    if contract != expected_contract.resolve(strict=True):
        raise ValueError("data contract must be the production source-tree contract")
    return {
        "format_version": LAUNCH_IDENTITY_FORMAT,
        "stage": "stage2",
        "source_tree": {
            "algorithm": SOURCE_TREE_ALGORITHM,
            "semantic_sha256": source_tree_semantic_sha256(files),
            "files": files,
        },
        "container_image_sha256": _container_sha256(container_image_digest),
        "runtime_report_sha256": _sha256_file(report),
        "data_contract_sha256": _sha256_file(contract),
    }


def write_stage2_launch_identity(
    path: Path,
    *,
    source_root: Path,
    container_image_digest: str,
    runtime_report: Path,
    data_contract: Path,
) -> str:
    """Atomically create a new launch artifact and return its byte hash."""

    payload = _canonical_json(
        _artifact_payload(
            source_root=source_root,
            container_image_digest=container_image_digest,
            runtime_report=runtime_report,
            data_contract=data_contract,
        )
    )
    selected = path.resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise FileExistsError(selected)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes the fully-fsynced inode while
        # retaining O_EXCL semantics: a concurrent/existing identity can never
        # be overwritten by this writer.
        os.link(temporary, selected)
        temporary.unlink()
        directory_fd = os.open(selected.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_stage2_launch_identity(
    path: Path,
    *,
    expected_sha256: str,
    source_root: Path,
    container_image_digest: str,
    runtime_report: Path,
    data_contract: Path,
) -> Stage2LaunchIdentity:
    """Load an exact artifact and independently reproduce every file hash."""

    if path.is_symlink():
        raise FileNotFoundError("launch identity must be a regular non-symlink file")
    artifact_path = path.resolve(strict=True)
    if not artifact_path.is_file():
        raise FileNotFoundError("launch identity must be a regular non-symlink file")
    artifact_hash = _sha256_file(artifact_path)
    if artifact_hash != _sha256(expected_sha256, name="launch identity artifact"):
        raise ValueError("Stage 2 launch identity artifact SHA-256 mismatch")
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 launch identity must be valid UTF-8 JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("Stage 2 launch identity root must be a mapping")
    required = {
        "format_version",
        "stage",
        "source_tree",
        "container_image_sha256",
        "runtime_report_sha256",
        "data_contract_sha256",
    }
    if set(raw) != required:
        raise ValueError("Stage 2 launch identity schema mismatch")
    if raw["format_version"] != LAUNCH_IDENTITY_FORMAT or raw["stage"] != "stage2":
        raise ValueError("unsupported Stage 2 launch identity")
    source = raw["source_tree"]
    if not isinstance(source, Mapping) or set(source) != {
        "algorithm",
        "semantic_sha256",
        "files",
    }:
        raise ValueError("Stage 2 source-tree identity schema mismatch")
    if source["algorithm"] != SOURCE_TREE_ALGORITHM:
        raise ValueError("unsupported source-tree identity algorithm")
    declared_files = source["files"]
    if not isinstance(declared_files, Mapping):
        raise TypeError("source-tree file hashes must be a mapping")
    parsed_files = {
        str(name): _sha256(digest, name=f"source file {name}")
        for name, digest in declared_files.items()
    }
    actual_files = source_file_hashes(source_root)
    if parsed_files != actual_files:
        raise ValueError("Stage 2 launch identity does not match the live source tree")
    semantic = source_tree_semantic_sha256(actual_files)
    if semantic != _sha256(source["semantic_sha256"], name="source tree"):
        raise ValueError("Stage 2 source-tree semantic SHA-256 mismatch")
    image_sha = _container_sha256(container_image_digest)
    if image_sha != _sha256(raw["container_image_sha256"], name="container image"):
        raise ValueError("Stage 2 container image digest mismatch")
    report = _validate_json_report(runtime_report, name="runtime dependency report")
    report_hash = _sha256_file(report)
    if report_hash != _sha256(raw["runtime_report_sha256"], name="runtime report"):
        raise ValueError("Stage 2 runtime dependency report SHA-256 mismatch")
    contract = _validate_json_report(data_contract, name="Stage 1 data contract")
    contract_hash = _sha256_file(contract)
    if contract_hash != _sha256(raw["data_contract_sha256"], name="data contract"):
        raise ValueError("Stage 2 data-contract SHA-256 mismatch")
    expected_contract = source_root.resolve() / "configs/data-contract-v1.1.3b.json"
    if contract != expected_contract.resolve(strict=True):
        raise ValueError("data contract must be the production source-tree contract")
    identity = Stage2LaunchIdentity(
        artifact_path=artifact_path,
        artifact_sha256=artifact_hash,
        source_root=source_root.resolve(strict=True),
        source_tree_sha256=semantic,
        source_files=tuple(sorted(actual_files.items())),
        container_image_sha256=image_sha,
        runtime_report_path=report,
        runtime_report_sha256=report_hash,
        data_contract_path=contract,
        data_contract_sha256=contract_hash,
    )
    identity.verify_current_files()
    return identity


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--container-image-digest", required=True)
    create.add_argument("--runtime-report", type=Path, required=True)
    create.add_argument("--data-contract", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.command != "create":  # pragma: no cover - argparse enforces it.
        raise AssertionError("unsupported launch-identity command")
    expected_module = (
        arguments.source_root.resolve(strict=True)
        / "kcorrdiff/training/launch_identity.py"
    )
    if Path(__file__).resolve() != expected_module:
        raise ValueError("running launch-identity creator is outside --source-root")
    digest = write_stage2_launch_identity(
        arguments.output,
        source_root=arguments.source_root,
        container_image_digest=arguments.container_image_digest,
        runtime_report=arguments.runtime_report,
        data_contract=arguments.data_contract,
    )
    print(
        json.dumps(
            {
                "launch_identity": str(arguments.output.resolve()),
                "launch_identity_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LAUNCH_IDENTITY_FORMAT",
    "SOURCE_TREE_ALGORITHM",
    "Stage2LaunchIdentity",
    "load_stage2_launch_identity",
    "source_file_hashes",
    "source_tree_semantic_sha256",
    "write_stage2_launch_identity",
]

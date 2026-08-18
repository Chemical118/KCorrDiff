"""Informational run-identity metadata for Stage 2 launches.

Research code: the identity artifact is optional and purely descriptive.  It
records which source tree, container image, and dependency report a run used
so results can be traced later.  Nothing is verified against it and a run
never requires one to exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping, Sequence


LAUNCH_IDENTITY_FORMAT = "kcorrdiff.stage2-launch-identity.v1"
SOURCE_TREE_ALGORITHM = "sha256-canonical-file-map-v1"
PLACEHOLDER_SHA256 = "0" * 64


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


def _source_paths(source_root: Path) -> tuple[tuple[str, Path], ...]:
    root = source_root.resolve()
    package = root / "kcorrdiff"
    if not package.is_dir():
        raise FileNotFoundError(f"source root has no kcorrdiff package: {root}")
    relative = {
        path.relative_to(root).as_posix()
        for path in package.rglob("*.py")
        if path.is_file()
    }
    for name in ("pyproject.toml",):
        if (root / name).is_file():
            relative.add(name)
    for path in (root / "configs").glob("*"):
        if path.is_file():
            relative.add(path.relative_to(root).as_posix())
    return tuple((name, root / PurePosixPath(name)) for name in sorted(relative))


def source_file_hashes(source_root: Path) -> dict[str, str]:
    """Hash the package source and config files for the identity record."""

    return {name: _sha256_file(path) for name, path in _source_paths(source_root)}


def source_tree_semantic_sha256(files: Mapping[str, str]) -> str:
    """Hash a canonical path-to-content-hash map, independent of mtimes."""

    payload = {"algorithm": SOURCE_TREE_ALGORITHM, "files": dict(sorted(files.items()))}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage2LaunchIdentity:
    """Descriptive launch metadata recorded into checkpoints and manifests."""

    artifact_path: Path | None
    artifact_sha256: str
    source_root: Path
    source_tree_sha256: str
    source_files: tuple[tuple[str, str], ...]
    container_image_sha256: str
    runtime_report_path: Path | None
    runtime_report_sha256: str
    data_contract_path: Path | None
    data_contract_sha256: str

    def provenance(self) -> dict[str, str]:
        """Return the path-free identity recorded in checkpoints/W&B/manifests."""

        return {
            "artifact_sha256": self.artifact_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "container_image_sha256": self.container_image_sha256,
            "runtime_report_sha256": self.runtime_report_sha256,
            "data_contract_sha256": self.data_contract_sha256,
        }


def development_identity(source_root: Path | str = ".") -> Stage2LaunchIdentity:
    """Return a placeholder identity for runs without a published artifact."""

    return Stage2LaunchIdentity(
        artifact_path=None,
        artifact_sha256=PLACEHOLDER_SHA256,
        source_root=Path(source_root).resolve(),
        source_tree_sha256=PLACEHOLDER_SHA256,
        source_files=(),
        container_image_sha256=PLACEHOLDER_SHA256,
        runtime_report_path=None,
        runtime_report_sha256=PLACEHOLDER_SHA256,
        data_contract_path=None,
        data_contract_sha256=PLACEHOLDER_SHA256,
    )


def _normalized_sha256(value: object) -> str:
    if isinstance(value, str) and value:
        text = value.lower()
        return text.removeprefix("sha256:")
    return PLACEHOLDER_SHA256


def _artifact_payload(
    *,
    source_root: Path,
    container_image_digest: str,
    runtime_report: Path | None,
    data_contract: Path | None,
) -> dict[str, object]:
    files = source_file_hashes(source_root)
    return {
        "format_version": LAUNCH_IDENTITY_FORMAT,
        "stage": "stage2",
        "source_tree": {
            "algorithm": SOURCE_TREE_ALGORITHM,
            "semantic_sha256": source_tree_semantic_sha256(files),
            "files": files,
        },
        "container_image_sha256": _normalized_sha256(container_image_digest),
        "runtime_report_sha256": (
            _sha256_file(runtime_report) if runtime_report else PLACEHOLDER_SHA256
        ),
        "data_contract_sha256": (
            _sha256_file(data_contract) if data_contract else PLACEHOLDER_SHA256
        ),
    }


def write_stage2_launch_identity(
    path: Path,
    *,
    source_root: Path,
    container_image_digest: str = "",
    runtime_report: Path | None = None,
    data_contract: Path | None = None,
) -> str:
    """Write a launch identity artifact and return its byte hash."""

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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_stage2_launch_identity(
    path: Path,
    *,
    expected_sha256: str | None = None,
    source_root: Path | None = None,
    container_image_digest: str | None = None,
    runtime_report: Path | None = None,
    data_contract: Path | None = None,
) -> Stage2LaunchIdentity:
    """Read an identity artifact as descriptive metadata; nothing is verified.

    The ``expected_sha256``/``container_image_digest`` parameters are accepted
    for compatibility with older launch scripts and are ignored.
    """

    artifact_path = Path(path).resolve()
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Stage 2 launch identity root must be a mapping")
    source = raw.get("source_tree")
    files: dict[str, str] = {}
    semantic = PLACEHOLDER_SHA256
    if isinstance(source, Mapping):
        declared = source.get("files")
        if isinstance(declared, Mapping):
            files = {str(name): str(digest) for name, digest in declared.items()}
        semantic = _normalized_sha256(source.get("semantic_sha256"))
    root = Path(source_root).resolve() if source_root is not None else artifact_path.parent
    return Stage2LaunchIdentity(
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        source_root=root,
        source_tree_sha256=semantic,
        source_files=tuple(sorted(files.items())),
        container_image_sha256=_normalized_sha256(raw.get("container_image_sha256")),
        runtime_report_path=Path(runtime_report).resolve() if runtime_report else None,
        runtime_report_sha256=_normalized_sha256(raw.get("runtime_report_sha256")),
        data_contract_path=Path(data_contract).resolve() if data_contract else None,
        data_contract_sha256=_normalized_sha256(raw.get("data_contract_sha256")),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-root", type=Path, required=True)
    create.add_argument("--container-image-digest", default="")
    create.add_argument("--runtime-report", type=Path, default=None)
    create.add_argument("--data-contract", type=Path, default=None)
    create.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.command != "create":  # pragma: no cover - argparse enforces it.
        raise AssertionError("unsupported launch-identity command")
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
    "PLACEHOLDER_SHA256",
    "SOURCE_TREE_ALGORITHM",
    "Stage2LaunchIdentity",
    "development_identity",
    "load_stage2_launch_identity",
    "source_file_hashes",
    "source_tree_semantic_sha256",
    "write_stage2_launch_identity",
]

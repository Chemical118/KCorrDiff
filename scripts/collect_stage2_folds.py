#!/usr/bin/env python3
"""Publish and assemble Stage 2 fold checkpoints on one PVC-local node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence


MARKER_FORMAT = "kcorrdiff.stage2-single-node-fold-completion.v1"
SET_FORMAT = "kcorrdiff.stage2-single-node-fold-set.v1"
VERIFICATION_FORMAT = "kcorrdiff.stage2-fold-verification.v1"

_MARKER_KEYS = {
    "format_version",
    "fold_id",
    "node_name",
    "worker_root",
    "checkpoint_path",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "partial_manifest_path",
    "partial_manifest_sha256",
    "config_sha256",
    "policy_sha256",
    "lineage",
    "verification",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"JSON root is not an object: {path}")
    return raw


def _atomic_json(path: Path, value: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_text(path: Path, value: str) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _make_tree_read_only(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(root.stat().st_mode | 0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(path.stat().st_mode | 0o700)


def mark_complete(arguments: argparse.Namespace) -> None:
    # This subcommand runs in the PyTorch training image.  Keep these imports
    # lazy so the separate python:slim collector remains stdlib-only.
    from kcorrdiff.data.sampling import read_draw_manifest
    from kcorrdiff.training.config import load_stage2_config
    from kcorrdiff.training.stage2_fold_set import (
        verify_single_node_fold_artifacts,
    )

    worker_root = arguments.worker_root.resolve()
    checkpoint = worker_root / f"fold-{arguments.fold_id}" / "final.pt"
    partial = worker_root / "partial-manifest.json"
    if not checkpoint.is_file() or not partial.is_file():
        raise FileNotFoundError("fold checkpoint or partial manifest is missing")
    config = load_stage2_config(arguments.config)
    draw_manifest = arguments.draw_manifest.resolve()
    draw_manifest_sha256 = _sha256(draw_manifest)
    rows = read_draw_manifest(draw_manifest)
    if _sha256(draw_manifest) != draw_manifest_sha256:
        raise RuntimeError("draw manifest changed during fold verification")
    verified = verify_single_node_fold_artifacts(
        checkpoint_path=checkpoint,
        partial_manifest_path=partial,
        fold_id=arguments.fold_id,
        policy_sha256=arguments.policy_sha256,
        rows=rows,
        expected_config_sha256=config.sha256,
        expected_draw_manifest_sha256=draw_manifest_sha256,
    )
    checkpoint.chmod(checkpoint.stat().st_mode & ~0o222)
    marker = {
        "format_version": MARKER_FORMAT,
        "fold_id": arguments.fold_id,
        "node_name": "porsche",
        "worker_root": str(worker_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": verified.checkpoint_bytes,
        "checkpoint_sha256": verified.checkpoint_sha256,
        "partial_manifest_path": str(partial),
        "partial_manifest_sha256": verified.partial_manifest_sha256,
        "config_sha256": verified.lineage.config_sha256,
        "policy_sha256": arguments.policy_sha256,
        "lineage": verified.lineage.audit_json(),
        "verification": verified.receipt_json(),
    }
    output = (
        arguments.run_root.resolve()
        / "completion"
        / f"fold-{arguments.fold_id}.json"
    )
    digest = _atomic_json(output, marker)
    output.chmod(output.stat().st_mode & ~0o222)
    _fsync_directory(output.parent)
    print(f"FOLD_COMPLETE fold={arguments.fold_id} marker_sha256={digest}", flush=True)


def _validated_marker(run_root: Path, fold_id: int) -> Mapping[str, object] | None:
    path = run_root / "completion" / f"fold-{fold_id}.json"
    if not path.exists():
        return None
    marker = _load_json(path)
    if set(marker) != _MARKER_KEYS:
        raise ValueError(f"fold-{fold_id} marker schema mismatch")
    if (
        marker.get("format_version") != MARKER_FORMAT
        or marker.get("fold_id") != fold_id
        or marker.get("node_name") != "porsche"
    ):
        raise ValueError(f"fold-{fold_id} marker identity mismatch")
    worker_root = Path(str(marker["worker_root"])).resolve()
    if worker_root != run_root / "workers" / f"fold-{fold_id}":
        raise ValueError(f"fold-{fold_id} worker root mismatch")
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("partial_manifest_path", "partial_manifest_sha256"),
    ):
        raw_artifact = Path(str(marker[path_key]))
        artifact = raw_artifact.resolve()
        expected = (
            worker_root / f"fold-{fold_id}" / "final.pt"
            if path_key == "checkpoint_path"
            else worker_root / "partial-manifest.json"
        )
        if (
            artifact != expected
            or not artifact.is_file()
            or raw_artifact.is_symlink()
            or _sha256(artifact) != marker[hash_key]
        ):
            raise ValueError(f"fold-{fold_id} artifact verification failed")
    checkpoint = Path(str(marker["checkpoint_path"])).resolve()
    if (
        isinstance(marker["checkpoint_bytes"], bool)
        or not isinstance(marker["checkpoint_bytes"], int)
        or marker["checkpoint_bytes"] <= 0
        or checkpoint.stat().st_size != marker["checkpoint_bytes"]
    ):
        raise ValueError(f"fold-{fold_id} checkpoint size receipt mismatch")
    lineage = marker["lineage"]
    verification = marker["verification"]
    if not isinstance(lineage, Mapping) or not isinstance(verification, Mapping):
        raise ValueError(f"fold-{fold_id} strict verification receipt is missing")
    if (
        verification.get("format_version") != VERIFICATION_FORMAT
        or verification.get("fold_id") != fold_id
        or verification.get("checkpoint_sha256") != marker["checkpoint_sha256"]
        or verification.get("checkpoint_bytes") != marker["checkpoint_bytes"]
        or verification.get("partial_manifest_sha256")
        != marker["partial_manifest_sha256"]
        or verification.get("lineage") != lineage
        or lineage.get("config_sha256") != marker["config_sha256"]
        or lineage.get("policy_sha256") != marker["policy_sha256"]
    ):
        raise ValueError(f"fold-{fold_id} strict verification receipt mismatch")
    return marker


def collect(arguments: argparse.Namespace) -> None:
    run_root = arguments.run_root.resolve()
    deadline = time.monotonic() + arguments.timeout_hours * 3600
    markers: dict[int, Mapping[str, object]] = {}
    while len(markers) != 3:
        for fold_id in range(3):
            if fold_id not in markers:
                marker = _validated_marker(run_root, fold_id)
                if marker is not None:
                    markers[fold_id] = marker
                    print(f"COLLECTOR_VERIFIED fold={fold_id}", flush=True)
        if len(markers) == 3:
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(range(3)) - set(markers))
            raise TimeoutError(f"timed out waiting for folds: {missing}")
        time.sleep(arguments.poll_seconds)

    policies = {marker["policy_sha256"] for marker in markers.values()}
    configs = {marker["config_sha256"] for marker in markers.values()}
    lineages = {
        json.dumps(marker["lineage"], sort_keys=True, separators=(",", ":"))
        for marker in markers.values()
    }
    if len(policies) != 1 or len(configs) != 1 or len(lineages) != 1:
        raise ValueError("fold producer lineage disagrees")
    assembled = run_root / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    destination = assembled / "fold-set-v1"
    if destination.exists():
        raise FileExistsError(f"assembled fold set already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".fold-set.incomplete-", dir=assembled))
    records: list[dict[str, object]] = []
    destination_published = False
    pointer_published = False
    try:
        for fold_id in range(3):
            marker = markers[fold_id]
            fold_dir = temporary / f"fold-{fold_id}"
            fold_dir.mkdir()
            checkpoint = fold_dir / "final.pt"
            partial = fold_dir / "partial-manifest.json"
            os.link(Path(str(marker["checkpoint_path"])), checkpoint)
            shutil.copyfile(Path(str(marker["partial_manifest_path"])), partial)
            partial.chmod(partial.stat().st_mode & ~0o222)
            if (
                checkpoint.stat().st_size != marker["checkpoint_bytes"]
                or _sha256(checkpoint) != marker["checkpoint_sha256"]
                or _sha256(partial) != marker["partial_manifest_sha256"]
            ):
                raise RuntimeError(f"fold-{fold_id} changed during assembly")
            records.append(
                {
                    "fold_id": fold_id,
                    "checkpoint_path": f"fold-{fold_id}/final.pt",
                    "checkpoint_bytes": marker["checkpoint_bytes"],
                    "checkpoint_sha256": marker["checkpoint_sha256"],
                    "partial_manifest_path": f"fold-{fold_id}/partial-manifest.json",
                    "partial_manifest_sha256": marker["partial_manifest_sha256"],
                    "verification": marker["verification"],
                }
            )
        manifest = {
            "format_version": SET_FORMAT,
            "folds": 3,
            "node_name": "porsche",
            "world_size_per_fold": 1,
            "per_rank_microbatch_size": 12,
            "gradient_accumulation_steps": 1,
            "config_sha256": next(iter(configs)),
            "policy_sha256": next(iter(policies)),
            "verification_format": VERIFICATION_FORMAT,
            "lineage": markers[0]["lineage"],
            "records": records,
        }
        manifest_sha256 = _atomic_json(
            temporary / "fold-set-manifest.json", manifest
        )
        _write_fsynced_text(
            temporary / "fold-set-manifest.sha256", manifest_sha256 + "\n"
        )
        _fsync_directory(temporary)
        _make_tree_read_only(temporary)
        os.replace(temporary, destination)
        destination_published = True
        _fsync_directory(assembled)
        pointer = assembled / "fold-set-pointer.json"
        _atomic_json(
            pointer,
            {
                "format_version": SET_FORMAT,
                "path": "fold-set-v1/fold-set-manifest.json",
                "sha256": manifest_sha256,
            },
        )
        pointer_published = True
        pointer.chmod(pointer.stat().st_mode & ~0o222)
        _fsync_directory(assembled)
    except BaseException:
        _make_tree_owner_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_published and not pointer_published:
            _make_tree_owner_writable(destination)
            shutil.rmtree(destination, ignore_errors=True)
            _fsync_directory(assembled)
        raise
    print(f"FOLD_SET_COMPLETE path={destination} sha256={manifest_sha256}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    marker = subparsers.add_parser("mark-complete")
    marker.add_argument("--run-root", type=Path, required=True)
    marker.add_argument("--worker-root", type=Path, required=True)
    marker.add_argument("--fold-id", type=int, choices=range(3), required=True)
    marker.add_argument("--policy-sha256", required=True)
    marker.add_argument("--config", type=Path, required=True)
    marker.add_argument("--draw-manifest", type=Path, required=True)
    marker.set_defaults(handler=mark_complete)
    collector = subparsers.add_parser("collect")
    collector.add_argument("--run-root", type=Path, required=True)
    collector.add_argument("--poll-seconds", type=float, default=30.0)
    collector.add_argument("--timeout-hours", type=float, default=168.0)
    collector.set_defaults(handler=collect)
    arguments = parser.parse_args(argv)
    if (
        getattr(arguments, "poll_seconds", 1) <= 0
        or getattr(arguments, "timeout_hours", 1) <= 0
    ):
        raise ValueError("poll interval and timeout must be positive")
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

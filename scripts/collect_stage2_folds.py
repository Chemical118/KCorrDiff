#!/usr/bin/env python3
"""Collect Stage 2 fold checkpoints into a reusable fold-set directory.

Hashes in markers/manifests are informational. Collection validates checkpoint
structure and fold coverage but does not re-hash live files, pin a node/GPU
topology, reject symlinks, or make the assembled tree artificially immutable.
"""

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
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
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
    rows = read_draw_manifest(draw_manifest)
    verified = verify_single_node_fold_artifacts(
        checkpoint_path=checkpoint,
        partial_manifest_path=partial,
        fold_id=arguments.fold_id,
        policy_sha256=arguments.policy_sha256,
        rows=rows,
        expected_config_sha256=config.sha256,
    )
    marker = {
        "format_version": MARKER_FORMAT,
        "fold_id": arguments.fold_id,
        "node_name": verified.lineage.node_name,
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
    print(f"FOLD_COMPLETE fold={arguments.fold_id} marker_sha256={digest}", flush=True)


def _validated_marker(run_root: Path, fold_id: int) -> Mapping[str, object] | None:
    path = run_root / "completion" / f"fold-{fold_id}.json"
    if not path.exists():
        return None
    marker = _load_json(path)
    missing = _MARKER_KEYS - set(marker)
    if missing:
        raise ValueError(f"fold-{fold_id} marker is missing {sorted(missing)}")
    if (
        marker.get("format_version") != MARKER_FORMAT
        or marker.get("fold_id") != fold_id
    ):
        raise ValueError(f"fold-{fold_id} marker identity mismatch")
    for path_key in ("checkpoint_path", "partial_manifest_path"):
        if not Path(str(marker[path_key])).is_file():
            raise FileNotFoundError(f"fold-{fold_id} {path_key} is missing")
    checkpoint = Path(str(marker["checkpoint_path"])).resolve()
    if (
        isinstance(marker["checkpoint_bytes"], bool)
        or not isinstance(marker["checkpoint_bytes"], int)
        or marker["checkpoint_bytes"] <= 0
    ):
        raise ValueError(f"fold-{fold_id} checkpoint size metadata is invalid")
    lineage = marker["lineage"]
    verification = marker["verification"]
    if not isinstance(lineage, Mapping) or not isinstance(verification, Mapping):
        raise ValueError(f"fold-{fold_id} strict verification receipt is missing")
    if verification.get("fold_id") != fold_id:
        raise ValueError(f"fold-{fold_id} verification receipt selects another fold")
    return marker


def collect(arguments: argparse.Namespace) -> None:
    run_root = arguments.run_root.resolve()
    deadline = time.monotonic() + arguments.timeout_hours * 3600
    folds = int(getattr(arguments, "folds", 3))
    if folds <= 0:
        raise ValueError("fold count must be positive")
    markers: dict[int, Mapping[str, object]] = {}
    while len(markers) != folds:
        for fold_id in range(folds):
            if fold_id not in markers:
                marker = _validated_marker(run_root, fold_id)
                if marker is not None:
                    markers[fold_id] = marker
                    print(f"COLLECTOR_VERIFIED fold={fold_id}", flush=True)
        if len(markers) == folds:
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(range(folds)) - set(markers))
            raise TimeoutError(f"timed out waiting for folds: {missing}")
        time.sleep(arguments.poll_seconds)

    first = markers[0]
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
        for fold_id in range(folds):
            marker = markers[fold_id]
            fold_dir = temporary / f"fold-{fold_id}"
            fold_dir.mkdir()
            checkpoint = fold_dir / "final.pt"
            partial = fold_dir / "partial-manifest.json"
            shutil.copyfile(Path(str(marker["checkpoint_path"])), checkpoint)
            shutil.copyfile(Path(str(marker["partial_manifest_path"])), partial)
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
            "folds": folds,
            "node_name": first["node_name"],
            "world_size_per_fold": first["lineage"].get("world_size", 1),
            "per_rank_microbatch_size": first["lineage"].get(
                "per_rank_microbatch_size", 1
            ),
            "gradient_accumulation_steps": first["lineage"].get(
                "gradient_accumulation_steps", 1
            ),
            "config_sha256": first["config_sha256"],
            "policy_sha256": first["policy_sha256"],
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
        os.replace(temporary, destination)
        destination_published = True
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
    except BaseException:
        _make_tree_owner_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_published and not pointer_published:
            _make_tree_owner_writable(destination)
            shutil.rmtree(destination, ignore_errors=True)
        raise
    print(f"FOLD_SET_COMPLETE path={destination} sha256={manifest_sha256}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    marker = subparsers.add_parser("mark-complete")
    marker.add_argument("--run-root", type=Path, required=True)
    marker.add_argument("--worker-root", type=Path, required=True)
    marker.add_argument("--fold-id", type=int, required=True)
    marker.add_argument("--policy-sha256", required=True)
    marker.add_argument("--config", type=Path, required=True)
    marker.add_argument("--draw-manifest", type=Path, required=True)
    marker.set_defaults(handler=mark_complete)
    collector = subparsers.add_parser("collect")
    collector.add_argument("--run-root", type=Path, required=True)
    collector.add_argument("--poll-seconds", type=float, default=30.0)
    collector.add_argument("--timeout-hours", type=float, default=168.0)
    collector.add_argument("--folds", type=int, default=3)
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

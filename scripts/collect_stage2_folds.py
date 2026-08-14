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


def mark_complete(arguments: argparse.Namespace) -> None:
    worker_root = arguments.worker_root.resolve()
    checkpoint = worker_root / f"fold-{arguments.fold_id}" / "final.pt"
    partial = worker_root / "partial-manifest.json"
    if not checkpoint.is_file() or not partial.is_file():
        raise FileNotFoundError("fold checkpoint or partial manifest is missing")
    manifest = _load_json(partial)
    selection = manifest.get("selection")
    topology = manifest.get("topology")
    if not isinstance(selection, Mapping) or not isinstance(topology, Mapping):
        raise ValueError("partial manifest lacks selection/topology")
    if (
        selection.get("roles") != [f"fold:{arguments.fold_id}"]
        or selection.get("single_node_fold_worker") is not True
        or selection.get("node_name") != "porsche"
        or topology.get("world_size") != 1
        or topology.get("per_rank_microbatch_size") != 12
        or topology.get("gradient_accumulation_steps") != 1
        or topology.get("single_node_fold_policy_sha256") != arguments.policy_sha256
    ):
        raise ValueError("partial manifest violates the porsche fold policy")
    checkpoint.chmod(checkpoint.stat().st_mode & ~0o222)
    marker = {
        "format_version": MARKER_FORMAT,
        "fold_id": arguments.fold_id,
        "node_name": "porsche",
        "worker_root": str(worker_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint),
        "partial_manifest_path": str(partial),
        "partial_manifest_sha256": _sha256(partial),
        "config_sha256": manifest.get("config_sha256"),
        "policy_sha256": arguments.policy_sha256,
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
    if (
        marker.get("format_version") != MARKER_FORMAT
        or marker.get("fold_id") != fold_id
    ):
        raise ValueError(f"fold-{fold_id} marker identity mismatch")
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("partial_manifest_path", "partial_manifest_sha256"),
    ):
        artifact = Path(str(marker[path_key])).resolve()
        if not artifact.is_relative_to(run_root) or _sha256(artifact) != marker[hash_key]:
            raise ValueError(f"fold-{fold_id} artifact verification failed")
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
    if len(policies) != 1 or len(configs) != 1:
        raise ValueError("fold provenance disagrees")
    assembled = run_root / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    destination = assembled / "fold-set-v1"
    if destination.exists():
        raise FileExistsError(f"assembled fold set already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".fold-set.incomplete-", dir=assembled))
    records: list[dict[str, object]] = []
    try:
        for fold_id in range(3):
            marker = markers[fold_id]
            fold_dir = temporary / f"fold-{fold_id}"
            fold_dir.mkdir()
            checkpoint = fold_dir / "final.pt"
            partial = fold_dir / "partial-manifest.json"
            os.link(Path(str(marker["checkpoint_path"])), checkpoint)
            shutil.copyfile(Path(str(marker["partial_manifest_path"])), partial)
            records.append(
                {
                    "fold_id": fold_id,
                    "checkpoint_path": f"fold-{fold_id}/final.pt",
                    "checkpoint_bytes": marker["checkpoint_bytes"],
                    "checkpoint_sha256": marker["checkpoint_sha256"],
                    "partial_manifest_path": f"fold-{fold_id}/partial-manifest.json",
                    "partial_manifest_sha256": marker["partial_manifest_sha256"],
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
            "records": records,
        }
        manifest_sha256 = _atomic_json(
            temporary / "fold-set-manifest.json", manifest
        )
        (temporary / "fold-set-manifest.sha256").write_text(
            manifest_sha256 + "\n", encoding="ascii"
        )
        os.replace(temporary, destination)
        _atomic_json(
            assembled / "fold-set-pointer.json",
            {
                "format_version": SET_FORMAT,
                "path": "fold-set-v1/fold-set-manifest.json",
                "sha256": manifest_sha256,
            },
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
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

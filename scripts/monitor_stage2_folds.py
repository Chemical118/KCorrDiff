#!/usr/bin/env python3
"""Show porsche A100 availability and Stage 2 fold placement."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import subprocess
import time
from typing import Mapping, Sequence


NAMESPACE = "ws-md93se6gk3270"
TRAINING_JOB = "kcorrdiff-stage2-folds-porsche-v1"
INSTANCE = "stage2-folds-porsche-v1"


def _kubectl_json(*arguments: str) -> Mapping[str, object]:
    result = subprocess.run(
        ["kubectl", *arguments, "-o", "json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-1024:]
        raise RuntimeError(f"kubectl {' '.join(arguments)} failed: {detail}")
    raw = json.loads(result.stdout)
    if not isinstance(raw, Mapping):
        raise TypeError("kubectl JSON root is not a mapping")
    return raw


def _gpu_limit(pod: Mapping[str, object]) -> int:
    spec = pod.get("spec")
    if not isinstance(spec, Mapping):
        return 0
    total = 0
    containers = spec.get("containers", [])
    if not isinstance(containers, list):
        return 0
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        resources = container.get("resources", {})
        if not isinstance(resources, Mapping):
            continue
        limits = resources.get("limits", {})
        if isinstance(limits, Mapping):
            total += int(limits.get("nvidia.com/gpu", 0))
    return total


def snapshot() -> str:
    nodes = _kubectl_json("get", "nodes")
    pods = _kubectl_json("get", "pods", "--all-namespaces")
    namespace_jobs = _kubectl_json(
        "get", "jobs", "-n", NAMESPACE,
        "-l", f"app.kubernetes.io/instance={INSTANCE}"
    )
    node_rows: dict[str, dict[str, object]] = {}
    for node in nodes.get("items", []):  # type: ignore[union-attr]
        if not isinstance(node, Mapping):
            continue
        metadata = node.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        labels = metadata.get("labels", {})
        if not isinstance(labels, Mapping):
            continue
        count = labels.get("nvidia.com/gpu.count")
        if count is None or metadata.get("name") != "porsche":
            continue
        name = str(metadata.get("name"))
        node_rows[name] = {
            "capacity": int(count),
            "allocated": 0,
            "memory_mib": labels.get("nvidia.com/gpu.memory", "?"),
        }
    fold_rows: list[str] = []
    for pod in pods.get("items", []):  # type: ignore[union-attr]
        if not isinstance(pod, Mapping):
            continue
        metadata = pod.get("metadata", {})
        spec = pod.get("spec", {})
        status = pod.get("status", {})
        if not all(isinstance(value, Mapping) for value in (metadata, spec, status)):
            continue
        phase = str(status.get("phase", "Unknown"))
        node_name = str(spec.get("nodeName", ""))
        if phase not in {"Succeeded", "Failed"} and node_name in node_rows:
            node_rows[node_name]["allocated"] = int(
                node_rows[node_name]["allocated"]
            ) + _gpu_limit(pod)
        labels = metadata.get("labels", {})
        if not isinstance(labels, Mapping) or labels.get("job-name") != TRAINING_JOB:
            continue
        annotations = metadata.get("annotations", {})
        index = annotations.get("batch.kubernetes.io/job-completion-index", "?") if isinstance(annotations, Mapping) else "?"
        fold_rows.append(
            f"  fold={index} pod={metadata.get('name')} phase={phase} node={node_name or '-'}"
        )
    lines = [f"[{datetime.now().isoformat(timespec='seconds')}] porsche A100 availability"]
    for name in sorted(node_rows):
        row = node_rows[name]
        capacity = int(row["capacity"])
        allocated = int(row["allocated"])
        lines.append(
            f"  {name:8s} {row['memory_mib']}MiB allocated={allocated}/{capacity} free={capacity-allocated}"
        )
    lines.append("Stage 2 fold pods:")
    lines.extend(sorted(fold_rows) or ["  none"])
    lines.append("Jobs:")
    for job in namespace_jobs.get("items", []):  # type: ignore[union-attr]
        if not isinstance(job, Mapping):
            continue
        metadata = job.get("metadata", {})
        status = job.get("status", {})
        if isinstance(metadata, Mapping) and isinstance(status, Mapping):
            lines.append(
                f"  {metadata.get('name')}: active={status.get('active', 0)} "
                f"succeeded={status.get('succeeded', 0)} failed={status.get('failed', 0)}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    arguments = parser.parse_args(argv)
    if arguments.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        print(snapshot(), flush=True)
        if not arguments.watch:
            return 0
        print(flush=True)
        time.sleep(arguments.interval)


if __name__ == "__main__":
    raise SystemExit(main())

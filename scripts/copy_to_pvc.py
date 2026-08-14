#!/usr/bin/env python3
"""Resumably copy local files into a Kubernetes-mounted PVC.

``kubectl cp`` uses one long tar stream, which is fragile for multi-GiB data.
This utility sends bounded chunks with byte-addressed writes, resumes a partial
destination, and compares the final local/remote SHA-256 before moving on.
It never deletes a remote file or PVC.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Iterable, Iterator, Sequence


DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024


def _safe_remote_root(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ValueError("remote destination must be an absolute, non-root path")
    return path


def _kubectl_prefix(namespace: str, pod: str, container: str | None) -> list[str]:
    command = ["kubectl", "-n", namespace, "exec", "-i", pod]
    if container:
        command += ["-c", container]
    command += ["--"]
    return command


def _run(
    command: Sequence[str],
    *,
    data: bytes | None = None,
    timeout: int = 120,
    capture: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=data,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        check=True,
        timeout=timeout,
    )


def _remote_size(prefix: Sequence[str], destination: PurePosixPath) -> int:
    result = _run(
        [*prefix, "bash", "-ceu", 'test ! -e "$1" || stat -c %s -- "$1"', "--", str(destination)]
    )
    output = result.stdout.strip()
    return int(output) if output else 0


def _remote_hash(prefix: Sequence[str], destination: PurePosixPath) -> str:
    result = _run([*prefix, "sha256sum", str(destination)], timeout=1800)
    return result.stdout.decode().split()[0]


def _local_hash(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            block = stream.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in {None, 0}:
        raise EOFError(path)
    return digest.hexdigest()


def copy_file(
    source: Path,
    destination: PurePosixPath,
    *,
    prefix: Sequence[str],
    chunk_bytes: int,
    retries: int,
) -> None:
    source_size = source.stat().st_size
    _run([*prefix, "mkdir", "-p", str(destination.parent)])
    offset = _remote_size(prefix, destination)
    if offset > source_size:
        raise RuntimeError(
            f"remote file is larger than source and will not be overwritten: {destination}"
        )
    if offset and _remote_hash(prefix, destination) != _local_hash(source, limit=offset):
        raise RuntimeError(
            f"remote prefix hash mismatch; refusing unsafe resume: {destination}"
        )
    if offset == source_size and source_size:
        local_digest = _local_hash(source)
        remote_digest = _remote_hash(prefix, destination)
        if local_digest == remote_digest:
            print(f"verified cached {source} -> {destination} bytes={source_size}", flush=True)
            return
        raise RuntimeError(
            f"same-size destination hash mismatch; refusing overwrite: {destination}"
        )
    if source_size == 0:
        _run([*prefix, "touch", str(destination)])
        print(f"verified empty {source} -> {destination}", flush=True)
        return

    with source.open("rb") as stream:
        while offset < source_size:
            stream.seek(offset)
            block = stream.read(min(chunk_bytes, source_size - offset))
            if not block:
                raise EOFError(source)
            expected_end = offset + len(block)
            advanced = False
            for attempt in range(1, retries + 1):
                try:
                    # ``cat`` reliably drains Kubernetes' framed stdin stream.
                    # GNU ``dd bs=...`` can interpret a short API frame as a
                    # short input record and terminate successfully after only
                    # a prefix, so byte-positioned dd is intentionally avoided.
                    _run(
                        [*prefix, "bash", "-ceu", 'cat >> "$1"', "--", str(destination)],
                        data=block,
                        timeout=120,
                        capture=False,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    # The API stream can fail after the remote ``cat`` has
                    # already persisted a prefix.  Re-read its exact length;
                    # retrying the whole chunk would silently duplicate bytes.
                    remote_size = _remote_size(prefix, destination)
                    if offset < remote_size <= expected_end:
                        offset = remote_size
                        advanced = True
                        break
                    if remote_size != offset or attempt == retries:
                        raise RuntimeError(
                            f"chunk transfer failed at offset {offset}: {source}"
                        ) from error
                    time.sleep(min(2**attempt, 15))
                    continue
                remote_size = _remote_size(prefix, destination)
                if not offset < remote_size <= expected_end:
                    raise RuntimeError(
                        f"remote append made invalid progress {offset}->{remote_size}: "
                        f"{destination}"
                    )
                offset = remote_size
                advanced = True
                break
            if not advanced:
                raise RuntimeError(f"copy made no progress at offset {offset}: {source}")
            print(
                f"copied {source.name} {offset}/{source_size} "
                f"({100.0 * offset / max(source_size, 1):.1f}%)",
                flush=True,
            )
    local_digest = _local_hash(source)
    remote_size = _remote_size(prefix, destination)
    remote_digest = _remote_hash(prefix, destination)
    if remote_size != source_size or remote_digest != local_digest:
        raise RuntimeError(
            f"final verification failed for {destination}: "
            f"size={remote_size}/{source_size}, sha256={remote_digest}/{local_digest}"
        )
    print(f"verified {source} -> {destination} sha256={local_digest}", flush=True)


def iter_files(source: Path) -> Iterator[tuple[Path, Path]]:
    if source.is_file():
        yield source, Path(source.name)
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source symlinks are forbidden: {path}")
        if path.is_file():
            yield path, path.relative_to(source)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--namespace", default="ws-md93se6gk3270")
    parser.add_argument("--pod", default="kcorrdiff-stager")
    parser.add_argument("--container")
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.chunk_mib <= 0 or arguments.retries <= 0:
        raise ValueError("chunk size and retries must be positive")
    destination_root = _safe_remote_root(arguments.destination)
    prefix = _kubectl_prefix(arguments.namespace, arguments.pod, arguments.container)
    for source, relative in iter_files(arguments.source.resolve()):
        destination = destination_root.joinpath(*relative.parts)
        copy_file(
            source,
            destination,
            prefix=prefix,
            chunk_bytes=arguments.chunk_mib * 1024 * 1024,
            retries=arguments.retries,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

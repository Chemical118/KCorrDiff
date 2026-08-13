"""Lossless, memory-mapped cache for compressed per-timestamp NPZ archives.

CPrecNet distributes one compressed ``.npy`` member per timestamp.  Reopening
and inflating many members in every DataLoader worker is CPU- and metadata-
bound.  The cache converts each source archive once into a contiguous float32
``.npy`` shard plus an immutable timestamp index.  Workers memory-map shards,
so the kernel page cache is shared without duplicating the dataset in RAM.

The cache deliberately keeps float32 values.  It is an I/O layout change, not
a precision or model-size fallback.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap


FORMAT_VERSION = "kcorrdiff.npy-shards.v1"
FLOAT32_BYTES = np.dtype(np.float32).itemsize


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class CacheShard:
    source: Literal["target", "condition"]
    archive_name: str
    archive_size: int
    archive_sha256: str
    values_file: str
    timestamps_file: str
    samples: int
    shape: tuple[int, int]
    dtype: str
    values_sha256: str


@dataclass(frozen=True, slots=True)
class CacheManifest:
    format_version: str
    dtype: str
    shards: tuple[CacheShard, ...]
    unique_timestamps: int
    estimated_value_bytes: int

    def to_json(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "dtype": self.dtype,
            "shards": [asdict(shard) for shard in self.shards],
            "unique_timestamps": self.unique_timestamps,
            "estimated_value_bytes": self.estimated_value_bytes,
        }


def paired_archives(
    target_root: Path, condition_root: Path
) -> tuple[tuple[Path, Path], ...]:
    targets = sorted(target_root.glob("hsr_22_SEOUL_NON_UNI_*.npz"))
    targets = [path for path in targets if not path.name.endswith("_cond.npz")]
    if not targets:
        raise FileNotFoundError(f"no target archives under {target_root}")
    pairs: list[tuple[Path, Path]] = []
    for target in targets:
        condition = condition_root / f"{target.stem}_cond.npz"
        if not condition.is_file():
            raise FileNotFoundError(f"missing matching condition archive: {condition}")
        pairs.append((target, condition))
    orphan_names = {
        path.name.removesuffix("_cond.npz")
        for path in condition_root.glob("hsr_22_SEOUL_NON_UNI_*_cond.npz")
    } - {path.stem for path in targets}
    if orphan_names:
        raise ValueError(f"orphan condition archives: {sorted(orphan_names)}")
    return tuple(pairs)


def inspect_archive(path: Path) -> tuple[tuple[str, ...], tuple[int, int]]:
    with np.load(path, allow_pickle=False) as archive:
        timestamps = tuple(archive.files)
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError(f"archive keys are not chronological: {path}")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError(f"archive contains duplicate keys: {path}")
        if not timestamps:
            raise ValueError(f"empty archive: {path}")
        sample = archive[timestamps[0]]
        if sample.ndim != 2:
            raise ValueError(f"expected 2-D radar fields in {path}, got {sample.shape}")
        shape = tuple(map(int, sample.shape))
    return timestamps, shape  # type: ignore[return-value]


def estimate_cache_bytes(pairs: Sequence[tuple[Path, Path]]) -> int:
    """Exact value-array bytes, excluding small indexes/manifests."""

    total = 0
    for target, condition in pairs:
        target_keys, target_shape = inspect_archive(target)
        condition_keys, condition_shape = inspect_archive(condition)
        if target_keys != condition_keys:
            missing_target = len(set(condition_keys) - set(target_keys))
            missing_condition = len(set(target_keys) - set(condition_keys))
            raise ValueError(
                f"target/condition key mismatch for {target.stem}: "
                f"target_missing={missing_target}, condition_missing={missing_condition}"
            )
        total += len(target_keys) * (
            int(np.prod(target_shape)) + int(np.prod(condition_shape))
        ) * FLOAT32_BYTES
    return total


def _write_shard(task: tuple[str, str, str, str]) -> CacheShard:
    source, archive_string, output_string, prefix = task
    archive_path = Path(archive_string)
    output_dir = Path(output_string)
    timestamps, shape = inspect_archive(archive_path)
    values_name = f"{prefix}.npy"
    timestamps_name = f"{prefix}.timestamps.json"
    values_path = output_dir / values_name
    values = open_memmap(
        values_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(timestamps), *shape),
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        for index, timestamp in enumerate(timestamps):
            field = archive[timestamp]
            if field.shape != shape:
                raise ValueError(
                    f"shape changed in {archive_path}:{timestamp}: {field.shape} != {shape}"
                )
            if not np.issubdtype(field.dtype, np.number):
                raise TypeError(f"non-numeric radar field in {archive_path}:{timestamp}")
            values[index] = field.astype(np.float32, copy=False)
    values.flush()
    del values
    with (output_dir / timestamps_name).open("w", encoding="utf-8") as stream:
        json.dump(timestamps, stream, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return CacheShard(
        source=source,  # type: ignore[arg-type]
        archive_name=archive_path.name,
        archive_size=archive_path.stat().st_size,
        archive_sha256=_hash_file(archive_path),
        values_file=values_name,
        timestamps_file=timestamps_name,
        samples=len(timestamps),
        shape=shape,
        dtype="float32",
        values_sha256=_hash_file(values_path),
    )


def build_radar_cache(
    *,
    target_root: Path,
    condition_root: Path,
    output_dir: Path,
    workers: int = 2,
    reserve_bytes: int = 20 * 1024**3,
) -> CacheManifest:
    """Build and atomically publish a lossless radar cache.

    Existing destinations are never overwritten.  An interrupted build leaves
    only a sibling ``.incomplete-*`` directory which is safe to inspect or
    remove explicitly.
    """

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"cache destination already exists: {output_dir}")
    pairs = paired_archives(target_root.resolve(), condition_root.resolve())
    estimated = estimate_cache_bytes(pairs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir.parent).free
    if estimated + reserve_bytes > free:
        raise OSError(
            f"insufficient free space: need cache={estimated} + reserve={reserve_bytes}, "
            f"free={free}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )
    try:
        tasks: list[tuple[str, str, str, str]] = []
        for target, condition in pairs:
            period = target.stem.removeprefix("hsr_22_SEOUL_NON_UNI_")
            tasks.extend(
                [
                    ("target", str(target), str(temporary), f"target-{period}"),
                    (
                        "condition",
                        str(condition),
                        str(temporary),
                        f"condition-{period}",
                    ),
                ]
            )
        shards: list[CacheShard] = []
        if workers == 1:
            shards = [_write_shard(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [executor.submit(_write_shard, task) for task in tasks]
                for future in as_completed(futures):
                    shards.append(future.result())
        shards.sort(key=lambda item: (item.archive_name, item.source))
        target_timestamps: set[str] = set()
        for shard in shards:
            if shard.source == "target":
                target_timestamps.update(
                    json.loads((temporary / shard.timestamps_file).read_text())
                )
        manifest = CacheManifest(
            format_version=FORMAT_VERSION,
            dtype="float32",
            shards=tuple(shards),
            unique_timestamps=len(target_timestamps),
            estimated_value_bytes=estimated,
        )
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest.to_json(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
        return manifest
    except BaseException:
        # Keep the exact incomplete directory for forensic inspection/resume;
        # never remove potentially expensive work implicitly.
        raise


class RadarCache:
    """Read-only memory-mapped access with an O(1) timestamp index."""

    def __init__(self, root: Path, *, verify_hashes: bool = False) -> None:
        self.root = root.resolve()
        raw = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if raw.get("format_version") != FORMAT_VERSION or raw.get("dtype") != "float32":
            raise ValueError(f"unsupported radar cache at {self.root}")
        self._locations: dict[tuple[str, str], tuple[str, int]] = {}
        self._arrays: dict[str, np.memmap] = {}
        for shard in raw["shards"]:
            source = str(shard["source"])
            if source not in {"target", "condition"}:
                raise ValueError(f"unsupported radar cache source: {source!r}")
            values_file = str(shard["values_file"])
            if verify_hashes:
                expected = str(shard.get("values_sha256", ""))
                if not expected:
                    raise ValueError(
                        f"radar cache has no values_sha256 for {values_file}"
                    )
                if _hash_file(self.root / values_file) != expected:
                    raise ValueError(f"radar cache hash mismatch: {values_file}")
            timestamps = json.loads(
                (self.root / shard["timestamps_file"]).read_text(encoding="utf-8")
            )
            if len(timestamps) != int(shard["samples"]):
                raise ValueError(f"timestamp count mismatch in {shard['timestamps_file']}")
            for index, timestamp in enumerate(timestamps):
                key = (source, str(timestamp))
                if key in self._locations:
                    raise ValueError(f"duplicate cache timestamp: {key}")
                self._locations[key] = (values_file, index)

    def __contains__(self, item: tuple[str, str]) -> bool:
        return item in self._locations

    def timestamps(self, source: Literal["target", "condition"]) -> tuple[str, ...]:
        return tuple(sorted(timestamp for kind, timestamp in self._locations if kind == source))

    def _array(self, filename: str) -> np.memmap:
        result = self._arrays.get(filename)
        if result is None:
            result = np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
            self._arrays[filename] = result
        return result

    def read(self, source: Literal["target", "condition"], timestamp: str) -> np.ndarray:
        try:
            filename, index = self._locations[(source, timestamp)]
        except KeyError as error:
            raise KeyError(f"missing {source} timestamp {timestamp}; gaps are not dry") from error
        return np.asarray(self._array(filename)[index])

    def read_many(
        self, source: Literal["target", "condition"], timestamps: Sequence[str]
    ) -> np.ndarray:
        # Preserve requested ordering while grouping by shard to minimize mmap
        # page churn and Python dispatch.
        locations: list[tuple[str, int]] = []
        for timestamp in timestamps:
            try:
                locations.append(self._locations[(source, timestamp)])
            except KeyError as error:
                raise KeyError(f"missing {source} timestamp {timestamp}; gaps are not dry") from error
        output: list[np.ndarray | None] = [None] * len(locations)
        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_index, (filename, row) in enumerate(locations):
            grouped.setdefault(filename, []).append((output_index, row))
        for filename, entries in grouped.items():
            array = self._array(filename)
            rows = [row for _, row in entries]
            values = np.asarray(array[rows])
            for value, (output_index, _) in zip(values, entries, strict=True):
                output[output_index] = value
        return np.stack(output, axis=0)  # type: ignore[arg-type]

    def close(self) -> None:
        self._arrays.clear()

    def __getstate__(self) -> dict[str, object]:
        # DataLoader spawn workers reopen mmaps locally; file descriptors are
        # never serialized across process boundaries.
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    manifest = build_radar_cache(
        target_root=arguments.target_root,
        condition_root=arguments.condition_root,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
        reserve_bytes=int(arguments.reserve_gib * 1024**3),
    )
    print(json.dumps(manifest.to_json(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

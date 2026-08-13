from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.data.cache import (
    FORMAT_VERSION,
    RadarCache,
    build_radar_cache,
    estimate_cache_bytes,
    paired_archives,
)


def make_pair(root: Path, keys: tuple[str, ...] = ("202201010000", "202201010005")):
    target_root = root / "targets"
    condition_root = root / "conditions"
    target_root.mkdir()
    condition_root.mkdir()
    target = target_root / "hsr_22_SEOUL_NON_UNI_2022_1-2_9-12.npz"
    condition = condition_root / "hsr_22_SEOUL_NON_UNI_2022_1-2_9-12_cond.npz"
    np.savez_compressed(
        target, **{key: np.full((2, 3), index, np.float32) for index, key in enumerate(keys)}
    )
    np.savez_compressed(
        condition,
        **{key: np.full((4, 2), 10 + index, np.float32) for index, key in enumerate(keys)},
    )
    return target_root, condition_root


def test_cache_is_lossless_indexed_and_atomic(tmp_path: Path) -> None:
    target_root, condition_root = make_pair(tmp_path)
    destination = tmp_path / "cache"
    manifest = build_radar_cache(
        target_root=target_root,
        condition_root=condition_root,
        output_dir=destination,
        workers=1,
        reserve_bytes=0,
    )
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.unique_timestamps == 2
    assert destination.is_dir()
    assert not list(tmp_path.glob(".cache.incomplete-*"))
    raw = json.loads((destination / "manifest.json").read_text())
    assert raw["dtype"] == "float32"

    cache = RadarCache(destination)
    assert np.array_equal(cache.read("target", "202201010005"), np.ones((2, 3)))
    many = cache.read_many("condition", ["202201010005", "202201010000"])
    assert many.dtype == np.float32
    assert many.shape == (2, 4, 2)
    assert np.all(many[0] == 11) and np.all(many[1] == 10)
    with pytest.raises(KeyError, match="not dry"):
        cache.read("target", "202201010010")


def test_pair_contract_and_exact_estimate(tmp_path: Path) -> None:
    target_root, condition_root = make_pair(tmp_path)
    pairs = paired_archives(target_root, condition_root)
    assert estimate_cache_bytes(pairs) == 2 * (2 * 3 + 4 * 2) * 4
    with np.load(pairs[0][1]) as archive:
        values = {key: archive[key] for key in archive.files[:-1]}
    np.savez_compressed(pairs[0][1], **values)
    with pytest.raises(ValueError, match="key mismatch"):
        estimate_cache_bytes(pairs)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    target_root, condition_root = make_pair(tmp_path)
    destination = tmp_path / "cache"
    destination.mkdir()
    marker = destination / "owned-by-user"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        build_radar_cache(
            target_root=target_root,
            condition_root=condition_root,
            output_dir=destination,
            workers=1,
            reserve_bytes=0,
        )
    assert marker.read_text() == "keep"


def test_reader_can_verify_persisted_value_hashes(tmp_path: Path) -> None:
    target_root, condition_root = make_pair(tmp_path)
    destination = tmp_path / "cache"
    build_radar_cache(
        target_root=target_root,
        condition_root=condition_root,
        output_dir=destination,
        workers=1,
        reserve_bytes=0,
    )
    RadarCache(destination, verify_hashes=True)

    raw = json.loads((destination / "manifest.json").read_text())
    values_path = destination / raw["shards"][0]["values_file"]
    with values_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 0x01]))

    with pytest.raises(ValueError, match="hash mismatch"):
        RadarCache(destination, verify_hashes=True)

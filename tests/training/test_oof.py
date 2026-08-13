from __future__ import annotations

import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from kcorrdiff.training.oof import (
    OOF_COMPRESSED_ENCODING,
    OOF_COMPRESSED_FORMAT_VERSION,
    OOFArtifact,
    OOFCompressedBuild,
    OOFIdentity,
    OOFPartitionSpec,
    OOFShardWriter,
    compressed_oof_storage_plan,
)


def identity(index: int) -> OOFIdentity:
    return OOFIdentity(
        sample_id=f"sample-{index}",
        t0_utc=f"2020-01-0{index + 1}T00:00:00+00:00",
        lead_hours=0.5,
        condition_signature="era5",
        block_id=f"block-{index}",
        fold_id=index % 2,
        regression_checkpoint_sha256="a" * 64,
    )


def test_oof_writer_is_bounded_atomic_sharded_and_verified(tmp_path: Path) -> None:
    output = tmp_path / "oof"
    writer = OOFShardWriter(
        output,
        expected_items=3,
        maximum_bytes=3 * 2 * 2 * 3 * 4,
        height=2,
        width=3,
        shard_items=2,
        draw_manifest_sha256="b" * 64,
        target_builder_version="v1",
    )
    for index in range(3):
        writer.append(
            identity(index),
            np.full((2, 3), index / 3, np.float32),
            np.full((2, 3), index, np.float32),
        )
    digest = writer.close()
    assert digest == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    artifact = OOFArtifact(output, verify_hashes=True)
    entries = list(artifact)
    assert len(entries) == 3
    assert np.array_equal(entries[2].mu_z, np.full((2, 3), 2, np.float32))
    assert len(json.loads((output / "manifest.json").read_text())["shards"]) == 2


def test_oof_writer_rejects_budget_duplicate_precision_and_incomplete_close(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError, match="exceeding"):
        OOFShardWriter(
            tmp_path / "small",
            expected_items=2,
            maximum_bytes=1,
            height=2,
            width=2,
            draw_manifest_sha256="x",
            target_builder_version="v1",
        )
    writer = OOFShardWriter(
        tmp_path / "bad",
        expected_items=2,
        maximum_bytes=64,
        height=2,
        width=2,
        draw_manifest_sha256="x",
        target_builder_version="v1",
    )
    field = np.zeros((2, 2), np.float32)
    writer.append(identity(0), field, field)
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(identity(0), field, field)
    with pytest.raises(ValueError, match="item count"):
        writer.close()
    other = OOFShardWriter(
        tmp_path / "dtype",
        expected_items=1,
        maximum_bytes=32,
        height=2,
        width=2,
        draw_manifest_sha256="x",
        target_builder_version="v1",
    )
    with pytest.raises(TypeError, match="float32"):
        other.append(identity(0), field.astype(np.float16), field)


def test_oof_reader_verifies_manifest_companion_hash(tmp_path: Path) -> None:
    output = tmp_path / "oof"
    writer = OOFShardWriter(
        output,
        expected_items=0,
        maximum_bytes=1,
        height=2,
        width=2,
        draw_manifest_sha256="b" * 64,
        target_builder_version="v1",
    )
    writer.close()
    (output / "manifest.sha256").write_text("0" * 64 + "  manifest.json\n")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        OOFArtifact(output, verify_hashes=True)


def _compressed_universe(
    count: int, *, world_size: int = 2
) -> tuple[tuple[OOFIdentity, ...], tuple[OOFPartitionSpec, ...]]:
    if count % world_size:
        raise ValueError("test universe must divide evenly by world size")
    per_partition = count // world_size
    identities: list[OOFIdentity] = []
    partitions: list[OOFPartitionSpec] = []
    for rank in range(world_size):
        checkpoint = f"{rank + 1:x}" * 64
        start = len(identities)
        for offset in range(per_partition):
            index = start + offset
            identities.append(
                OOFIdentity(
                    sample_id=f"compressed-{index}",
                    t0_utc=f"2020-02-{index + 1:02d}T00:00:00+00:00",
                    lead_hours=0.5,
                    condition_signature="era5",
                    block_id=f"compressed-block-{index}",
                    fold_id=rank,
                    regression_checkpoint_sha256=checkpoint,
                )
            )
        partitions.append(
            OOFPartitionSpec(
                partition_id=f"fold-{rank}-rank-{rank}",
                fold_id=rank,
                rank=rank,
                world_size=world_size,
                row_indices=tuple(range(start, len(identities))),
                regression_checkpoint_sha256=checkpoint,
            )
        )
    return tuple(identities), tuple(partitions)


def _compressed_build(
    tmp_path: Path,
    identities: tuple[OOFIdentity, ...],
    partitions: tuple[OOFPartitionSpec, ...],
    *,
    maximum_bytes: int = 2_000_000,
    maximum_ratio: float = 10.0,
    shard_items: int = 2,
) -> OOFCompressedBuild:
    return OOFCompressedBuild(
        tmp_path / "compressed-artifact",
        identities=identities,
        partitions=partitions,
        maximum_compressed_bytes=maximum_bytes,
        maximum_compression_ratio=maximum_ratio,
        compression_probe_bytes=1,
        concurrent_writers=2,
        height=8,
        width=8,
        shard_items=shard_items,
        draw_manifest_sha256="a" * 64,
        target_builder_version="target-v1",
        config_sha256="b" * 64,
        launch_identity_sha256="c" * 64,
        initialize=True,
    )


def _fields(index: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(64, dtype=np.float32).reshape(8, 8)
    probability = np.clip((grid + index) / 128.0, 0.0, 1.0).astype(np.float32)
    mean = np.square((grid + index) / 32.0, dtype=np.float32)
    return probability, mean


def test_lossless_compressed_oof_is_single_payload_bitwise_and_bounded(
    tmp_path: Path,
) -> None:
    identities, partitions = _compressed_universe(8)
    build = _compressed_build(tmp_path, identities, partitions)
    for partition in partitions:
        writer = build.partition(partition.partition_id, commit_items=1)
        for index in partition.row_indices:
            writer.append(index, identities[index], *_fields(index))
        writer.finish()
    digest = build.seal()

    artifact = OOFArtifact(build.output_dir, verify_hashes=True)
    assert artifact.format_version == OOF_COMPRESSED_FORMAT_VERSION
    assert artifact.manifest["encoding"] == OOF_COMPRESSED_ENCODING
    assert digest == hashlib.sha256(
        (build.output_dir / "manifest.json").read_bytes()
    ).hexdigest()
    for index, entry in enumerate(artifact):
        expected_probability, expected_mean = _fields(index)
        assert np.array_equal(
            entry.occurrence_probability.view(np.uint32),
            expected_probability.view(np.uint32),
        )
        assert np.array_equal(entry.mu_z.view(np.uint32), expected_mean.view(np.uint32))
    assert not tuple(build.output_dir.rglob("*.active.npy"))
    assert not tuple(build.output_dir.rglob("*.receipts.jsonl"))
    actual_archives = sum(
        path.stat().st_size for path in build.output_dir.glob("fields-*.npz")
    )
    assert artifact.manifest["actual_compressed_bytes"] == actual_archives
    plan = compressed_oof_storage_plan(
        identities,
        partitions,
        maximum_compressed_bytes=2_000_000,
        concurrent_writers=2,
        height=8,
        width=8,
        shard_items=2,
    )
    assert plan.logical_dense_bytes == 8 * 2 * 8 * 8 * 4
    assert plan.peak_persistent_bytes < 2 * plan.logical_dense_bytes + 20_000_000


def test_compressed_oof_crash_resume_repairs_torn_tail_and_binds_launch(
    tmp_path: Path,
) -> None:
    identities, partitions = _compressed_universe(4, world_size=1)
    build = _compressed_build(
        tmp_path, identities, partitions, shard_items=4
    )
    writer = build.partition(partitions[0].partition_id, commit_items=2)
    for index in (0, 1):
        writer.append(index, identities[index], *_fields(index))
    writer.abort()
    journal = next(build.build_dir.rglob("*.receipts.jsonl"))
    with journal.open("ab") as stream:
        stream.write(b'{"row":2')

    resumed = build.partition(partitions[0].partition_id, commit_items=2)
    for index in range(4):
        resumed.append(index, identities[index], *_fields(index))
    resumed.finish()
    build.seal()
    assert len(list(OOFArtifact(build.output_dir, verify_hashes=True))) == 4

    with pytest.raises(ValueError, match="identity universe mismatch|provenance"):
        OOFCompressedBuild(
            build.output_dir,
            identities=identities,
            partitions=partitions,
            maximum_compressed_bytes=2_000_000,
            maximum_compression_ratio=10.0,
            compression_probe_bytes=1,
            concurrent_writers=2,
            height=8,
            width=8,
            shard_items=4,
            draw_manifest_sha256="a" * 64,
            target_builder_version="target-v1",
            config_sha256="b" * 64,
            launch_identity_sha256="d" * 64,
            initialize=True,
        )


def test_compressed_oof_rejects_missing_duplicate_and_corrupt_partition_rows(
    tmp_path: Path,
) -> None:
    identities, partitions = _compressed_universe(4, world_size=1)
    missing = (
        OOFPartitionSpec(
            partition_id=partitions[0].partition_id,
            fold_id=0,
            rank=0,
            world_size=1,
            row_indices=(0, 1, 2),
            regression_checkpoint_sha256="1" * 64,
        ),
    )
    with pytest.raises(ValueError, match="missing canonical"):
        _compressed_build(tmp_path / "missing", identities, missing)

    duplicate = (
        OOFPartitionSpec(
            partition_id="first",
            fold_id=0,
            rank=0,
            world_size=1,
            row_indices=(0, 1, 2),
            regression_checkpoint_sha256="1" * 64,
        ),
        OOFPartitionSpec(
            partition_id="second",
            fold_id=0,
            rank=0,
            world_size=1,
            row_indices=(2, 3),
            regression_checkpoint_sha256="1" * 64,
        ),
    )
    with pytest.raises(ValueError, match="duplicate canonical"):
        _compressed_build(tmp_path / "duplicate", identities, duplicate)

    build = _compressed_build(tmp_path / "journal", identities, partitions, shard_items=4)
    writer = build.partition(partitions[0].partition_id, commit_items=1)
    writer.append(0, identities[0], *_fields(0))
    writer.abort()
    journal = next(build.build_dir.rglob("*.receipts.jsonl"))
    journal.write_bytes(journal.read_bytes() + journal.read_bytes())
    with pytest.raises(ValueError, match="sequence mismatch"):
        corrupted = build.partition(partitions[0].partition_id)
        corrupted.append(0, identities[0], *_fields(0))


def test_compressed_oof_rank_partitions_write_concurrently(tmp_path: Path) -> None:
    identities, partitions = _compressed_universe(8)
    build = _compressed_build(tmp_path, identities, partitions, shard_items=2)

    def write_partition(partition: OOFPartitionSpec) -> None:
        writer = build.partition(partition.partition_id, commit_items=1)
        for index in partition.row_indices:
            writer.append(index, identities[index], *_fields(index))
        writer.finish()

    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(pool.map(write_partition, partitions))
    build.seal()
    assert [entry.identity for entry in OOFArtifact(build.output_dir)] == list(
        identities
    )


def test_compressed_oof_hard_cap_and_ratio_gate_abort_before_publish(
    tmp_path: Path,
) -> None:
    identities, partitions = _compressed_universe(2, world_size=1)
    build = _compressed_build(
        tmp_path / "cap",
        identities,
        partitions,
        maximum_bytes=100,
        maximum_ratio=100.0,
        shard_items=2,
    )
    writer = build.partition(partitions[0].partition_id)
    writer.append(0, identities[0], *_fields(0))
    with pytest.raises(OSError, match="hard byte cap"):
        writer.append(1, identities[1], *_fields(1))
    assert sum(path.stat().st_size for path in build.build_dir.glob("*.npz")) <= 100
    assert tuple(build.build_dir.rglob("*.active.npy"))

    gated = _compressed_build(
        tmp_path / "ratio",
        identities,
        partitions,
        maximum_bytes=2_000_000,
        maximum_ratio=0.0001,
        shard_items=2,
    )
    writer = gated.partition(partitions[0].partition_id)
    writer.append(0, identities[0], *_fields(0))
    with pytest.raises(OSError, match="compression ratio"):
        writer.append(1, identities[1], *_fields(1))

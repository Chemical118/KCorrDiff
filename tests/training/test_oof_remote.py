from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from kcorrdiff.training.oof import (
    OOF_REMOTE_FORMAT_VERSION,
    OOFArtifact,
    OOFCompressedBuild,
    OOFIdentity,
    OOFPartitionSpec,
)

from kcorrdiff.training.oof_remote import (
    DEFAULT_PVC_RESERVE_BYTES,
    PVCRemoteSpillController,
    RemoteShardReceipt,
    RemoteStoreIdentity,
    RsyncSSHShardStore,
)


class FakeStore:
    def __init__(self) -> None:
        self.identity = RemoteStoreIdentity(
            ssh_host="hyunwoo-home",
            remote_root="/hyunwoo/kcorrdiff/oof",
        )
        self.objects: dict[str, bytes] = {}
        self.fail_upload = False
        self.verify_calls = 0

    def upload_verified(
        self, source: Path, *, relative_path: str, expected_sha256: str
    ) -> RemoteShardReceipt:
        if self.fail_upload:
            raise OSError("injected upload failure")
        payload = source.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        existing = self.objects.setdefault(relative_path, payload)
        if existing != payload:
            raise ValueError("remote content mismatch")
        return RemoteShardReceipt(
            remote_store_sha256=self.identity.semantic_sha256,
            relative_path=relative_path,
            bytes=len(payload),
            sha256=expected_sha256,
        )

    def verify(self, receipt: RemoteShardReceipt) -> None:
        self.verify_calls += 1
        payload = self.objects.get(receipt.relative_path)
        if payload is None or len(payload) != receipt.bytes:
            raise ValueError("remote object missing")
        if hashlib.sha256(payload).hexdigest() != receipt.sha256:
            raise ValueError("remote object corrupt")

    def download_verified(
        self, receipt: RemoteShardReceipt, destination: Path
    ) -> Path:
        self.verify(receipt)
        destination.write_bytes(self.objects[receipt.relative_path])
        return destination


def _sealed_shard(
    root: Path, *, index: int, start: int, payload: bytes, intent: str
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    partition = root / ".partitions"
    partition.mkdir(exist_ok=True)
    filename = f"fields-{index:05d}.npz"
    shard = root / filename
    shard.write_bytes(payload)
    record = {
        "format_version": "kcorrdiff.regression-oof-compressed-partition.v1",
        "build_intent_sha256": intent,
        "partition_id": "fold-0-rank-0",
        "file": filename,
        "start": start,
        "items": 1,
        "shape": [1, 2, 2, 2],
        "logical_bytes": 32,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "array_sha256": "b" * 64,
        "row_sha256": ["c" * 64],
        "encoding": "npz-deflate-byte-shuffle-v1",
    }
    (partition / f"{filename}.json").write_text(json.dumps(record), encoding="utf-8")
    return shard


def test_spills_oldest_one_by_one_until_ten_gib_reserve_plus_next_write(
    tmp_path: Path,
) -> None:
    intent = "1" * 64
    first = _sealed_shard(tmp_path, index=0, start=0, payload=b"first", intent=intent)
    second = _sealed_shard(tmp_path, index=1, start=1, payload=b"second", intent=intent)
    store = FakeStore()
    free_values = iter(
        [
            DEFAULT_PVC_RESERVE_BYTES - 1,
            DEFAULT_PVC_RESERVE_BYTES + 50,
            DEFAULT_PVC_RESERVE_BYTES + 500,
        ]
    )
    controller = PVCRemoteSpillController(
        tmp_path,
        build_intent_sha256=intent,
        store=store,
        free_bytes=lambda _: next(free_values),
    )
    receipts = controller.ensure_headroom(next_write_bound_bytes=100)
    assert [item.relative_path for item in receipts] == [
        f"{intent}/fields-00000.npz",
        f"{intent}/fields-00001.npz",
    ]
    assert not first.exists() and not second.exists()
    assert store.verify_calls == 0
    assert controller.receipt_for("fields-00000.npz", verify_remote=True) == receipts[0]


def test_never_deletes_local_shard_when_upload_or_readback_fails(tmp_path: Path) -> None:
    intent = "2" * 64
    shard = _sealed_shard(tmp_path, index=0, start=0, payload=b"payload", intent=intent)
    store = FakeStore()
    store.fail_upload = True
    controller = PVCRemoteSpillController(
        tmp_path,
        build_intent_sha256=intent,
        store=store,
        free_bytes=lambda _: 0,
    )
    with pytest.raises(OSError, match="injected upload failure"):
        controller.ensure_headroom(next_write_bound_bytes=1)
    assert shard.is_file()
    assert not tuple((tmp_path / ".remote").glob("*.json"))


def test_crash_after_upload_before_receipt_is_idempotent(tmp_path: Path) -> None:
    intent = "3" * 64
    shard = _sealed_shard(tmp_path, index=0, start=0, payload=b"same", intent=intent)
    store = FakeStore()
    relative = f"{intent}/{shard.name}"
    store.objects[relative] = shard.read_bytes()
    controller = PVCRemoteSpillController(
        tmp_path,
        build_intent_sha256=intent,
        store=store,
        free_bytes=lambda _: DEFAULT_PVC_RESERVE_BYTES + 1,
    )
    receipt = controller.spill_one()
    assert receipt.relative_path == relative
    assert not shard.exists()
    assert controller.receipt_for(shard.name, verify_remote=True) == receipt


def test_remote_mutation_is_detected_and_receipt_is_provenance_bound(
    tmp_path: Path,
) -> None:
    intent = "4" * 64
    shard = _sealed_shard(tmp_path, index=0, start=0, payload=b"original", intent=intent)
    store = FakeStore()
    controller = PVCRemoteSpillController(
        tmp_path, build_intent_sha256=intent, store=store
    )
    receipt = controller.spill_one()
    store.objects[receipt.relative_path] = b"mutated"
    with pytest.raises(ValueError, match="remote object (missing|corrupt)"):
        controller.receipt_for(shard.name, verify_remote=True)

    path = tmp_path / ".remote" / f"{shard.name}.remote.json"
    raw = json.loads(path.read_text())
    raw["relative_path"] = "../escape"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert controller.receipt_for(shard.name, verify_remote=False) is not None


def test_no_spill_when_reserve_and_next_write_are_already_safe(tmp_path: Path) -> None:
    intent = "5" * 64
    shard = _sealed_shard(tmp_path, index=0, start=0, payload=b"local", intent=intent)
    store = FakeStore()
    controller = PVCRemoteSpillController(
        tmp_path,
        build_intent_sha256=intent,
        store=store,
        free_bytes=lambda _: DEFAULT_PVC_RESERVE_BYTES + 101,
    )
    assert controller.ensure_headroom(next_write_bound_bytes=100) == ()
    assert shard.is_file()
    assert not store.objects


def test_eager_mirror_copies_sealed_shards_and_retains_hot_tier(tmp_path: Path) -> None:
    intent = "6" * 64
    first = _sealed_shard(tmp_path, index=0, start=0, payload=b"first", intent=intent)
    second = _sealed_shard(tmp_path, index=1, start=1, payload=b"second", intent=intent)
    store = FakeStore()
    controller = PVCRemoteSpillController(
        tmp_path,
        build_intent_sha256=intent,
        store=store,
        free_bytes=lambda _: 10 * DEFAULT_PVC_RESERVE_BYTES,
    )

    receipts = controller.mirror_all()

    assert [receipt.relative_path for receipt in receipts] == [
        f"{intent}/fields-00000.npz",
        f"{intent}/fields-00001.npz",
    ]
    assert first.is_file() and second.is_file()
    assert len(store.objects) == 2


def test_rsync_identity_and_command_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SSH host alias"):
        RemoteStoreIdentity(
            ssh_host="bad host",
            remote_root="/hyunwoo/oof",
        )

    identity = RemoteStoreIdentity(
        ssh_host="hyunwoo-home",
        remote_root="/hyunwoo/kcorrdiff/oof",
    )
    assert identity.transport == "rsync+ssh"
    assert identity.remote_root == "/hyunwoo/kcorrdiff/oof"
    store = RsyncSSHShardStore(
        ssh_host=identity.ssh_host,
        remote_root=identity.remote_root,
        ssh_config=tmp_path / "config",
        timeout_seconds=600,
    )
    assert "ControlPersist=600" in store._rsync_shell()


def test_compressed_build_spills_each_shard_and_reads_v3_bitwise(
    tmp_path: Path,
) -> None:
    checkpoint = "d" * 64
    identities = tuple(
        OOFIdentity(
            sample_id=f"remote-{index}",
            t0_utc=f"2020-01-0{index + 1}T00:00:00+00:00",
            lead_hours=0.5,
            condition_signature="era5",
            block_id=f"block-{index}",
            fold_id=0,
            regression_checkpoint_sha256=checkpoint,
        )
        for index in range(2)
    )
    partition = OOFPartitionSpec(
        partition_id="fold-0-rank-0",
        fold_id=0,
        rank=0,
        world_size=1,
        row_indices=(0, 1),
        regression_checkpoint_sha256=checkpoint,
    )
    store = FakeStore()
    build = OOFCompressedBuild(
        tmp_path / "artifact",
        identities=identities,
        partitions=(partition,),
        maximum_compressed_bytes=1_000_000,
        maximum_compression_ratio=10.0,
        compression_probe_bytes=1,
        concurrent_writers=1,
        height=2,
        width=2,
        shard_items=1,
        draw_manifest_sha256="a" * 64,
        target_builder_version="target-v1",
        config_sha256="b" * 64,
        launch_identity_sha256="c" * 64,
        remote_store=store,
        local_reserve_bytes=1,
        initialize=True,
    )
    # Simulate a filesystem that has headroom only while no sealed shard is
    # resident.  This forces precisely one verified spill after each seal.
    build.remote_spill = PVCRemoteSpillController(
        build.build_dir,
        build_intent_sha256=build.intent_sha256,
        store=store,
        local_reserve_bytes=1,
        free_bytes=lambda root: (
            0 if tuple(root.glob("fields-*.npz")) else 1_000_000
        ),
    )
    expected: list[tuple[np.ndarray, np.ndarray]] = []
    writer = build.partition(partition.partition_id)
    for index, identity in enumerate(identities):
        probability = np.full((2, 2), 0.25 + index * 0.5, dtype=np.float32)
        mean = np.full((2, 2), 1.5 + index, dtype=np.float32)
        expected.append((probability, mean))
        writer.append(index, identity, probability, mean)
    writer.finish()
    build.seal()

    assert not tuple(build.output_dir.glob("fields-*.npz"))
    artifact = OOFArtifact(
        build.output_dir, verify_hashes=True, remote_store=store
    )
    assert artifact.format_version == OOF_REMOTE_FORMAT_VERSION
    assert all(shard["storage"] == "remote_rsync" for shard in artifact.manifest["shards"])
    for entry, (probability, mean) in zip(artifact, expected, strict=True):
        assert np.array_equal(entry.occurrence_probability, probability)
        assert np.array_equal(entry.mu_z, mean)

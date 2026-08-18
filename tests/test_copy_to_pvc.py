from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess

import pytest

from scripts import copy_to_pvc


def test_remote_root_is_an_unpinned_user_path() -> None:
    assert str(copy_to_pvc._safe_remote_root("/workspace/data")) == "/workspace/data"
    assert str(copy_to_pvc._safe_remote_root("relative")) == "relative"
    assert str(copy_to_pvc._safe_remote_root("/")) == "/"


def test_copy_resumes_a_partially_persisted_failed_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"0123456789abcdef"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    remote = bytearray(payload[:3])
    first_append = True

    def remote_size(*_args: object) -> int:
        return len(remote)

    def run(command: list[str], *, data: bytes | None = None, **_kwargs: object):
        nonlocal first_append
        if "cat >> \"$1\"" in command:
            assert data is not None
            if first_append:
                first_append = False
                remote.extend(data[:2])
                raise subprocess.TimeoutExpired(command, 1)
            remote.extend(data)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(copy_to_pvc, "_remote_size", remote_size)
    monkeypatch.setattr(copy_to_pvc, "_run", run)
    monkeypatch.setattr(copy_to_pvc.time, "sleep", lambda _seconds: None)

    copy_to_pvc.copy_file(
        source,
        PurePosixPath("/workspace/data/source.bin"),
        prefix=("kubectl", "exec"),
        chunk_bytes=6,
        retries=3,
    )
    assert bytes(remote) == payload


def test_copy_resumes_by_size_without_hash_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected payload")
    remote = bytearray(b"evil")
    monkeypatch.setattr(copy_to_pvc, "_remote_size", lambda *_args: len(remote))
    def append(command: list[str], *, data: bytes | None = None, **_kwargs: object):
        if "cat >> \"$1\"" in command:
            assert data is not None
            remote.extend(data)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(copy_to_pvc, "_run", append)
    copy_to_pvc.copy_file(
        source,
        PurePosixPath("/workspace/data/source.bin"),
        prefix=("kubectl", "exec"),
        chunk_bytes=4,
        retries=2,
    )
    assert len(remote) == source.stat().st_size

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import subprocess

import pytest

from scripts import copy_to_pvc


def test_remote_root_must_be_absolute_bounded_path() -> None:
    assert str(copy_to_pvc._safe_remote_root("/workspace/data")) == "/workspace/data"
    for invalid in ("relative", "/", "/workspace/../other"):
        with pytest.raises(ValueError):
            copy_to_pvc._safe_remote_root(invalid)


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

    def remote_hash(*_args: object) -> str:
        return hashlib.sha256(remote).hexdigest()

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
    monkeypatch.setattr(copy_to_pvc, "_remote_hash", remote_hash)
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


def test_copy_rejects_mismatched_remote_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected payload")
    monkeypatch.setattr(copy_to_pvc, "_remote_size", lambda *_args: 4)
    monkeypatch.setattr(
        copy_to_pvc,
        "_remote_hash",
        lambda *_args: hashlib.sha256(b"evil").hexdigest(),
    )
    monkeypatch.setattr(
        copy_to_pvc,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, b"", b""),
    )

    with pytest.raises(RuntimeError, match="prefix hash mismatch"):
        copy_to_pvc.copy_file(
            source,
            PurePosixPath("/workspace/data/source.bin"),
            prefix=("kubectl", "exec"),
            chunk_bytes=4,
            retries=2,
        )

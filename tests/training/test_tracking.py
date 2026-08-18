from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kcorrdiff.training.tracking import WandbAuditRun, initialize_tracking


class Backend:
    def __init__(self) -> None:
        self.values = []
        self.exit_code = None

    def log(self, values, *, step: int) -> None:
        self.values.append((values, step))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


def test_audit_run_is_durable_monotonic_and_secret_safe(tmp_path: Path) -> None:
    backend = Backend()
    path = tmp_path / "metrics.jsonl"
    stream = path.open("x", encoding="utf-8")
    run = WandbAuditRun(backend, stream, "run")
    run.log({"loss": 1.25}, step=3)
    with pytest.raises(ValueError, match="monotonic"):
        run.log({"loss": 1.0}, step=2)
    with pytest.raises(ValueError, match="secret"):
        run.log({"api_key_copy": 1}, step=4)
    run.finish(exit_code=0)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["metrics"] == {"loss": 1.25}
    assert records[-1]["event"] == "finish"
    assert backend.exit_code == 0


def test_nonzero_rank_never_imports_or_initializes_wandb(tmp_path: Path) -> None:
    run = initialize_tracking(
        enabled=True,
        rank=1,
        run_dir=tmp_path,
        run_id="run",
        project="project",
        job_type="job",
        config={},
        config_sha256="a" * 64,
    )
    run.log({"ignored": 1.0}, step=0)
    run.finish()
    assert not list(tmp_path.iterdir())


def test_resume_appends_same_run_and_restores_monotonic_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backends: list[Backend] = []
    calls: list[dict[str, object]] = []

    def initialize(**kwargs: object) -> Backend:
        calls.append(dict(kwargs))
        backend = Backend()
        backends.append(backend)
        return backend

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=initialize,
            Settings=lambda **kwargs: dict(kwargs),
        ),
    )
    kwargs = {
        "enabled": True,
        "rank": 0,
        "run_dir": tmp_path,
        "run_id": "stable-run",
        "project": "project",
        "job_type": "job",
        "config": {"seed": 1},
        "config_sha256": "a" * 64,
        "mode": "offline",
        "resume": "allow",
    }
    first = initialize_tracking(**kwargs)
    first.log({"loss": 2.0}, step=7)
    first.finish()

    second = initialize_tracking(**kwargs)
    assert second.last_step == 7
    with pytest.raises(ValueError, match="monotonic"):
        second.log({"loss": 1.5}, step=6)
    second.log({"loss": 1.0}, step=8)
    second.finish()

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "start",
        "metrics",
        "finish",
        "resume",
        "metrics",
        "finish",
    ]
    assert all(record["run_id"] == "stable-run" for record in records)
    assert calls[0]["resume"] == calls[1]["resume"] == "allow"


def test_distinct_backend_runs_share_one_logical_audit_and_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def initialize(**kwargs: object) -> Backend:
        calls.append(dict(kwargs))
        return Backend()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=initialize, Settings=lambda **kwargs: dict(kwargs)),
    )
    kwargs = {
        "enabled": True,
        "rank": 0,
        "run_dir": tmp_path,
        "run_id": "logical-run",
        "project": "project",
        "job_type": "job",
        "config": {},
        "config_sha256": "a" * 64,
        "mode": "offline",
        "resume": "allow",
    }
    first = initialize_tracking(**kwargs, backend_run_id="pod-a")
    first.log({"loss": 2.0}, step=0)
    first.finish()
    second = initialize_tracking(**kwargs, backend_run_id="pod-b")
    assert second.last_step == 0
    second.log({"loss": 1.0}, step=1)
    second.finish()

    assert [call["id"] for call in calls] == ["pod-a", "pod-b"]
    assert [call["group"] for call in calls] == ["logical-run", "logical-run"]
    assert [call["resume"] for call in calls] == ["never", "never"]


def test_resume_rejects_foreign_run_but_tolerates_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: Backend(),
            Settings=lambda **kwargs: dict(kwargs),
        ),
    )
    base = dict(
        enabled=True,
        rank=0,
        run_dir=tmp_path,
        run_id="run-a",
        project="project",
        job_type="job",
        config={},
        config_sha256="a" * 64,
        launch_identity_sha256="c" * 64,
        mode="offline",
        resume="allow",
    )
    run = initialize_tracking(**base)
    run.finish()
    # Interleaving a different run into the same audit is a real bug.
    with pytest.raises(ValueError, match="run_id"):
        initialize_tracking(**{**base, "run_id": "run-b"})
    # Config/launch identity changes are informational and never block resume.
    run = initialize_tracking(
        **{**base, "config_sha256": "b" * 64, "launch_identity_sha256": "d" * 64}
    )
    run.finish()


def test_external_log_failure_keeps_durable_monotonic_resume_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingBackend(Backend):
        def log(self, values, *, step: int) -> None:
            del values, step
            raise RuntimeError("external backend unavailable")

    backends = iter((FailingBackend(), Backend()))
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: next(backends),
            Settings=lambda **kwargs: dict(kwargs),
        ),
    )
    arguments = dict(
        enabled=True,
        rank=0,
        run_dir=tmp_path,
        run_id="durable-run",
        project="project",
        job_type="job",
        config={"seed": 1},
        config_sha256="a" * 64,
        mode="offline",
        resume="allow",
    )
    first = initialize_tracking(**arguments)
    with pytest.raises(RuntimeError, match="external backend"):
        first.log({"loss": 1.0}, step=5)
    assert first.last_step == 5
    first.finish(exit_code=1)

    resumed = initialize_tracking(**arguments)
    assert resumed.last_step == 5
    resumed.log({"loss": 0.5}, step=6)
    resumed.finish()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record.get("step") for record in records if record["event"] == "metrics"] == [
        5,
        6,
    ]

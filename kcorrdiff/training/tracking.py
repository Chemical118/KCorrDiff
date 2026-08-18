"""Rank-zero experiment tracking with a PVC-local, append-only audit log."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Literal, Mapping, Protocol


class TrackingRun(Protocol):
    last_step: int
    finished: bool

    def log(self, data: Mapping[str, object], *, step: int) -> None: ...
    def finish(self, *, exit_code: int = 0) -> None: ...


def _safe_metric_value(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tracking metrics must be finite")
        return value
    try:
        scalar = float(value)  # NumPy scalar / zero-dimensional tensor.
    except (TypeError, ValueError) as error:
        raise TypeError(f"unsupported tracking metric type: {type(value).__name__}") from error
    if not math.isfinite(scalar):
        raise ValueError("tracking metrics must be finite")
    return scalar


class _NullRun:
    last_step = -1
    finished = False

    def log(self, data: Mapping[str, object], *, step: int) -> None:
        return None

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = True


def _read_resume_state(
    audit_path: Path,
    *,
    run_id: str,
) -> int:
    """Read the last metric step from an existing append-only audit."""

    last_step = -1
    with audit_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"tracking audit has invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise TypeError("tracking audit records must be mappings")
            if record.get("run_id") != run_id:
                raise ValueError("tracking audit run_id mismatch")
            event = record.get("event")
            if event in {"start", "resume"}:
                pass
            elif event == "metrics":
                step = record.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step < last_step:
                    raise ValueError("tracking audit metric steps are not monotonic")
                last_step = step
            elif event == "finish":
                step = record.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step != last_step:
                    raise ValueError("tracking audit finish step disagrees with metric history")
            else:
                raise ValueError(f"unsupported tracking audit event: {event!r}")
    return last_step


def _write_lifecycle_record(
    stream: object,
    *,
    event: Literal["start", "resume"],
    run_id: str,
    config_sha256: str,
    launch_identity_sha256: str | None,
    step: int,
) -> None:
    stream.write(  # type: ignore[attr-defined]
        json.dumps(
            {
                "event": event,
                "config_sha256": config_sha256,
                "launch_identity_sha256": launch_identity_sha256,
                "run_id": run_id,
                "step": step,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()  # type: ignore[attr-defined]
    os.fsync(stream.fileno())  # type: ignore[attr-defined]


@dataclass(slots=True)
class WandbAuditRun:
    """Mirror every W&B metric into a durable JSONL file on the PVC."""

    backend_run: object
    audit_stream: object
    run_id: str
    last_step: int = -1
    finished: bool = False

    def log(self, data: Mapping[str, object], *, step: int) -> None:
        if self.finished:
            raise RuntimeError("tracking run is already finished")
        if isinstance(step, bool) or step < self.last_step:
            raise ValueError("tracking step must be monotonic")
        safe = {str(key): _safe_metric_value(value) for key, value in data.items()}
        if any("api_key" in key.lower() or "secret" in key.lower() for key in safe):
            raise ValueError("secret-like fields are forbidden in tracking metrics")
        record = {
            "event": "metrics",
            "monotonic_time_seconds": time.monotonic(),
            "run_id": self.run_id,
            "step": int(step),
            "metrics": safe,
        }
        self.audit_stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.audit_stream.flush()
        os.fsync(self.audit_stream.fileno())
        # Advance the durable/audit cursor before the external call.  If W&B
        # fails, the orchestrator aborts all ranks, but the append-only audit
        # still resumes from the attempted metric step without being overwritten.
        self.last_step = step
        self.backend_run.log(safe, step=step)

    def finish(self, *, exit_code: int = 0) -> None:
        if self.finished:
            return
        record = {
            "event": "finish",
            "exit_code": int(exit_code),
            "run_id": self.run_id,
            "step": self.last_step,
        }
        self.audit_stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.audit_stream.flush()
        os.fsync(self.audit_stream.fileno())
        self.audit_stream.close()
        self.finished = True
        # The local terminal record is authoritative.  Mark the wrapper closed
        # before calling the external service so a backend exception cannot
        # append a duplicate finish record during orchestrator cleanup.
        self.backend_run.finish(exit_code=exit_code)


def initialize_tracking(
    *,
    enabled: bool,
    rank: int,
    run_dir: Path,
    run_id: str,
    project: str,
    job_type: str,
    config: Mapping[str, object],
    config_sha256: str,
    launch_identity_sha256: str | None = None,
    mode: str = "online",
    resume: Literal["never", "allow"] = "never",
    backend_run_id: str | None = None,
) -> TrackingRun:
    """Initialize W&B only on rank zero and never expose its API key."""

    if rank != 0 or not enabled:
        return _NullRun()
    if not run_id or not project or not job_type or not config_sha256:
        raise ValueError("tracking identity/provenance fields must be non-empty")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("unsupported W&B mode")
    if resume not in {"never", "allow"}:
        raise ValueError("tracking resume must be 'never' or 'allow'")
    if backend_run_id is not None and not backend_run_id:
        raise ValueError("W&B backend run ID must be non-empty when provided")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for online tracking")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("the training image does not contain wandb") from error
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "metrics.jsonl"
    existed = audit_path.exists()
    if existed and resume == "never":
        raise FileExistsError(f"tracking audit already exists: {audit_path}")
    last_step = (
        _read_resume_state(audit_path, run_id=run_id) if existed else -1
    )
    public_config = dict(config)
    public_config["config_sha256"] = config_sha256
    effective_backend_run_id = backend_run_id or run_id
    public_config["logical_run_id"] = run_id
    public_config["wandb_backend_run_id"] = effective_backend_run_id
    if launch_identity_sha256 is not None:
        public_config["launch_identity_sha256"] = launch_identity_sha256
    if any("api_key" in str(key).lower() or "secret" in str(key).lower() for key in public_config):
        raise ValueError("secret-like config fields are forbidden")
    try:
        backend = wandb.init(
            project=project,
            id=effective_backend_run_id,
            name=effective_backend_run_id,
            group=(run_id if effective_backend_run_id != run_id else None),
            job_type=job_type,
            dir=str(run_dir),
            config=public_config,
            mode=mode,
            resume=(resume if effective_backend_run_id == run_id else "never"),
            settings=wandb.Settings(start_method="thread"),
        )
        if backend is None:
            raise RuntimeError("wandb.init returned no run")
        stream = audit_path.open("a" if existed else "x", encoding="utf-8")
        _write_lifecycle_record(
            stream,
            event="resume" if existed else "start",
            run_id=run_id,
            config_sha256=config_sha256,
            launch_identity_sha256=launch_identity_sha256,
            step=last_step,
        )
    except BaseException:
        if "stream" in locals():
            stream.close()
        if not existed:
            audit_path.unlink(missing_ok=True)
        if "backend" in locals() and backend is not None:
            backend.finish(exit_code=1)
        raise
    return WandbAuditRun(backend, stream, run_id, last_step=last_step)


__all__ = ["TrackingRun", "WandbAuditRun", "initialize_tracking"]

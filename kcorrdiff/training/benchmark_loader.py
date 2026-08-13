"""Strict real-cache DataLoader and pinned host-to-device benchmark CLI.

The CLI is intentionally narrower than a training entry point.  It validates
the immutable radar/ERA/static artifacts, reads a pre-materialized draw
manifest, and measures a declared loader grid without reducing spatial sizes,
model-width metadata, precision, or device requirements.  Results are first
published as JSONL and then committed by an atomic summary JSON on the PVC.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Mapping, Sequence

import torch
import yaml

from kcorrdiff.data.sampling import read_draw_manifest
from kcorrdiff.training.cache_benchmark import (
    BenchmarkGrid,
    CacheBenchmarkContract,
    CacheDatasetSpec,
    PRODUCTION_CONTEXT_WIDTHS,
    PRODUCTION_ERA_LATENT_CHANNELS,
    PRODUCTION_ERA_SHAPE,
    PRODUCTION_RADAR_SHAPE,
    PRODUCTION_TARGET_WIDTHS,
    run_benchmark_grid,
    validate_cache_artifacts,
)


SUMMARY_FORMAT = "kcorrdiff.loader-benchmark-summary.v1"
RESULT_FORMAT = "kcorrdiff.loader-benchmark-result.v1"
_FALLBACK_ENVIRONMENT = (
    "KCORRDIFF_ALLOW_CPU_FALLBACK",
    "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK",
    "KCORRDIFF_ALLOW_PRECISION_FALLBACK",
    "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK",
)


def _csv_integers(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    try:
        converted = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    return converted


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--era5-cache-root", type=Path, required=True)
    parser.add_argument("--draw-manifest", type=Path, required=True)
    parser.add_argument("--static-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--target-widths", type=_csv_integers, required=True)
    parser.add_argument("--context-widths", type=_csv_integers, required=True)
    parser.add_argument("--era-latent-channels", type=int, required=True)
    parser.add_argument("--era-grid-size", type=int, required=True)
    parser.add_argument("--batch-sizes", type=_csv_integers, required=True)
    parser.add_argument("--worker-counts", type=_csv_integers, required=True)
    parser.add_argument("--prefetch-factors", type=_csv_integers, required=True)
    parser.add_argument("--warmup-batches", type=int, required=True)
    parser.add_argument("--measured-batches", type=int, required=True)
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("auto", "disabled", "offline", "online"),
        default="auto",
        help="auto enables online logging only when WANDB_API_KEY is present",
    )
    parser.add_argument(
        "--verify-cache-hashes",
        action="store_true",
        help="hash every cache mmap in each worker (normally too expensive)",
    )
    return parser.parse_args(argv)


def _environment_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"invalid boolean environment value: {value!r}")


def validate_strict_arguments(
    arguments: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> CacheBenchmarkContract:
    """Reject every precision, device, width, and spatial fallback."""

    environment = os.environ if environ is None else environ
    if arguments.device not in {"cuda", "cuda:0"}:
        raise ValueError("production loader benchmark requires device cuda:0")
    if arguments.precision != "float32":
        raise ValueError("production loader benchmark requires float32")
    if not arguments.disable_tf32:
        raise ValueError("--disable-tf32 is required")
    if not arguments.fail_on_fallback:
        raise ValueError("--fail-on-fallback is required")
    if tuple(arguments.target_widths) != PRODUCTION_TARGET_WIDTHS:
        raise ValueError("target widths must remain full production widths")
    if tuple(arguments.context_widths) != PRODUCTION_CONTEXT_WIDTHS:
        raise ValueError("context widths must remain full production widths")
    if arguments.era_latent_channels != PRODUCTION_ERA_LATENT_CHANNELS:
        raise ValueError("ERA latent width must remain 128")
    if arguments.era_grid_size != PRODUCTION_ERA_SHAPE[0]:
        raise ValueError("ERA grid must remain 33x33")
    for name in _FALLBACK_ENVIRONMENT:
        raw = environment.get(name)
        if raw is not None and _environment_flag(raw):
            raise ValueError(f"fallback environment flag must be disabled: {name}")
    expected_environment = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    for name, expected in expected_environment.items():
        raw = environment.get(name)
        if raw is not None and raw.strip().lower() != expected:
            raise ValueError(
                f"strict environment contract mismatch for {name}: {raw!r}"
            )
    return CacheBenchmarkContract(
        radar_shape=PRODUCTION_RADAR_SHAPE,
        era_shape=PRODUCTION_ERA_SHAPE,
        target_widths=tuple(arguments.target_widths),
        context_widths=tuple(arguments.context_widths),
        era_latent_channels=arguments.era_latent_channels,
        precision=arguments.precision,
        tf32_enabled=False,
    )


def _load_config(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark config: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("benchmark config must be a YAML mapping")
    return loaded


_MISSING = object()


def _nested(config: Mapping[str, object], path: str) -> object:
    value: object = config
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _normal_form(value: object) -> object:
    if isinstance(value, list):
        return tuple(_normal_form(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normal_form(item) for item in value)
    return value


def validate_config_contract(
    config: Mapping[str, object],
    *,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Validate every benchmark-relevant config field which is declared."""

    expected: dict[str, object] = {
        "runtime.target_widths": tuple(arguments.target_widths),
        "runtime.context_widths": tuple(arguments.context_widths),
        "runtime.era_latent_channels": arguments.era_latent_channels,
        "runtime.era_grid_size": arguments.era_grid_size,
        "runtime.precision": "float32",
        "runtime.tf32": False,
        "runtime.allow_cpu_fallback": False,
        "runtime.allow_model_width_fallback": False,
        "runtime.allow_precision_fallback": False,
        "runtime.allow_era_grid_fallback": False,
        "data.radar_timezone": "Asia/Seoul",
        "data.history_frames": 12,
        "data.future_target_frames": 7,
        "data.target_shape": PRODUCTION_RADAR_SHAPE,
        "data.context_shape": PRODUCTION_RADAR_SHAPE,
        "data.era_shape": PRODUCTION_ERA_SHAPE,
        "data.mmap_lazy_per_worker": True,
        "data.pin_memory": True,
        "data.persistent_workers": True,
        "loader_tuning.candidates.batch_sizes": tuple(arguments.batch_sizes),
        "loader_tuning.candidates.worker_counts": tuple(arguments.worker_counts),
        "loader_tuning.candidates.prefetch_factors": tuple(
            arguments.prefetch_factors
        ),
    }
    candidate_paths = {
        "loader_tuning.candidates.batch_sizes",
        "loader_tuning.candidates.worker_counts",
        "loader_tuning.candidates.prefetch_factors",
    }
    validated: dict[str, object] = {}
    for path, wanted in expected.items():
        actual = _nested(config, path)
        if actual is _MISSING:
            raise ValueError(
                f"config is missing strict benchmark contract field {path}"
            )
        normalized_actual = _normal_form(actual)
        normalized_wanted = _normal_form(wanted)
        if path in candidate_paths:
            if not isinstance(normalized_actual, tuple):
                raise TypeError(f"config candidate grid must be a sequence at {path}")
            positions = {value: index for index, value in enumerate(normalized_actual)}
            if any(value not in positions for value in normalized_wanted):
                raise ValueError(
                    f"CLI grid contains an undeclared candidate at {path}"
                )
            indices = tuple(positions[value] for value in normalized_wanted)
            if indices != tuple(sorted(indices)):
                raise ValueError(f"CLI grid reorders candidates at {path}")
        elif normalized_actual != normalized_wanted:
            raise ValueError(
                f"config contradicts strict benchmark contract at {path}: "
                f"expected {wanted!r}, got {actual!r}"
            )
        validated[path] = wanted
    return validated


def configure_cuda_device(device_text: str) -> torch.device:
    """Configure FP32 CUDA execution and assert exactly one visible GPU."""

    device = torch.device(device_text)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("benchmark device must be the sole visible CUDA GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one visible CUDA GPU, got "
            f"{torch.cuda.device_count()}"
        )
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("PyTorch default dtype must remain float32")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("failed to disable TF32")
    return torch.device("cuda:0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_provenance(arguments: argparse.Namespace) -> dict[str, object]:
    paths = {
        "config": arguments.config,
        "radar_manifest": arguments.radar_cache_root / "manifest.json",
        "era5_manifest": arguments.era5_cache_root / "manifest.json",
        "draw_manifest": arguments.draw_manifest,
        "static_npz": arguments.static_path,
    }
    result: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_bytes(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def derive_jsonl_path(summary_path: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    if summary_path.suffix.lower() == ".json":
        return summary_path.with_suffix(".jsonl")
    return summary_path.with_name(summary_path.name + ".jsonl")


def preflight_output_paths(*, jsonl_path: Path, summary_path: Path) -> None:
    """Fail before a long benchmark if its immutable destinations are unsafe."""

    if jsonl_path.resolve() == summary_path.resolve():
        raise ValueError("JSONL and summary destinations must differ")
    for path in (jsonl_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark output: {path}")


def publish_results_atomic(
    *,
    jsonl_path: Path,
    summary_path: Path,
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> str:
    """Publish JSONL first and summary last as the atomic completion marker."""

    preflight_output_paths(jsonl_path=jsonl_path, summary_path=summary_path)
    jsonl_payload = b"".join(
        _json_bytes({"format_version": RESULT_FORMAT, **dict(result)})
        for result in results
    )
    digest = hashlib.sha256(jsonl_payload).hexdigest()
    committed_summary = dict(summary)
    committed_summary["results_jsonl"] = {
        "path": str(jsonl_path.resolve()),
        "sha256": digest,
        "records": len(results),
        "bytes": len(jsonl_payload),
    }
    summary_payload = _json_bytes(committed_summary)
    jsonl_temporary = _stage_bytes(jsonl_path, jsonl_payload)
    summary_temporary: Path | None = None
    try:
        summary_temporary = _stage_bytes(summary_path, summary_payload)
        # Summary is the commit marker, so it is always renamed last.
        os.replace(jsonl_temporary, jsonl_path)
        _fsync_directory(jsonl_path.parent)
        os.replace(summary_temporary, summary_path)
        _fsync_directory(summary_path.parent)
    finally:
        jsonl_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)
    return digest


class _WandbLogger:
    """Optional W&B bridge which never logs or returns an API key."""

    def __init__(
        self,
        *,
        mode: str,
        safe_config: Mapping[str, object],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        environment = os.environ if environ is None else environ
        selected = mode
        if selected == "auto":
            selected = "online" if environment.get("WANDB_API_KEY") else "disabled"
        if selected == "online" and not environment.get("WANDB_API_KEY"):
            raise RuntimeError("W&B online mode requires WANDB_API_KEY")
        self._run: Any | None = None
        self.mode = selected
        if selected == "disabled":
            return
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                f"W&B {selected} logging was requested but wandb is unavailable"
            ) from error
        self._run = wandb.init(
            project=environment.get("WANDB_PROJECT", "kcorrdiff"),
            job_type=environment.get("WANDB_JOB_TYPE", "loader-benchmark"),
            name=environment.get("WANDB_NAME"),
            id=environment.get("WANDB_RUN_ID"),
            resume="never",
            mode=selected,
            dir=environment.get("WANDB_DIR"),
            config=dict(safe_config),
        )

    def log(self, result: Mapping[str, object], *, index: int) -> None:
        if self._run is None:
            return
        flattened: dict[str, object] = {"grid_index": index}
        for name in (
            "batch_size",
            "worker_count",
            "prefetch_factor",
            "samples_per_second",
            "batches_per_second",
            "logical_input_gib_per_second",
            "total_rss_peak_bytes",
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "gpu_memory_allocated_peak_bytes",
            "gpu_memory_reserved_peak_bytes",
        ):
            if name in result:
                flattened[name] = result[name]
        if result.get("status") == "ok":
            for prefix in ("data_wait_seconds", "h2d_seconds"):
                distribution = result.get(prefix)
                if isinstance(distribution, Mapping):
                    for statistic, value in distribution.items():
                        flattened[f"{prefix}/{statistic}"] = value
        flattened["status_ok"] = int(result.get("status") == "ok")
        self._run.log(flattened, step=index)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def _safe_contract_summary(contract: CacheBenchmarkContract) -> dict[str, object]:
    value = asdict(contract)
    value.pop("allow_test_override", None)
    return value


def _best_result(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        raise RuntimeError("benchmark produced no successful grid cell")
    best = max(successful, key=lambda value: float(value["samples_per_second"]))
    return {
        name: best[name]
        for name in (
            "batch_size",
            "worker_count",
            "prefetch_factor",
            "samples_per_second",
            "logical_input_gib_per_second",
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    started = datetime.now(UTC)
    summary_path = arguments.output_json
    jsonl_path = derive_jsonl_path(summary_path, arguments.output_jsonl)
    preflight_output_paths(jsonl_path=jsonl_path, summary_path=summary_path)
    contract = validate_strict_arguments(arguments)
    config = _load_config(arguments.config)
    validated_config = validate_config_contract(config, arguments=arguments)
    artifacts = _artifact_provenance(arguments)
    artifact_contract = validate_cache_artifacts(
        radar_cache_root=arguments.radar_cache_root,
        era5_cache_root=arguments.era5_cache_root,
        static_path=arguments.static_path,
        contract=contract,
    )
    rows = read_draw_manifest(arguments.draw_manifest)
    grid = BenchmarkGrid(
        batch_sizes=tuple(arguments.batch_sizes),
        worker_counts=tuple(arguments.worker_counts),
        prefetch_factors=tuple(arguments.prefetch_factors),
        warmup_batches=arguments.warmup_batches,
        measured_batches=arguments.measured_batches,
    )
    device = configure_cuda_device(arguments.device)
    safe_wandb_config: dict[str, object] = {
        "precision": "float32",
        "tf32": False,
        "radar_shape": list(contract.radar_shape),
        "era_shape": list(contract.era_shape),
        "target_widths": list(contract.target_widths),
        "context_widths": list(contract.context_widths),
        "era_latent_channels": contract.era_latent_channels,
        "draw_rows": len(rows),
        "batch_sizes": list(grid.batch_sizes),
        "worker_counts": list(grid.worker_counts),
        "prefetch_factors": list(grid.prefetch_factors),
    }
    wandb_logger = _WandbLogger(
        mode=arguments.wandb_mode,
        safe_config=safe_wandb_config,
    )
    try:
        results = run_benchmark_grid(
            CacheDatasetSpec(
                radar_cache_root=arguments.radar_cache_root,
                era5_cache_root=arguments.era5_cache_root,
                static_path=arguments.static_path,
                rows=rows,
                contract=contract,
                radar_timezone=str(
                    _nested(config, "data.radar_timezone")
                    if _nested(config, "data.radar_timezone") is not _MISSING
                    else "Asia/Seoul"
                ),
                verify_cache_hashes=arguments.verify_cache_hashes,
            ),
            grid,
            device=device,
            pin_memory=True,
        )
        for index, result in enumerate(results):
            wandb_logger.log(result, index=index)
    finally:
        wandb_logger.finish()

    finished = datetime.now(UTC)
    summary: dict[str, object] = {
        "format_version": SUMMARY_FORMAT,
        "complete": True,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "contract": _safe_contract_summary(contract),
        "validated_config_fields": validated_config,
        "artifacts": artifacts,
        "artifact_contract": artifact_contract,
        "draw_manifest_rows": len(rows),
        "draw_identity": ["t0_utc", "lead_hours", "condition_signature"],
        "future_target_frames_benchmark_only": 7,
        "era_native_hours": 8,
        "grid": asdict(grid),
        "grid_cells": len(results),
        "successful_grid_cells": sum(
            result.get("status") == "ok" for result in results
        ),
        "best": _best_result(results),
        "runtime": {
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "visible_cuda_devices": torch.cuda.device_count(),
            "device": str(device),
            "default_dtype": str(torch.get_default_dtype()),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "wandb_mode": wandb_logger.mode,
        },
    }
    digest = publish_results_atomic(
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        results=results,
        summary=summary,
    )
    # Deliberately print only safe publication metadata; never dump environment.
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary_path.resolve()),
                "results_jsonl": str(jsonl_path.resolve()),
                "results_sha256": digest,
                "grid_cells": len(results),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "RESULT_FORMAT",
    "SUMMARY_FORMAT",
    "configure_cuda_device",
    "derive_jsonl_path",
    "main",
    "preflight_output_paths",
    "publish_results_atomic",
    "validate_config_contract",
    "validate_strict_arguments",
]


if __name__ == "__main__":
    raise SystemExit(main())

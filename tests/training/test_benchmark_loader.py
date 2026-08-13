from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcorrdiff.training import benchmark_loader
from kcorrdiff.training.benchmark_loader import (
    RESULT_FORMAT,
    SUMMARY_FORMAT,
    derive_jsonl_path,
    publish_results_atomic,
    validate_config_contract,
    validate_strict_arguments,
)
from kcorrdiff.training.cache_benchmark import (
    PRODUCTION_CONTEXT_WIDTHS,
    PRODUCTION_TARGET_WIDTHS,
)


def k8s_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "stage2-full-width.yaml"),
        "--radar-cache-root",
        str(tmp_path / "radar-v1"),
        "--era5-cache-root",
        str(tmp_path / "era5-v1"),
        "--draw-manifest",
        str(tmp_path / "training-draw-manifest.jsonl"),
        "--static-path",
        str(tmp_path / "target_static.npz"),
        "--output-json",
        str(tmp_path / "loader-benchmark.json"),
        "--device",
        "cuda",
        "--precision",
        "float32",
        "--disable-tf32",
        "--target-widths",
        "64,128,256,384,512",
        "--context-widths",
        "32,64,128,256,384",
        "--era-latent-channels",
        "128",
        "--era-grid-size",
        "33",
        "--batch-sizes",
        "1,2,4,8",
        "--worker-counts",
        "0,4,8,12,16,24",
        "--prefetch-factors",
        "2,4",
        "--warmup-batches",
        "5",
        "--measured-batches",
        "25",
        "--fail-on-fallback",
    ]


def strict_environment() -> dict[str, str]:
    return {
        "NVIDIA_TF32_OVERRIDE": "0",
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
        "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
        "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
        "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
    }


def test_parser_matches_k8s_manifest_argument_contract(tmp_path: Path) -> None:
    arguments = benchmark_loader._parse_args(k8s_cli_args(tmp_path))
    assert arguments.target_widths == PRODUCTION_TARGET_WIDTHS
    assert arguments.context_widths == PRODUCTION_CONTEXT_WIDTHS
    assert arguments.batch_sizes == (1, 2, 4, 8)
    assert arguments.worker_counts == (0, 4, 8, 12, 16, 24)
    assert arguments.prefetch_factors == (2, 4)
    assert arguments.disable_tf32 and arguments.fail_on_fallback
    contract = validate_strict_arguments(
        arguments, environ=strict_environment()
    )
    assert contract.radar_shape == (256, 256)
    assert contract.era_shape == (33, 33)
    assert contract.precision == "float32"
    assert contract.tf32_enabled is False


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("--device", "cpu"), "cuda"),
        (("--precision", "float16"), "float32"),
        (("--target-widths", "8,16,32,64,128"), "target widths"),
        (("--era-grid-size", "17"), "33x33"),
    ],
)
def test_strict_cli_rejects_fallback_values(
    tmp_path: Path,
    replacement: tuple[str, str],
    message: str,
) -> None:
    argv = k8s_cli_args(tmp_path)
    option, value = replacement
    argv[argv.index(option) + 1] = value
    arguments = benchmark_loader._parse_args(argv)
    with pytest.raises(ValueError, match=message):
        validate_strict_arguments(arguments, environ=strict_environment())


def test_strict_cli_rejects_enabled_fallback_environment(tmp_path: Path) -> None:
    arguments = benchmark_loader._parse_args(k8s_cli_args(tmp_path))
    environment = strict_environment()
    environment["KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK"] = "1"
    with pytest.raises(ValueError, match="MODEL_WIDTH"):
        validate_strict_arguments(arguments, environ=environment)


def test_config_validation_accepts_full_contract_and_rejects_silent_change(
    tmp_path: Path,
) -> None:
    arguments = benchmark_loader._parse_args(k8s_cli_args(tmp_path))
    config = {
        "runtime": {
            "target_widths": list(PRODUCTION_TARGET_WIDTHS),
            "context_widths": list(PRODUCTION_CONTEXT_WIDTHS),
            "era_latent_channels": 128,
            "era_grid_size": 33,
            "precision": "float32",
            "tf32": False,
            "allow_cpu_fallback": False,
            "allow_model_width_fallback": False,
            "allow_precision_fallback": False,
            "allow_era_grid_fallback": False,
        },
        "data": {
            "radar_timezone": "Asia/Seoul",
            "history_frames": 12,
            "future_target_frames": 7,
            "target_shape": [256, 256],
            "context_shape": [256, 256],
            "era_shape": [33, 33],
            "mmap_lazy_per_worker": True,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "loader_tuning": {
            "candidates": {
                "batch_sizes": [1, 2, 4, 8, 16, 32],
                "worker_counts": [0, 4, 8, 12, 16, 24],
                "prefetch_factors": [2, 4],
            }
        },
    }
    validated = validate_config_contract(config, arguments=arguments)
    assert validated["data.history_frames"] == 12
    reordered = k8s_cli_args(tmp_path)
    reordered[reordered.index("--batch-sizes") + 1] = "2,1"
    with pytest.raises(ValueError, match="reorders"):
        validate_config_contract(
            config, arguments=benchmark_loader._parse_args(reordered)
        )
    config["runtime"]["target_widths"] = [8, 16, 32, 64, 128]  # type: ignore[index]
    with pytest.raises(ValueError, match="runtime.target_widths"):
        validate_config_contract(config, arguments=arguments)


def test_atomic_jsonl_and_summary_are_new_files_and_hash_linked(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "run" / "loader-benchmark.json"
    jsonl_path = derive_jsonl_path(summary_path)
    secret = "do-not-leak-wandb-key"
    results = [
        {
            "status": "ok",
            "batch_size": 1,
            "worker_count": 0,
            "prefetch_factor": 2,
            "samples_per_second": 4.0,
        },
        {
            "status": "oom",
            "batch_size": 16,
            "worker_count": 4,
            "prefetch_factor": 2,
            "error": "CUDA out of memory at unchanged full-width contract",
        },
    ]
    summary = {
        "format_version": SUMMARY_FORMAT,
        "complete": True,
        "runtime": {"wandb_mode": "disabled"},
    }
    digest = publish_results_atomic(
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        results=results,
        summary=summary,
    )
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    committed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [record["format_version"] for record in records] == [
        RESULT_FORMAT,
        RESULT_FORMAT,
    ]
    assert committed["results_jsonl"]["sha256"] == digest
    assert committed["results_jsonl"]["records"] == 2
    assert secret not in jsonl_path.read_text(encoding="utf-8")
    assert secret not in summary_path.read_text(encoding="utf-8")
    assert not tuple((tmp_path / "run").glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_results_atomic(
            jsonl_path=jsonl_path,
            summary_path=summary_path,
            results=results,
            summary=summary,
        )


def test_wandb_auto_disables_without_key_and_online_requires_key() -> None:
    logger = benchmark_loader._WandbLogger(
        mode="auto", safe_config={"precision": "float32"}, environ={}
    )
    assert logger.mode == "disabled"
    logger.finish()
    with pytest.raises(RuntimeError, match="requires WANDB_API_KEY"):
        benchmark_loader._WandbLogger(
            mode="online", safe_config={}, environ={}
        )

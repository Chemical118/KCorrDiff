from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kcorrdiff.models.regression_system import RegressionSystem
from kcorrdiff.training import model_benchmark
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.data_factory import RankLocalBatch
from kcorrdiff.training.model_benchmark import (
    ForwardLossResult,
    ModelBenchmarkGrid,
    run_model_candidate,
    run_model_grid,
    summarize_selection,
    validate_model_instance,
    validate_stage2_contract,
    validate_strict_arguments,
)
from kcorrdiff.training.plan import DistributedDrawPlan, DrawSlot
from kcorrdiff.training.production_benchmark import (
    CandidatePreflight,
    LoaderCandidate,
    PipelinePayloadLimits,
    ShmUsage,
    TimedRankLocalBatch,
    estimate_pipeline_payload,
    tensor_tree_metrics,
)
from kcorrdiff.training.train_stage2 import regression_system_config_from_stage2


SHA = "a" * 64


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "stage2.yaml"),
        "--radar-cache-root",
        str(tmp_path / "radar"),
        "--era5-cache-root",
        str(tmp_path / "era5"),
        "--target-static",
        str(tmp_path / "static.npz"),
        "--candidate-manifest",
        str(tmp_path / "candidate.json"),
        "--candidate-manifest-sha256",
        SHA,
        "--draw-manifest",
        str(tmp_path / "draw.jsonl"),
        "--draw-manifest-sha256",
        "b" * 64,
        "--bundle-metadata",
        str(tmp_path / "bundle.json"),
        "--bundle-metadata-sha256",
        "c" * 64,
        "--target-coordinates-sha256",
        "d" * 64,
        "--context-coordinates-sha256",
        "e" * 64,
        "--output-json",
        str(tmp_path / "model-benchmark.json"),
        "--device",
        "cuda:0",
        "--precision",
        "float32",
        "--disable-tf32",
        "--fail-on-fallback",
        "--pin-memory",
        "--batch-sizes",
        "1,2",
        "--warmup-steps",
        "1",
        "--measured-steps",
        "2",
        "--worker-count",
        "12",
        "--prefetch-factor",
        "2",
        "--maximum-grid-cells",
        "2",
        "--maximum-host-pipeline-bytes",
        str(64 * 1024**3),
        "--maximum-shm-pipeline-bytes",
        str(48 * 1024**3),
        "--logical-world-size",
        "2",
    ]


def _strict_environment() -> dict[str, str]:
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


def test_cli_help_exposes_training_step_and_provenance_contract(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        model_benchmark._parse_args(["--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--batch-sizes",
        "--warmup-steps",
        "--measured-steps",
        "--candidate-manifest-sha256",
        "--draw-manifest-sha256",
        "--bundle-metadata-sha256",
        "--target-coordinates-sha256",
        "--context-coordinates-sha256",
        "--fail-on-fallback",
        "--wandb-mode",
    ):
        assert option in output


def test_strict_contract_binds_full_precision_bounded_grid(tmp_path: Path) -> None:
    arguments = model_benchmark._parse_args(_cli_args(tmp_path))
    contract = validate_strict_arguments(
        arguments, environ=_strict_environment()
    )
    assert contract.device == "cuda:0"
    assert contract.precision == "float32"
    assert contract.tf32_enabled is False
    assert contract.grid.batch_sizes == (1, 2)
    assert contract.grid.warmup_steps == 1
    assert contract.grid.measured_steps == 2
    assert contract.worker_count == 12
    assert contract.prefetch_factor == 2
    assert contract.logical_world_size == 2

    repository = Path(__file__).resolve().parents[2]
    config = load_stage2_config(repository / "configs" / "stage2-full-width.yaml")
    model_contract, validated = validate_stage2_contract(config, contract)
    assert model_contract.system.activation_checkpoint is True
    assert validated["optimizer"]["class"] == "AdamW"
    assert validated["loader"]["candidate_grid"]["batch_sizes"] == (1, 2)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--device", "cpu", "cuda:0"),
        ("--precision", "float16", "float32"),
        ("--warmup-steps", "0", "warmup_steps"),
        ("--maximum-grid-cells", "1", "maximum-grid-cells"),
    ],
)
def test_strict_contract_rejects_fallback_or_unbounded_grid(
    tmp_path: Path, option: str, value: str, message: str
) -> None:
    argv = _cli_args(tmp_path)
    argv[argv.index(option) + 1] = value
    arguments = model_benchmark._parse_args(argv)
    with pytest.raises(ValueError, match=message):
        validate_strict_arguments(arguments, environ=_strict_environment())


def test_strict_contract_rejects_fallback_environment_and_reordered_grid(
    tmp_path: Path,
) -> None:
    arguments = model_benchmark._parse_args(_cli_args(tmp_path))
    environment = _strict_environment()
    environment["KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK"] = "1"
    with pytest.raises(ValueError, match="MODEL_WIDTH_FALLBACK"):
        validate_strict_arguments(arguments, environ=environment)

    argv = _cli_args(tmp_path)
    argv[argv.index("--batch-sizes") + 1] = "2,1"
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strict_arguments(
            model_benchmark._parse_args(argv), environ=_strict_environment()
        )

    argv = _cli_args(tmp_path)
    argv[argv.index("--batch-sizes") + 1] = "1,2,3,4"
    with pytest.raises(ValueError, match="powers of two"):
        validate_strict_arguments(
            model_benchmark._parse_args(argv), environ=_strict_environment()
        )


def test_model_identity_rejects_width_and_precision_fallback() -> None:
    model = nn.Linear(1, 1)
    identity = validate_model_instance(
        model, expected_parameter_count=2, require_exact_class=False
    )
    assert identity.parameter_count == 2
    assert identity.parameter_bytes == 8

    with pytest.raises(ValueError, match="parameter count changed"):
        validate_model_instance(
            model, expected_parameter_count=3, require_exact_class=False
        )
    with pytest.raises(TypeError, match="non-float32"):
        validate_model_instance(
            nn.Linear(1, 1).double(),
            expected_parameter_count=2,
            require_exact_class=False,
        )


def test_production_model_identity_is_exact_full_width_fp32() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = load_stage2_config(repository / "configs" / "stage2-full-width.yaml")
    model = RegressionSystem(regression_system_config_from_stage2(config).system)
    identity = validate_model_instance(model)
    assert identity.parameter_count == 62_008_276
    assert identity.trainable_parameter_count == 62_008_276
    assert identity.parameter_bytes == 248_033_104
    assert identity.component_parameter_counts == {
        "condition_bank": 31_537_202,
        "era_encoder": 200_000,
        "regression": 30_271_074,
    }


@dataclass(frozen=True)
class _FakeTraining:
    inputs: torch.Tensor
    targets: torch.Tensor

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "_FakeTraining":
        return _FakeTraining(
            self.inputs.to(device=device, non_blocking=non_blocking),
            self.targets.to(device=device, non_blocking=non_blocking),
        )


def _rank_batch(batch_size: int) -> TimedRankLocalBatch:
    inputs = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1) + 1.0
    targets = inputs * 2.0
    slots = tuple(DrawSlot(index, index) for index in range(batch_size))
    return TimedRankLocalBatch(
        rank_local=RankLocalBatch(
            training=_FakeTraining(inputs, targets),
            slots=slots,
            active_slot_indices=tuple(range(batch_size)),
            loss_weights=torch.ones(batch_size, dtype=torch.float32),
        ),
        collate_seconds=0.001,
        worker_pid=os.getpid(),
    )


def _plan(slots: int = 32) -> DistributedDrawPlan:
    rank_slots = tuple(DrawSlot(index, index) for index in range(slots))
    return DistributedDrawPlan(
        role="deployment",
        fold_id=None,
        source_row_indices=tuple(range(slots)),
        world_size=1,
        per_rank_microbatch_size=1,
        rank_slots=(rank_slots,),
        semantic_sha256="f" * 64,
    )


def _preflight(candidate: LoaderCandidate) -> CandidatePreflight:
    metrics = tensor_tree_metrics(_rank_batch(candidate.batch_size).rank_local)
    estimate = estimate_pipeline_payload(
        metrics,
        candidate,
        limits=PipelinePayloadLimits(
            maximum_grid_cells=8,
            maximum_host_pipeline_bytes=10_000_000,
            maximum_shm_pipeline_bytes=10_000_000,
            safety_factor=1.0,
        ),
        shm_snapshot=ShmUsage("/dev/shm", 10_000_000, 0, 10_000_000),
        pin_memory=False,
    )
    assert estimate.approved
    return CandidatePreflight(estimate, 0.0, 0.0, 0.0)


def test_cpu_injected_candidate_executes_forward_backward_and_adamw() -> None:
    candidate = LoaderCandidate(batch_size=2, worker_count=0, prefetch_factor=2)
    config = SimpleNamespace(
        seed=17,
        loss=SimpleNamespace(mean_warmup_steps=0),
        optimization=SimpleNamespace(
            learning_rate=0.01,
            weight_decay=0.0001,
            betas=(0.9, 0.999),
            gradient_clip_norm=1.0,
        ),
    )
    created: list[nn.Linear] = []
    initial: list[torch.Tensor] = []

    def model_builder(_config: object) -> nn.Module:
        model = nn.Linear(1, 1)
        created.append(model)
        initial.append(model.weight.detach().clone())
        return model

    def loader_builder(
        factory: object,
        plan: object,
        *,
        rank: int,
        candidate: LoaderCandidate,
        pin_memory: bool,
        maximum_batches: int,
    ) -> list[TimedRankLocalBatch]:
        del factory, plan, rank, pin_memory
        return [_rank_batch(candidate.batch_size) for _ in range(maximum_batches)]

    def forward_loss(
        model: nn.Module,
        training: object,
        _config: object,
        global_step: int,
    ) -> ForwardLossResult:
        assert isinstance(training, _FakeTraining)
        prediction = model(training.inputs)
        loss = (prediction - training.targets).square().mean()
        return ForwardLossResult(
            loss=loss,
            metrics={
                "loss_total": loss.detach(),
                "global_step": torch.tensor(
                    float(global_step), dtype=torch.float32
                ),
            },
        )

    result = run_model_candidate(
        object(),  # type: ignore[arg-type]
        _plan(),
        candidate,
        rank=0,
        warmup_steps=1,
        measured_steps=2,
        device=torch.device("cpu"),
        preflight=_preflight(candidate),
        config=config,  # type: ignore[arg-type]
        system_config=SimpleNamespace(activation_checkpoint=True),  # type: ignore[arg-type]
        pin_memory=False,
        allow_cpu_test=True,
        loader_builder=loader_builder,  # type: ignore[arg-type]
        model_builder=model_builder,  # type: ignore[arg-type]
        forward_loss=forward_loss,  # type: ignore[arg-type]
        expected_parameter_count=2,
        require_exact_model_class=False,
    )
    assert result["status"] == "ok"
    assert result["accepted"] is True
    assert result["optimizer"] == "AdamW"
    assert result["optimizer_steps_executed"] == 3
    assert result["samples"] == 4
    assert result["samples_per_second"] > 0.0
    assert result["forward_loss_seconds"]["max"] >= 0.0
    assert result["backward_seconds"]["max"] >= 0.0
    assert result["adamw_step_seconds"]["max"] >= 0.0
    assert result["gpu_memory_allocated_peak_bytes"] == 0
    assert result["fallback_used"] is False
    assert not torch.equal(created[0].weight.detach(), initial[0])


class _FakePreflight:
    def __init__(self, candidate: LoaderCandidate, approved: bool = True) -> None:
        self.approved = approved
        self.payload = SimpleNamespace(candidate=candidate)

    def as_json(self) -> dict[str, object]:
        return {"approved": self.approved}


def test_grid_records_oom_as_rejected_cleans_and_continues() -> None:
    grid = ModelBenchmarkGrid((1, 2, 4), warmup_steps=1, measured_steps=1)
    memory = {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "peak_allocated_bytes": 900,
        "peak_reserved_bytes": 1_000,
    }
    cleaned: list[int] = []
    visited: list[int] = []
    logged: list[tuple[int, str]] = []

    def preflight(candidate: LoaderCandidate) -> _FakePreflight:
        return _FakePreflight(candidate)

    def runner(
        candidate: LoaderCandidate, _preflight: object
    ) -> dict[str, object]:
        visited.append(candidate.batch_size)
        if candidate.batch_size == 2:
            memory["allocated_bytes"] = 800
            memory["reserved_bytes"] = 1_000
            raise torch.cuda.OutOfMemoryError("injected CUDA out of memory")
        return {
            "format_version": model_benchmark.RESULT_FORMAT,
            "status": "ok",
            "accepted": True,
            "batch_size": candidate.batch_size,
            "worker_count": candidate.worker_count,
            "prefetch_factor": candidate.prefetch_factor,
            "samples_per_second": float(candidate.batch_size * 10),
            "optimizer_steps_per_second": 10.0,
            "gpu_memory_allocated_peak_bytes": candidate.batch_size * 100,
            "gpu_memory_reserved_peak_bytes": candidate.batch_size * 120,
            "precision": "float32",
            "tf32_enabled": False,
            "fallback_used": False,
        }

    def reset(_device: torch.device) -> None:
        memory["allocated_bytes"] = 0
        memory["reserved_bytes"] = 0

    def cleanup(_device: torch.device) -> None:
        cleaned.append(len(cleaned))
        memory["allocated_bytes"] = 0
        memory["reserved_bytes"] = 0

    results = run_model_grid(
        grid,
        worker_count=12,
        prefetch_factor=2,
        device=torch.device("cpu"),
        preflight_runner=preflight,  # type: ignore[arg-type]
        candidate_runner=runner,  # type: ignore[arg-type]
        result_callback=lambda result, index: logged.append(
            (index, str(result["status"]))
        ),
        memory_probe=lambda _device: dict(memory),
        reset_peak=reset,
        cleanup=cleanup,
    )
    assert visited == [1, 2, 4]
    assert [result["status"] for result in results] == [
        "ok",
        "rejected_oom",
        "ok",
    ]
    assert results[1]["accepted"] is False
    assert results[1]["rejection_reason"] == "cuda_out_of_memory"
    assert results[1]["gpu_memory_allocated_peak_bytes"] == 900
    assert results[1]["post_cleanup_gpu_memory_allocated_bytes"] == 0
    assert len(cleaned) == 3
    assert logged == [(0, "ok"), (1, "rejected_oom"), (2, "ok")]

    selection = summarize_selection(results)
    assert selection["accepted_cells"] == 2
    assert selection["oom_rejected_cells"] == 1
    assert selection["maximum_accepted"]["batch_size"] == 4
    assert selection["throughput_best_accepted"]["batch_size"] == 4
    assert selection["scope"] == "benchmark_observation_only"
    assert selection["changes_production_training_config"] is False
    assert selection["minimum_oom_rejected_batch_size"] == 2


def test_atomic_publisher_commits_jsonl_hash_before_summary(tmp_path: Path) -> None:
    results = [
        {
            "format_version": model_benchmark.RESULT_FORMAT,
            "status": "ok",
            "accepted": True,
            "batch_size": 1,
        }
    ]
    summary = {
        "format_version": model_benchmark.SUMMARY_FORMAT,
        "complete": True,
    }
    jsonl_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    digest = model_benchmark.publish_results_atomic(
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        results=results,
        summary=summary,
    )
    committed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert committed["complete"] is True
    assert committed["results_jsonl"]["sha256"] == digest
    assert committed["results_jsonl"]["records"] == 1
    assert jsonl_path.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        model_benchmark.publish_results_atomic(
            jsonl_path=jsonl_path,
            summary_path=summary_path,
            results=results,
            summary=summary,
        )

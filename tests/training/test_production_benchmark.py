from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kcorrdiff.data.advection import AdvectionGeometry
from kcorrdiff.training import production_benchmark
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.data_factory import (
    FactoryInputs,
    PlannedSample,
    RankLocalBatch,
)
from kcorrdiff.training.plan import DistributedDrawPlan, DrawSlot
from kcorrdiff.training.production_benchmark import (
    ExpectedArtifactHashes,
    LoaderCandidate,
    PipelinePayloadLimits,
    ShmUsage,
    TimedRankLocalBatch,
    TimedRankSlotCollator,
    estimate_pipeline_payload,
    preflight_candidate_payload,
    require_float32_tensor_tree,
    run_production_candidate,
    tensor_tree_metrics,
    validate_config_contract,
    validate_strict_arguments,
    verify_expected_manifest_hashes,
)


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
        str(tmp_path / "candidates.json"),
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
        str(tmp_path / "benchmark.json"),
        "--device",
        "cuda:0",
        "--precision",
        "float32",
        "--disable-tf32",
        "--fail-on-fallback",
        "--pin-memory",
        "--batch-sizes",
        "1,2,4,8",
        "--worker-counts",
        "0,4,8,12,16,24",
        "--prefetch-factors",
        "2,4",
        "--warmup-batches",
        "2",
        "--measured-batches",
        "3",
        "--maximum-grid-cells",
        "48",
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


def test_cli_help_exposes_content_hashes_and_payload_bounds(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        production_benchmark._parse_args(["--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--candidate-manifest-sha256",
        "--draw-manifest-sha256",
        "--bundle-metadata-sha256",
        "--target-coordinates-sha256",
        "--context-coordinates-sha256",
        "--maximum-host-pipeline-bytes",
        "--maximum-shm-pipeline-bytes",
    ):
        assert option in output


def test_strict_cli_contract_is_full_precision_and_bounded(tmp_path: Path) -> None:
    arguments = production_benchmark._parse_args(_cli_args(tmp_path))
    contract = validate_strict_arguments(
        arguments, environ=_strict_environment()
    )
    assert contract.device == "cuda:0"
    assert contract.precision == "float32"
    assert contract.tf32_enabled is False
    assert contract.pin_memory is True
    assert contract.grid.cell_count == 48
    assert contract.logical_world_size == 2
    assert arguments.candidate_manifest_sha256 == SHA


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--device", "cpu", "cuda:0"),
        ("--precision", "float16", "float32"),
    ],
)
def test_strict_cli_rejects_fallback_or_unbounded_grid(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    argv = _cli_args(tmp_path)
    argv[argv.index(option) + 1] = value
    arguments = production_benchmark._parse_args(argv)
    with pytest.raises(ValueError, match=message):
        validate_strict_arguments(arguments, environ=_strict_environment())


def test_environment_flags_do_not_pin_benchmark_config(tmp_path: Path) -> None:
    arguments = production_benchmark._parse_args(_cli_args(tmp_path))
    environment = _strict_environment()
    environment["KCORRDIFF_ALLOW_PRECISION_FALLBACK"] = "1"
    assert validate_strict_arguments(arguments, environ=environment).precision == "float32"


def test_stage2_config_accepts_arbitrary_positive_cli_grid(tmp_path: Path) -> None:
    arguments = production_benchmark._parse_args(_cli_args(tmp_path))
    contract = validate_strict_arguments(
        arguments, environ=_strict_environment()
    )
    repository = Path(__file__).resolve().parents[2]
    config = load_stage2_config(repository / "configs" / "stage2-full-width.yaml")
    validated = validate_config_contract(config, contract)
    assert validated["precision"] == "float32"
    assert validated["candidate_grid"]["worker_counts"] == (
        0,
        4,
        8,
        12,
        16,
        24,
    )

    minimal = production_benchmark.ProductionBenchmarkGrid(
        batch_sizes=(1, 2),
        worker_counts=(0, 8, 12),
        prefetch_factors=(2,),
        warmup_batches=1,
        measured_batches=1,
    )
    subset = replace(contract, grid=minimal)
    subset_validated = validate_config_contract(config, subset)
    assert subset_validated["candidate_grid"]["worker_counts"] == (0, 8, 12)

    reordered = replace(
        contract,
        grid=replace(contract.grid, batch_sizes=(2, 1)),
    )
    assert validate_config_contract(config, reordered)["candidate_grid"][
        "batch_sizes"
    ] == (2, 1)

    arbitrary = production_benchmark.ProductionBenchmarkGrid(
        batch_sizes=(1, 2, 3, 4),
        worker_counts=(0,),
        prefetch_factors=(2,),
        warmup_batches=0,
        measured_batches=1,
    )
    assert arbitrary.batch_sizes == (1, 2, 3, 4)


@dataclass(frozen=True)
class _TensorTree:
    base: torch.Tensor
    view: torch.Tensor
    flag: torch.Tensor
    repeated: torch.Tensor


def test_tensor_metrics_distinguish_references_objects_and_storages() -> None:
    base = torch.arange(8, dtype=torch.float32)
    tree = _TensorTree(
        base=base,
        view=base[:4],
        flag=torch.ones(4, dtype=torch.bool),
        repeated=base,
    )
    metrics = tensor_tree_metrics(tree)
    assert metrics.tensor_references == 4
    assert metrics.unique_tensor_objects == 3
    assert metrics.unique_storages == 2
    assert metrics.logical_bytes == 8 * 4 + 4 * 4 + 4
    assert metrics.storage_bytes == 8 * 4 + 4
    assert metrics.dtype_tensor_counts == {"bool": 1, "float32": 2}
    assert require_float32_tensor_tree(tree) == metrics

    with pytest.raises(TypeError, match="non-float32"):
        require_float32_tensor_tree(torch.ones(2, dtype=torch.float64))


def test_immutable_advection_axes_are_cpu_metadata_not_h2d_payload() -> None:
    geometry = AdvectionGeometry(
        [0.0, 1.0], [0.0, 1.0], [-1.0, 2.0], [-1.0, 2.0]
    )
    tree = {"values": torch.ones(3, dtype=torch.float32), "geometry": geometry}
    metrics = require_float32_tensor_tree(tree)
    assert metrics.dtype_tensor_counts == {"float32": 1}
    assert metrics.storage_bytes == 3 * 4


def _shm(*, available: int = 10_000, used: int = 0) -> ShmUsage:
    return ShmUsage(
        path="/dev/shm",
        total_bytes=available + used,
        used_bytes=used,
        available_bytes=available,
    )


def test_payload_estimate_bounds_worker_prefetch_and_pinned_queues() -> None:
    metrics = tensor_tree_metrics(torch.zeros(9, dtype=torch.float32))
    candidate = LoaderCandidate(batch_size=2, worker_count=3, prefetch_factor=4)
    accepted = estimate_pipeline_payload(
        metrics,
        candidate,
        limits=PipelinePayloadLimits(
            maximum_grid_cells=1,
            maximum_host_pipeline_bytes=1_000,
            maximum_shm_pipeline_bytes=1_000,
            safety_factor=1.0,
        ),
        shm_snapshot=_shm(),
        pin_memory=True,
    )
    assert accepted.worker_prefetch_batches == 12
    assert accepted.pinned_queue_batches == 2
    assert accepted.estimated_shm_bytes == 36 * 12
    assert accepted.estimated_host_pipeline_bytes == 36 * 15
    assert accepted.approved

    advisory = estimate_pipeline_payload(
        metrics,
        candidate,
        limits=PipelinePayloadLimits(
            maximum_grid_cells=1,
            maximum_host_pipeline_bytes=1_000,
            maximum_shm_pipeline_bytes=400,
            safety_factor=1.0,
        ),
        shm_snapshot=_shm(),
        pin_memory=True,
    )
    assert advisory.approved
    assert "configured /dev/shm reference exceeded" in advisory.rejection_reasons


def test_timed_collator_retains_explicit_rank_slots() -> None:
    collator = TimedRankSlotCollator(lambda samples: tuple(samples))
    result = collator(
        [
            PlannedSample(slot=DrawSlot(0, 0), sample="a"),
            PlannedSample(slot=DrawSlot(1, 1), sample="b"),
        ]
    )
    assert result.rank_local.training == ("a", "b")
    assert result.rank_local.loss_weights.tolist() == [1.0, 1.0]
    assert result.rank_local.padding_count == 0
    assert result.worker_pid == os.getpid()
    assert result.collate_seconds >= 0.0


@dataclass(frozen=True)
class _FakeTraining:
    values: torch.Tensor

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "_FakeTraining":
        return _FakeTraining(
            self.values.to(device=device, non_blocking=non_blocking)
        )


def _rank_batch(batch_size: int) -> TimedRankLocalBatch:
    slots = tuple(DrawSlot(index, index) for index in range(batch_size))
    rank_local = RankLocalBatch(
        training=_FakeTraining(torch.arange(batch_size * 4, dtype=torch.float32)),
        slots=slots,
        active_slot_indices=tuple(range(batch_size)),
        loss_weights=torch.ones(batch_size, dtype=torch.float32),
    )
    return TimedRankLocalBatch(
        rank_local=rank_local,
        collate_seconds=0.001,
        worker_pid=os.getpid(),
    )


class _InjectedLoaderBuilder:
    def __init__(self, batches: int = 16) -> None:
        self.batches = batches
        self.calls: list[tuple[LoaderCandidate, bool, int]] = []

    def __call__(
        self,
        factory: object,
        plan: object,
        *,
        rank: int,
        candidate: LoaderCandidate,
        pin_memory: bool,
        maximum_batches: int,
    ) -> list[TimedRankLocalBatch]:
        del factory, plan, rank
        self.calls.append((candidate, pin_memory, maximum_batches))
        return [
            _rank_batch(candidate.batch_size)
            for _ in range(min(self.batches, maximum_batches))
        ]


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


def test_injected_cpu_candidate_reports_real_nested_payload_metrics() -> None:
    builder = _InjectedLoaderBuilder()
    candidate = LoaderCandidate(batch_size=2, worker_count=3, prefetch_factor=2)
    limits = PipelinePayloadLimits(
        maximum_grid_cells=1,
        maximum_host_pipeline_bytes=10_000_000,
        maximum_shm_pipeline_bytes=10_000_000,
    )
    preflight = preflight_candidate_payload(
        object(),  # type: ignore[arg-type]
        _plan(),
        candidate,
        rank=0,
        limits=limits,
        pin_memory=False,
        loader_builder=builder,  # type: ignore[arg-type]
        shm_probe=_shm,
    )
    assert preflight.approved
    assert builder.calls[0][0].worker_count == 0
    assert builder.calls[0][2] == 1
    result = run_production_candidate(
        object(),  # type: ignore[arg-type]
        _plan(),
        candidate,
        rank=0,
        warmup_batches=1,
        measured_batches=2,
        device=torch.device("cpu"),
        preflight=preflight,
        pin_memory=False,
        allow_cpu_test=True,
        loader_builder=builder,  # type: ignore[arg-type]
        shm_probe=_shm,
    )
    assert result["status"] == "ok"
    assert result["samples"] == 4
    assert result["precision"] == "float32"
    assert result["tf32_enabled"] is False
    assert result["non_blocking_h2d"] is False
    assert result["data_wait_seconds"]["max"] >= 0.0
    assert result["worker_collate_seconds"]["mean"] == pytest.approx(0.001)
    assert result["host_batch_tensor_metrics"]["unique_storages"] == 2
    assert result["device_batch_tensor_metrics"]["unique_storages"] == 2
    assert result["gpu_memory_allocated_peak_bytes"] == 0
    assert builder.calls[1][2] == 3


def test_candidate_capacity_fails_before_loader_construction() -> None:
    builder = _InjectedLoaderBuilder()
    candidate = LoaderCandidate(batch_size=4, worker_count=0, prefetch_factor=2)
    metrics = tensor_tree_metrics(_rank_batch(4).rank_local)
    payload = estimate_pipeline_payload(
        metrics,
        candidate,
        limits=PipelinePayloadLimits(
            maximum_grid_cells=1,
            maximum_host_pipeline_bytes=10_000,
            maximum_shm_pipeline_bytes=10_000,
        ),
        shm_snapshot=_shm(),
        pin_memory=False,
    )
    preflight = SimpleNamespace(approved=payload.approved, payload=payload)
    with pytest.raises(ValueError, match="too few slots"):
        run_production_candidate(
            object(),  # type: ignore[arg-type]
            _plan(slots=4),
            candidate,
            rank=0,
            warmup_batches=1,
            measured_batches=1,
            device=torch.device("cpu"),
            preflight=preflight,  # type: ignore[arg-type]
            pin_memory=False,
            allow_cpu_test=True,
            loader_builder=builder,  # type: ignore[arg-type]
            shm_probe=_shm,
        )
    assert not builder.calls


def test_manifest_expected_hashes_are_informational(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    draw = tmp_path / "draw.jsonl"
    bundle = tmp_path / "bundle.json"
    for path, value in ((candidate, b"a"), (draw, b"b"), (bundle, b"c")):
        path.write_bytes(value)
    inputs = FactoryInputs(
        radar_cache_root=tmp_path / "radar",
        era5_cache_root=tmp_path / "era5",
        target_static=tmp_path / "static.npz",
        candidate_manifest=candidate,
        draw_manifest=draw,
        bundle_metadata=bundle,
        expected_target_coordinates_sha256="d" * 64,
        expected_context_coordinates_sha256="e" * 64,
    )
    expected = ExpectedArtifactHashes(
        candidate_manifest=production_benchmark.sha256_file(candidate),
        draw_manifest=production_benchmark.sha256_file(draw),
        bundle_metadata=production_benchmark.sha256_file(bundle),
        target_coordinates="d" * 64,
        context_coordinates="e" * 64,
    )
    assert verify_expected_manifest_hashes(inputs, expected) == {
        "candidate_manifest": expected.candidate_manifest,
        "draw_manifest": expected.draw_manifest,
        "bundle_metadata": expected.bundle_metadata,
    }
    bad = ExpectedArtifactHashes(
        candidate_manifest="0" * 64,
        draw_manifest=expected.draw_manifest,
        bundle_metadata=expected.bundle_metadata,
        target_coordinates=expected.target_coordinates,
        context_coordinates=expected.context_coordinates,
    )
    assert verify_expected_manifest_hashes(inputs, bad)["candidate_manifest"] == (
        expected.candidate_manifest
    )

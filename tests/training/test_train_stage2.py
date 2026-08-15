from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn

from kcorrdiff.data.sampling import DrawRow
from kcorrdiff.training.checkpoints import CheckpointProvenance, TrainingCursor
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.crossfit import CheckpointRecord
from kcorrdiff.training.data_factory import RankLocalBatch
from kcorrdiff.training.losses import (
    accumulation_window_denominators,
    distributed_direct_loss_contribution,
)
from kcorrdiff.training.plan import DrawSlot
from kcorrdiff.training.stage2_fold_set import (
    FoldProducerLineage,
    VerifiedFoldCheckpoint,
    VerifiedStage2FoldSet,
)
import kcorrdiff.training.train_stage2 as stage2
from kcorrdiff.training.train_stage2 import (
    DistributedRuntime,
    RoleTrainingResult,
    RunSelection,
    execution_config_for_topology,
    preflight_storage,
    reduce_gradients_sum,
    regression_system_config_from_stage2,
    run_optimizer_loop,
    run_stage2,
    validate_launch_arguments,
)
from tests.training.test_batch import make_batch


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "stage2-full-width.yaml"
SOURCE_ROOT = CONFIG_PATH.parents[1]


class FakeLaunchIdentity:
    artifact_sha256 = "d" * 64
    source_tree_sha256 = "e" * 64
    container_image_sha256 = "f" * 64
    runtime_report_sha256 = "0" * 64
    data_contract_sha256 = "1" * 64

    def provenance(self) -> dict[str, str]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "container_image_sha256": self.container_image_sha256,
            "runtime_report_sha256": self.runtime_report_sha256,
            "data_contract_sha256": self.data_contract_sha256,
        }

    def verify_current_files(self) -> None:
        return None


class AuditRun:
    def __init__(self) -> None:
        self.values: list[tuple[dict[str, object], int]] = []
        self.last_step = -1
        self.finished = False

    def log(self, data: object, *, step: int) -> None:
        assert isinstance(data, dict)
        assert step >= self.last_step
        self.values.append((dict(data), step))
        self.last_step = step

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = True


class ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.weight


def _tiny_loop_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimization=SimpleNamespace(
            gradient_accumulation_steps=1,
            per_rank_microbatch_size=1,
            gradient_clip_norm=10.0,
            checkpoint_every_optimizer_steps=1,
            log_every_optimizer_steps=1,
        )
    )


def test_tiny_cpu_optimizer_orchestration_records_audit_and_bound(
    tmp_path: Path,
) -> None:
    batch = make_batch(leads=(0.5,))
    rank_batch = RankLocalBatch(
        training=batch,
        slots=(DrawSlot(logical_position=0, row_index=0),),
        active_slot_indices=(0,),
        loss_weights=torch.ones(1, dtype=torch.float32),
    )
    model = ScalarModel()
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tracking = AuditRun()
    checkpoints: list[TrainingCursor] = []

    def forward_loss(
        selected_model: nn.Module,
        selected_batch: object,
        denominators: object,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del global_step
        assert isinstance(selected_batch, type(batch))
        prediction = selected_model().reshape(1, 1, 1, 1)  # type: ignore[operator]
        target = selected_batch.labels.model_target_mm[:, :, :1, 1:2]
        validity = selected_batch.labels.target_validity[:, :, :1, 1:2]
        omega = selected_batch.labels.omega
        contribution, metric = distributed_direct_loss_contribution(
            statistic="mean",
            prediction_mm=prediction,
            target_mm=target,
            target_validity=validity,
            omega=omega,
            denominators=denominators,  # type: ignore[arg-type]
        )
        return contribution, metric.reshape(1)

    result = run_optimizer_loop(
        model=model,
        loader=(rank_batch,),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_tiny_loop_config(),  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=1,
        forward_loss=forward_loss,
        tracking=tracking,
        tracking_step=-1,
        role_name="direct-mean",
        checkpoint_callback=lambda cursor, complete: checkpoints.append(cursor),
    )
    assert result.cursor == TrainingCursor(0, 1, 1, 0)
    assert result.optimizer_steps_run == 1
    assert result.stopped_by_limit is True
    assert not torch.equal(model.weight.detach(), original)
    assert checkpoints == [result.cursor]
    metrics = tracking.values[0][0]
    for suffix in (
        "data_wait_s",
        "h2d_s",
        "forward_s",
        "backward_s",
        "optimizer_s",
        "samples_per_s",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    ):
        assert f"direct-mean/{suffix}" in metrics


def test_optimizer_cursor_is_rank_local_accumulation_aligned() -> None:
    config = _tiny_loop_config()
    config.optimization.gradient_accumulation_steps = 2
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="rank-local slot count"):
        run_optimizer_loop(
            model=model,
            loader=(),
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,  # type: ignore[arg-type]
            runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
            start_cursor=TrainingCursor(0, 1, 1, 0),
            max_optimizer_steps=1,
            forward_loss=lambda *args: (model.weight * 0.0, model.weight.reshape(1)),
            tracking=AuditRun(),
            tracking_step=-1,
            role_name="direct-mean",
            checkpoint_callback=lambda cursor, complete: None,
        )


def test_dataloader_iterator_seed_does_not_perturb_restored_model_rng() -> None:
    class SeedConsumingEmpty:
        def __iter__(self):
            torch.rand(1)
            return iter(())

    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    torch.manual_seed(812)
    expected_state = torch.get_rng_state().clone()
    result = run_optimizer_loop(
        model=model,
        loader=SeedConsumingEmpty(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_tiny_loop_config(),  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=None,
        forward_loss=lambda *args: (model.weight * 0.0, model.weight.reshape(1)),
        tracking=AuditRun(),
        tracking_step=-1,
        role_name="direct-mean",
        checkpoint_callback=lambda cursor, complete: None,
    )
    assert result.optimizer_steps_run == 0
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_metric_audit_honors_log_interval_and_flushes_last_step() -> None:
    batch = make_batch(leads=(0.5,))
    batches = tuple(
        RankLocalBatch(
            training=batch,
            slots=(DrawSlot(logical_position=index, row_index=index),),
            active_slot_indices=(0,),
            loss_weights=torch.ones(1),
        )
        for index in range(3)
    )
    config = _tiny_loop_config()
    config.optimization.log_every_optimizer_steps = 2
    config.optimization.checkpoint_every_optimizer_steps = 99
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tracking = AuditRun()

    def loss(
        selected_model: nn.Module,
        selected_batch: object,
        denominators: object,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del selected_batch, denominators, global_step
        value = selected_model()  # type: ignore[operator]
        return value.square(), value.detach().reshape(1)

    run_optimizer_loop(
        model=model,
        loader=batches,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=None,
        forward_loss=loss,
        tracking=tracking,
        tracking_step=-1,
        role_name="direct-mean",
        checkpoint_callback=lambda cursor, complete: None,
    )
    assert len(tracking.values) == 2
    assert [record[0]["direct-mean/optimizer_step"] for record in tracking.values] == [
        2,
        3,
    ]


def _gloo_padding_worker(
    rank: int, init_path: str, output_dir: str
) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    runtime = DistributedRuntime(rank, rank, 2, torch.device("cpu"), True)
    try:
        model = nn.Linear(1, 1, bias=True, dtype=torch.float32)
        with torch.no_grad():
            model.weight.fill_(0.25)
            model.bias.fill_(-0.1)
        labels = []
        if rank == 0:
            labels = [
                SimpleNamespace(
                    target_validity=torch.ones((1, 1, 1, 1), dtype=torch.bool),
                    target_wet=torch.ones((1, 1, 1, 1), dtype=torch.bool),
                    omega=torch.tensor([2.0], dtype=torch.float32),
                )
            ]
        denominators = accumulation_window_denominators(
            labels, device="cpu", reduce_sum=runtime.reduce_sum
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            prediction = model(torch.tensor([[2.0]], dtype=torch.float32)).reshape(
                1, 1, 1, 1
            )
            contribution, _ = distributed_direct_loss_contribution(
                statistic="mean",
                prediction_mm=prediction,
                target_mm=torch.tensor([[[[1.2]]]], dtype=torch.float32),
                target_validity=torch.ones((1, 1, 1, 1), dtype=torch.bool),
                omega=torch.tensor([2.0], dtype=torch.float32),
                denominators=denominators,
            )
            contribution.backward()
        else:
            stage2._connected_zero(model).backward()
        reduce_gradients_sum(model, runtime, bucket_bytes=8)
        optimizer.step()
        torch.save(model.state_dict(), Path(output_dir) / f"rank-{rank}.pt")
    finally:
        runtime.close()


def test_two_rank_gloo_padding_gradient_equals_concatenated_objective(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "gloo-init"
    multiprocessing.spawn(
        _gloo_padding_worker,
        args=(str(init_path), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    torch.testing.assert_close(states[0]["weight"], states[1]["weight"])
    torch.testing.assert_close(states[0]["bias"], states[1]["bias"])

    reference = nn.Linear(1, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        reference.weight.fill_(0.25)
        reference.bias.fill_(-0.1)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
    labels = SimpleNamespace(
        target_validity=torch.ones((1, 1, 1, 1), dtype=torch.bool),
        target_wet=torch.ones((1, 1, 1, 1), dtype=torch.bool),
        omega=torch.tensor([2.0], dtype=torch.float32),
    )
    denominators = accumulation_window_denominators([labels], device="cpu")
    prediction = reference(torch.tensor([[2.0]], dtype=torch.float32)).reshape(
        1, 1, 1, 1
    )
    loss, _ = distributed_direct_loss_contribution(
        statistic="mean",
        prediction_mm=prediction,
        target_mm=torch.tensor([[[[1.2]]]], dtype=torch.float32),
        target_validity=labels.target_validity,
        omega=labels.omega,
        denominators=denominators,
    )
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(states[0]["weight"], reference.weight)
    torch.testing.assert_close(states[0]["bias"], reference.bias)


def test_production_model_mapping_is_not_silently_ignored() -> None:
    config = load_stage2_config(CONFIG_PATH)
    contract = regression_system_config_from_stage2(config)
    assert contract.direct_mean_required is True
    assert contract.direct_q50_required is True
    assert contract.system.activation_checkpoint is True
    assert contract.system.regression_query_chunk_size == 128
    assert asdict(contract.system.flow_config) == {
        "pyramid_levels": 4,
        "warps_per_level": 3,
        "irls_iterations": 2,
        "window_size": 7,
        "gaussian_sigma": 1.5,
        "regularization": 0.001,
        "charbonnier_epsilon": 0.01,
        "minimum_eigenvalue": 0.0002,
        "photometric_scale": 0.15,
        "fb_error_scale_km": 1.2,
        "maximum_speed_km_per_hour": 150.0,
        "maximum_increment_pixels": 2.0,
        "recency_half_life_minutes": 30.0,
        "minimum_source_valid_fraction": 1e-06,
    }


def _launch_arguments() -> SimpleNamespace:
    config = load_stage2_config(CONFIG_PATH)
    return SimpleNamespace(
        source_root=SOURCE_ROOT,
        config=CONFIG_PATH,
        launch_identity=Path("launch-identity.json"),
        launch_identity_sha256="d" * 64,
        container_image_digest="sha256:" + "f" * 64,
        runtime_report=Path("pip-report.json"),
        data_contract=SOURCE_ROOT / "configs/data-contract-v1.1.3b.json",
        require_world_size=2,
        precision="float32",
        disable_tf32=True,
        fail_on_fallback=True,
        target_widths=config.runtime.target_widths,
        context_widths=config.runtime.context_widths,
        era_latent_channels=128,
        era_grid_size=33,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        num_workers=12,
        prefetch_factor=2,
        max_optimizer_steps=1,
        bounded_training_year=None,
        expected_training_items=None,
        phases=("crossfit",),
        roles=("fold:0",),
        run_id="bounded-tune",
    )


def test_launch_contract_accepts_bounded_fold_and_rejects_fallback() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    selection = validate_launch_arguments(
        arguments,
        config,
        environ={
            "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
            "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
            "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
            "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    assert selection.partial is True
    assert selection.includes_role("fold", 0)
    with pytest.raises(ValueError, match="fallback"):
        validate_launch_arguments(
            arguments,
            config,
            environ={"KCORRDIFF_ALLOW_CPU_FALLBACK": "1"},
        )


def test_launch_contract_accepts_one_gpu_with_equivalent_global_batch() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.require_world_size = 1
    arguments.gradient_accumulation_steps = 2
    selection = validate_launch_arguments(
        arguments,
        config,
        environ={
            "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
            "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
            "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
            "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    execution = execution_config_for_topology(
        config,
        world_size=1,
        per_rank_microbatch_size=8,
        gradient_accumulation_steps=2,
    )
    assert selection.partial is True
    assert execution.optimization.gradient_accumulation_steps == 2
    assert (
        execution.optimization.global_effective_batch_size
        == 1
        * execution.optimization.per_rank_microbatch_size
        * execution.optimization.gradient_accumulation_steps
        == 16
    )
    assert execution.sha256 == config.sha256
    assert execution.raw["optimization"]["gradient_accumulation_steps"] == 2  # type: ignore[index]
    assert execution.raw["optimization"]["global_effective_batch_size"] == 16  # type: ignore[index]
    assert config.raw["optimization"]["gradient_accumulation_steps"] == 1  # type: ignore[index]
    with pytest.raises(TypeError):
        execution.raw["optimization"]["gradient_accumulation_steps"] = 4  # type: ignore[index]

    arguments.gradient_accumulation_steps = 1
    with pytest.raises(ValueError, match="preserve the configured global"):
        validate_launch_arguments(arguments, config, environ={
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        })


def test_execution_topology_rejects_non_power_of_two_microbatch() -> None:
    config = load_stage2_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="power of two"):
        execution_config_for_topology(
            config,
            world_size=1,
            per_rank_microbatch_size=3,
            gradient_accumulation_steps=2,
        )


def test_bounded_2020_deployment_accepts_b12_without_accumulation() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.require_world_size = 1
    arguments.per_rank_microbatch_size = 12
    arguments.gradient_accumulation_steps = 1
    arguments.max_optimizer_steps = None
    arguments.phases = ("deployment",)
    arguments.roles = ("deployment",)
    arguments.bounded_training_year = 2020
    arguments.expected_training_items = 220_260
    selection = validate_launch_arguments(
        arguments,
        config,
        environ={
            "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
            "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
            "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
            "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    execution = execution_config_for_topology(
        config,
        world_size=1,
        per_rank_microbatch_size=12,
        gradient_accumulation_steps=1,
        allow_bounded_microbatch_override=True,
    )
    assert selection.partial is True
    assert selection.bounded_training_year == 2020
    assert execution.optimization.per_rank_microbatch_size == 12
    assert execution.optimization.gradient_accumulation_steps == 1
    assert execution.optimization.global_effective_batch_size == 12
    assert execution.raw["loader_tuning"]["selected_batch_size_per_rank"] == 12


def test_bounded_year_override_rejects_crossfit_or_accumulation() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.require_world_size = 1
    arguments.per_rank_microbatch_size = 12
    arguments.gradient_accumulation_steps = 1
    arguments.bounded_training_year = 2020
    arguments.expected_training_items = 220_260
    with pytest.raises(ValueError, match="deployment-only"):
        validate_launch_arguments(arguments, config, environ={})

    with pytest.raises(ValueError, match="no accumulation"):
        execution_config_for_topology(
            config,
            world_size=1,
            per_rank_microbatch_size=12,
            gradient_accumulation_steps=2,
            allow_bounded_microbatch_override=True,
        )


def test_single_node_fold_worker_uses_porsche_b12_without_accumulation() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.require_world_size = 1
    arguments.per_rank_microbatch_size = 12
    arguments.gradient_accumulation_steps = 1
    arguments.max_optimizer_steps = None
    arguments.single_node_fold_worker = True
    arguments.node_name = "porsche"
    selection = validate_launch_arguments(
        arguments,
        config,
        environ={
            "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
            "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
            "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
            "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    execution = execution_config_for_topology(
        config,
        world_size=1,
        per_rank_microbatch_size=12,
        gradient_accumulation_steps=1,
        allow_single_node_fold_override=True,
    )
    assert selection.single_node_fold_worker is True
    assert selection.node_name == "porsche"
    assert execution.optimization.global_effective_batch_size == 12
    assert execution.raw["optimization"]["global_effective_batch_size"] == 12
    assert execution.raw["loader_tuning"]["selected_batch_size_per_rank"] == 12
    assert execution.sha256 == config.sha256


def test_single_node_fold_worker_rejects_node_batch_or_partial_run() -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.require_world_size = 1
    arguments.per_rank_microbatch_size = 12
    arguments.gradient_accumulation_steps = 1
    arguments.max_optimizer_steps = None
    arguments.single_node_fold_worker = True
    arguments.node_name = "bentley"
    with pytest.raises(ValueError, match="PVC-local GPU node"):
        validate_launch_arguments(arguments, config, environ={})

    arguments.node_name = "porsche"
    arguments.max_optimizer_steps = 1
    with pytest.raises(ValueError, match="no truncation"):
        validate_launch_arguments(arguments, config, environ={})

    with pytest.raises(ValueError, match="B12/accum1"):
        execution_config_for_topology(
            config,
            world_size=1,
            per_rank_microbatch_size=24,
            gradient_accumulation_steps=1,
            allow_single_node_fold_override=True,
        )


def test_single_rank_skips_model_broadcast_and_multi_rank_is_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def broadcast(_value: torch.Tensor, *, src: int) -> None:
        assert src == 0
        calls.append(torch.is_grad_enabled())

    monkeypatch.setattr(distributed, "broadcast", broadcast)
    model = nn.Linear(2, 2)
    single = DistributedRuntime(0, 0, 1, torch.device("cpu"), True)
    single.broadcast_model(model)
    assert calls == []

    multi = DistributedRuntime(0, 0, 2, torch.device("cpu"), True)
    multi.broadcast_model(model)
    assert calls and calls == [False] * len(calls)


def test_launch_contract_binds_loader_tuning_and_selection_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    environment = {
        "KCORRDIFF_ALLOW_CPU_FALLBACK": "0",
        "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK": "0",
        "KCORRDIFF_ALLOW_PRECISION_FALLBACK": "0",
        "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK": "0",
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    arguments.num_workers = 8
    with pytest.raises(ValueError, match="worker/prefetch"):
        validate_launch_arguments(arguments, config, environ=environment)

    arguments.num_workers = 12
    original_sha256_file = stage2.sha256_file

    def changed_selection_hash(path: Path) -> str:
        if path.name == "stage2-loader-selection.json":
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(stage2, "sha256_file", changed_selection_hash)
    with pytest.raises(ValueError, match="selection artifact SHA-256 mismatch"):
        validate_launch_arguments(arguments, config, environ=environment)


def test_launch_contract_rejects_hash_valid_but_semantically_forged_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    environment = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    original_sha256_file = stage2.sha256_file

    def forged_hash(path: Path) -> str:
        if path.name == "stage2-loader-selection.json":
            return str(config.raw["loader_tuning"]["selection_artifact_sha256"])  # type: ignore[index]
        return original_sha256_file(path)

    monkeypatch.setattr(stage2, "sha256_file", forged_hash)
    called: dict[str, object] = {}

    def reject_semantics(*args: object, **kwargs: object) -> object:
        called["path"] = args[0]
        raise ValueError("selection artifact disagrees with Stage 2 launch/config")

    monkeypatch.setattr(stage2, "load_stage2_selection_contract", reject_semantics)
    with pytest.raises(ValueError, match="selection artifact disagrees"):
        validate_launch_arguments(arguments, config, environ=environment)
    assert Path(called["path"]) == SOURCE_ROOT / "configs/stage2-loader-selection.json"


def test_cli_help_and_tuning_switches(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        stage2._parse_args(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--source-root",
        "--launch-identity",
        "--launch-identity-sha256",
        "--container-image-digest",
        "--runtime-report",
        "--max-optimizer-steps",
        "--phases",
        "--roles",
        "--wandb-mode",
        "--output-dir",
        "--oof-output-dir",
        "--fold-set-manifest",
        "--fold-set-manifest-sha256",
    ):
        assert option in help_text


def test_fold_set_launch_arguments_require_pair_and_reject_fold_worker_ambiguity(
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    arguments = _launch_arguments()
    arguments.fold_set_manifest = Path("fold-set-manifest.json")
    arguments.fold_set_manifest_sha256 = None
    with pytest.raises(ValueError, match="required together"):
        validate_launch_arguments(arguments, config, environ={})

    arguments.fold_set_manifest = None
    arguments.fold_set_manifest_sha256 = "a" * 64
    with pytest.raises(ValueError, match="required together"):
        validate_launch_arguments(arguments, config, environ={})

    arguments.fold_set_manifest = Path("fold-set-manifest.json")
    arguments.fold_set_manifest_sha256 = "a" * 64
    with pytest.raises(ValueError, match="cannot select fold workers"):
        validate_launch_arguments(arguments, config, environ={})

    arguments.roles = ()
    arguments.single_node_fold_worker = True
    arguments.node_name = "porsche"
    with pytest.raises(ValueError, match="cannot select fold workers"):
        validate_launch_arguments(arguments, config, environ={})

    arguments.phases = ("oof",)
    arguments.single_node_fold_worker = False
    arguments.node_name = None
    arguments.max_optimizer_steps = None
    selection = validate_launch_arguments(
        arguments,
        config,
        environ={
            "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
            "KCORRDIFF_REQUIRE_PRECISION": "float32",
            "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    assert selection.phases == ("oof",)


def test_storage_preflight_counts_atomic_peak_and_partial_distinction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "checkpoints"
    oof = tmp_path / "oof"
    full = preflight_storage(
        output_dir=output,
        oof_output_dir=oof,
        model_parameter_count=10,
        selected_checkpoint_count=6,
        oof_dense_bytes=100,
        include_oof=True,
        reserve_bytes=1,
    )
    partial = preflight_storage(
        output_dir=output,
        oof_output_dir=oof,
        model_parameter_count=10,
        selected_checkpoint_count=1,
        oof_dense_bytes=100,
        include_oof=False,
        reserve_bytes=1,
    )
    assert full.checkpoint_bytes == 10 * 16 * 7
    assert partial.checkpoint_bytes == 10 * 16 * 2
    # Lossless direct compression has one capped final artifact, not dense
    # rank-partials plus a second dense merge target.
    assert full.oof_peak_bytes == 100
    assert full.oof_compressed_cap_bytes == 100
    assert full.oof_staging_bytes == 0
    assert partial.oof_peak_bytes == 0

    monkeypatch.setattr(
        stage2.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1, used=0, free=1),
    )
    with pytest.raises(OSError, match="byte budget"):
        preflight_storage(
            output_dir=output,
            oof_output_dir=oof,
            model_parameter_count=10,
            selected_checkpoint_count=6,
            oof_dense_bytes=100,
            include_oof=True,
            reserve_bytes=1,
        )


def test_partial_selection_storage_counts_every_possible_role_and_selected_oof() -> None:
    selection = RunSelection(stage2._ALL_PHASES, (), 100)
    assert selection.partial is True
    assert stage2._selected_training_role_count(selection, folds=3) == 6
    assert "oof" in selection.phases


def _draw_row(index: int, *, duplicate: bool = False) -> DrawRow:
    semantic = 0 if duplicate else index
    return DrawRow(
        global_example_index=index,
        sample_id=f"sample-{semantic}",
        t0_utc="2020-01-01T00:00:00+00:00",
        lead_hours=0.5,
        condition_signature="era5_oracle:era=1:tp=1:full_trajectory",
        block_id="block-0",
        stratum="event",
        split="outer_train",
        p_target=0.5,
        p_draw=0.5,
        omega=1.0,
        fold_id=0,
    )


def test_residual_sidecar_binds_hash_provenance_and_draw_multiplicity(
    tmp_path: Path,
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    rows = (_draw_row(0), _draw_row(1, duplicate=True))
    selected = ((0, rows[0]),)
    multiplicities = stage2._draw_multiplicities(rows)
    accumulator = stage2.ResidualScaleAccumulator(
        epsilon_scale=0.001,
        minimum_independent_blocks=1,
        minimum_block_ess=1.0,
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=rows[0].condition_signature,
        block_id=rows[0].block_id,
        target_z=[1.0],
        mu_z_oof=[0.0],
        target_validity=[True],
        omega=1.0,
        multiplicity=2,
    )
    factory = SimpleNamespace(
        artifacts=SimpleNamespace(artifact_hashes={"draw_manifest": "a" * 64})
    )
    checkpoint = tmp_path / "fold.pt"
    checkpoint.write_bytes(b"checkpoint")
    fold_result = RoleTrainingResult(
        role="fold",
        fold_id=0,
        complete=True,
        checkpoint_path=checkpoint,
        checkpoint_sha256="b" * 64,
        cursor=TrainingCursor(0, 1, 1, 0),
        optimizer_steps_run=0,
        metrics_step=0,
        record=None,
    )
    path = tmp_path / "residual-state.json"
    stage2._write_residual_state(
        path,
        accumulator,
        config=config,
        factory=factory,  # type: ignore[arg-type]
        fold_result=fold_result,
        fold_id=0,
        rank=0,
        selected=selected,
        multiplicities=multiplicities,
        oof_partial_manifest_sha256="c" * 64,
    )
    restored = stage2._read_residual_state(
        path,
        config=config,
        factory=factory,  # type: ignore[arg-type]
        fold_result=fold_result,
        fold_id=0,
        rank=0,
        selected=selected,
        multiplicities=multiplicities,
        oof_partial_manifest_sha256="c" * 64,
    )
    assert restored.result()["records"][0]["items"] == 2  # type: ignore[index]
    raw = json.loads(path.read_text())
    raw["rank"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        stage2._read_residual_state(
            path,
            config=config,
            factory=factory,  # type: ignore[arg-type]
            fold_result=fold_result,
            fold_id=0,
            rank=0,
            selected=selected,
            multiplicities=multiplicities,
            oof_partial_manifest_sha256="c" * 64,
        )


def test_draw_multiplicity_rejects_changed_duplicate_weight_or_identity() -> None:
    first = _draw_row(0)
    changed_weight = DrawRow(
        **{
            **asdict(_draw_row(1, duplicate=True)),
            "omega": 2.0,
        }
    )
    with pytest.raises(ValueError, match="repeated OOF identity changed metadata"):
        stage2._draw_multiplicities((first, changed_weight))

    changed_block = DrawRow(
        **{
            **asdict(_draw_row(1, duplicate=True)),
            "block_id": "block-1",
        }
    )
    with pytest.raises(ValueError, match="repeated OOF identity changed metadata"):
        stage2._draw_multiplicities((first, changed_block))


def test_existing_residual_scales_must_exactly_reproduce_verified_moments(
    tmp_path: Path,
) -> None:
    accumulator = stage2.ResidualScaleAccumulator(
        epsilon_scale=0.001,
        minimum_independent_blocks=1,
        minimum_block_ess=1.0,
    )
    accumulator.update(
        lead_hours=0.5,
        condition_signature=_draw_row(0).condition_signature,
        block_id=_draw_row(0).block_id,
        target_z=[2.0],
        mu_z_oof=[1.0],
        target_validity=[True],
        omega=2.0,
    )
    path = tmp_path / "residual-scales.json"
    digest = stage2.write_residual_scales(
        path,
        accumulator,
        oof_manifest_sha256="a" * 64,
        regression_checkpoint_set_sha256="b" * 64,
    )
    assert stage2._validate_existing_residual_scales(
        path,
        accumulator,
        oof_manifest_sha256="a" * 64,
        regression_checkpoint_set_sha256="b" * 64,
    ) == digest

    tampered = json.loads(path.read_text())
    tampered["records"][0]["scale"] = 9.0
    path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not reproducible"):
        stage2._validate_existing_residual_scales(
            path,
            accumulator,
            oof_manifest_sha256="a" * 64,
            regression_checkpoint_set_sha256="b" * 64,
        )


def _fabricated_verified_fold_set(
    tmp_path: Path, *, config_sha256: str, artifact_hashes: dict[str, str]
) -> VerifiedStage2FoldSet:
    launch = FakeLaunchIdentity().provenance()
    lineage = FoldProducerLineage(
        protocol_version="v1.1.3b",
        config_sha256=config_sha256,
        draw_manifest_sha256=artifact_hashes["draw_manifest"],
        launch_identity=launch,
        artifact_hashes=artifact_hashes,
        node_name="porsche",
        world_size=1,
        per_rank_microbatch_size=12,
        gradient_accumulation_steps=1,
        global_effective_batch_size=12,
        policy_sha256=stage2.SINGLE_NODE_FOLD_POLICY_SHA256,
    )
    manifest = tmp_path / "fold-set-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    folds: list[VerifiedFoldCheckpoint] = []
    for fold_id in range(3):
        checkpoint = tmp_path / f"fold-{fold_id}" / "final.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(f"fold-{fold_id}".encode())
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        partial_manifest = checkpoint.parent / "partial-manifest.json"
        partial_manifest.write_text("{}\n", encoding="utf-8")
        partial_sha256 = hashlib.sha256(partial_manifest.read_bytes()).hexdigest()
        cursor = TrainingCursor(0, 12, 1, 0)
        blocks_sha256 = hashlib.sha256(f"blocks-{fold_id}".encode()).hexdigest()
        record = CheckpointRecord(
            fold_id=fold_id,
            role="fold",
            path=str(checkpoint.resolve()),
            sha256=checkpoint_sha256,
            training_blocks_sha256=blocks_sha256,
            config_sha256=config_sha256,
            draw_manifest_sha256=artifact_hashes["draw_manifest"],
            global_step=cursor.optimizer_step,
        )
        folds.append(
            VerifiedFoldCheckpoint(
                fold_id=fold_id,
                checkpoint_path=checkpoint,
                checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_sha256=checkpoint_sha256,
                partial_manifest_path=partial_manifest,
                partial_manifest_sha256=partial_sha256,
                provenance=CheckpointProvenance(
                    protocol_version="v1.1.3b",
                    config_sha256=config_sha256,
                    draw_manifest_sha256=artifact_hashes["draw_manifest"],
                    launch_identity_sha256=launch["artifact_sha256"],
                    source_tree_sha256=launch["source_tree_sha256"],
                    container_image_sha256=launch["container_image_sha256"],
                    runtime_report_sha256=launch["runtime_report_sha256"],
                    data_contract_sha256=launch["data_contract_sha256"],
                    role="fold",
                    fold_id=fold_id,
                    world_size=1,
                    per_rank_microbatch_size=12,
                    gradient_accumulation_steps=1,
                ),
                cursor=cursor,
                plan_semantic_sha256=str(fold_id) * 64,
                plan_optimizer_steps=cursor.optimizer_step,
                training_blocks=(f"block-{fold_id}",),
                training_blocks_sha256=blocks_sha256,
                lineage=lineage,
                record=record,
            )
        )
    return VerifiedStage2FoldSet(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        lineage=lineage,
        folds=tuple(folds),
    )


def test_imported_fold_set_seeds_crossfit_and_is_forwarded_to_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    artifact_hashes = {"draw_manifest": "a" * 64}
    imported = _fabricated_verified_fold_set(
        tmp_path,
        config_sha256=config.sha256,
        artifact_hashes=artifact_hashes,
    )
    factory = SimpleNamespace(
        artifacts=SimpleNamespace(artifact_hashes=artifact_hashes, rows=())
    )
    monkeypatch.setattr(stage2, "initialize_tracking", lambda **_kwargs: AuditRun())

    def unexpected_train_role(**_kwargs: object) -> RoleTrainingResult:
        raise AssertionError(
            "an imported fold must never enter training/resume loading"
        )

    monkeypatch.setattr(stage2, "train_role", unexpected_train_role)
    captured: dict[str, object] = {}

    def fake_infer_oof_rank_partials(**kwargs: object) -> tuple[Path, ...]:
        captured["imported_fold_set"] = kwargs["imported_fold_set"]
        captured["fold_results"] = kwargs["fold_results"]
        return ()

    def fake_merge_oof_partials(**kwargs: object) -> stage2.OOFMergeResult:
        fold_results = kwargs["fold_results"]
        assert isinstance(fold_results, dict)
        assert set(fold_results) == {0, 1, 2}
        return stage2.OOFMergeResult(
            artifact_dir=tmp_path / "oof" / "artifact",
            manifest_sha256="9" * 64,
            residual_accumulator=stage2.ResidualScaleAccumulator(
                epsilon_scale=config.crossfit.epsilon_residual_scale
            ),
            items=0,
        )

    monkeypatch.setattr(stage2, "infer_oof_rank_partials", fake_infer_oof_rank_partials)
    monkeypatch.setattr(stage2, "merge_oof_partials", fake_merge_oof_partials)

    result = run_stage2(
        config=config,
        factory=factory,  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 2, torch.device("cpu"), False),
        launch_identity=FakeLaunchIdentity(),  # type: ignore[arg-type]
        output_dir=tmp_path / "checkpoints",
        oof_output_dir=tmp_path / "oof",
        selection=RunSelection(("crossfit", "oof"), (), None),
        num_workers=0,
        prefetch_factor=2,
        wandb_mode="disabled",
        run_id="imported-oof",
        imported_fold_set=imported,
    )

    assert captured["imported_fold_set"] is imported
    seeded = captured["fold_results"]
    assert isinstance(seeded, dict)
    assert set(seeded) == {0, 1, 2}
    assert all(value.optimizer_steps_run == 0 for value in seeded.values())
    assert tuple(value.checkpoint_path for value in seeded.values()) == tuple(
        fold.checkpoint_path for fold in imported.folds
    )
    assert tuple(role.fold_id for role in result.role_results) == (0, 1, 2)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["imported_fold_set"] == imported.audit_json()


def test_bounded_orchestration_writes_only_partial_manifest_with_latest_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    checkpoint = tmp_path / "checkpoints" / "fold-0" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resumable checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def fake_train_role(**kwargs: object) -> RoleTrainingResult:
        assert kwargs["role"] == "fold"
        assert kwargs["fold_id"] == 0
        assert kwargs["max_optimizer_steps"] == 1
        return RoleTrainingResult(
            role="fold",
            fold_id=0,
            complete=False,
            checkpoint_path=checkpoint,
            checkpoint_sha256=digest,
            cursor=TrainingCursor(0, 4, 1, 0),
            optimizer_steps_run=1,
            metrics_step=0,
            record=None,
        )

    monkeypatch.setattr(stage2, "train_role", fake_train_role)
    tracking_initialization: dict[str, object] = {}

    def fake_initialize_tracking(**kwargs: object) -> AuditRun:
        tracking_initialization.update(kwargs)
        return AuditRun()

    monkeypatch.setattr(stage2, "initialize_tracking", fake_initialize_tracking)
    factory = SimpleNamespace(
        artifacts=SimpleNamespace(
            artifact_hashes={
                "draw_manifest": "a" * 64,
                "candidate_manifest": "b" * 64,
                "normalization": "c" * 64,
            }
        )
    )
    output = tmp_path / "checkpoints"
    result = run_stage2(
        config=config,
        factory=factory,  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 2, torch.device("cpu"), False),
        launch_identity=FakeLaunchIdentity(),  # type: ignore[arg-type]
        output_dir=output,
        oof_output_dir=tmp_path / "oof",
        selection=RunSelection(("crossfit",), ("fold:0",), 1),
        num_workers=0,
        prefetch_factor=2,
        wandb_mode="disabled",
        run_id="bounded",
    )
    assert result.partial is True
    assert result.stopped_by_limit is True
    assert result.manifest_path.name == "partial-manifest.json"
    assert not (output / "stage2-manifest.json").exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["partial"] is True and manifest["complete"] is False
    assert manifest["launch_identity"] == FakeLaunchIdentity().provenance()
    assert tracking_initialization["launch_identity_sha256"] == "d" * 64
    assert tracking_initialization["config"]["launch_identity"] == (  # type: ignore[index]
        FakeLaunchIdentity().provenance()
    )
    assert manifest["role_checkpoints"] == []
    assert manifest["role_states"] == [
        {
            "role": "fold",
            "fold_id": 0,
            "complete": False,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": digest,
            "cursor": asdict(TrainingCursor(0, 4, 1, 0)),
            "optimizer_steps_run_this_invocation": 1,
        }
    ]
    assert manifest["topology"]["cursor_global_example_index_semantics"] == (
        "rank_local_plan_slots_consumed_including_padding"
    )


def test_optimizer_step_limit_stops_before_creating_next_role_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    output = tmp_path / "checkpoints"
    calls: list[tuple[str, int | None, int | None]] = []

    def fake_train_role(**kwargs: object) -> RoleTrainingResult:
        role = str(kwargs["role"])
        fold_id = kwargs["fold_id"]
        limit = kwargs["max_optimizer_steps"]
        calls.append((role, fold_id, limit))  # type: ignore[arg-type]
        checkpoint = output / "fold-0" / "final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"complete first role")
        return RoleTrainingResult(
            role="fold",
            fold_id=0,
            complete=True,
            checkpoint_path=checkpoint,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            cursor=TrainingCursor(0, 4, 1, 0),
            optimizer_steps_run=1,
            metrics_step=0,
            record=None,
        )

    monkeypatch.setattr(stage2, "train_role", fake_train_role)
    factory = SimpleNamespace(
        artifacts=SimpleNamespace(
            artifact_hashes={"draw_manifest": "a" * 64}
        )
    )
    result = run_stage2(
        config=config,
        factory=factory,  # type: ignore[arg-type]
        runtime=DistributedRuntime(0, 0, 2, torch.device("cpu"), False),
        launch_identity=FakeLaunchIdentity(),  # type: ignore[arg-type]
        output_dir=output,
        oof_output_dir=tmp_path / "oof",
        selection=RunSelection(
            ("crossfit",), ("fold:0", "fold:1"), 1
        ),
        num_workers=0,
        prefetch_factor=2,
        wandb_mode="disabled",
        run_id="boundary",
    )
    assert result.partial is True and result.stopped_by_limit is True
    assert calls == [("fold", 0, 1)]
    assert not (output / "fold-1").exists()


def test_full_default_phase_set_contains_direct_comparison_arms() -> None:
    selection = RunSelection(stage2._ALL_PHASES, (), None)
    assert selection.partial is False
    assert selection.phases[-2:] == ("direct_mean", "direct_q50")
    with pytest.raises(ValueError, match="requires phase"):
        RunSelection(("crossfit",), ("direct_mean",), 1)


def test_complete_manifest_loader_rejects_partial_name_and_duplicate_roles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stage2-manifest.json"
    records = [
        {"role": "fold", "fold_id": fold_id}
        for fold_id in range(3)
    ] + [
        {"role": "deployment", "fold_id": None},
        {"role": "direct_mean", "fold_id": None},
        {"role": "direct_q50", "fold_id": None},
    ]
    payload = {
        "format_version": stage2.STAGE2_MANIFEST_FORMAT,
        "protocol_version": "v1.1.3b",
        "complete": True,
        "partial": False,
        "launch_identity": FakeLaunchIdentity().provenance(),
        "role_checkpoints": records,
        "oof_manifest_sha256": "a" * 64,
        "residual_scales_sha256": "b" * 64,
        "fold_checkpoint_set_sha256": "c" * 64,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = stage2.load_complete_stage2_manifest(path, expected_sha256=digest)
    assert loaded["complete"] is True

    without_identity = dict(payload)
    without_identity.pop("launch_identity")
    path.write_text(json.dumps(without_identity, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="launch identity schema"):
        stage2.load_complete_stage2_manifest(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    partial = tmp_path / "partial-manifest.json"
    partial.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="exact stage2-manifest"):
        stage2.load_complete_stage2_manifest(
            partial,
            expected_sha256=hashlib.sha256(partial.read_bytes()).hexdigest(),
        )

    payload["role_checkpoints"] = records + [records[0]]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed/duplicate"):
        stage2.load_complete_stage2_manifest(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_resume_rejects_existing_partial_manifest_from_other_launch(
    tmp_path: Path,
) -> None:
    config = load_stage2_config(CONFIG_PATH)
    output = tmp_path / "checkpoints"
    output.mkdir()
    identity = FakeLaunchIdentity().provenance()
    identity["source_tree_sha256"] = "2" * 64
    (output / "partial-manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": config.sha256,
                "draw_manifest_sha256": "a" * 64,
                "launch_identity": identity,
            }
        ),
        encoding="utf-8",
    )
    factory = SimpleNamespace(
        artifacts=SimpleNamespace(artifact_hashes={"draw_manifest": "a" * 64})
    )
    with pytest.raises(ValueError, match="immutable launch identity mismatch"):
        run_stage2(
            config=config,
            factory=factory,  # type: ignore[arg-type]
            runtime=DistributedRuntime(0, 0, 2, torch.device("cpu"), False),
            launch_identity=FakeLaunchIdentity(),  # type: ignore[arg-type]
            output_dir=output,
            oof_output_dir=tmp_path / "oof",
            selection=RunSelection(("crossfit",), ("fold:0",), 1),
            num_workers=0,
            prefetch_factor=2,
            wandb_mode="disabled",
            run_id="must-not-resume",
        )


def test_production_runtime_initialization_refuses_non_torchrun_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="torchrun environment"):
        stage2.initialize_distributed_runtime(require_world_size=2)

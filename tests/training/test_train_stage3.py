from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
import yaml

from kcorrdiff.training.checkpoints import TrainingCursor
from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    save_training_checkpoint,
)
from kcorrdiff.training.config import load_stage2_config
import kcorrdiff.training.train_stage3 as stage3
from kcorrdiff.training.train_stage2 import DistributedRuntime
from kcorrdiff.training.train_stage3 import (
    COMMON_ENSEMBLE_SEED,
    DEPLOYMENT_TRAINING_SEED,
    EDMForwardResult,
    FINAL_SELECTION_FORMAT,
    REPEAT_TRAINING_SEEDS,
    SCREENING_EVALUATION_FORMAT,
    SCREEN_TRAINING_SEED,
    CheckpointIdentity,
    Stage3CheckpointProvenance,
    Stage3NoiseConfig,
    counter_keyed_training_noise,
    load_final_model_selection_decision,
    load_screening_evaluation_result,
    load_stage3_checkpoint,
    load_stage3_config,
    preflight_stage3_storage,
    run_edm_optimizer_loop,
    save_stage3_checkpoint,
    source_tree_sha256,
    stage3_architecture_sha256,
    validate_launch_arguments,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "stage3-full-width.yaml"
STAGE2_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "stage2-full-width.yaml"


class AuditRun:
    def __init__(self) -> None:
        self.last_step = -1
        self.finished = False
        self.values: list[tuple[dict[str, object], int]] = []

    def log(self, data: object, *, step: int) -> None:
        assert isinstance(data, dict)
        assert step >= self.last_step
        self.last_step = step
        self.values.append((dict(data), step))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = True


class ScalarModel(nn.Module):
    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float32))


class ScalarOperations:
    def is_padding(self, batch: object) -> bool:
        return batch is None

    def valid_importance_mass(
        self, batch: object, *, device: torch.device
    ) -> torch.Tensor:
        if batch is None:
            return torch.zeros((), dtype=torch.float32, device=device)
        target, mass = batch
        del target
        return torch.tensor(float(mass), dtype=torch.float32, device=device)

    def forward(
        self,
        *,
        model: nn.Module,
        deployment: nn.Module,
        batch: object,
        global_valid_importance_mass: torch.Tensor,
        training_seed: int,
    ) -> EDMForwardResult:
        del deployment, training_seed
        assert isinstance(model, ScalarModel)
        target, mass = batch
        numerator = (model.weight - float(target)).square() * float(mass)
        return EDMForwardResult(
            loss_contribution=numerator / global_valid_importance_mass,
            weighted_square_error_sum=numerator.detach(),
            valid_importance_mass=torch.tensor(float(mass), dtype=torch.float32),
            samples=1,
        )


def _tiny_config(*, accumulation: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        optimization=SimpleNamespace(
            gradient_accumulation_steps=accumulation,
            per_rank_microbatch_size=1,
            gradient_clip_norm=100.0,
            checkpoint_every_optimizer_steps=99,
            log_every_optimizer_steps=1,
        )
    )


def _canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, payload: dict[str, object]) -> None:
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _canonical(path, payload)


def _checkpoint_records(*, seeds: tuple[int, ...]) -> tuple[CheckpointIdentity, ...]:
    return tuple(
        CheckpointIdentity(variant, seed, hashlib.sha256(f"{variant}-{seed}".encode()).hexdigest())
        for variant in ("edm_a", "edm_b")
        for seed in seeds
    )


def test_full_width_config_and_frozen_seed_contract() -> None:
    config = load_stage3_config(CONFIG_PATH)
    assert config.seed == SCREEN_TRAINING_SEED
    assert config.target_widths == (64, 128, 256, 384, 512)
    assert config.context_widths == (32, 64, 128, 256, 384)
    assert config.activation_checkpoint is True
    assert config.query_chunk_size == 128
    assert config.condition_augmentation_sha256 == (
        "813d7395ac59897a7a258ed07205fff289c262b7e8df93cd2cef4deeace9c9f8"
    )
    assert config.condition_signatures == (
        "era5_oracle:era=1:tp=0:full_trajectory",
        "era5_oracle:era=1:tp=1:full_trajectory",
        "null_provider:era=0:tp=0:no_era_access",
    )
    assert (
        2
        * config.optimization.per_rank_microbatch_size
        * config.optimization.gradient_accumulation_steps
        == config.optimization.global_effective_batch_size
    )
    assert config.optimization.per_rank_microbatch_size & (
        config.optimization.per_rank_microbatch_size - 1
    ) == 0
    assert REPEAT_TRAINING_SEEDS == (11103, 11105, 11106)
    assert COMMON_ENSEMBLE_SEED == 11104
    assert DEPLOYMENT_TRAINING_SEED == 11103


def test_stage3_raw_config_is_deeply_immutable_and_hash_bound() -> None:
    config = load_stage3_config(CONFIG_PATH)
    config_sha256 = config.sha256

    with pytest.raises(TypeError):
        config.raw["data"]["draw_manifest"] = "/tmp/unhashed-draw.jsonl"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.raw["model"]["candidates"][0]["name"] = "unhashed-arm"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.raw["data"]["condition_signatures"][0] = "unhashed-signature"  # type: ignore[index]

    assert config.sha256 == config_sha256
    assert config.raw["data"]["draw_manifest"] != "/tmp/unhashed-draw.jsonl"  # type: ignore[index]


def test_stage3_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    duplicated = source.replace("seed: 11103", "seed: 99999\nseed: 11103", 1)
    path = tmp_path / "duplicate-seed.yaml"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'seed'"):
        load_stage3_config(path)


@pytest.mark.parametrize(
    "section",
    (
        "stage2_contract",
        "model",
        "sampling_profiles",
        "model_selection",
        "calibration",
        "inference",
        "publication",
    ),
)
def test_stage3_rejects_unknown_keys_in_hashed_contract_sections(
    tmp_path: Path, section: str
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw[section]["unimplemented_contract_option"] = True
    path = tmp_path / f"invalid-{section}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_stage3_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    (
        ("model", "physical_attention_heads", 4, "frozen architecture"),
        ("model", "state_channels", True, "frozen architecture"),
        ("sampling_profiles", "solver", "euler", "sampler contract"),
        (
            "model_selection",
            "sampler_bias_d_candidates",
            ["disabled"],
            "model-selection governance",
        ),
        ("inference", "ensemble_median", "interpolated", "inference contract"),
        ("inference", "inverse_transform_a0_mm", True, "inference contract"),
        (
            "publication",
            "require_stage2_hash_lineage",
            False,
            "publication requirement",
        ),
    ),
)
def test_stage3_rejects_changed_fixed_hashed_contract_values(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    match: str,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / f"changed-{section}-{key}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_stage3_config(path)


def test_stage3_rejects_changed_candidate_architecture(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["candidates"][0]["static_and_advection_conditions"] = False
    path = tmp_path / "changed-candidate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate architecture"):
        load_stage3_config(path)


def test_stage3_allows_only_runtime_wired_architecture_tuning(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["physical_attention_query_chunk_size"] = 64
    path = tmp_path / "query-chunk.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert load_stage3_config(path).query_chunk_size == 64


def test_stage3_rejects_non_power_of_two_batch_choice(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["optimization"]["per_rank_microbatch_size"] = 3
    raw["optimization"]["global_effective_batch_size"] = 24
    raw["loader_tuning"]["selected_batch_size_per_rank"] = 3
    path = tmp_path / "invalid-batch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="positive power of two"):
        load_stage3_config(path)


def test_stage3_rejects_single_or_changed_condition_support(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["condition_signatures"] = [
        "era5_oracle:era=1:tp=1:full_trajectory"
    ]
    path = tmp_path / "invalid-condition-support.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="condition-signature support"):
        load_stage3_config(path)


def test_launch_contract_requires_exact_two_rank_fp32_no_fallback() -> None:
    config = load_stage3_config(CONFIG_PATH)
    arguments = SimpleNamespace(
        require_world_size=2,
        precision="float32",
        disable_tf32=True,
        fail_on_fallback=True,
        target_widths=config.target_widths,
        context_widths=config.context_widths,
        era_latent_channels=128,
        era_grid_size=33,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        num_workers=config.selected_num_workers,
        prefetch_factor=config.selected_prefetch_factor,
        max_optimizer_steps=None,
        run_id="stage3-test",
        phase="screen",
        screening_evaluation=None,
        final_decision=None,
        container_image_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        runtime_report_sha256="c" * 64,
    )
    environment = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
        "KCORRDIFF_CONTAINER_IMAGE": "example/image@sha256:" + "a" * 64,
    }
    validate_launch_arguments(arguments, config, environ=environment)
    arguments.require_world_size = 1
    with pytest.raises(ValueError, match="two ranks"):
        validate_launch_arguments(arguments, config, environ=environment)
    arguments.require_world_size = 2
    with pytest.raises(ValueError, match="fallback environment"):
        validate_launch_arguments(
            arguments,
            config,
            environ={**environment, "KCORRDIFF_ALLOW_CPU_FALLBACK": "1"},
        )


def test_source_tree_identity_is_path_sorted_and_content_sensitive(
    tmp_path: Path,
) -> None:
    (tmp_path / "kcorrdiff").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "kcorrdiff" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tmp_path / "kcorrdiff" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "stage3.yaml").write_text("stage: 3\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    assert first == source_tree_sha256(tmp_path)
    (tmp_path / "kcorrdiff" / "a.py").write_text("a = 9\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first


def test_counter_noise_is_order_topology_and_candidate_independent() -> None:
    config = load_stage3_config(CONFIG_PATH)
    identities = (
        (17, "sample-z", 5.5),
        (2, "sample-a", 0.5),
        (91, "sample-q", 2.0),
    )
    direct = counter_keyed_training_noise(
        training_seed=11103,
        noise_config=config.noise,
        global_example_indices=[item[0] for item in identities],
        sample_ids=[item[1] for item in identities],
        lead_hours=[item[2] for item in identities],
        spatial_shape=(4, 5),
        device="cpu",
    )
    permutation = (2, 0, 1)
    shuffled = counter_keyed_training_noise(
        training_seed=11103,
        noise_config=config.noise,
        global_example_indices=[identities[item][0] for item in permutation],
        sample_ids=[identities[item][1] for item in permutation],
        lead_hours=[identities[item][2] for item in permutation],
        spatial_shape=(4, 5),
        device="cpu",
    )
    inverse = tuple(permutation.index(index) for index in range(3))
    assert torch.equal(direct.sigma, shuffled.sigma[list(inverse)])
    assert torch.equal(direct.gaussian, shuffled.gaussian[list(inverse)])
    # Calling the same function for EDM-A and EDM-B is exactly common random
    # numbers because no variant/checkpoint argument exists in the API.
    repeated = counter_keyed_training_noise(
        training_seed=11103,
        noise_config=config.noise,
        global_example_indices=[item[0] for item in identities],
        sample_ids=[item[1] for item in identities],
        lead_hours=[item[2] for item in identities],
        spatial_shape=(4, 5),
        device="cpu",
    )
    assert torch.equal(direct.sigma, repeated.sigma)
    assert torch.equal(direct.gaussian, repeated.gaussian)
    assert bool((direct.sigma >= config.noise.minimum_sigma).all())
    assert bool((direct.sigma <= config.noise.maximum_sigma).all())
    changed = counter_keyed_training_noise(
        training_seed=11105,
        noise_config=config.noise,
        global_example_indices=[item[0] for item in identities],
        sample_ids=[item[1] for item in identities],
        lead_hours=[item[2] for item in identities],
        spatial_shape=(4, 5),
        device="cpu",
    )
    assert not torch.equal(direct.gaussian, changed.gaussian)


def test_counter_noise_rejects_unregistered_seed_or_purpose() -> None:
    noise = Stage3NoiseConfig(-1.2, 1.2, 0.002, 80.0, "edm-training-noise-v1")
    kwargs = dict(
        noise_config=noise,
        global_example_indices=[1],
        sample_ids=["s"],
        lead_hours=[0.5],
        spatial_shape=(2, 2),
        device="cpu",
    )
    with pytest.raises(ValueError, match="preregistered"):
        counter_keyed_training_noise(training_seed=7, **kwargs)
    with pytest.raises(ValueError, match="purpose"):
        counter_keyed_training_noise(
            training_seed=11103,
            **{**kwargs, "noise_config": replace(noise, purpose_id="worker-rng")},
        )


def test_optimizer_loop_uses_one_global_weighted_denominator_and_padding() -> None:
    model = ScalarModel(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    audit = AuditRun()
    result = run_edm_optimizer_loop(
        model=model,
        deployment=nn.Identity(),
        loader=((2.0, 3.0),),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_tiny_config(),
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=1,
        training_seed=11103,
        tracking=audit,
        role_name="edm-a-seed-11103",
        checkpoint_callback=lambda cursor, complete: None,
        batch_operations=ScalarOperations(),
    )
    # d((w-2)^2*3/3)/dw at 0.5 is -3; SGD(0.1) -> 0.8.
    assert model.weight.item() == pytest.approx(0.8)
    assert result.cursor == TrainingCursor(0, 1, 1, 0)
    assert audit.values[0][0]["edm-a-seed-11103/loss_edm"] == pytest.approx(2.25)

    padding_model = ScalarModel(0.5)
    padding_optimizer = torch.optim.SGD(padding_model.parameters(), lr=0.1)
    padding_scheduler = torch.optim.lr_scheduler.LambdaLR(padding_optimizer, lambda _: 1.0)
    run_edm_optimizer_loop(
        model=padding_model,
        deployment=nn.Identity(),
        loader=(None, (2.0, 3.0)),
        optimizer=padding_optimizer,
        scheduler=padding_scheduler,
        config=_tiny_config(accumulation=2),
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=None,
        training_seed=11103,
        tracking=AuditRun(),
        role_name="edm-a-seed-11103",
        checkpoint_callback=lambda cursor, complete: None,
        batch_operations=ScalarOperations(),
    )
    assert padding_model.weight.item() == pytest.approx(0.8)


def test_optimizer_cursor_is_accumulation_aligned_and_iterator_preserves_rng() -> None:
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="rank-local slots"):
        run_edm_optimizer_loop(
            model=model,
            deployment=nn.Identity(),
            loader=(),
            optimizer=optimizer,
            scheduler=scheduler,
            config=_tiny_config(accumulation=2),
            runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
            start_cursor=TrainingCursor(0, 1, 1, 0),
            max_optimizer_steps=None,
            training_seed=11103,
            tracking=AuditRun(),
            role_name="x",
            checkpoint_callback=lambda cursor, complete: None,
            batch_operations=ScalarOperations(),
        )

    class EmptySeedConsumer:
        def __iter__(self):
            torch.rand(1)
            random.random()
            np.random.random()
            return iter(())

    random.seed(708)
    np.random.seed(708)
    torch.manual_seed(709)
    from kcorrdiff.training.checkpoints import capture_rng_state, restore_rng_state

    before = capture_rng_state()
    result = run_edm_optimizer_loop(
        model=model,
        deployment=nn.Identity(),
        loader=EmptySeedConsumer(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_tiny_config(),
        runtime=DistributedRuntime(0, 0, 1, torch.device("cpu"), False),
        start_cursor=TrainingCursor(0, 0, 0, 0),
        max_optimizer_steps=None,
        training_seed=11103,
        tracking=AuditRun(),
        role_name="x",
        checkpoint_callback=lambda cursor, complete: None,
        batch_operations=ScalarOperations(),
    )
    assert result.optimizer_steps_run == 0
    actual = (random.random(), float(np.random.random()), torch.rand(1))
    restore_rng_state(before)
    expected = (random.random(), float(np.random.random()), torch.rand(1))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def _provenance() -> Stage3CheckpointProvenance:
    config = load_stage3_config(CONFIG_PATH)
    return Stage3CheckpointProvenance(
        protocol_version="v1.1.3b",
        config_sha256="1" * 64,
        stage3_data_sha256="2" * 64,
        stage2_deployment_checkpoint_sha256="3" * 64,
        stage2_launch_identity_sha256="4" * 64,
        stage2_source_tree_sha256="5" * 64,
        stage2_container_image_sha256="6" * 64,
        stage2_runtime_report_sha256="7" * 64,
        stage2_data_contract_sha256="8" * 64,
        stage3_source_tree_sha256="9" * 64,
        container_image_sha256="a" * 64,
        runtime_report_sha256="b" * 64,
        draw_manifest_sha256="c" * 64,
        plan_sha256="d" * 64,
        variant="edm_a",
        training_seed=11103,
        world_size=2,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        wandb_mode="online",
        wandb_run_id="stage3-checkpoint-test",
    )


def test_checkpoint_round_trip_binds_rng_topology_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    model = ScalarModel(0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    torch.manual_seed(17)
    from kcorrdiff.training.checkpoints import capture_rng_state

    state = capture_rng_state()
    config = load_stage3_config(CONFIG_PATH)
    rank_slots_per_step = (
        config.optimization.per_rank_microbatch_size
        * config.optimization.gradient_accumulation_steps
    )
    digest = save_stage3_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=TrainingCursor(0, rank_slots_per_step, 1, 0),
        provenance=_provenance(),
        rank_rng_states=(state, state),
        complete=False,
        plan_optimizer_steps=2,
    )
    assert len(digest) == 64
    model.weight.data.zero_()
    cursor, complete, plan_steps = load_stage3_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_provenance=_provenance(),
        restore_rank_rng=1,
    )
    assert model.weight.item() == pytest.approx(0.7)
    assert cursor == TrainingCursor(0, rank_slots_per_step, 1, 0)
    assert complete is False and plan_steps == 2
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_stage3_checkpoint(
            path,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_provenance=replace(_provenance(), plan_sha256="6" * 64),
            restore_rank_rng=None,
        )


def test_frozen_stage2_deployment_checkpoint_is_loaded_with_exact_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stage2-deployment.pt"
    stage2_config = load_stage2_config(STAGE2_CONFIG_PATH)
    model = ScalarModel(0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    provenance = CheckpointProvenance(
        protocol_version="v1.1.3b",
        config_sha256="1" * 64,
        draw_manifest_sha256="2" * 64,
        launch_identity_sha256="3" * 64,
        source_tree_sha256="4" * 64,
        container_image_sha256="5" * 64,
        runtime_report_sha256="6" * 64,
        data_contract_sha256="7" * 64,
        role="deployment",
        fold_id=None,
        world_size=2,
        per_rank_microbatch_size=stage2_config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=stage2_config.optimization.gradient_accumulation_steps,
    )
    digest = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=TrainingCursor(
            0,
            2
            * stage2_config.optimization.per_rank_microbatch_size
            * stage2_config.optimization.gradient_accumulation_steps,
            2,
            0,
        ),
        provenance=provenance,
        extra={"complete": True},
    )
    loaded = ScalarModel(0.0)
    stage3._load_stage2_deployment_weights(
        loaded,  # type: ignore[arg-type]
        path,
        expected_sha256=digest,
        expected_config_sha256="1" * 64,
        expected_draw_manifest_sha256="2" * 64,
        expected_launch_identity_sha256="3" * 64,
        expected_source_tree_sha256="4" * 64,
        expected_container_image_sha256="5" * 64,
        expected_runtime_report_sha256="6" * 64,
        expected_data_contract_sha256="7" * 64,
        expected_global_step=2,
        per_rank_microbatch_size=stage2_config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=stage2_config.optimization.gradient_accumulation_steps,
    )
    assert loaded.weight.item() == pytest.approx(0.7)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="provenance/topology mismatch"):
        stage3._load_stage2_deployment_weights(
            ScalarModel(),  # type: ignore[arg-type]
            path,
            expected_sha256=digest,
            expected_config_sha256="3" * 64,
            expected_draw_manifest_sha256="2" * 64,
            expected_launch_identity_sha256="3" * 64,
            expected_source_tree_sha256="4" * 64,
            expected_container_image_sha256="5" * 64,
            expected_runtime_report_sha256="6" * 64,
            expected_data_contract_sha256="7" * 64,
            expected_global_step=2,
            per_rank_microbatch_size=stage2_config.optimization.per_rank_microbatch_size,
            gradient_accumulation_steps=stage2_config.optimization.gradient_accumulation_steps,
        )


def test_storage_preflight_counts_all_states_and_atomic_copy(tmp_path: Path) -> None:
    result = preflight_stage3_storage(
        output_dir=tmp_path,
        checkpoint_parameter_count=100,
        checkpoint_count=6,
        reserve_bytes=1,
        manifest_bytes=2,
    )
    assert result.checkpoint_bytes == 100 * 16 * 6
    assert result.atomic_temporary_bytes == 100 * 16
    with pytest.raises(OSError, match="exceeds free space"):
        preflight_stage3_storage(
            output_dir=tmp_path,
            checkpoint_parameter_count=10**20,
            checkpoint_count=6,
            reserve_bytes=0,
        )
    decision_only = preflight_stage3_storage(
        output_dir=tmp_path,
        checkpoint_parameter_count=100,
        checkpoint_count=0,
        reserve_bytes=0,
        manifest_bytes=2,
    )
    assert decision_only.checkpoint_bytes == 0
    assert decision_only.atomic_temporary_bytes == 0


def test_durable_tracking_audit_requires_one_run_and_clean_finish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    records = (
        {"event": "start", "run_id": "run-a", "config_sha256": "1" * 64, "step": -1},
        {"event": "metrics", "run_id": "run-a", "step": 1, "metrics": {"loss": 1.0}},
        {"event": "finish", "run_id": "run-a", "step": 1, "exit_code": 0},
    )
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    assert len(stage3._validated_tracking_audit_sha256(path, run_id="run-a")) == 64
    failed = list(records)
    failed[-1] = {**failed[-1], "exit_code": 1}
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in failed),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed exit code"):
        stage3._validated_tracking_audit_sha256(path, run_id="run-a")

    recovered = failed + [
        {
            "event": "resume",
            "run_id": "run-a",
            "config_sha256": "1" * 64,
            "step": 1,
        },
        {
            "event": "metrics",
            "run_id": "run-a",
            "step": 2,
            "metrics": {"checkpoint_recovered": True},
        },
        {"event": "finish", "run_id": "run-a", "step": 2, "exit_code": 0},
    ]
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in recovered),
        encoding="utf-8",
    )
    assert len(stage3._validated_tracking_audit_sha256(path, run_id="run-a")) == 64


def test_screening_adapter_requires_external_holdout_and_both_exact_arms(
    tmp_path: Path,
) -> None:
    checkpoints = _checkpoint_records(seeds=(11103,))
    path = tmp_path / "screen.json"
    payload: dict[str, object] = {
        "format_version": SCREENING_EVALUATION_FORMAT,
        "protocol_version": "v1.1.3b",
        "split": "model_selection",
        "complete": True,
        "screening_training_manifest_sha256": "a" * 64,
        "evaluation_result_sha256": "b" * 64,
        "sampling_profile": "selection_signature",
        "members": 16,
        "edm_steps": 8,
        "common_ensemble_seed": 11104,
        "labels_used_for_parameter_training": False,
        "calibration_parameters_fit_from_model_selection_labels": False,
        "checkpoint_sha256s": [
            {"variant": item.variant, "training_seed": item.training_seed, "sha256": item.sha256}
            for item in checkpoints
        ],
        "finalist_variants": ["edm_a", "edm_b"],
        "evaluator_id": "evaluation-pipeline-v1",
    }
    _artifact(path, payload)
    result = load_screening_evaluation_result(
        path,
        expected_screening_manifest_sha256="a" * 64,
        expected_checkpoints=checkpoints,
    )
    assert result.finalist_variants == ("edm_a", "edm_b")
    payload["labels_used_for_parameter_training"] = True
    payload.pop("artifact_sha256")
    _artifact(path, payload)
    with pytest.raises(ValueError, match="holdout boundary"):
        load_screening_evaluation_result(
            path,
            expected_screening_manifest_sha256="a" * 64,
            expected_checkpoints=checkpoints,
        )


def test_external_checkpoint_evidence_requires_exact_final_bytes(
    tmp_path: Path,
) -> None:
    records = _checkpoint_records(seeds=(11103,))
    for record in records:
        path = (
            tmp_path
            / "checkpoints"
            / f"{record.variant}-seed-{record.training_seed}"
            / "final.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(record.variant.encode("utf-8"))
    actual = tuple(
        CheckpointIdentity(
            record.variant,
            record.training_seed,
            hashlib.sha256(record.variant.encode("utf-8")).hexdigest(),
        )
        for record in records
    )
    stage3._verify_checkpoint_files(actual, output_dir=tmp_path)
    with pytest.raises(ValueError, match="missing/changed"):
        stage3._verify_checkpoint_files(records, output_dir=tmp_path)


def test_final_decision_requires_all_three_seeds_and_edm_b_dispersion_gate(
    tmp_path: Path,
) -> None:
    checkpoints = _checkpoint_records(seeds=REPEAT_TRAINING_SEEDS)
    path = tmp_path / "decision.json"
    config = load_stage3_config(CONFIG_PATH)
    architecture_hashes = {
        variant: stage3_architecture_sha256(config, variant=variant)
        for variant in ("edm_a", "edm_b")
    }
    payload: dict[str, object] = {
        "format_version": FINAL_SELECTION_FORMAT,
        "protocol_version": "v1.1.3b",
        "split": "model_selection",
        "complete": True,
        "finalist_training_manifest_sha256": "c" * 64,
        "evaluation_result_sha256": "d" * 64,
        "sampling_profile": "selection_signature",
        "members": 16,
        "edm_steps": 8,
        "common_ensemble_seed": 11104,
        "repeat_training_seeds": list(REPEAT_TRAINING_SEEDS),
        "deployment_training_seed": 11103,
        "labels_used_for_parameter_training": False,
        "calibration_parameters_fit_from_model_selection_labels": False,
        "checkpoint_sha256s": [
            {"variant": item.variant, "training_seed": item.training_seed, "sha256": item.sha256}
            for item in checkpoints
        ],
        "selected_variant": "edm_b",
        "architecture_sha256": architecture_hashes["edm_b"],
        "d_enabled": False,
        "probability_mapping_family": "monotone_logit_linear",
        "pooling_order": [
            "full_cell",
            "lead_provider_era_present",
            "lead_provider",
            "lead_only",
        ],
        "mandatory_guardrails_passed": True,
        "absolute_gates_passed": True,
        "edm_b_dispersion_noninferiority_passed": True,
        "evaluator_id": "evaluation-pipeline-v1",
    }
    _artifact(path, payload)
    decision = load_final_model_selection_decision(
        path,
        expected_finalist_manifest_sha256="c" * 64,
        expected_checkpoints=checkpoints,
        expected_architecture_sha256s=architecture_hashes,
    )
    assert decision.selected_variant == "edm_b"
    frozen = decision.frozen_calibration_decision()
    assert frozen.decision_sha256 == decision.file_sha256
    assert frozen.architecture_sha256 == architecture_hashes["edm_b"]
    payload["edm_b_dispersion_noninferiority_passed"] = False
    payload.pop("artifact_sha256")
    _artifact(path, payload)
    with pytest.raises(ValueError, match="dispersion"):
        load_final_model_selection_decision(
            path,
            expected_finalist_manifest_sha256="c" * 64,
            expected_checkpoints=checkpoints,
            expected_architecture_sha256s=architecture_hashes,
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _gloo_worker(rank: int, port: int, output: str) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        model = ScalarModel(0.5)
        for parameter in model.parameters():
            distributed.broadcast(parameter.data, src=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        # Rank 0 has mass 1/target 1; rank 1 has mass 3/target 3. Global
        # derivative at w=.5 is 2*(.5-1)*1/4 + 2*(.5-3)*3/4 = -4.
        batch = (1.0, 1.0) if rank == 0 else (3.0, 3.0)
        runtime = DistributedRuntime(rank, rank, 2, torch.device("cpu"), True)
        result = run_edm_optimizer_loop(
            model=model,
            deployment=nn.Identity(),
            loader=(batch,),
            optimizer=optimizer,
            scheduler=scheduler,
            config=_tiny_config(),
            runtime=runtime,
            start_cursor=TrainingCursor(0, 0, 0, 0),
            max_optimizer_steps=1,
            training_seed=11103,
            tracking=AuditRun(),
            role_name="gloo",
            checkpoint_callback=lambda cursor, complete: None,
            batch_operations=ScalarOperations(),
        )
        values = [None, None]
        distributed.all_gather_object(values, (model.weight.item(), result.cursor.optimizer_step))
        if rank == 0:
            Path(output).write_text(json.dumps(values), encoding="utf-8")
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_available(), reason="torch.distributed unavailable")
def test_two_process_gloo_global_weighted_objective(tmp_path: Path) -> None:
    output = tmp_path / "gloo.json"
    multiprocessing.spawn(
        _gloo_worker,
        args=(_free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    values = json.loads(output.read_text(encoding="utf-8"))
    assert values[0][0] == pytest.approx(0.9)
    assert values[1][0] == pytest.approx(0.9)
    assert values[0][1] == values[1][1] == 1

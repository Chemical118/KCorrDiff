"""Fail-closed Stage 2 loader/model benchmark selection contract.

The YAML binds the selection artifact by SHA-256, but a digest alone does not
prove that the JSON contains the selected topology or the data lineage used by
the training factory.  This module validates both the semantic decision and,
when requested, every small benchmark summary/JSONL file cited by it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.training.config import Stage2Config


SELECTION_FORMAT = "kcorrdiff.loader-and-model-selection.v2"
EXPECTED_REGRESSION_PARAMETER_COUNT = 62_008_276


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object], *, required: set[str], name: str
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _power_of_two(value: int, *, name: str) -> int:
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two")
    return value


def _integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(_positive_int(item, name=f"{name} item") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


@dataclass(frozen=True, slots=True)
class Stage2SelectionContract:
    artifact_path: Path
    artifact_sha256: str
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    global_effective_batch_size: int
    num_workers: int
    prefetch_factor: int
    artifact_lineage: Mapping[str, str]
    benchmark_artifacts: tuple[tuple[Path, str], ...]

    def validate_factory_lineage(self, actual: Mapping[str, str]) -> None:
        mapping = {
            "candidate_manifest_sha256": "candidate_manifest",
            "draw_manifest_sha256": "draw_manifest",
            "bundle_metadata_sha256": "bundle_metadata",
            "radar_cache_manifest_sha256": "radar_cache_manifest",
            "era5_cache_manifest_sha256": "era5_cache_manifest",
            "normalization_sha256": "normalization",
        }
        for selection_name, factory_name in mapping.items():
            observed = actual.get(factory_name)
            if observed != self.artifact_lineage[selection_name]:
                raise ValueError(
                    f"selection artifact/factory lineage mismatch: {selection_name}"
                )

    def verify_benchmark_artifacts(self) -> None:
        for path, expected in self.benchmark_artifacts:
            if not path.is_file():
                raise FileNotFoundError(f"selection benchmark artifact is missing: {path}")
            if sha256_file(path) != expected:
                raise ValueError(f"selection benchmark artifact SHA-256 mismatch: {path}")


def load_stage2_selection_contract(
    path: Path,
    *,
    expected_sha256: str,
    config: Stage2Config,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    verify_benchmark_artifacts: bool = False,
) -> Stage2SelectionContract:
    """Load and semantically bind the exact Stage 2 tuning decision."""

    selected_path = path.resolve()
    expected = _sha(expected_sha256, name="selection artifact SHA-256")
    actual_digest = sha256_file(selected_path)
    if actual_digest != expected:
        raise ValueError("loader selection artifact SHA-256 mismatch")
    raw = json.loads(selected_path.read_text(encoding="utf-8"))
    root = _mapping(raw, name="selection artifact")
    _exact_keys(
        root,
        required={
            "format_version",
            "selected",
            "decision",
            "loader_benchmark",
            "model_benchmarks",
            "runtime_contract",
            "artifact_lineage",
        },
        name="selection artifact",
    )
    if root["format_version"] != SELECTION_FORMAT:
        raise ValueError("unsupported Stage 2 selection artifact format")

    selected = _mapping(root["selected"], name="selection selected")
    _exact_keys(
        selected,
        required={
            "per_rank_microbatch_size",
            "gradient_accumulation_steps",
            "global_effective_batch_size",
            "num_workers",
            "prefetch_factor",
            "model_samples_per_second",
        },
        name="selection selected",
    )
    batch = _power_of_two(
        _positive_int(selected["per_rank_microbatch_size"], name="selected batch"),
        name="selected batch",
    )
    accumulation = _positive_int(
        selected["gradient_accumulation_steps"], name="selected accumulation"
    )
    global_batch = _positive_int(
        selected["global_effective_batch_size"], name="selected global batch"
    )
    workers = _positive_int(selected["num_workers"], name="selected num_workers")
    prefetch = _positive_int(
        selected["prefetch_factor"], name="selected prefetch_factor"
    )
    throughput = selected["model_samples_per_second"]
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float)) or throughput <= 0:
        raise ValueError("selected model throughput must be positive")
    if global_batch != world_size * batch * accumulation:
        raise ValueError("selection global batch disagrees with topology")
    if (
        batch != config.optimization.per_rank_microbatch_size
        or accumulation != config.optimization.gradient_accumulation_steps
        or global_batch != config.optimization.global_effective_batch_size
        or workers != num_workers
        or prefetch != prefetch_factor
    ):
        raise ValueError("selection artifact disagrees with Stage 2 launch/config")

    decision = _mapping(root["decision"], name="selection decision")
    _exact_keys(
        decision,
        required={
            "batch_search_policy",
            "power_of_two_batch_sizes_tested",
            "power_of_two_batch_sizes_accepted",
            "minimum_oom_rejected_batch_size",
            "selected_batch_size_per_rank",
            "rationale",
            "ignored_historical_non_power_of_two_batch_sizes",
            "selection_evidence_uses_only_power_of_two_batch_sizes",
        },
        name="selection decision",
    )
    if (
        decision["batch_search_policy"] != "positive_powers_of_two_only"
        or decision["selection_evidence_uses_only_power_of_two_batch_sizes"] is not True
    ):
        raise ValueError("selection does not enforce power-of-two-only B evidence")
    tested = _integer_sequence(
        decision["power_of_two_batch_sizes_tested"], name="tested batch sizes"
    )
    accepted = _integer_sequence(
        decision["power_of_two_batch_sizes_accepted"], name="accepted batch sizes"
    )
    if tested != tuple(sorted(tested)) or accepted != tuple(sorted(accepted)):
        raise ValueError("selection batch grids must be strictly increasing")
    for value in (*tested, *accepted):
        _power_of_two(value, name="selection batch size")
    if not set(accepted).issubset(tested) or batch not in accepted:
        raise ValueError("selected/accepted batches disagree with tested grid")
    rejected = _power_of_two(
        _positive_int(
            decision["minimum_oom_rejected_batch_size"], name="minimum OOM batch"
        ),
        name="minimum OOM batch",
    )
    if rejected not in tested or rejected <= max(accepted):
        raise ValueError("OOM boundary does not follow the accepted power grid")
    if decision["selected_batch_size_per_rank"] != batch:
        raise ValueError("decision selected batch disagrees with selected section")
    ignored = _integer_sequence(
        decision["ignored_historical_non_power_of_two_batch_sizes"],
        name="ignored historical batch sizes",
    )
    if any(not (value & (value - 1)) for value in ignored):
        raise ValueError("ignored historical B entries must be non-powers-of-two")
    if not isinstance(decision["rationale"], str) or not decision["rationale"].strip():
        raise ValueError("selection decision requires a rationale")

    runtime = _mapping(root["runtime_contract"], name="selection runtime contract")
    _exact_keys(
        runtime,
        required={
            "device",
            "precision",
            "tf32",
            "world_size",
            "model_parameter_count",
            "fallback_used",
        },
        name="selection runtime contract",
    )
    if (
        runtime["device"] != "NVIDIA A100-SXM4-40GB"
        or runtime["precision"] != "float32"
        or runtime["tf32"] is not False
        or runtime["fallback_used"] is not False
        or runtime["world_size"] != world_size
        or runtime["model_parameter_count"] != EXPECTED_REGRESSION_PARAMETER_COUNT
    ):
        raise ValueError("selection runtime/model contract changed")

    loader = _mapping(root["loader_benchmark"], name="loader benchmark")
    loader_required = {
        "run_id", "wandb_url", "summary_path", "summary_sha256",
        "results_jsonl_sha256", "successful_grid_cells", "tested_batch_sizes",
        "tested_worker_counts", "tested_prefetch_factors", "selected_num_workers",
        "selected_prefetch_factor", "selected_samples_per_second_at_batch_one",
    }
    _exact_keys(loader, required=loader_required, name="loader benchmark")
    loader_batches = _integer_sequence(loader["tested_batch_sizes"], name="loader batches")
    if any(value & (value - 1) for value in loader_batches):
        raise ValueError("loader benchmark contains a non-power-of-two B")
    if loader["selected_num_workers"] != workers or loader["selected_prefetch_factor"] != prefetch:
        raise ValueError("loader benchmark selection disagrees with selected settings")

    benchmark_files: list[tuple[Path, str]] = [
        (Path(str(loader["summary_path"])).resolve(), _sha(loader["summary_sha256"], name="loader summary SHA")),
        (Path(str(loader["summary_path"])).resolve().with_suffix(".jsonl"), _sha(loader["results_jsonl_sha256"], name="loader JSONL SHA")),
    ]
    models = _mapping(root["model_benchmarks"], name="model benchmarks")
    _exact_keys(
        models,
        required={
            "power_grid_through_batch_8",
            "power_observation_batch_16",
            "power_observation_batch_32",
        },
        name="model benchmarks",
    )
    for name, expected_batch, expected_accepted in (
        ("power_grid_through_batch_8", 8, True),
        ("power_observation_batch_16", 16, True),
        ("power_observation_batch_32", 32, False),
    ):
        record = _mapping(models[name], name=f"model benchmark {name}")
        summary_path = Path(str(record.get("summary_path", ""))).resolve()
        benchmark_files.extend(
            (
                (summary_path, _sha(record.get("summary_sha256"), name=f"{name} summary SHA")),
                (summary_path.with_suffix(".jsonl"), _sha(record.get("results_jsonl_sha256"), name=f"{name} JSONL SHA")),
            )
        )
        if name == "power_grid_through_batch_8":
            values = _integer_sequence(record.get("accepted_batch_sizes"), name="B1-8 accepted batches")
            if values != (1, 2, 4, 8):
                raise ValueError("B1-8 benchmark grid changed")
        else:
            if record.get("batch_size") != expected_batch or record.get("accepted") is not expected_accepted:
                raise ValueError(f"{name} acceptance boundary changed")

    lineage_raw = _mapping(root["artifact_lineage"], name="selection artifact lineage")
    lineage_keys = {
        "candidate_manifest_sha256",
        "draw_manifest_sha256",
        "bundle_metadata_sha256",
        "radar_cache_manifest_sha256",
        "era5_cache_manifest_sha256",
        "normalization_sha256",
    }
    _exact_keys(lineage_raw, required=lineage_keys, name="selection artifact lineage")
    lineage = {name: _sha(lineage_raw[name], name=name) for name in lineage_keys}
    result = Stage2SelectionContract(
        artifact_path=selected_path,
        artifact_sha256=actual_digest,
        per_rank_microbatch_size=batch,
        gradient_accumulation_steps=accumulation,
        global_effective_batch_size=global_batch,
        num_workers=workers,
        prefetch_factor=prefetch,
        artifact_lineage=lineage,
        benchmark_artifacts=tuple(benchmark_files),
    )
    if verify_benchmark_artifacts:
        result.verify_benchmark_artifacts()
    return result


__all__ = [
    "EXPECTED_REGRESSION_PARAMETER_COUNT",
    "SELECTION_FORMAT",
    "Stage2SelectionContract",
    "load_stage2_selection_contract",
]

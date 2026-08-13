from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import pickle

import numpy as np
from numpy.lib.format import open_memmap
import pytest
import torch

from kcorrdiff.data.availability import (
    format_radar_timestamp,
    history_times,
    target_times,
)
from kcorrdiff.data.cache import FORMAT_VERSION as RADAR_CACHE_FORMAT
from kcorrdiff.data.era5_reader import ERA5_CACHE_FORMAT
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.sampling import DrawRow
from kcorrdiff.training.cache_benchmark import (
    BenchmarkGrid,
    CacheBenchmarkContract,
    CacheBenchmarkDataset,
    CacheDatasetSpec,
    STATIC_FLOAT_CHANNELS,
    collate_benchmark_samples,
    run_benchmark_grid,
    validate_cache_artifacts,
)


RADAR_SHAPE = (4, 4)
ERA_SHAPE = (3, 3)
SIGNATURE = "era5_oracle:era=1:tp=1:full_trajectory"


def _hours_in_year(year: int) -> int:
    return int(
        (
            datetime(year + 1, 1, 1, tzinfo=UTC)
            - datetime(year, 1, 1, tzinfo=UTC)
        )
        / timedelta(hours=1)
    )


def _write_radar_cache(root: Path, *, t0: datetime, missing: str | None = None) -> None:
    root.mkdir()
    history = tuple(format_radar_timestamp(value) for value in history_times(t0))
    future = tuple(
        format_radar_timestamp(value) for value in target_times(t0, 0.5)
    )
    target_keys = tuple(dict.fromkeys((*history, *future)))
    condition_keys = history
    shards: list[dict[str, object]] = []
    for source, keys in (("target", target_keys), ("condition", condition_keys)):
        selected = tuple(key for key in keys if key != missing)
        values_name = f"{source}-synthetic.npy"
        timestamps_name = f"{source}-synthetic.timestamps.json"
        values = np.stack(
            [
                np.full(RADAR_SHAPE, index + (0 if source == "target" else 100))
                for index in range(len(selected))
            ]
        ).astype(np.float32)
        np.save(root / values_name, values, allow_pickle=False)
        (root / timestamps_name).write_text(
            json.dumps(selected, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        shards.append(
            {
                "source": source,
                "archive_name": f"{source}.npz",
                "archive_size": 1,
                "archive_sha256": "",
                "values_file": values_name,
                "timestamps_file": timestamps_name,
                "samples": len(selected),
                "shape": list(RADAR_SHAPE),
                "dtype": "float32",
                "values_sha256": "",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": RADAR_CACHE_FORMAT,
                "dtype": "float32",
                "shards": shards,
                "unique_timestamps": len(target_keys),
                "estimated_value_bytes": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_era_cache(root: Path, *, year: int) -> None:
    root.mkdir()
    hours = _hours_in_year(year)
    instant_name = f"instantaneous-{year}.npy"
    tp_name = f"tp-{year}.npy"
    data_valid_name = f"data-valid-inst-{year}.npy"
    tp_valid_name = f"tp-valid-{year}.npy"

    instantaneous = open_memmap(
        root / instant_name,
        mode="w+",
        dtype=np.float32,
        shape=(hours, 23, *ERA_SHAPE),
    )
    instantaneous[:8] = np.arange(23, dtype=np.float32)[None, :, None, None]
    instantaneous.flush()
    del instantaneous
    tp = open_memmap(
        root / tp_name,
        mode="w+",
        dtype=np.float32,
        shape=(hours, 1, *ERA_SHAPE),
    )
    tp[:8] = np.arange(8, dtype=np.float32)[:, None, None, None]
    tp.flush()
    del tp
    data_valid = open_memmap(
        root / data_valid_name, mode="w+", dtype=np.bool_, shape=(hours,)
    )
    data_valid[:8] = True
    data_valid.flush()
    del data_valid
    tp_valid = open_memmap(
        root / tp_valid_name, mode="w+", dtype=np.bool_, shape=(hours,)
    )
    tp_valid[:8] = True
    tp_valid.flush()
    del tp_valid
    np.save(root / "latitude.npy", np.linspace(37.0, 38.0, ERA_SHAPE[0]))
    np.save(root / "longitude.npy", np.linspace(126.0, 127.0, ERA_SHAPE[1]))

    manifest = {
        "format_version": ERA5_CACHE_FORMAT,
        "dtype": "float32",
        "grid_shape": list(ERA_SHAPE),
        "latitude_file": "latitude.npy",
        "longitude_file": "longitude.npy",
        "latitude_orientation": "south_to_north",
        "longitude_orientation": "west_to_east",
        "instantaneous_channel_names": list(ERA5_CHANNEL_NAMES[:-1]),
        "tp_channel_name": "tp",
        "tp_units": "mm per 1 hour",
        "shards": [
            {
                "year": year,
                "start_utc": f"{year}-01-01T00:00:00+00:00",
                "end_utc_exclusive": f"{year + 1}-01-01T00:00:00+00:00",
                "hours": hours,
                "instantaneous_file": instant_name,
                "tp_file": tp_name,
                "data_valid_inst_file": data_valid_name,
                "tp_valid_file": tp_valid_name,
                "valid_instantaneous_hours": 8,
                "valid_tp_hours": 8,
                "instantaneous_sha256": "",
                "tp_sha256": "",
            }
        ],
        "provenance": {
            "provider_id": "synthetic-era5",
            "provider_version": "test",
            "dataset": "synthetic",
            "sources": ["test"],
            "transformations": [],
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


def _write_static(path: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        name: np.full(RADAR_SHAPE, index, dtype=np.float32)
        for index, name in enumerate(STATIC_FLOAT_CHANNELS)
    }
    # The real immutable artifact may preserve coordinate geometry as float64;
    # the benchmark must still construct an explicitly FP32 H2D payload.
    arrays["lcc_x_normalized"] = np.zeros(RADAR_SHAPE, dtype=np.float64)
    arrays["lcc_y_normalized"] = np.ones(RADAR_SHAPE, dtype=np.float64)
    arrays["land_mask"] = np.ones(RADAR_SHAPE, dtype=np.bool_)
    arrays["coverage"] = np.ones(RADAR_SHAPE, dtype=np.bool_)
    np.savez(path, **arrays)


def build_synthetic_artifacts(
    tmp_path: Path, *, missing_radar_timestamp: str | None = None
) -> tuple[CacheDatasetSpec, DrawRow]:
    t0 = datetime(2020, 1, 1, 0, 5, tzinfo=UTC)
    radar_root = tmp_path / "radar-v1"
    era_root = tmp_path / "era5-v1"
    static_path = tmp_path / "target_static.npz"
    _write_radar_cache(
        radar_root, t0=t0, missing=missing_radar_timestamp
    )
    _write_era_cache(era_root, year=t0.year)
    _write_static(static_path)
    row = DrawRow(
        global_example_index=0,
        sample_id="synthetic-0",
        t0_utc=t0.isoformat(),
        lead_hours=0.5,
        condition_signature=SIGNATURE,
        block_id="block-0",
        stratum="event",
        split="outer_train",
        p_target=1.0,
        p_draw=1.0,
        omega=1.0,
        fold_id=0,
    )
    contract = CacheBenchmarkContract(
        radar_shape=RADAR_SHAPE,
        era_shape=ERA_SHAPE,
        target_widths=(4, 8, 12, 16, 20),
        context_widths=(2, 4, 6, 8, 10),
        era_latent_channels=8,
        allow_test_override=True,
    )
    return (
        CacheDatasetSpec(
            radar_cache_root=radar_root,
            era5_cache_root=era_root,
            static_path=static_path,
            rows=(row,),
            contract=contract,
        ),
        row,
    )


@pytest.fixture()
def synthetic_spec(tmp_path: Path) -> CacheDatasetSpec:
    spec, _ = build_synthetic_artifacts(tmp_path)
    return spec


def test_contract_rejects_precision_tf32_and_spatial_fallbacks() -> None:
    with pytest.raises(ValueError, match="float32"):
        CacheBenchmarkContract(precision="float16")
    with pytest.raises(ValueError, match="TF32"):
        CacheBenchmarkContract(tf32_enabled=True)
    with pytest.raises(ValueError, match="fallback"):
        CacheBenchmarkContract(radar_shape=RADAR_SHAPE)


def test_artifact_preflight_and_worker_lazy_dataset_round_trip(
    synthetic_spec: CacheDatasetSpec,
) -> None:
    report = validate_cache_artifacts(
        radar_cache_root=synthetic_spec.radar_cache_root,
        era5_cache_root=synthetic_spec.era5_cache_root,
        static_path=synthetic_spec.static_path,
        contract=synthetic_spec.contract,
    )
    assert report["radar_shape"] == list(RADAR_SHAPE)
    assert report["era5_shape"] == list(ERA_SHAPE)

    dataset = CacheBenchmarkDataset(synthetic_spec)
    assert not dataset.handles_open
    sample = dataset[0]
    assert dataset.handles_open
    tensors = sample["tensors"]
    assert isinstance(tensors, dict)
    assert tensors["target_history"].shape == (12, *RADAR_SHAPE)
    assert tensors["context_history"].shape == (12, *RADAR_SHAPE)
    assert tensors["target_future_benchmark_only"].shape == (7, *RADAR_SHAPE)
    assert tensors["era_values"].shape == (8, 24, *ERA_SHAPE)
    assert tensors["target_static"].shape == (7, *RADAR_SHAPE)
    assert tensors["era_data_valid_inst"].tolist() == [True] * 8
    assert sample["identity"] == (
        "synthetic-0",
        "2020-01-01T00:05:00+00:00",
        0.5,
        SIGNATURE,
    )
    assert sample["logical_bytes"] > 0
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored.handles_open
    dataset.close()
    assert not dataset.handles_open


def test_collate_preserves_fp32_and_named_benchmark_only_future(
    synthetic_spec: CacheDatasetSpec,
) -> None:
    dataset = CacheBenchmarkDataset(synthetic_spec)
    sample = dataset[0]
    batch = collate_benchmark_samples((sample, sample))
    assert batch.batch_size == 2
    assert batch.tensors["target_history"].shape == (2, 12, *RADAR_SHAPE)
    assert batch.tensors["target_future_benchmark_only"].shape == (
        2,
        7,
        *RADAR_SHAPE,
    )
    for tensor in batch.tensors.values():
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32
    dataset.close()


def test_missing_radar_key_fails_instead_of_becoming_dry(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 1, 0, 5, tzinfo=UTC)
    missing = format_radar_timestamp(history_times(t0)[0])
    spec, _ = build_synthetic_artifacts(
        tmp_path, missing_radar_timestamp=missing
    )
    with pytest.raises(KeyError, match="gaps are not dry"):
        CacheBenchmarkDataset(spec)[0]


def test_cpu_grid_exercises_zero_and_persistent_worker_paths(
    synthetic_spec: CacheDatasetSpec,
) -> None:
    results = run_benchmark_grid(
        synthetic_spec,
        BenchmarkGrid(
            batch_sizes=(2,),
            worker_counts=(0, 1),
            prefetch_factors=(2,),
            warmup_batches=1,
            measured_batches=2,
        ),
        device=torch.device("cpu"),
        pin_memory=False,
        allow_cpu_test=True,
    )
    assert len(results) == 2
    assert all(result["status"] == "ok" for result in results)
    assert all(result["samples"] == 4 for result in results)
    assert all(result["samples_per_second"] > 0 for result in results)
    assert all(result["logical_input_bytes"] > 0 for result in results)
    assert results[0]["observed_worker_pids"] == 0
    assert results[1]["observed_worker_pids"] == 1
    assert results[1]["prefetch_effective"] == 2


def test_grid_rejects_non_power_of_two_batch_size() -> None:
    with pytest.raises(ValueError, match="powers of two"):
        BenchmarkGrid(
            batch_sizes=(1, 2, 3, 4),
            worker_counts=(0,),
            prefetch_factors=(2,),
            warmup_batches=0,
            measured_batches=1,
        )


def test_production_runner_refuses_cpu(synthetic_spec: CacheDatasetSpec) -> None:
    with pytest.raises(RuntimeError, match="CPU fallback"):
        run_benchmark_grid(
            synthetic_spec,
            BenchmarkGrid(
                batch_sizes=(1,),
                worker_counts=(0,),
                prefetch_factors=(2,),
                warmup_batches=0,
                measured_batches=1,
            ),
            device=torch.device("cpu"),
            pin_memory=False,
        )

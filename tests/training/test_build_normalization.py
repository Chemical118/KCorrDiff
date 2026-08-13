from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.lib.format import open_memmap
import pytest

from kcorrdiff.data.availability import format_radar_timestamp, history_times
from kcorrdiff.data.cache import build_radar_cache
from kcorrdiff.data.normalization import read_normalization_artifact, sha256_file
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES, schema_manifest
from kcorrdiff.data.split_manifest import make_item
from kcorrdiff.training.build_normalization import (
    build_normalization_artifact,
    derive_candidate_universe,
    main,
)


T0 = datetime(2020, 1, 2, 6, tzinfo=UTC)
HOURS_2020 = 366 * 24


def _write_candidate(directory: Path, *, reverse: bool = False) -> Path:
    directory.mkdir(parents=True)
    rows = [
        replace(
            make_item(
                t0=T0,
                lead_hours=lead,
                condition_signature="era5_oracle:era=1:tp=1:full_trajectory",
                split="outer_train",
                block_id="event-a",
                stratum="event",
                p_target=0.5,
                p_draw=0.5,
            ),
            fold_id=0,
        )
        for lead in (0.5, 1.0)
    ]
    if reverse:
        rows.reverse()
    metadata = {
        "protocol_version": "v1.1.3b",
        "outer_train_folds": 3,
        "split_intervals": [
            {
                "name": "outer_train",
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2021-01-01T00:00:00+00:00",
            }
        ],
        "config_sha256": "1" * 64,
        "event_index_semantic_sha256": "2" * 64,
        "eligible_universe_sample_ids_sha256": "3" * 64,
        "candidate_semantic_sha256": "4" * 64,
        "fold_map_sha256": "5" * 64,
    }
    candidate = directory / "candidate-manifest.json"
    candidate.write_text(
        json.dumps(
            {
                "items": [asdict(row) for row in rows],
                "metadata": metadata,
                "version": "kcorrdiff.split-manifest.v1",
            },
            sort_keys=True,
        )
        + "\n"
    )
    candidate_hash = sha256_file(candidate)
    bundle = directory / "bundle-metadata.json"
    bundle.write_text(
        json.dumps(
            {
                "artifacts": {
                    "candidate_manifest": {
                        "path": "candidate-manifest.json",
                        "sha256": candidate_hash,
                    }
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    bundle_hash = sha256_file(bundle)
    (directory / "bundle-metadata.sha256").write_text(
        f"{bundle_hash}  bundle-metadata.json\n"
    )
    return candidate


def _write_radar_cache(directory: Path) -> Path:
    target_root = directory / "target-source"
    condition_root = directory / "condition-source"
    target_root.mkdir()
    condition_root.mkdir()
    target_fields: dict[str, np.ndarray] = {}
    context_fields: dict[str, np.ndarray] = {}
    spatial = np.linspace(0.0, 0.05, 256 * 256, dtype=np.float32).reshape(256, 256)
    for index, value in enumerate(history_times(T0)):
        key = format_radar_timestamp(value, timezone="Asia/Seoul")
        target_fields[key] = np.asarray(0.05 + index * 0.01 + spatial, np.float32)
        context_fields[key] = np.asarray(0.08 + index * 0.01 + spatial, np.float32)
    name = "hsr_22_SEOUL_NON_UNI_2020.npz"
    np.savez(target_root / name, **target_fields)
    np.savez(condition_root / name.replace(".npz", "_cond.npz"), **context_fields)
    output = directory / "radar-v1"
    build_radar_cache(
        target_root=target_root,
        condition_root=condition_root,
        output_dir=output,
        workers=1,
        reserve_bytes=0,
    )
    return output


def _write_era_cache(directory: Path) -> Path:
    root = directory / "era5-v1"
    root.mkdir()
    instantaneous_file = root / "instantaneous-2020.npy"
    tp_file = root / "tp-2020.npy"
    instantaneous = open_memmap(
        instantaneous_file,
        mode="w+",
        dtype=np.float32,
        shape=(HOURS_2020, 23, 2, 2),
    )
    tp = open_memmap(
        tp_file,
        mode="w+",
        dtype=np.float32,
        shape=(HOURS_2020, 1, 2, 2),
    )
    inst_valid = np.zeros(HOURS_2020, dtype=np.bool_)
    tp_valid = np.zeros(HOURS_2020, dtype=np.bool_)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    base_offset = int((T0 - start).total_seconds() // 3600)
    spatial = np.arange(4, dtype=np.float32).reshape(2, 2) / 10.0
    for token in range(7):
        offset = base_offset + token
        instantaneous[offset] = (
            np.arange(23, dtype=np.float32)[:, None, None]
            + token
            + spatial
        )
        tp[offset, 0] = token + spatial
        inst_valid[offset] = token != 2
        tp_valid[offset] = token != 3
    instantaneous.flush()
    tp.flush()
    del instantaneous, tp
    np.save(root / "data-valid-inst-2020.npy", inst_valid, allow_pickle=False)
    np.save(root / "tp-valid-2020.npy", tp_valid, allow_pickle=False)
    np.save(root / "latitude.npy", np.asarray([37.0, 38.0]), allow_pickle=False)
    np.save(root / "longitude.npy", np.asarray([126.0, 127.0]), allow_pickle=False)
    manifest = {
        "format_version": "kcorrdiff.era5-yearly-mmap.v1",
        "dtype": "float32",
        "grid_shape": [2, 2],
        "latitude_file": "latitude.npy",
        "longitude_file": "longitude.npy",
        "channels": schema_manifest(),
        "instantaneous_channel_names": list(ERA5_CHANNEL_NAMES[:-1]),
        "tp_channel_name": "tp",
        "tp_units": "mm per 1 hour",
        "tp_interval": "(valid_time - 1 hour, valid_time]",
        "shards": [
            {
                "year": 2020,
                "start_utc": "2020-01-01T00:00:00+00:00",
                "end_utc_exclusive": "2021-01-01T00:00:00+00:00",
                "hours": HOURS_2020,
                "instantaneous_file": instantaneous_file.name,
                "tp_file": tp_file.name,
                "data_valid_inst_file": "data-valid-inst-2020.npy",
                "tp_valid_file": "tp-valid-2020.npy",
                "valid_instantaneous_hours": 6,
                "valid_tp_hours": 6,
                "instantaneous_sha256": sha256_file(instantaneous_file),
                "tp_sha256": sha256_file(tp_file),
            }
        ],
        "provenance": {
            "provider_id": "era5",
            "provider_version": "synthetic",
            "dataset": "synthetic-test",
            "sources": ["synthetic.grib"],
            "transformations": ["tp:m_to_mm"],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return root


def _write_static_and_coordinates(directory: Path) -> tuple[Path, Path]:
    axis = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    coverage = np.ones((256, 256), dtype=np.uint8)
    coverage[0, 0] = 0
    static = directory / "target_static.npz"
    np.savez(
        static,
        elevation=100.0 + 10.0 * x + 20.0 * y,
        slope_x=x,
        slope_y=y,
        local_relief=np.square(x) + np.square(y),
        land_mask=(x > 0).astype(np.uint8),
        lcc_x_normalized=x,
        lcc_y_normalized=y,
        coverage=coverage,
    )
    coordinates = directory / "condition-grid-coordinates.npz"
    latitude = np.broadcast_to(np.linspace(36.0, 38.0, 256)[:, None], (256, 256)).copy()
    longitude = np.broadcast_to(np.linspace(126.0, 128.0, 256)[None], (256, 256)).copy()
    latitude[0, 0] = np.nan
    np.savez(coordinates, latitude=latitude, longitude=longitude)
    return static, coordinates


def _synthetic_inputs(tmp_path: Path, *, reverse: bool = False) -> dict[str, Path]:
    candidate = _write_candidate(tmp_path / ("bundle-reversed" if reverse else "bundle"), reverse=reverse)
    radar = _write_radar_cache(tmp_path)
    era = _write_era_cache(tmp_path)
    static, coordinates = _write_static_and_coordinates(tmp_path)
    source = tmp_path / "cprecnet-source-manifest.json"
    source.write_text(json.dumps({"dataset": "synthetic", "artifacts": []}) + "\n")
    return {
        "candidate": candidate,
        "radar": radar,
        "era": era,
        "static": static,
        "coordinates": coordinates,
        "source": source,
    }


def test_cli_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kcorrdiff.training.build_normalization", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--candidate-manifest" in result.stdout
    assert "--source-manifest" in result.stdout
    assert "--output-json" in result.stdout


def test_small_cli_smoke_uses_only_outer_train_unique_input_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "normalization.json"
    assert main(
        [
            "--candidate-manifest",
            str(paths["candidate"]),
            "--radar-cache-root",
            str(paths["radar"]),
            "--era5-cache-root",
            str(paths["era"]),
            "--target-static",
            str(paths["static"]),
            "--context-coordinates",
            str(paths["coordinates"]),
            "--source-manifest",
            str(paths["source"]),
            "--output-json",
            str(output),
            "--chunk-size",
            "5",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["sha256"] == sha256_file(output)

    artifact = read_normalization_artifact(output)
    counts = artifact.provenance["time_universe_counts"]
    assert counts == {
        "candidate_rows_total": 2,
        "outer_train_candidate_rows": 2,
        "outer_train_unique_t0": 1,
        "radar_history_unique_timestamps": 12,
        "era_instantaneous_requested_unique_hours": 7,
        "era_instantaneous_valid_unique_hours": 6,
        "era_tp_requested_unique_hours": 7,
        "era_tp_valid_unique_hours": 6,
        "target_static_valid_pixels": 256 * 256 - 1,
        "context_geometry_valid_pixels": 256 * 256 - 1,
    }
    assert tuple(artifact.statistics["era5_instantaneous"]) == ERA5_CHANNEL_NAMES[:-1]
    assert tuple(artifact.statistics["era5_tp"]) == ("tp",)
    assert artifact.statistics["era5_instantaneous"]["t925"].count == 6 * 2 * 2
    assert artifact.statistics["era5_tp"]["tp"].count == 6 * 2 * 2
    assert artifact.statistics["target_radar"]["cprecnet_normalized"].count == 12 * (256 * 256 - 1)
    assert artifact.statistics["context_radar"]["rain_rate_mm_per_hour"].count == 12 * (256 * 256 - 1)
    assert set(artifact.statistics["target_static"]) == {
        "elevation",
        "slope_x",
        "slope_y",
        "local_relief",
    }
    hashes = artifact.provenance["artifact_hashes"]
    assert hashes["candidate_manifest"] == sha256_file(paths["candidate"])
    assert hashes["radar"]["cache_manifest"] == sha256_file(paths["radar"] / "manifest.json")
    assert hashes["era5"]["cache_manifest"] == sha256_file(paths["era"] / "manifest.json")
    assert hashes["static"]["target_static"] == sha256_file(paths["static"])
    assert hashes["source_manifests"][paths["source"].name] == sha256_file(paths["source"])


def test_candidate_row_order_does_not_change_universe_or_statistics(tmp_path: Path) -> None:
    paths = _synthetic_inputs(tmp_path)
    reversed_candidate = _write_candidate(tmp_path / "bundle-reversed", reverse=True)
    first_universe = derive_candidate_universe(paths["candidate"])
    second_universe = derive_candidate_universe(reversed_candidate)
    assert first_universe == second_universe

    common = {
        "radar_cache_root": paths["radar"],
        "era5_cache_root": paths["era"],
        "target_static": paths["static"],
        "context_coordinates": paths["coordinates"],
        "source_manifests": [paths["source"]],
        "chunk_size": 7,
    }
    first = build_normalization_artifact(
        candidate_manifest=paths["candidate"], **common
    )
    second = build_normalization_artifact(
        candidate_manifest=reversed_candidate, **common
    )
    assert first.to_json()["statistics"] == second.to_json()["statistics"]
    assert (
        first.provenance["time_universe_counts"]
        == second.provenance["time_universe_counts"]
    )
    assert (
        first.provenance["artifact_hashes"]["candidate_outer_train_semantic"]
        == second.provenance["artifact_hashes"]["candidate_outer_train_semantic"]
    )
    # Exact source bytes are still tracked, so reversing JSON rows changes only
    # the candidate artifact hash, not the fitted estimand/statistics.
    assert (
        first.provenance["artifact_hashes"]["candidate_manifest"]
        != second.provenance["artifact_hashes"]["candidate_manifest"]
    )


def test_outer_train_dependency_leakage_fails_before_cache_scan(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "bundle")
    raw = json.loads(candidate.read_text())
    raw["metadata"]["split_intervals"][0]["start"] = T0.isoformat()
    candidate.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="dependency crosses split"):
        derive_candidate_universe(candidate)

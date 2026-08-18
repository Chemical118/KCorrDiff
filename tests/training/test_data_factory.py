from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

import kcorrdiff.training.data_factory as data_factory
from kcorrdiff.data.cache import FORMAT_VERSION as RADAR_CACHE_FORMAT
from kcorrdiff.data.condition_augmentation import (
    ConditionAugmentationPolicy,
    ConditionMixtureEntry,
)
from kcorrdiff.data.context_regrid import build_context_regrid_operator
from kcorrdiff.data.coordinates import normalize_lcc_coordinates, target_center_from_axes
from kcorrdiff.data.era5_reader import ERA5_CACHE_FORMAT
from kcorrdiff.data.event_manifest import (
    build_event_pretrain_index,
    write_event_training_bundle,
)
from kcorrdiff.data.normalization import (
    ChannelStatistics,
    NormalizationArtifact,
    read_normalization_artifact,
    sha256_file,
    write_normalization_artifact,
)
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES, schema_manifest
from kcorrdiff.data.sampling import CandidateItem, DrawRow, write_draw_manifest
from kcorrdiff.data.split_manifest import (
    SplitInterval,
    canonical_sample_id,
    write_manifest,
)
from kcorrdiff.models.context_encoder import CONTEXT_STATIC_CHANNEL_NAMES
from kcorrdiff.training.config import (
    Stage2Config,
    load_stage2_config,
    thaw_config_mapping,
)
from kcorrdiff.training.data_factory import (
    DatasetDependencies,
    EraArtifactNormalization,
    FactoryInputs,
    KCorrDiffDataFactory,
    LazyRankSlotDataset,
    PlannedSample,
    RankSlotCollator,
    build_factory_artifacts,
    dataloader_options,
    load_coordinate_artifact,
)
from kcorrdiff.training.plan import DrawSlot


SIGNATURE = "era5_oracle:era=1:tp=1:full_trajectory"
HASH_KEYS = (
    "config_sha256",
    "event_index_semantic_sha256",
    "eligible_universe_sample_ids_sha256",
    "candidate_semantic_sha256",
    "fold_map_sha256",
)


class _RecordingPinLeaf:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def pin_memory(self) -> "_RecordingPinLeaf":
        self.calls.append(self.name)
        return self


class _ForbiddenRecursivePin:
    def pin_memory(self) -> None:
        raise AssertionError("custom dataclass contents were recursively pinned")


@dataclass(frozen=True)
class _GenericPinTree:
    pair: tuple[object, ...]
    mapping: dict[str, object]


@dataclass(frozen=True)
class _CustomPinDataclass:
    calls: list[str]
    child: object

    def pin_memory(self) -> "_CustomPinDataclass":
        self.calls.append("custom")
        return self


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _stats(mean: float = 1.0, standard_deviation: float = 2.0) -> ChannelStatistics:
    return ChannelStatistics(
        count=4,
        mean=mean,
        variance=standard_deviation**2,
        standard_deviation=standard_deviation,
        minimum=mean - standard_deviation,
        maximum=mean + standard_deviation,
    )


def _coordinate_npz(path: Path, *, nonuniform: bool = False) -> str:
    target = np.arange(256, dtype=np.float64) * 0.5 - 63.75
    if nonuniform:
        increments = np.linspace(0.25, 0.75, 255, dtype=np.float64)
        x_km = np.concatenate(([target[0]], target[0] + np.cumsum(increments)))
        y_km = x_km.copy()
    else:
        x_km = target
        y_km = target.copy()
    longitude, latitude = np.meshgrid(
        126.0 + x_km / 100.0, 37.0 + y_km / 100.0, indexing="xy"
    )
    np.savez(
        path,
        lcc_x_m=x_km * 1000.0,
        lcc_y_m=y_km * 1000.0,
        latitude=latitude,
        longitude=longitude,
    )
    return sha256_file(path)


def _target_static(path: Path, target_coordinates: Path) -> str:
    with np.load(target_coordinates, allow_pickle=False) as archive:
        x = np.asarray(archive["lcc_x_m"], dtype=np.float64) / 1000.0
        y = np.asarray(archive["lcc_y_m"], dtype=np.float64) / 1000.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    center_x, center_y = target_center_from_axes(x, y)
    shared_x, shared_y = normalize_lcc_coordinates(
        xx,
        yy,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
    )
    np.savez(
        path,
        elevation=100.0 + xx + yy,
        slope_x=xx / 100.0,
        slope_y=yy / 100.0,
        local_relief=np.hypot(xx, yy),
        land_mask=(xx >= 0).astype(np.float32),
        lcc_x_normalized=shared_x,
        lcc_y_normalized=shared_y,
        coverage=np.ones((256, 256), dtype=np.bool_),
        lcc_x_m=x * 1000.0,
        lcc_y_m=y * 1000.0,
    )
    return sha256_file(path)


def _radar_cache(root: Path) -> dict[str, str]:
    root.mkdir()
    shards = []
    for source in ("target", "condition"):
        timestamps = f"{source}.timestamps.json"
        _write_json(root / timestamps, ["20200101_0000"])
        np.save(root / f"{source}.npy", np.zeros((1, 1, 1), dtype=np.float32))
        shards.append(
            {
                "source": source,
                "archive_name": f"{source}.npz",
                "archive_size": 1,
                "archive_sha256": ("1" if source == "target" else "2") * 64,
                "values_file": f"{source}.npy",
                "timestamps_file": timestamps,
                "samples": 1,
                "shape": [256, 256],
                "dtype": "float32",
                "values_sha256": ("3" if source == "target" else "4") * 64,
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "format_version": RADAR_CACHE_FORMAT,
            "dtype": "float32",
            "shards": shards,
            "unique_timestamps": 1,
            "estimated_value_bytes": 2 * 256 * 256 * 4,
        },
    )
    return data_factory._radar_cache_hashes(root)


def _era_cache(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    root.mkdir()
    latitude = np.linspace(32.0, 40.0, 33, dtype=np.float64)
    longitude = np.linspace(123.0, 131.0, 33, dtype=np.float64)
    np.save(root / "latitude.npy", latitude)
    np.save(root / "longitude.npy", longitude)
    np.save(root / "inst-valid-2020.npy", np.ones(1, dtype=np.bool_))
    np.save(root / "tp-valid-2020.npy", np.ones(1, dtype=np.bool_))
    (root / "instantaneous-2020.npy").touch()
    (root / "tp-2020.npy").touch()
    provenance = {
        "provider_id": "era5",
        "provider_version": "test-provider-v1",
        "dataset": "test-era5",
        "sources": ["test.grib"],
        "transformations": ["tp_m_to_mm"],
    }
    _write_json(
        root / "manifest.json",
        {
            "format_version": ERA5_CACHE_FORMAT,
            "dtype": "float32",
            "grid_shape": [33, 33],
            "latitude_file": "latitude.npy",
            "longitude_file": "longitude.npy",
            "latitude_orientation": "south_to_north",
            "longitude_orientation": "west_to_east",
            "channels": schema_manifest(),
            "instantaneous_channel_names": list(ERA5_CHANNEL_NAMES[:-1]),
            "tp_channel_name": "tp",
            "tp_units": "mm per 1 hour",
            "tp_interval": "(valid_time - 1 hour, valid_time]",
            "time_zone": "UTC",
            "shards": [
                {
                    "year": 2020,
                    "start_utc": "2020-01-01T00:00:00+00:00",
                    "end_utc_exclusive": "2021-01-01T00:00:00+00:00",
                    "hours": 8784,
                    "instantaneous_file": "instantaneous-2020.npy",
                    "tp_file": "tp-2020.npy",
                    "data_valid_inst_file": "inst-valid-2020.npy",
                    "tp_valid_file": "tp-valid-2020.npy",
                    "valid_instantaneous_hours": 8784,
                    "valid_tp_hours": 8784,
                    "instantaneous_sha256": "5" * 64,
                    "tp_sha256": "6" * 64,
                }
            ],
            "provenance": provenance,
        },
    )
    hashes, _, _, _ = data_factory._era_cache_hashes_and_geometry(root)
    return hashes, provenance


@dataclass(frozen=True)
class ArtifactFixture:
    config: Stage2Config
    inputs: FactoryInputs
    era_provenance: dict[str, str]


@pytest.fixture(scope="module")
def artifact_fixture(tmp_path_factory: pytest.TempPathFactory) -> ArtifactFixture:
    root = tmp_path_factory.mktemp("data-factory")
    target_coordinates = root / "target-coordinates.npz"
    context_coordinates = root / "context-coordinates.npz"
    target_coordinate_hash = _coordinate_npz(target_coordinates)
    context_coordinate_hash = _coordinate_npz(context_coordinates, nonuniform=True)
    target_static = root / "target-static.npz"
    target_static_hash = _target_static(target_static, target_coordinates)
    radar_root = root / "radar-v1"
    radar_hashes = _radar_cache(radar_root)
    era_root = root / "era5-v1"
    era_hashes, era_provenance = _era_cache(era_root)

    metadata_hashes = {key: str(index + 7) * 64 for index, key in enumerate(HASH_KEYS)}
    candidates = []
    rows = []
    for index in range(3):
        t0 = datetime(2020, 6, 1, index, tzinfo=UTC)
        lead = 0.5 + 0.5 * index
        sample_id = canonical_sample_id(t0, lead, SIGNATURE)
        item = CandidateItem(
            sample_id=sample_id,
            t0_utc=t0.isoformat(),
            lead_hours=lead,
            condition_signature=SIGNATURE,
            block_id=f"block-{index}",
            stratum="event",
            split="outer_train",
            p_target=1.0 / 3.0,
            fold_id=index,
        )
        candidates.append(item)
        rows.append(
            DrawRow(
                global_example_index=index,
                **asdict(item),
                p_draw=1.0 / 3.0,
                omega=1.0,
            )
        )
    candidate_path = root / "candidate-manifest.json"
    candidate_hash = write_manifest(
        candidate_path,
        candidates,
        metadata={
            "protocol_version": "v1.1.3b",
            "outer_train_folds": 3,
            "population": "cprecnet_event_conditioned_archive",
            "strata": ["event"],
            "probability_scope": "within_split",
            "target_policy": "eligible_items_uniform",
            "draw_policy": "uniform",
            **metadata_hashes,
        },
    )
    draw_path = root / "training-draw-manifest.jsonl"
    draw_hash = write_draw_manifest(draw_path, rows)
    bundle_path = root / "bundle-metadata.json"
    bundle = {
        "format_version": "kcorrdiff.event-training-bundle.v1",
        "protocol_version": "v1.1.3b",
        "population": "cprecnet_event_conditioned_archive",
        "estimand": "cprecnet_event_conditioned_pretraining",
        "strata": ["event"],
        "condition_signatures": [SIGNATURE],
        **metadata_hashes,
        "sampling": {
            "draw_split": "outer_train",
            "target_policy": "eligible_items_uniform",
            "probability_scope": "within_split",
            "draw_policy": "uniform",
            "weight_clipping": None,
        },
        "folds": {"split": "outer_train", "count": 3},
        "artifacts": {
            "candidate_manifest": {
                "path": candidate_path.name,
                "bytes": candidate_path.stat().st_size,
                "sha256": candidate_hash,
                "rows": 3,
            },
            "training_draw_manifest": {
                "path": draw_path.name,
                "bytes": draw_path.stat().st_size,
                "sha256": draw_hash,
                "rows": 3,
            },
        },
    }
    _write_json(bundle_path, bundle)
    bundle_hash = sha256_file(bundle_path)
    (root / "bundle-metadata.sha256").write_text(
        f"{bundle_hash}  bundle-metadata.json\n", encoding="ascii"
    )

    normalization_path = root / "normalization.json"
    artifact = NormalizationArtifact(
        statistics={
            "era5_instantaneous": {
                name: _stats(index + 1.0, 2.0)
                for index, name in enumerate(ERA5_CHANNEL_NAMES[:-1])
            },
            "era5_tp": {"tp": _stats(2.0, 0.5)},
            "target_static": {
                name: _stats(0.0, 1.0)
                for name in ("elevation", "slope_x", "slope_y", "local_relief")
            },
        },
        provenance={
            "artifact_hashes": {
                "candidate_manifest": candidate_hash,
                "candidate_outer_train_semantic": "a" * 64,
                "split_definition_semantic": "b" * 64,
                "radar": radar_hashes,
                "era5": era_hashes,
                "static": {
                    "target_static": target_static_hash,
                    "context_coordinates": context_coordinate_hash,
                },
                "source_manifests": {"source.json": "c" * 64},
                "bundle_metadata": bundle_hash,
            },
            "candidate_metadata_hashes": metadata_hashes,
            "time_universe_counts": {"outer_train_candidate_rows": 3},
            "cache_hash_verification": True,
        },
        excluded_fields={
            "target_static": ("land_mask", "lcc_x_normalized", "lcc_y_normalized")
        },
    )
    write_normalization_artifact(normalization_path, artifact)

    raw = yaml.safe_load(
        (Path(__file__).parents[2] / "configs" / "stage2-full-width.yaml").read_text()
    )
    raw["data"]["target_coordinates"] = str(target_coordinates)
    raw["data"]["context_coordinates"] = str(context_coordinates)
    raw["data"]["normalization"] = str(normalization_path)
    # The shared fixture intentionally exercises the legacy immutable
    # single-signature bundle. Production augmentation has its own end-to-end
    # fixture below and must not silently reinterpret this source artifact.
    raw["data"].pop("condition_augmentation", None)
    config_path = root / "stage2.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_stage2_config(config_path)
    inputs = FactoryInputs(
        radar_cache_root=radar_root,
        era5_cache_root=era_root,
        target_static=target_static,
        candidate_manifest=candidate_path,
        draw_manifest=draw_path,
        bundle_metadata=bundle_path,
        expected_target_coordinates_sha256=target_coordinate_hash,
        expected_context_coordinates_sha256=context_coordinate_hash,
        verify_cache_hashes=False,
    )
    return ArtifactFixture(config=config, inputs=inputs, era_provenance=era_provenance)


@pytest.fixture(scope="module")
def built(artifact_fixture: ArtifactFixture):
    calls = []

    def counting_builder(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return build_context_regrid_operator(*args, **kwargs)

    artifacts = build_factory_artifacts(
        artifact_fixture.config,
        artifact_fixture.inputs,
        operator_builder=counting_builder,
    )
    assert len(calls) == 1
    return artifacts


def test_coordinate_hash_axes_and_no_small_grid_override(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.npz"
    digest = _coordinate_npz(path)
    artifact = load_coordinate_artifact(
        path, expected_sha256=digest, kind="target"
    )
    assert artifact.x_lcc_km.shape == (256,)
    assert np.allclose(np.diff(artifact.x_lcc_km), 0.5)
    assert load_coordinate_artifact(
        path, expected_sha256="0" * 64, kind="target"
    ).x_lcc_km.shape == (256,)
    small = tmp_path / "small.npz"
    axis = np.arange(16, dtype=np.float64)
    np.savez(
        small,
        lcc_x_m=axis,
        lcc_y_m=axis,
        latitude=np.zeros((16, 16)),
        longitude=np.zeros((16, 16)),
    )
    with pytest.raises(ValueError, match="256"):
        load_coordinate_artifact(
            small, expected_sha256=sha256_file(small), kind="target"
        )


def test_factory_preflight_builds_exact_named_static_and_common_lcc(built) -> None:
    assert len(built.rows) == 3
    assert built.target_static.values.shape == (7, 256, 256)
    assert built.context_static_fields.shape == (6, 256, 256)
    assert built.context_static_channel_names == CONTEXT_STATIC_CHANNEL_NAMES
    assert built.geometry.input_size == 256
    assert np.asarray(built.geometry.era_latitude_degrees).shape == (33,)
    assert built.era_normalization.artifact_id == built.artifact_hashes["normalization"]
    assert not built.target_static.values.flags.writeable
    assert not built.context_static_fields.flags.writeable


def test_factory_ignores_bundle_sidecar_and_coordinate_hash_metadata(
    artifact_fixture: ArtifactFixture,
) -> None:
    bad_inputs = FactoryInputs(
        **{
            name: getattr(artifact_fixture.inputs, name)
            for name in artifact_fixture.inputs.__dataclass_fields__
            if name != "expected_target_coordinates_sha256"
        },
        expected_target_coordinates_sha256="0" * 64,
    )
    assert build_factory_artifacts(artifact_fixture.config, bad_inputs).rows

    sidecar = artifact_fixture.inputs.bundle_metadata.with_suffix(".sha256")
    original = sidecar.read_text()
    try:
        sidecar.write_text(f"{'0' * 64}  bundle-metadata.json\n")
        assert build_factory_artifacts(
            artifact_fixture.config, artifact_fixture.inputs
        ).rows
    finally:
        sidecar.write_text(original)


def test_candidate_universe_independently_rejects_mutated_draw_metadata(
    artifact_fixture: ArtifactFixture, built
) -> None:
    bundle = json.loads(artifact_fixture.inputs.bundle_metadata.read_text())
    forged = (replace(built.rows[0], p_target=0.25), *built.rows[1:])
    with pytest.raises(ValueError, match="p_target disagrees"):
        data_factory._audit_draw_membership(
            path=artifact_fixture.inputs.candidate_manifest,
            rows=forged,
            bundle=bundle,
            config=artifact_fixture.config,
        )


def test_augmented_bundle_factory_consumes_row_level_signatures(
    tmp_path: Path,
    artifact_fixture: ArtifactFixture,
) -> None:
    starts = (
        datetime(2020, 6, 1, tzinfo=UTC),
        datetime(2020, 7, 1, tzinfo=UTC),
        datetime(2020, 8, 1, tzinfo=UTC),
    )
    timestamps = tuple(
        start + timedelta(minutes=5 * offset)
        for start in starts
        for offset in range(84)
    )
    intervals = (
        SplitInterval(
            "outer_train",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
    )
    event_rows, event_audit = build_event_pretrain_index(
        timestamps,
        intervals,
        condition_signature=SIGNATURE,
    )
    tp_off = "era5_oracle:era=1:tp=0:full_trajectory"
    null = "null_provider:era=0:tp=0:no_era_access"
    policy = ConditionAugmentationPolicy(
        policy_id="factory-row-signature-test-v1",
        protocol_version="v1.1.3b",
        source_condition_signature=SIGNATURE,
        temporal_access_strategy="matched_arm",
        entries=(
            ConditionMixtureEntry(SIGNATURE, 0.6, 0.5),
            ConditionMixtureEntry(tp_off, 0.25, 0.25),
            ConditionMixtureEntry(null, 0.15, 0.25),
        ),
    )
    bundle_dir = tmp_path / "augmented-bundle"
    write_event_training_bundle(
        bundle_dir,
        event_rows,
        event_audit,
        intervals,
        draw_count=40,
        draw_policy="uniform",
        seed=11103,
        purpose_id="factory-base-draw-v1",
        signature_purpose_id="factory-condition-signature-v1",
        condition_augmentation=policy,
        maximum_oof_bytes=100_000_000,
        config_sha256="a" * 64,
        protocol_version="v1.1.3b",
    )
    bundle_path = bundle_dir / "bundle-metadata.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    original_normalization = read_normalization_artifact(
        artifact_fixture.config.data.normalization
    )
    provenance = deepcopy(dict(original_normalization.provenance))
    artifact_hashes = deepcopy(dict(provenance["artifact_hashes"]))
    artifact_hashes["candidate_manifest"] = sha256_file(
        bundle_dir / "candidate-manifest.json"
    )
    artifact_hashes["bundle_metadata"] = sha256_file(bundle_path)
    provenance["artifact_hashes"] = artifact_hashes
    provenance["candidate_metadata_hashes"] = {
        key: bundle[key] for key in HASH_KEYS
    } | {
        "condition_augmentation_policy_sha256": policy.semantic_sha256
    }
    provenance["time_universe_counts"] = {
        "outer_train_candidate_rows": bundle["artifacts"][
            "candidate_manifest"
        ]["rows"]
    }
    normalization = NormalizationArtifact(
        statistics=original_normalization.statistics,
        provenance=provenance,
        excluded_fields=original_normalization.excluded_fields,
    )
    normalization_path = tmp_path / "normalization.json"
    write_normalization_artifact(normalization_path, normalization)

    raw_config = thaw_config_mapping(artifact_fixture.config.raw)
    raw_config["data"]["normalization"] = str(normalization_path)
    raw_config["data"]["condition_augmentation"] = policy.to_json()
    config_path = tmp_path / "stage2-augmented.yaml"
    config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8"
    )
    config = load_stage2_config(config_path)
    inputs = FactoryInputs(
        radar_cache_root=artifact_fixture.inputs.radar_cache_root,
        era5_cache_root=artifact_fixture.inputs.era5_cache_root,
        target_static=artifact_fixture.inputs.target_static,
        candidate_manifest=bundle_dir / "candidate-manifest.json",
        draw_manifest=bundle_dir / "training-draw-manifest.jsonl",
        bundle_metadata=bundle_path,
        expected_target_coordinates_sha256=(
            artifact_fixture.inputs.expected_target_coordinates_sha256
        ),
        expected_context_coordinates_sha256=(
            artifact_fixture.inputs.expected_context_coordinates_sha256
        ),
        verify_cache_hashes=False,
    )
    artifacts = build_factory_artifacts(config, inputs)

    assert artifacts.condition_signatures == config.data.condition_signatures
    assert {row.condition_signature for row in artifacts.rows} == {
        SIGNATURE,
        tp_off,
        null,
    }
    assert artifacts.artifact_hashes["condition_augmentation_policy"] == (
        policy.semantic_sha256
    )
    factory = KCorrDiffDataFactory(config, inputs, artifacts)
    plan = factory.plan(role="deployment", fold_id=None, world_size=2)
    rank_dataset = factory.rank_dataset(plan, rank=0)
    assert rank_dataset.condition_signatures == config.data.condition_signatures


def test_era_normalization_preserves_separate_validity_and_presence(
    built, artifact_fixture: ArtifactFixture
) -> None:
    normalizer: EraArtifactNormalization = built.era_normalization
    values = np.empty((2, 8, 24, 33, 33), dtype=np.float32)
    for index, name in enumerate(ERA5_CHANNEL_NAMES[:-1]):
        values[:, :, index] = normalizer.instantaneous[name].mean
    values[:, :, -1] = normalizer.tp["tp"].mean
    values[0, 1, 0] = np.nan
    values[1, :, -1] = np.nan
    inst_valid = np.ones((2, 8), dtype=np.bool_)
    inst_valid[0, 1] = False
    tp_valid = np.ones((2, 8), dtype=np.bool_)
    provenance = tuple(
        SimpleNamespace(
            cache_manifest=str(artifact_fixture.inputs.era5_cache_root / "manifest.json"),
            condition_signature=SIGNATURE,
            **artifact_fixture.era_provenance,
        )
        for _ in range(2)
    )
    normalized = normalizer.normalize(
        values,
        channel_names=ERA5_CHANNEL_NAMES,
        data_valid_inst=inst_valid,
        tp_valid=tp_valid,
        era_present=np.ones(2, dtype=np.bool_),
        tp_present=np.asarray([True, False], dtype=np.bool_),
        provenance=provenance,
    )
    assert normalized.dtype == np.float32
    assert normalized.shape == values.shape
    assert np.all(normalized == 0.0)
    assert inst_valid[0, 1] is np.False_
    assert tp_valid.all()

    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid ERA channel"):
        normalizer.normalize(
            values,
            channel_names=ERA5_CHANNEL_NAMES,
            data_valid_inst=inst_valid,
            tp_valid=tp_valid,
            era_present=np.ones(2, dtype=np.bool_),
            tp_present=np.asarray([True, False], dtype=np.bool_),
            provenance=provenance,
        )


class _FakeCache:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeDecodedDataset:
    def __init__(self, **kwargs: object) -> None:
        self.rows = kwargs["rows"]

    def __getitem__(self, index: int) -> object:
        row = self.rows[index]
        return (row.global_example_index, row.sample_id)


def _fake_radar(path: Path, verify_hashes: bool) -> _FakeCache:
    return _FakeCache(f"radar:{path}:{verify_hashes}")


def _fake_era(path: Path, verify_hashes: bool) -> _FakeCache:
    return _FakeCache(f"era:{path}:{verify_hashes}")


def _fake_dataset(**kwargs: object) -> _FakeDecodedDataset:
    return _FakeDecodedDataset(**kwargs)


def _lazy_dataset(
    artifact_fixture: ArtifactFixture,
    built,
    slots: tuple[DrawSlot, ...],
) -> LazyRankSlotDataset:
    return LazyRankSlotDataset(
        rows=built.rows,
        slots=slots,
        radar_cache_root=artifact_fixture.inputs.radar_cache_root,
        era5_cache_root=artifact_fixture.inputs.era5_cache_root,
        context_regrid_operator=built.context_regrid_operator,
        static_conditions=built.static_conditions,
        condition_signature=SIGNATURE,
        radar_timezone="Asia/Seoul",
        context_detail_mode="local_max",
        context_wet_threshold_mm_per_hour=0.0,
        verify_cache_hashes=False,
        dependencies=DatasetDependencies(
            radar_cache=_fake_radar,
            era5_cache=_fake_era,
            dataset=_fake_dataset,
        ),
    )


def test_padding_slot_never_opens_cache_or_substitutes_sample(
    artifact_fixture: ArtifactFixture, built
) -> None:
    dataset = _lazy_dataset(
        artifact_fixture,
        built,
        (DrawSlot(logical_position=3, row_index=None), DrawSlot(0, 0)),
    )
    padding = dataset[0]
    assert padding == PlannedSample(slot=DrawSlot(3, None), sample=None)
    assert not dataset.cache_open
    real = dataset[1]
    assert real.sample == (0, built.rows[0].sample_id)
    assert dataset.cache_open
    decoded = dataset._dataset
    assert isinstance(decoded, _FakeDecodedDataset)
    assert decoded.rows is dataset.rows


def test_rank_collator_emits_graph_connected_zero_loss_sentinel() -> None:
    collator = RankSlotCollator(lambda samples: tuple(samples))
    sentinel = collator(
        [PlannedSample(slot=DrawSlot(logical_position=9, row_index=None), sample=None)]
    )
    assert sentinel.is_zero_loss_sentinel
    assert sentinel.loss_weights.tolist() == [0.0]
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    loss = sentinel.connected_zero_loss([parameter])
    loss.backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == 0.0

    mixed = collator(
        [
            PlannedSample(slot=DrawSlot(0, 0), sample="real"),
            PlannedSample(slot=DrawSlot(1, None), sample=None),
        ]
    )
    assert mixed.training == ("real",)
    assert mixed.active_slot_indices == (0,)
    assert mixed.loss_weights.tolist() == [1.0, 0.0]


def test_pin_tree_prefers_custom_method_without_recursive_reentry() -> None:
    calls: list[str] = []
    value = _CustomPinDataclass(calls, _ForbiddenRecursivePin())

    assert data_factory._pin_tree(value) is value
    assert calls == ["custom"]


def test_pin_tree_retains_generic_dataclass_container_behavior() -> None:
    calls: list[str] = []
    first = _RecordingPinLeaf(calls, "first")
    second = _RecordingPinLeaf(calls, "second")
    value = _GenericPinTree(pair=(first, 7), mapping={"leaf": second})

    pinned = data_factory._pin_tree(value)

    assert isinstance(pinned, _GenericPinTree)
    assert pinned is not value
    assert pinned.pair == (first, 7)
    assert pinned.mapping == {"leaf": second}
    assert calls == ["first", "second"]


def test_worker_options_and_topology_preserve_rank_slot_order(
    artifact_fixture: ArtifactFixture, built
) -> None:
    assert dataloader_options(
        num_workers=0,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    ) == {"num_workers": 0, "pin_memory": True}
    worker_options = dataloader_options(
        num_workers=1,
        prefetch_factor=3,
        pin_memory=False,
        persistent_workers=False,
    )
    assert worker_options["prefetch_factor"] == 3
    assert worker_options["persistent_workers"] is False

    slots = (
        DrawSlot(0, 0),
        DrawSlot(1, 1),
        DrawSlot(2, 2),
        DrawSlot(3, None),
    )
    collator = RankSlotCollator(lambda samples: tuple(samples))
    serial = DataLoader(
        _lazy_dataset(artifact_fixture, built, slots),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    parallel = DataLoader(
        _lazy_dataset(artifact_fixture, built, slots),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=2,
    )
    serial_order = [batch.slots[0].logical_position for batch in serial]
    parallel_order = [batch.slots[0].logical_position for batch in parallel]
    assert serial_order == parallel_order == [0, 1, 2, 3]


def test_factory_rejects_foreign_or_mutated_plan(
    artifact_fixture: ArtifactFixture, built
) -> None:
    factory = KCorrDiffDataFactory(
        artifact_fixture.config, artifact_fixture.inputs, built
    )
    plan = factory.plan(role="deployment", fold_id=None, world_size=2)
    global_batch = artifact_fixture.config.optimization.global_effective_batch_size
    assert plan.padding_slots == global_batch - len(built.rows)
    assert plan.optimizer_steps == 1
    microbatch = artifact_fixture.config.optimization.per_rank_microbatch_size
    assert [slot.logical_position for slot in plan.rank_slots[0]] == list(
        range(0, microbatch)
    )
    assert [slot.logical_position for slot in plan.rank_slots[1]] == list(
        range(microbatch, 2 * microbatch)
    )
    rank_one = factory.rank_dataset(plan, rank=1)
    assert rank_one.slots[-1].is_padding
    with pytest.raises(ValueError, match="does not match"):
        factory.rank_dataset(
            type(plan)(
                role=plan.role,
                fold_id=plan.fold_id,
                source_row_indices=plan.source_row_indices,
                world_size=plan.world_size,
                per_rank_microbatch_size=plan.per_rank_microbatch_size,
                rank_slots=(plan.rank_slots[1], plan.rank_slots[0]),
                semantic_sha256=plan.semantic_sha256,
            ),
            rank=0,
        )

"""Immutable, portable Stage 2 release index for Stage 3 consumers.

The Stage 2 training manifest intentionally records content identity rather
than deployment paths.  This module adds the missing promotion boundary: one
canonical index maps every Stage 3 input to a path relative to an explicitly
supplied containment root and binds that path to immutable content.

No absolute path is serialized.  Indexed files and directory roots must be
regular, symlink-free descendants of the containment root.  Cache and OOF
directories are transitively identified by their verified ``manifest.json``;
their manifests remain responsible for binding all payload shards.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Mapping

import yaml

from kcorrdiff.data.condition_augmentation import ConditionAugmentationPolicy


RELEASE_INDEX_FORMAT = "kcorrdiff.stage2-release-index.v2"
PROTOCOL_VERSION = "v1.1.3b"
PATH_CONTRACT = "posix-relative-to-explicit-containment-root.v2"
STAGE2_MANIFEST_FORMAT = "kcorrdiff.stage2-training.v1"
STAGE2_LAUNCH_IDENTITY_FORMAT = "kcorrdiff.stage2-launch-identity.v1"
IMPORTED_FOLD_SET_FORMAT = "kcorrdiff.stage2-verified-fold-set-import.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILE_RECORD_KEYS = frozenset({"path", "sha256", "bytes"})
_ROOT_RECORD_KEYS = frozenset(
    {"path", "manifest_artifact", "manifest_sha256", "identity_sha256"}
)
_CHECKPOINT_ARTIFACT_KEYS = frozenset(
    {"role", "fold_id", "path", "sha256", "bytes"}
)
_IMPORTED_FOLD_ARTIFACT_KEYS = frozenset({"manifest", "partials"})
_IMPORTED_PARTIAL_ARTIFACT_KEYS = frozenset(
    {"fold_id", "path", "sha256", "bytes"}
)
_INDEX_KEYS = frozenset(
    {
        "format_version",
        "protocol_version",
        "stage",
        "path_contract",
        "artifacts",
        "model_checkpoints",
        "imported_fold_set_artifacts",
        "roots",
        "identities",
        "semantic_sha256",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "stage2_config_semantic_sha256",
        "condition_augmentation_policy_semantic_sha256",
    }
)
_FILE_ROLES = (
    "stage2_config",
    "launch_identity",
    "data_contract",
    "augmented_candidate_manifest",
    "augmented_draw_manifest",
    "augmented_bundle_metadata",
    "condition_augmentation_policy",
    "normalization",
    "radar_cache_manifest",
    "era5_cache_manifest",
    "target_static",
    "target_coordinates",
    "context_coordinates",
    "stage2_manifest",
    "oof_artifact_manifest",
    "residual_scales",
    "runtime_report",
)
_ROOT_BINDINGS = {
    "radar_cache_root": "radar_cache_manifest",
    "era5_cache_root": "era5_cache_manifest",
    "oof_artifact_root": "oof_artifact_manifest",
}
_STAGE2_ARTIFACT_HASH_BINDINGS = {
    "candidate_manifest": "augmented_candidate_manifest",
    "draw_manifest": "augmented_draw_manifest",
    "bundle_metadata": "augmented_bundle_metadata",
    "normalization": "normalization",
    "radar_cache_manifest": "radar_cache_manifest",
    "era5_cache_manifest": "era5_cache_manifest",
    "target_static": "target_static",
    "target_coordinates": "target_coordinates",
    "context_coordinates": "context_coordinates",
}
_CHECKPOINT_IDENTITIES = (
    ("fold", 0),
    ("fold", 1),
    ("fold", 2),
    ("deployment", None),
    ("direct_mean", None),
    ("direct_q50", None),
)
_IMPORTED_FOLD_SET_KEYS = frozenset(
    {"format_version", "manifest_path", "manifest_sha256", "producer", "folds"}
)
_IMPORTED_FOLD_KEYS = frozenset(
    {
        "fold_id",
        "checkpoint_path",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "partial_manifest_path",
        "partial_manifest_sha256",
        "cursor",
        "plan_semantic_sha256",
        "plan_optimizer_steps",
        "training_blocks_sha256",
    }
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, name: str
) -> None:
    actual = {str(key) for key in value}
    if actual != set(expected):
        raise ValueError(
            f"{name} schema mismatch; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _read_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    return _mapping(value, name=name)


def _canonical_root(value: Path) -> Path:
    raw = Path(value)
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if raw.is_symlink() or not absolute.exists():
        raise FileNotFoundError("release containment root must exist without symlinks")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute or not resolved.is_dir():
        raise ValueError("release containment root must be a canonical directory")
    if resolved == Path(resolved.anchor):
        raise ValueError("filesystem root is too broad for release containment")
    return resolved


def _relative_input(root: Path, value: Path, *, name: str) -> tuple[str, Path]:
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError(f"{name} path cannot contain '..'")
    lexical = Path(
        os.path.abspath(os.fspath(raw if raw.is_absolute() else root / raw))
    )
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} path escapes release containment root") from error
    if not relative.parts:
        raise ValueError(f"{name} cannot equal the containment root")
    current = root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"indexed {name} is missing: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"indexed {name} path contains a symlink: {current}")
    if lexical.resolve(strict=True) != lexical:
        raise ValueError(f"indexed {name} path is not canonical")
    posix = relative.as_posix()
    if PurePosixPath(posix).as_posix() != posix or "\\" in posix:
        raise ValueError(f"indexed {name} path is not canonical POSIX")
    return posix, lexical


def _regular_file(root: Path, value: Path, *, name: str) -> tuple[str, Path]:
    relative, selected = _relative_input(root, value, name=name)
    if not stat.S_ISREG(selected.stat().st_mode):
        raise FileNotFoundError(f"indexed {name} must be a regular file")
    return relative, selected


def _directory(root: Path, value: Path, *, name: str) -> tuple[str, Path]:
    relative, selected = _relative_input(root, value, name=name)
    if not stat.S_ISDIR(selected.stat().st_mode):
        raise NotADirectoryError(f"indexed {name} must be a directory")
    return relative, selected


def _relative_record_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty canonical POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise ValueError(f"{name} must be a contained canonical relative path")
    return value


@dataclass(frozen=True, slots=True)
class Stage2ReleaseCheckpointPaths:
    """The six exact regression checkpoints required by Stage 3."""

    fold_0: Path
    fold_1: Path
    fold_2: Path
    deployment: Path
    direct_mean: Path
    direct_q50: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, Path(getattr(self, field.name)))

    def by_identity(self) -> dict[tuple[str, int | None], Path]:
        return {
            ("fold", 0): self.fold_0,
            ("fold", 1): self.fold_1,
            ("fold", 2): self.fold_2,
            ("deployment", None): self.deployment,
            ("direct_mean", None): self.direct_mean,
            ("direct_q50", None): self.direct_q50,
        }


@dataclass(frozen=True, slots=True)
class Stage2ImportedFoldSetPaths:
    """Relocated immutable audit artifacts for an imported B12 fold set."""

    manifest: Path
    fold_0_partial: Path
    fold_1_partial: Path
    fold_2_partial: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, Path(getattr(self, field.name)))

    def partials_by_fold(self) -> dict[int, Path]:
        return {
            0: self.fold_0_partial,
            1: self.fold_1_partial,
            2: self.fold_2_partial,
        }


@dataclass(frozen=True, slots=True)
class Stage2ReleasePaths:
    """All exact files and manifest-rooted directories in a Stage 2 release."""

    stage2_config: Path
    launch_identity: Path
    data_contract: Path
    augmented_candidate_manifest: Path
    augmented_draw_manifest: Path
    augmented_bundle_metadata: Path
    condition_augmentation_policy: Path
    normalization: Path
    radar_cache_manifest: Path
    era5_cache_manifest: Path
    target_static: Path
    target_coordinates: Path
    context_coordinates: Path
    stage2_manifest: Path
    oof_artifact_manifest: Path
    residual_scales: Path
    runtime_report: Path
    radar_cache_root: Path
    era5_cache_root: Path
    oof_artifact_root: Path
    model_checkpoints: Stage2ReleaseCheckpointPaths
    imported_fold_set: Stage2ImportedFoldSetPaths | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"model_checkpoints", "imported_fold_set"}:
                continue
            object.__setattr__(self, field.name, Path(getattr(self, field.name)))
        if not isinstance(self.model_checkpoints, Stage2ReleaseCheckpointPaths):
            raise TypeError("model_checkpoints must be Stage2ReleaseCheckpointPaths")
        if self.imported_fold_set is not None and not isinstance(
            self.imported_fold_set, Stage2ImportedFoldSetPaths
        ):
            raise TypeError("imported_fold_set must be Stage2ImportedFoldSetPaths")


@dataclass(frozen=True, slots=True)
class Stage2ReleaseCheckpoint:
    role: str
    fold_id: int | None
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Stage2ImportedFoldSetArtifacts:
    manifest: Path
    partials: tuple[tuple[int, Path], ...]


@dataclass(frozen=True, slots=True)
class Stage3CLIReleaseInputs:
    """Explicit values consumed by ``kcorrdiff.training.train_stage3``."""

    release_index: Path
    release_index_sha256: str
    release_root: Path
    stage2_config: Path
    stage2_manifest: Path
    stage2_manifest_sha256: str
    oof_artifact: Path
    residual_scales: Path
    radar_cache_root: Path
    era5_cache_root: Path
    draw_manifest: Path
    candidate_manifest: Path
    bundle_metadata: Path
    data_contract: Path
    static_path: Path
    target_coordinates: Path
    context_coordinates: Path
    normalization: Path
    checkpoints: tuple[Stage2ReleaseCheckpoint, ...]
    imported_fold_set: Stage2ImportedFoldSetArtifacts | None

    def as_cli_mapping(self) -> dict[str, str]:
        return {
            "--stage2-release-index": str(self.release_index),
            "--stage2-release-index-sha256": self.release_index_sha256,
            "--stage2-release-root": str(self.release_root),
        }


@dataclass(frozen=True, slots=True)
class Stage2ReleaseIndex:
    index_path: Path
    index_sha256: str
    semantic_sha256: str
    containment_root: Path
    paths: Stage2ReleasePaths
    artifact_sha256: tuple[tuple[str, str], ...]
    stage2_config_semantic_sha256: str
    condition_augmentation_policy_semantic_sha256: str
    checkpoints: tuple[Stage2ReleaseCheckpoint, ...]
    imported_fold_set: Stage2ImportedFoldSetArtifacts | None

    def verify_current_files(self) -> None:
        """Revalidate the index and every content/root binding in place."""

        _load_verified(
            self.index_path,
            expected_sha256=self.index_sha256,
            containment_root=self.containment_root,
        )

    def stage3_cli_inputs(self) -> Stage3CLIReleaseInputs:
        hashes = dict(self.artifact_sha256)
        return Stage3CLIReleaseInputs(
            release_index=self.index_path,
            release_index_sha256=self.index_sha256,
            release_root=self.containment_root,
            stage2_config=self.paths.stage2_config,
            stage2_manifest=self.paths.stage2_manifest,
            stage2_manifest_sha256=hashes["stage2_manifest"],
            oof_artifact=self.paths.oof_artifact_root,
            residual_scales=self.paths.residual_scales,
            radar_cache_root=self.paths.radar_cache_root,
            era5_cache_root=self.paths.era5_cache_root,
            draw_manifest=self.paths.augmented_draw_manifest,
            candidate_manifest=self.paths.augmented_candidate_manifest,
            bundle_metadata=self.paths.augmented_bundle_metadata,
            data_contract=self.paths.data_contract,
            static_path=self.paths.target_static,
            target_coordinates=self.paths.target_coordinates,
            context_coordinates=self.paths.context_coordinates,
            normalization=self.paths.normalization,
            checkpoints=self.checkpoints,
            imported_fold_set=self.imported_fold_set,
        )


def _file_records(
    root: Path, paths: Stage2ReleasePaths
) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    records: dict[str, dict[str, object]] = {}
    selected: dict[str, Path] = {}
    for role in _FILE_ROLES:
        relative, path = _regular_file(root, getattr(paths, role), name=role)
        records[role] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        selected[role] = path
    if len({record["path"] for record in records.values()}) != len(records):
        raise ValueError("release artifacts must use distinct canonical paths")
    return records, selected


def _checkpoint_records(
    root: Path, paths: Stage2ReleaseCheckpointPaths
) -> tuple[list[dict[str, object]], dict[tuple[str, int | None], Path]]:
    records: list[dict[str, object]] = []
    selected: dict[tuple[str, int | None], Path] = {}
    supplied = paths.by_identity()
    if set(supplied) != set(_CHECKPOINT_IDENTITIES):  # pragma: no cover
        raise AssertionError("checkpoint path identities are incomplete")
    for role, fold_id in _CHECKPOINT_IDENTITIES:
        identity = (role, fold_id)
        relative, path = _regular_file(
            root,
            supplied[identity],
            name=f"{role} checkpoint" if fold_id is None else f"fold {fold_id} checkpoint",
        )
        records.append(
            {
                "role": role,
                "fold_id": fold_id,
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        selected[identity] = path
    checkpoint_paths = [str(record["path"]) for record in records]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError("release model checkpoints must use distinct canonical paths")
    return records, selected


def _imported_fold_records(
    root: Path, paths: Stage2ImportedFoldSetPaths | None
) -> tuple[dict[str, object] | None, Stage2ImportedFoldSetArtifacts | None]:
    if paths is None:
        return None, None
    manifest_relative, manifest = _regular_file(
        root, paths.manifest, name="imported fold-set manifest"
    )
    partial_records: list[dict[str, object]] = []
    selected_partials: list[tuple[int, Path]] = []
    for fold_id, supplied in paths.partials_by_fold().items():
        relative, path = _regular_file(
            root, supplied, name=f"imported fold {fold_id} partial manifest"
        )
        partial_records.append(
            {
                "fold_id": fold_id,
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        selected_partials.append((fold_id, path))
    all_paths = [manifest_relative] + [
        str(record["path"]) for record in partial_records
    ]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("imported fold-set artifacts must use distinct paths")
    return (
        {
            "manifest": {
                "path": manifest_relative,
                "sha256": _sha256_file(manifest),
                "bytes": manifest.stat().st_size,
            },
            "partials": partial_records,
        },
        Stage2ImportedFoldSetArtifacts(
            manifest=manifest, partials=tuple(selected_partials)
        ),
    )


def _root_records(
    root: Path,
    paths: Stage2ReleasePaths,
    artifacts: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    records: dict[str, dict[str, object]] = {}
    selected: dict[str, Path] = {}
    for role, manifest_role in _ROOT_BINDINGS.items():
        relative, path = _directory(root, getattr(paths, role), name=role)
        manifest = artifacts[manifest_role]
        identity = {
            "path": relative,
            "manifest_artifact": manifest_role,
            "manifest_sha256": manifest["sha256"],
        }
        records[role] = {**identity, "identity_sha256": _semantic_sha256(identity)}
        selected[role] = path
    if len({record["path"] for record in records.values()}) != len(records):
        raise ValueError("release directory roots must use distinct canonical paths")
    return records, selected


def _artifact_from_bundle(
    bundle: Mapping[str, object], name: str
) -> Mapping[str, object]:
    artifacts = _mapping(bundle.get("artifacts"), name="augmented bundle artifacts")
    return _mapping(artifacts.get(name), name=f"augmented bundle artifact {name}")


def _stage2_checkpoint_lineage(
    stage2: Mapping[str, object],
    checkpoints: Mapping[tuple[str, int | None], Path],
) -> None:
    raw_records = stage2.get("role_checkpoints")
    if not isinstance(raw_records, list):
        raise TypeError("Stage 2 role_checkpoints must be a list")
    declared: dict[tuple[str, int | None], Mapping[str, object]] = {}
    for index, value in enumerate(raw_records):
        record = _mapping(value, name=f"Stage 2 checkpoint record {index}")
        role = record.get("role")
        fold_id = record.get("fold_id")
        identity = (str(role), fold_id if isinstance(fold_id, int) else None)
        if (
            role not in {item[0] for item in _CHECKPOINT_IDENTITIES}
            or isinstance(fold_id, bool)
            or identity not in _CHECKPOINT_IDENTITIES
            or identity in declared
            or not isinstance(record.get("path"), str)
            or not record.get("path")
        ):
            raise ValueError("Stage 2 checkpoint role/fold identity is invalid")
        declared[identity] = record
    if set(declared) != set(_CHECKPOINT_IDENTITIES) or set(checkpoints) != set(
        _CHECKPOINT_IDENTITIES
    ):
        raise ValueError("release must contain all six exact model checkpoints")
    for identity in _CHECKPOINT_IDENTITIES:
        record = declared[identity]
        expected = _sha256(
            record.get("sha256"), name=f"Stage 2 {identity} checkpoint"
        )
        if _sha256_file(checkpoints[identity]) != expected:
            raise ValueError(f"release checkpoint role/content mismatch: {identity}")


def _imported_fold_lineage(
    stage2: Mapping[str, object],
    artifacts: Stage2ImportedFoldSetArtifacts | None,
    checkpoints: Mapping[tuple[str, int | None], Path],
) -> None:
    value = stage2.get("imported_fold_set")
    if value is None:
        if artifacts is not None:
            raise ValueError(
                "release carries imported fold-set artifacts absent from Stage 2"
            )
        return
    if artifacts is None:
        raise ValueError("imported B12 Stage 2 release is missing its audit artifacts")
    imported = _mapping(value, name="Stage 2 imported_fold_set")
    _exact_keys(imported, _IMPORTED_FOLD_SET_KEYS, name="Stage 2 imported_fold_set")
    if imported.get("format_version") != IMPORTED_FOLD_SET_FORMAT:
        raise ValueError("Stage 2 imported fold-set format mismatch")
    expected_manifest = _sha256(
        imported.get("manifest_sha256"), name="imported fold-set manifest"
    )
    if _sha256_file(artifacts.manifest) != expected_manifest:
        raise ValueError("release imported fold-set manifest SHA-256 mismatch")
    partials = dict(artifacts.partials)
    if set(partials) != {0, 1, 2}:
        raise ValueError("release imported fold partial coverage is incomplete")
    raw_folds = imported.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 3:
        raise ValueError("Stage 2 imported fold-set must contain exactly three folds")
    observed: set[int] = set()
    for index, value in enumerate(raw_folds):
        fold = _mapping(value, name=f"Stage 2 imported fold {index}")
        _exact_keys(fold, _IMPORTED_FOLD_KEYS, name=f"Stage 2 imported fold {index}")
        fold_id = fold.get("fold_id")
        if (
            isinstance(fold_id, bool)
            or not isinstance(fold_id, int)
            or fold_id not in partials
            or fold_id in observed
        ):
            raise ValueError("Stage 2 imported fold identities are invalid")
        observed.add(fold_id)
        expected = _sha256(
            fold.get("partial_manifest_sha256"),
            name=f"imported fold {fold_id} partial manifest",
        )
        if _sha256_file(partials[fold_id]) != expected:
            raise ValueError(
                f"release imported fold {fold_id} partial SHA-256 mismatch"
            )
        checkpoint_hash = _sha256(
            fold.get("checkpoint_sha256"),
            name=f"imported fold {fold_id} checkpoint",
        )
        checkpoint_bytes = fold.get("checkpoint_bytes")
        checkpoint = checkpoints[("fold", fold_id)]
        if (
            isinstance(checkpoint_bytes, bool)
            or not isinstance(checkpoint_bytes, int)
            or checkpoint_bytes <= 0
            or checkpoint.stat().st_size != checkpoint_bytes
            or _sha256_file(checkpoint) != checkpoint_hash
        ):
            raise ValueError(
                f"release imported fold {fold_id} checkpoint lineage mismatch"
            )
    if observed != {0, 1, 2}:  # pragma: no cover
        raise AssertionError("imported fold coverage check failed")


def _validate_content_lineage(
    *,
    files: Mapping[str, Path],
    roots: Mapping[str, Path],
    hashes: Mapping[str, str],
    checkpoints: Mapping[tuple[str, int | None], Path],
    imported_fold_set: Stage2ImportedFoldSetArtifacts | None,
    require_original_config_paths: bool,
) -> tuple[str, str]:
    """Cross-check every identity already published by Stage 2 producers."""

    for root_role, manifest_role in _ROOT_BINDINGS.items():
        if files[manifest_role] != roots[root_role] / "manifest.json":
            raise ValueError(
                f"{root_role} must be identified by its direct manifest.json"
            )
    if files["stage2_manifest"].name != "stage2-manifest.json":
        raise ValueError("complete Stage 2 manifest must be named stage2-manifest.json")

    # Parse through the production validators so config and policy semantic
    # identities cannot be replaced by byte-compatible-looking metadata.
    from kcorrdiff.training.config import load_stage2_config
    from kcorrdiff.training.train_stage2 import load_complete_stage2_manifest

    config = load_stage2_config(files["stage2_config"])
    if config.protocol_version != PROTOCOL_VERSION:
        raise ValueError("release Stage 2 config protocol mismatch")
    policy = ConditionAugmentationPolicy.from_mapping(
        _read_json(
            files["condition_augmentation_policy"],
            name="condition augmentation policy",
        )
    )
    if config.data.condition_augmentation is None:
        raise ValueError("release Stage 2 config has no production augmentation policy")
    if config.data.condition_augmentation.semantic_sha256 != policy.semantic_sha256:
        raise ValueError("Stage 2 config and indexed augmentation policy differ")
    if require_original_config_paths:
        # Promotion may copy immutable inputs into a self-contained release
        # tree.  The YAML keeps its producer paths (and semantic hash), so bind
        # an available producer file by content rather than requiring the
        # deployment path to be identical.  Stage 2's artifact_hashes below
        # remain authoritative even when the producer mount is later gone.
        for configured, role in (
            (config.data.normalization, "normalization"),
            (config.data.target_coordinates, "target_coordinates"),
            (config.data.context_coordinates, "context_coordinates"),
        ):
            producer_path = configured.resolve()
            if producer_path.is_file() and _sha256_file(producer_path) != hashes[role]:
                raise ValueError(
                    f"Stage 2 config producer content disagrees with indexed {role}"
                )

    stage2 = load_complete_stage2_manifest(
        files["stage2_manifest"], expected_sha256=hashes["stage2_manifest"]
    )
    if stage2.get("config_sha256") != config.sha256:
        raise ValueError("Stage 2 manifest config semantic SHA-256 mismatch")
    if stage2.get("draw_manifest_sha256") != hashes["augmented_draw_manifest"]:
        raise ValueError("Stage 2 manifest draw SHA-256 mismatch")
    artifact_hashes = _mapping(
        stage2.get("artifact_hashes"), name="Stage 2 manifest artifact hashes"
    )
    for stage2_name, release_role in _STAGE2_ARTIFACT_HASH_BINDINGS.items():
        if artifact_hashes.get(stage2_name) != hashes[release_role]:
            raise ValueError(
                f"Stage 2 manifest artifact hash mismatch: {stage2_name}"
            )
    if artifact_hashes.get("condition_augmentation_policy") != policy.semantic_sha256:
        raise ValueError("Stage 2 manifest augmentation policy identity mismatch")
    if stage2.get("oof_manifest_sha256") != hashes["oof_artifact_manifest"]:
        raise ValueError("Stage 2 manifest OOF SHA-256 mismatch")
    if stage2.get("residual_scales_sha256") != hashes["residual_scales"]:
        raise ValueError("Stage 2 manifest residual-scale SHA-256 mismatch")
    _stage2_checkpoint_lineage(stage2, checkpoints)
    _imported_fold_lineage(stage2, imported_fold_set, checkpoints)

    launch = _read_json(files["launch_identity"], name="Stage 2 launch identity")
    if (
        launch.get("format_version") != STAGE2_LAUNCH_IDENTITY_FORMAT
        or launch.get("stage") != "stage2"
    ):
        raise ValueError("unsupported indexed Stage 2 launch identity")
    declared_launch = _mapping(
        stage2.get("launch_identity"), name="Stage 2 manifest launch identity"
    )
    if declared_launch.get("artifact_sha256") != hashes["launch_identity"]:
        raise ValueError("Stage 2 manifest launch artifact SHA-256 mismatch")
    for launch_name, role in (
        ("runtime_report_sha256", "runtime_report"),
        ("data_contract_sha256", "data_contract"),
    ):
        if launch.get(launch_name) != hashes[role] or declared_launch.get(
            launch_name
        ) != hashes[role]:
            raise ValueError(f"Stage 2 launch {launch_name} mismatch")

    # These are JSON artifacts by contract; parsing here prevents an index
    # from promoting opaque bytes under a trusted role name.
    _read_json(files["data_contract"], name="data contract")
    _read_json(files["runtime_report"], name="runtime report")
    bundle = _read_json(
        files["augmented_bundle_metadata"], name="augmented bundle metadata"
    )
    if (
        bundle.get("protocol_version") != PROTOCOL_VERSION
        or bundle.get("condition_augmentation_policy_sha256")
        != policy.semantic_sha256
    ):
        raise ValueError("augmented bundle policy/protocol identity mismatch")
    bundle_bindings = {
        "candidate_manifest": "augmented_candidate_manifest",
        "training_draw_manifest": "augmented_draw_manifest",
        "condition_policy": "condition_augmentation_policy",
    }
    bundle_root = files["augmented_bundle_metadata"].parent
    for bundle_name, role in bundle_bindings.items():
        record = _artifact_from_bundle(bundle, bundle_name)
        relative = _relative_record_path(
            record.get("path"), name=f"bundle artifact {bundle_name} path"
        )
        if (bundle_root / relative).resolve() != files[role]:
            raise ValueError(f"augmented bundle path mismatch: {bundle_name}")
        if record.get("sha256") != hashes[role]:
            raise ValueError(f"augmented bundle SHA-256 mismatch: {bundle_name}")

    oof = _read_json(files["oof_artifact_manifest"], name="OOF manifest")
    if (
        oof.get("protocol_version") != PROTOCOL_VERSION
        or oof.get("draw_manifest_sha256") != hashes["augmented_draw_manifest"]
    ):
        raise ValueError("OOF manifest draw/protocol identity mismatch")
    scales = _read_json(files["residual_scales"], name="residual scales")
    if scales.get("oof_manifest_sha256") != hashes["oof_artifact_manifest"]:
        raise ValueError("residual scales do not bind the indexed OOF manifest")
    return config.sha256, policy.semantic_sha256


def _payload(
    *, root: Path, paths: Stage2ReleasePaths
) -> tuple[dict[str, object], dict[str, Path], dict[str, Path]]:
    artifacts, files = _file_records(root, paths)
    checkpoint_records, checkpoint_paths = _checkpoint_records(
        root, paths.model_checkpoints
    )
    imported_records, imported_paths = _imported_fold_records(
        root, paths.imported_fold_set
    )
    occupied_paths = {
        str(record["path"]) for record in artifacts.values()
    } | {str(record["path"]) for record in checkpoint_records}
    if imported_records is not None:
        imported_manifest = _mapping(
            imported_records["manifest"], name="imported release manifest"
        )
        imported_partials = imported_records["partials"]
        assert isinstance(imported_partials, list)
        extra_paths = {str(imported_manifest["path"])} | {
            str(_mapping(record, name="imported partial")["path"])
            for record in imported_partials
        }
        if occupied_paths & extra_paths:
            raise ValueError("release file artifacts must use distinct paths")
        occupied_paths |= extra_paths
    if len(occupied_paths) != len(artifacts) + len(checkpoint_records) + (
        0 if imported_records is None else 4
    ):
        raise ValueError("release file artifacts must use distinct paths")
    root_records, roots = _root_records(root, paths, artifacts)
    hashes = {role: str(record["sha256"]) for role, record in artifacts.items()}
    config_sha, policy_sha = _validate_content_lineage(
        files=files,
        roots=roots,
        hashes=hashes,
        checkpoints=checkpoint_paths,
        imported_fold_set=imported_paths,
        require_original_config_paths=True,
    )
    payload: dict[str, object] = {
        "format_version": RELEASE_INDEX_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "stage2",
        "path_contract": PATH_CONTRACT,
        "artifacts": artifacts,
        "model_checkpoints": checkpoint_records,
        "imported_fold_set_artifacts": imported_records,
        "roots": root_records,
        "identities": {
            "stage2_config_semantic_sha256": config_sha,
            "condition_augmentation_policy_semantic_sha256": policy_sha,
        },
    }
    payload["semantic_sha256"] = _semantic_sha256(payload)
    return payload, files, roots


def _output_path(root: Path, value: Path) -> Path:
    raw = Path(value)
    if raw.name != "release-index.json" or ".." in raw.parts:
        raise ValueError("release index must use the exact release-index.json filename")
    selected = Path(
        os.path.abspath(os.fspath(raw if raw.is_absolute() else root / raw))
    )
    try:
        selected.relative_to(root)
    except ValueError as error:
        raise ValueError("release index path escapes containment root") from error
    _directory(root, selected.parent, name="release index parent")
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(selected)
    return selected


def write_stage2_release_index(
    path: Path,
    *,
    containment_root: Path,
    release_paths: Stage2ReleasePaths,
) -> str:
    """Atomically create a no-clobber canonical release index.

    Returns the SHA-256 of the exact canonical index bytes.  The separate
    semantic SHA-256 is embedded in the index and excludes only itself.
    """

    root = _canonical_root(containment_root)
    payload, _, _ = _payload(root=root, paths=release_paths)
    encoded = _canonical_json(payload)
    selected = _output_path(root, path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, selected)
        temporary.unlink()
        directory = os.open(selected.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    if selected.read_bytes() != encoded:
        raise RuntimeError("published Stage 2 release index bytes changed")
    return hashlib.sha256(encoded).hexdigest()


def _paths_from_verified(
    artifacts: Mapping[str, Path],
    roots: Mapping[str, Path],
    checkpoints: Mapping[tuple[str, int | None], Path],
    imported_fold_set: Stage2ImportedFoldSetArtifacts | None,
) -> Stage2ReleasePaths:
    return Stage2ReleasePaths(
        **{role: artifacts[role] for role in _FILE_ROLES},
        **{role: roots[role] for role in _ROOT_BINDINGS},
        model_checkpoints=Stage2ReleaseCheckpointPaths(
            fold_0=checkpoints[("fold", 0)],
            fold_1=checkpoints[("fold", 1)],
            fold_2=checkpoints[("fold", 2)],
            deployment=checkpoints[("deployment", None)],
            direct_mean=checkpoints[("direct_mean", None)],
            direct_q50=checkpoints[("direct_q50", None)],
        ),
        imported_fold_set=(
            None
            if imported_fold_set is None
            else Stage2ImportedFoldSetPaths(
                manifest=imported_fold_set.manifest,
                fold_0_partial=dict(imported_fold_set.partials)[0],
                fold_1_partial=dict(imported_fold_set.partials)[1],
                fold_2_partial=dict(imported_fold_set.partials)[2],
            )
        ),
    )


def _load_verified(
    path: Path, *, expected_sha256: str, containment_root: Path
) -> Stage2ReleaseIndex:
    root = _canonical_root(containment_root)
    _, selected = _regular_file(root, path, name="release index")
    if selected.name != "release-index.json":
        raise ValueError("Stage 2 release index must be named release-index.json")
    encoded = selected.read_bytes()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != _sha256(expected_sha256, name="expected release index"):
        raise ValueError("Stage 2 release index SHA-256 mismatch")
    try:
        decoded = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 release index is not valid UTF-8 JSON") from error
    raw = _mapping(decoded, name="Stage 2 release index")
    if encoded != _canonical_json(raw):
        raise ValueError("Stage 2 release index must use canonical JSON bytes")
    _exact_keys(raw, _INDEX_KEYS, name="Stage 2 release index")
    if (
        raw["format_version"] != RELEASE_INDEX_FORMAT
        or raw["protocol_version"] != PROTOCOL_VERSION
        or raw["stage"] != "stage2"
        or raw["path_contract"] != PATH_CONTRACT
    ):
        raise ValueError("unsupported Stage 2 release index contract")
    declared_semantic = _sha256(
        raw["semantic_sha256"], name="release index semantic identity"
    )
    semantic_payload = {
        key: value for key, value in raw.items() if key != "semantic_sha256"
    }
    if _semantic_sha256(semantic_payload) != declared_semantic:
        raise ValueError("Stage 2 release index semantic SHA-256 mismatch")

    raw_artifacts = _mapping(raw["artifacts"], name="release index artifacts")
    _exact_keys(raw_artifacts, set(_FILE_ROLES), name="release index artifacts")
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    seen_files: set[str] = set()
    for role in _FILE_ROLES:
        record = _mapping(raw_artifacts[role], name=f"release artifact {role}")
        _exact_keys(record, _FILE_RECORD_KEYS, name=f"release artifact {role}")
        relative = _relative_record_path(record["path"], name=f"artifact {role} path")
        if relative in seen_files:
            raise ValueError("release index contains duplicate artifact paths")
        seen_files.add(relative)
        _, file_path = _regular_file(root, Path(relative), name=role)
        declared_bytes = record["bytes"]
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
            or file_path.stat().st_size != declared_bytes
        ):
            raise ValueError(f"release artifact byte count mismatch: {role}")
        digest = _sha256(record["sha256"], name=f"release artifact {role}")
        if _sha256_file(file_path) != digest:
            raise ValueError(f"release artifact SHA-256 mismatch: {role}")
        files[role] = file_path
        hashes[role] = digest

    raw_checkpoints = raw["model_checkpoints"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != len(
        _CHECKPOINT_IDENTITIES
    ):
        raise ValueError("release model checkpoint coverage is incomplete")
    checkpoints: dict[tuple[str, int | None], Path] = {}
    checkpoint_exports: list[Stage2ReleaseCheckpoint] = []
    for expected_identity, value in zip(
        _CHECKPOINT_IDENTITIES, raw_checkpoints, strict=True
    ):
        record = _mapping(value, name=f"release checkpoint {expected_identity}")
        _exact_keys(
            record,
            _CHECKPOINT_ARTIFACT_KEYS,
            name=f"release checkpoint {expected_identity}",
        )
        identity = (record["role"], record["fold_id"])
        if identity != expected_identity:
            raise ValueError("release checkpoint order/role identity mismatch")
        relative = _relative_record_path(
            record["path"], name=f"release checkpoint {identity} path"
        )
        if relative in seen_files:
            raise ValueError("release index contains duplicate file paths")
        seen_files.add(relative)
        _, checkpoint_path = _regular_file(
            root, Path(relative), name=f"release checkpoint {identity}"
        )
        declared_bytes = record["bytes"]
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes <= 0
            or checkpoint_path.stat().st_size != declared_bytes
        ):
            raise ValueError(f"release checkpoint byte count mismatch: {identity}")
        digest = _sha256(record["sha256"], name=f"release checkpoint {identity}")
        if _sha256_file(checkpoint_path) != digest:
            raise ValueError(f"release checkpoint SHA-256 mismatch: {identity}")
        checkpoints[identity] = checkpoint_path
        checkpoint_exports.append(
            Stage2ReleaseCheckpoint(
                role=str(identity[0]),
                fold_id=identity[1],
                path=checkpoint_path,
                sha256=digest,
            )
        )

    raw_imported = raw["imported_fold_set_artifacts"]
    imported_fold_set: Stage2ImportedFoldSetArtifacts | None = None
    if raw_imported is not None:
        imported = _mapping(
            raw_imported, name="release imported fold-set artifacts"
        )
        _exact_keys(
            imported,
            _IMPORTED_FOLD_ARTIFACT_KEYS,
            name="release imported fold-set artifacts",
        )
        manifest_record = _mapping(
            imported["manifest"], name="release imported fold-set manifest"
        )
        _exact_keys(
            manifest_record,
            _FILE_RECORD_KEYS,
            name="release imported fold-set manifest",
        )
        manifest_relative = _relative_record_path(
            manifest_record["path"], name="release imported fold-set manifest path"
        )
        if manifest_relative in seen_files:
            raise ValueError("release index contains duplicate file paths")
        seen_files.add(manifest_relative)
        _, imported_manifest = _regular_file(
            root,
            Path(manifest_relative),
            name="release imported fold-set manifest",
        )
        manifest_bytes = manifest_record["bytes"]
        if (
            isinstance(manifest_bytes, bool)
            or not isinstance(manifest_bytes, int)
            or manifest_bytes <= 0
            or imported_manifest.stat().st_size != manifest_bytes
        ):
            raise ValueError("release imported fold-set manifest byte count mismatch")
        manifest_hash = _sha256(
            manifest_record["sha256"], name="release imported fold-set manifest"
        )
        if _sha256_file(imported_manifest) != manifest_hash:
            raise ValueError("release imported fold-set manifest SHA-256 mismatch")
        raw_partials = imported["partials"]
        if not isinstance(raw_partials, list) or len(raw_partials) != 3:
            raise ValueError("release imported fold partial coverage is incomplete")
        partials: list[tuple[int, Path]] = []
        for expected_fold, value in zip(range(3), raw_partials, strict=True):
            record = _mapping(value, name=f"release imported fold {expected_fold}")
            _exact_keys(
                record,
                _IMPORTED_PARTIAL_ARTIFACT_KEYS,
                name=f"release imported fold {expected_fold}",
            )
            if record["fold_id"] != expected_fold:
                raise ValueError("release imported fold order/identity mismatch")
            relative = _relative_record_path(
                record["path"], name=f"release imported fold {expected_fold} path"
            )
            if relative in seen_files:
                raise ValueError("release index contains duplicate file paths")
            seen_files.add(relative)
            _, partial = _regular_file(
                root, Path(relative), name=f"release imported fold {expected_fold}"
            )
            declared_bytes = record["bytes"]
            if (
                isinstance(declared_bytes, bool)
                or not isinstance(declared_bytes, int)
                or declared_bytes <= 0
                or partial.stat().st_size != declared_bytes
            ):
                raise ValueError(
                    f"release imported fold {expected_fold} byte count mismatch"
                )
            digest = _sha256(
                record["sha256"], name=f"release imported fold {expected_fold}"
            )
            if _sha256_file(partial) != digest:
                raise ValueError(
                    f"release imported fold {expected_fold} SHA-256 mismatch"
                )
            partials.append((expected_fold, partial))
        imported_fold_set = Stage2ImportedFoldSetArtifacts(
            manifest=imported_manifest, partials=tuple(partials)
        )

    raw_roots = _mapping(raw["roots"], name="release index roots")
    _exact_keys(raw_roots, set(_ROOT_BINDINGS), name="release index roots")
    roots: dict[str, Path] = {}
    seen_roots: set[str] = set()
    for role, expected_manifest in _ROOT_BINDINGS.items():
        record = _mapping(raw_roots[role], name=f"release root {role}")
        _exact_keys(record, _ROOT_RECORD_KEYS, name=f"release root {role}")
        relative = _relative_record_path(record["path"], name=f"root {role} path")
        if relative in seen_roots:
            raise ValueError("release index contains duplicate directory roots")
        seen_roots.add(relative)
        _, directory = _directory(root, Path(relative), name=role)
        manifest_role = record["manifest_artifact"]
        if manifest_role != expected_manifest:
            raise ValueError(f"release root manifest role mismatch: {role}")
        manifest_sha = _sha256(
            record["manifest_sha256"], name=f"release root {role} manifest"
        )
        if manifest_sha != hashes[expected_manifest]:
            raise ValueError(f"release root manifest SHA-256 mismatch: {role}")
        identity_payload = {
            "path": relative,
            "manifest_artifact": expected_manifest,
            "manifest_sha256": manifest_sha,
        }
        if _semantic_sha256(identity_payload) != _sha256(
            record["identity_sha256"], name=f"release root {role} identity"
        ):
            raise ValueError(f"release root semantic identity mismatch: {role}")
        roots[role] = directory

    identities = _mapping(raw["identities"], name="release index identities")
    _exact_keys(identities, _IDENTITY_KEYS, name="release index identities")
    expected_config = _sha256(
        identities["stage2_config_semantic_sha256"],
        name="Stage 2 config semantic identity",
    )
    expected_policy = _sha256(
        identities["condition_augmentation_policy_semantic_sha256"],
        name="condition policy semantic identity",
    )
    config_sha, policy_sha = _validate_content_lineage(
        files=files,
        roots=roots,
        hashes=hashes,
        checkpoints=checkpoints,
        imported_fold_set=imported_fold_set,
        require_original_config_paths=False,
    )
    if config_sha != expected_config or policy_sha != expected_policy:
        raise ValueError("release index semantic identities changed")
    return Stage2ReleaseIndex(
        index_path=selected,
        index_sha256=actual,
        semantic_sha256=declared_semantic,
        containment_root=root,
        paths=_paths_from_verified(
            files, roots, checkpoints, imported_fold_set
        ),
        artifact_sha256=tuple(sorted(hashes.items())),
        stage2_config_semantic_sha256=config_sha,
        condition_augmentation_policy_semantic_sha256=policy_sha,
        checkpoints=tuple(checkpoint_exports),
        imported_fold_set=imported_fold_set,
    )


def load_stage2_release_index(
    path: Path, *, expected_sha256: str, containment_root: Path
) -> Stage2ReleaseIndex:
    """Load and fully revalidate an exact Stage 2 release index."""

    return _load_verified(
        path, expected_sha256=expected_sha256, containment_root=containment_root
    )


def resolve_stage3_release_inputs(
    path: Path, *, expected_sha256: str, containment_root: Path
) -> Stage3CLIReleaseInputs:
    """Resolve verified explicit Stage 3 CLI inputs without path guessing."""

    return load_stage2_release_index(
        path, expected_sha256=expected_sha256, containment_root=containment_root
    ).stage3_cli_inputs()


__all__ = [
    "PATH_CONTRACT",
    "PROTOCOL_VERSION",
    "RELEASE_INDEX_FORMAT",
    "Stage2ImportedFoldSetArtifacts",
    "Stage2ImportedFoldSetPaths",
    "Stage2ReleaseCheckpoint",
    "Stage2ReleaseCheckpointPaths",
    "Stage2ReleaseIndex",
    "Stage2ReleasePaths",
    "Stage3CLIReleaseInputs",
    "load_stage2_release_index",
    "resolve_stage3_release_inputs",
    "write_stage2_release_index",
]

"""Portable Stage 2 path index for Stage 3 consumers.

The index is a convenience map from artifact roles to paths. Hashes and byte
counts are emitted as informational metadata only: loading an index never
compares them, refuses symlinks, enforces a containment boundary, or re-hashes
live files. Scientific role coverage and duplicate identities are still
checked because confusing fold/deployment checkpoints changes the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from kcorrdiff.data.condition_augmentation import ConditionAugmentationPolicy
from kcorrdiff.training.checkpoints import load_checkpoint_provenance


RELEASE_INDEX_FORMAT = "kcorrdiff.stage2-release-index.v2"
PROTOCOL_VERSION = "v1.1.3b"
PATH_CONTRACT = "posix-relative-to-explicit-containment-root.v2"

_REQUIRED_FILE_ROLES = (
    "stage2_config",
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
)
_OPTIONAL_PROVENANCE_FILE_ROLES = ("launch_identity", "runtime_report")
_FILE_ROLES = _REQUIRED_FILE_ROLES + _OPTIONAL_PROVENANCE_FILE_ROLES
_ROOT_BINDINGS = {
    "radar_cache_root": "radar_cache_manifest",
    "era5_cache_root": "era5_cache_manifest",
    "oof_artifact_root": "oof_artifact_manifest",
}
_REQUIRED_NONFOLD_CHECKPOINT_IDENTITIES = (("deployment", None),)
_PLACEHOLDER_SHA256 = "0" * 64


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_text(value: object, default: str = _PLACEHOLDER_SHA256) -> str:
    return value if isinstance(value, str) and value else default


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _read_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    return _mapping(value, name=name)


def _root(value: Path) -> Path:
    selected = Path(value).resolve()
    if not selected.is_dir():
        raise NotADirectoryError(selected)
    return selected


def _path_text(root: Path, value: Path) -> str:
    """Prefer relocatable relative paths, while allowing explicit outside paths."""

    selected = Path(value).resolve()
    try:
        return selected.relative_to(root).as_posix()
    except ValueError:
        return os.path.relpath(selected, root)


def _resolve_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} path must be a non-empty string")
    selected = Path(value)
    return selected.resolve() if selected.is_absolute() else (root / selected).resolve()


def _require_file(path: Path, *, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _require_directory(path: Path, *, name: str) -> Path:
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is missing: {path}")
    return path


@dataclass(frozen=True, slots=True)
class Stage2ReleaseCheckpointPaths:
    folds: tuple[tuple[int, Path], ...]
    deployment: Path
    direct_mean: Path | None = None
    direct_q50: Path | None = None

    def __post_init__(self) -> None:
        normalized = tuple(sorted((int(fold), Path(path)) for fold, path in self.folds))
        fold_ids = [fold for fold, _ in normalized]
        if not fold_ids or fold_ids != list(range(len(fold_ids))):
            raise ValueError("release fold paths must use contiguous non-negative IDs")
        object.__setattr__(self, "folds", normalized)
        object.__setattr__(self, "deployment", Path(self.deployment))
        for name in ("direct_mean", "direct_q50"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else Path(value))

    def by_identity(self) -> dict[tuple[str, int | None], Path]:
        result = {
            **{("fold", fold): path for fold, path in self.folds},
            ("deployment", None): self.deployment,
        }
        if self.direct_mean is not None:
            result[("direct_mean", None)] = self.direct_mean
        if self.direct_q50 is not None:
            result[("direct_q50", None)] = self.direct_q50
        return result

    def fold_path(self, fold_id: int) -> Path:
        return dict(self.folds)[fold_id]


@dataclass(frozen=True, slots=True)
class Stage2ImportedFoldSetPaths:
    manifest: Path
    partials: tuple[tuple[int, Path], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", Path(self.manifest))
        normalized = tuple(sorted((int(fold), Path(path)) for fold, path in self.partials))
        fold_ids = [fold for fold, _ in normalized]
        if not fold_ids or fold_ids != list(range(len(fold_ids))):
            raise ValueError("imported partials must use contiguous non-negative fold IDs")
        object.__setattr__(self, "partials", normalized)

    def partials_by_fold(self) -> dict[int, Path]:
        return dict(self.partials)


@dataclass(frozen=True, slots=True)
class Stage2ReleasePaths:
    stage2_config: Path
    launch_identity: Path | None
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
    runtime_report: Path | None
    radar_cache_root: Path
    era5_cache_root: Path
    oof_artifact_root: Path
    model_checkpoints: Stage2ReleaseCheckpointPaths
    imported_fold_set: Stage2ImportedFoldSetPaths | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"model_checkpoints", "imported_fold_set"}:
                continue
            value = getattr(self, field.name)
            if field.name in _OPTIONAL_PROVENANCE_FILE_ROLES and value is None:
                continue
            object.__setattr__(self, field.name, Path(value))
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
        """Confirm indexed paths still exist; do not verify their hashes."""

        load_stage2_release_index(
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
            stage2_manifest_sha256=hashes.get("stage2_manifest", _PLACEHOLDER_SHA256),
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


def _file_record(root: Path, path: Path, *, name: str) -> dict[str, object]:
    selected = _require_file(Path(path).resolve(), name=name)
    return {
        "path": _path_text(root, selected),
        "sha256": _sha256_file(selected),
        "bytes": selected.stat().st_size,
    }


def _policy_identity(path: Path) -> str:
    try:
        policy = ConditionAugmentationPolicy.from_mapping(
            _read_json(path, name="condition augmentation policy")
        )
    except (OSError, TypeError, ValueError, KeyError):
        return _sha256_file(path)
    return policy.semantic_sha256


def _build_payload(root: Path, paths: Stage2ReleasePaths) -> dict[str, object]:
    artifacts = {
        role: _file_record(root, getattr(paths, role), name=role)
        for role in _REQUIRED_FILE_ROLES
    }
    for role in _OPTIONAL_PROVENANCE_FILE_ROLES:
        path = getattr(paths, role)
        if path is not None:
            artifacts[role] = _file_record(root, path, name=role)
    checkpoint_records: list[dict[str, object]] = []
    checkpoint_paths = paths.model_checkpoints.by_identity()
    for role, fold_id in sorted(
        checkpoint_paths,
        key=lambda item: (item[0] != "fold", item[0], -1 if item[1] is None else item[1]),
    ):
        path = _require_file(
            checkpoint_paths[(role, fold_id)].resolve(),
            name=f"{role} checkpoint",
        )
        checkpoint_records.append(
            {
                "role": role,
                "fold_id": fold_id,
                "path": _path_text(root, path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    imported: dict[str, object] | None = None
    if paths.imported_fold_set is not None:
        imported = {
            "manifest": _file_record(
                root,
                paths.imported_fold_set.manifest,
                name="imported fold-set manifest",
            ),
            "partials": [
                {
                    "fold_id": fold_id,
                    **_file_record(
                        root, path, name=f"imported fold {fold_id} partial"
                    ),
                }
                for fold_id, path in paths.imported_fold_set.partials_by_fold().items()
            ],
        }
    roots: dict[str, dict[str, object]] = {}
    for role, manifest_role in _ROOT_BINDINGS.items():
        directory = _require_directory(Path(getattr(paths, role)).resolve(), name=role)
        roots[role] = {
            "path": _path_text(root, directory),
            "manifest_artifact": manifest_role,
            "manifest_sha256": artifacts[manifest_role]["sha256"],
            "identity_sha256": _PLACEHOLDER_SHA256,
        }
    payload: dict[str, object] = {
        "format_version": RELEASE_INDEX_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "stage2",
        "path_contract": PATH_CONTRACT,
        "artifacts": artifacts,
        "model_checkpoints": checkpoint_records,
        "imported_fold_set_artifacts": imported,
        "roots": roots,
        "identities": {
            "stage2_config_semantic_sha256": artifacts["stage2_config"]["sha256"],
            "condition_augmentation_policy_semantic_sha256": _policy_identity(
                Path(paths.condition_augmentation_policy).resolve()
            ),
        },
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def write_stage2_release_index(
    path: Path,
    *,
    containment_root: Path,
    release_paths: Stage2ReleasePaths,
) -> str:
    """Create a no-clobber path index and return its informational digest."""

    root = _root(containment_root)
    selected = Path(path)
    selected = selected.resolve() if selected.is_absolute() else (root / selected).resolve()
    if selected.exists():
        raise FileExistsError(selected)
    selected.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(_build_payload(root, release_paths))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        if selected.exists():
            raise FileExistsError(selected)
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _paths_from_loaded(
    files: Mapping[str, Path],
    roots: Mapping[str, Path],
    checkpoints: Mapping[tuple[str, int | None], Path],
    imported: Stage2ImportedFoldSetArtifacts | None,
) -> Stage2ReleasePaths:
    return Stage2ReleasePaths(
        **{role: files[role] for role in _REQUIRED_FILE_ROLES},
        **{role: files.get(role) for role in _OPTIONAL_PROVENANCE_FILE_ROLES},
        **{role: roots[role] for role in _ROOT_BINDINGS},
        model_checkpoints=Stage2ReleaseCheckpointPaths(
            folds=tuple(
                sorted(
                    (int(fold), path)
                    for (role, fold), path in checkpoints.items()
                    if role == "fold" and fold is not None
                )
            ),
            deployment=checkpoints[("deployment", None)],
            direct_mean=checkpoints.get(("direct_mean", None)),
            direct_q50=checkpoints.get(("direct_q50", None)),
        ),
        imported_fold_set=(
            None
            if imported is None
            else Stage2ImportedFoldSetPaths(
                manifest=imported.manifest,
                partials=imported.partials,
            )
        ),
    )


def load_stage2_release_index(
    path: Path,
    *,
    expected_sha256: str | None = None,
    containment_root: Path,
) -> Stage2ReleaseIndex:
    """Load a path index; expected hashes and containment are not gates."""

    del expected_sha256
    root = _root(containment_root)
    selected = Path(path)
    selected = selected.resolve() if selected.is_absolute() else (root / selected).resolve()
    _require_file(selected, name="release index")
    raw = _read_json(selected, name="Stage 2 release index")
    if raw.get("format_version") != RELEASE_INDEX_FORMAT or raw.get("stage") != "stage2":
        raise ValueError("unsupported Stage 2 release index format")

    raw_artifacts = _mapping(raw.get("artifacts"), name="release artifacts")
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in _REQUIRED_FILE_ROLES:
        record = _mapping(raw_artifacts.get(role), name=f"release artifact {role}")
        file_path = _resolve_path(root, record.get("path"), name=role)
        files[role] = _require_file(file_path, name=role)
        hashes[role] = _metadata_text(record.get("sha256"))
    for role in _OPTIONAL_PROVENANCE_FILE_ROLES:
        value = raw_artifacts.get(role)
        if value is None:
            continue
        record = _mapping(value, name=f"release artifact {role}")
        file_path = _resolve_path(root, record.get("path"), name=role)
        files[role] = _require_file(file_path, name=role)
        hashes[role] = _metadata_text(record.get("sha256"))

    raw_checkpoints = raw.get("model_checkpoints")
    if not isinstance(raw_checkpoints, list):
        raise TypeError("release model_checkpoints must be a list")
    checkpoints: dict[tuple[str, int | None], Path] = {}
    checkpoint_exports: list[Stage2ReleaseCheckpoint] = []
    for value in raw_checkpoints:
        record = _mapping(value, name="release checkpoint")
        identity = (str(record.get("role")), record.get("fold_id"))
        if identity in checkpoints:
            raise ValueError(f"duplicate release checkpoint identity: {identity}")
        checkpoint = _require_file(
            _resolve_path(root, record.get("path"), name=f"checkpoint {identity}"),
            name=f"checkpoint {identity}",
        )
        provenance = load_checkpoint_provenance(checkpoint)
        if provenance is not None and (provenance.role, provenance.fold_id) != identity:
            raise ValueError(
                f"checkpoint role/content mismatch: index={identity}, "
                f"checkpoint={(provenance.role, provenance.fold_id)}"
            )
        checkpoints[identity] = checkpoint
        checkpoint_exports.append(
            Stage2ReleaseCheckpoint(
                role=identity[0],
                fold_id=identity[1] if isinstance(identity[1], int) else None,
                path=checkpoint,
                sha256=_metadata_text(record.get("sha256")),
            )
        )
    fold_ids = sorted(
        int(fold)
        for role, fold in checkpoints
        if role == "fold" and fold is not None
    )
    if not fold_ids or fold_ids != list(range(len(fold_ids))):
        raise ValueError("release fold checkpoint IDs must be contiguous")
    missing = set(_REQUIRED_NONFOLD_CHECKPOINT_IDENTITIES) - set(checkpoints)
    if missing:
        raise ValueError(f"release checkpoint coverage is incomplete: {sorted(missing)}")

    raw_imported = raw.get("imported_fold_set_artifacts")
    imported: Stage2ImportedFoldSetArtifacts | None = None
    if raw_imported is not None:
        imported_mapping = _mapping(raw_imported, name="imported fold-set artifacts")
        manifest_record = _mapping(
            imported_mapping.get("manifest"), name="imported fold-set manifest"
        )
        manifest = _require_file(
            _resolve_path(root, manifest_record.get("path"), name="imported manifest"),
            name="imported manifest",
        )
        raw_partials = imported_mapping.get("partials")
        if not isinstance(raw_partials, list):
            raise TypeError("imported fold partials must be a list")
        partials: list[tuple[int, Path]] = []
        seen_folds: set[int] = set()
        for value in raw_partials:
            record = _mapping(value, name="imported fold partial")
            fold_id = record.get("fold_id")
            if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
                raise ValueError("imported fold_id must be a non-negative integer")
            if fold_id in seen_folds:
                raise ValueError("imported fold partials contain a duplicate fold")
            seen_folds.add(fold_id)
            partials.append(
                (
                    fold_id,
                    _require_file(
                        _resolve_path(
                            root, record.get("path"), name=f"fold {fold_id} partial"
                        ),
                        name=f"fold {fold_id} partial",
                    ),
                )
            )
        imported = Stage2ImportedFoldSetArtifacts(manifest, tuple(sorted(partials)))

    raw_roots = _mapping(raw.get("roots"), name="release roots")
    roots: dict[str, Path] = {}
    for role in _ROOT_BINDINGS:
        record = _mapping(raw_roots.get(role), name=f"release root {role}")
        roots[role] = _require_directory(
            _resolve_path(root, record.get("path"), name=role), name=role
        )

    identities = raw.get("identities")
    identities = identities if isinstance(identities, Mapping) else {}
    actual_index_sha = _sha256_file(selected)
    semantic = _metadata_text(raw.get("semantic_sha256"), actual_index_sha)
    config_identity = _metadata_text(
        identities.get("stage2_config_semantic_sha256"),
        hashes["stage2_config"],
    )
    policy_identity = _metadata_text(
        identities.get("condition_augmentation_policy_semantic_sha256"),
        hashes["condition_augmentation_policy"],
    )
    return Stage2ReleaseIndex(
        index_path=selected,
        index_sha256=actual_index_sha,
        semantic_sha256=semantic,
        containment_root=root,
        paths=_paths_from_loaded(files, roots, checkpoints, imported),
        artifact_sha256=tuple(sorted(hashes.items())),
        stage2_config_semantic_sha256=config_identity,
        condition_augmentation_policy_semantic_sha256=policy_identity,
        checkpoints=tuple(checkpoint_exports),
        imported_fold_set=imported,
    )


def resolve_stage3_release_inputs(
    path: Path,
    *,
    expected_sha256: str | None = None,
    containment_root: Path,
) -> Stage3CLIReleaseInputs:
    return load_stage2_release_index(
        path,
        expected_sha256=expected_sha256,
        containment_root=containment_root,
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

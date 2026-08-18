"""Atomically augment an existing Stage-1 training bundle with signatures.

This command consumes a base bundle, an explicit policy JSON, and a distinct
signature-purpose ID.  It publishes a new directory containing a
signature-expanded candidate manifest, the untouched base draw manifest, the
materialized augmented draw manifest, and bundle metadata with recorded
lineage.  The source directory is never modified and an existing output is
never replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from .condition_augmentation import (
    ConditionAugmentationPolicy,
    canonical_draw_rows_sha256,
    expand_manifest_condition_support,
    materialize_condition_signatures,
)
from .normalization import sha256_file
from .sampling import (
    CandidateItem,
    enforce_byte_budget,
    full_coverage_shuffled_draw_manifest,
    oof_dense_byte_budget,
    read_draw_manifest,
    write_draw_manifest,
)
from .split_manifest import (
    ManifestItem,
    SplitInterval,
    canonical_sample_id,
    parse_utc,
    write_manifest,
)


AUGMENTED_BUNDLE_FORMAT_VERSION = "kcorrdiff.condition-augmented-bundle.v1"
BundleStage = Literal["pilot", "production"]
DrawDerivationPolicy = Literal[
    "preserve-source",
    "full-coverage-shuffled",
]


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name=name)


def _resolve_artifact(
    source_dir: Path,
    bundle: Mapping[str, object],
    *,
    name: str,
) -> tuple[Path, Mapping[str, object]]:
    artifacts = _mapping(bundle.get("artifacts"), name="source bundle artifacts")
    record = _mapping(artifacts.get(name), name=f"source artifact {name}")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"source artifact {name} has no relative path")
    path = source_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"source artifact {name} is missing")
    return path, record


def _stream_candidate_items(
    path: Path,
) -> tuple[Iterable[ManifestItem], Callable[[], Mapping[str, object]]]:
    """Decode canonical streamed candidate JSON without loading its item list."""

    stream = path.open("r", encoding="utf-8")
    prefix = stream.read(len('{"items":['))
    if prefix != '{"items":[':
        stream.close()
        raise ValueError("candidate manifest is not canonical streamed JSON")
    decoder = json.JSONDecoder()
    buffer = ""
    offset = 0
    tail: dict[str, Mapping[str, object]] = {}

    def fill() -> bool:
        nonlocal buffer, offset
        if offset:
            buffer = buffer[offset:]
            offset = 0
        chunk = stream.read(1024 * 1024)
        buffer += chunk
        return bool(chunk)

    def items() -> Iterable[ManifestItem]:
        nonlocal buffer, offset
        try:
            fill()
            first = True
            while True:
                while True:
                    while offset < len(buffer) and buffer[offset].isspace():
                        offset += 1
                    if offset < len(buffer):
                        break
                    if not fill():
                        raise ValueError("candidate manifest ended inside items")
                if buffer[offset] == "]":
                    offset += 1
                    suffix = buffer[offset:] + stream.read()
                    raw_tail = json.loads('{"items":[]' + suffix)
                    tail["value"] = _mapping(
                        raw_tail, name="candidate manifest"
                    )
                    return
                if first:
                    first = False
                else:
                    if buffer[offset] != ",":
                        raise ValueError(
                            "candidate manifest item separator is invalid"
                        )
                    offset += 1
                while True:
                    try:
                        value, next_offset = decoder.raw_decode(buffer, offset)
                    except json.JSONDecodeError:
                        if not fill():
                            raise ValueError(
                                "candidate manifest has a truncated item"
                            ) from None
                        continue
                    offset = next_offset
                    yield ManifestItem(
                        **dict(_mapping(value, name="candidate item"))
                    )
                    break
        finally:
            stream.close()

    iterator = items()

    def read_tail() -> Mapping[str, object]:
        if "value" not in tail:
            raise RuntimeError(
                "candidate iterator must be exhausted before metadata"
            )
        raw = tail["value"]
        if raw.get("version") != "kcorrdiff.split-manifest.v1":
            raise ValueError("source candidate manifest version mismatch")
        return _mapping(raw.get("metadata"), name="candidate metadata")

    return iterator, read_tail


def _artifact_record(
    path: Path, root: Path, *, rows: int | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _semantic_sha256(values: Iterable[object]) -> str:
    """Match the streamed event-bundle semantic hash exactly."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(
                value,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _split_intervals(
    metadata: Mapping[str, object],
) -> tuple[SplitInterval, ...]:
    raw = metadata.get("split_intervals")
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidate metadata has no split intervals")
    result: list[SplitInterval] = []
    for index, value in enumerate(raw):
        item = _mapping(value, name=f"split interval {index}")
        result.append(
            SplitInterval(
                name=str(item.get("name", "")),  # type: ignore[arg-type]
                start=parse_utc(str(item.get("start", ""))),
                end=parse_utc(str(item.get("end", ""))),
            )
        )
    return tuple(result)


def _base_candidate(item: ManifestItem) -> CandidateItem:
    return CandidateItem(
        sample_id=item.sample_id,
        t0_utc=item.t0_utc,
        lead_hours=item.lead_hours,
        condition_signature=item.condition_signature,
        block_id=item.block_id,
        stratum=item.stratum,
        split=item.split,
        p_target=item.p_target,
        fold_id=item.fold_id,
    )


def _digest_semantic(digest: "hashlib._Hash", value: object) -> None:
    digest.update(
        json.dumps(
            value,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


class _BaseCandidateScan:
    """Small scan result; only production's 431,400 sampler rows stay resident."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, object],
        outer_candidates: Sequence[CandidateItem],
        source_rows: int,
        expanded_rows: int,
        outer_base_rows: int,
        expanded_candidate_semantic_sha256: str,
        expanded_eligible_sha256: str,
        candidate_audit: Mapping[str, object],
    ) -> None:
        self.metadata = metadata
        self.outer_candidates = tuple(outer_candidates)
        self.source_rows = source_rows
        self.expanded_rows = expanded_rows
        self.outer_base_rows = outer_base_rows
        self.expanded_candidate_semantic_sha256 = (
            expanded_candidate_semantic_sha256
        )
        self.expanded_eligible_sha256 = expanded_eligible_sha256
        self.candidate_audit = candidate_audit


def _expanded_items_for_source(
    item: ManifestItem,
    *,
    policy: ConditionAugmentationPolicy,
    draw_policy: DrawDerivationPolicy,
) -> tuple[ManifestItem, ...]:
    if item.condition_signature != policy.source_signature.key:
        raise ValueError(
            "source candidate signatures disagree with condition policy"
        )
    expected_id = canonical_sample_id(
        parse_utc(item.t0_utc), item.lead_hours, item.condition_signature
    )
    if item.sample_id != expected_id:
        raise ValueError("source candidate has a non-canonical sample_id")
    derived = item
    if draw_policy == "full-coverage-shuffled":
        derived = replace(item, p_draw=item.p_target, omega=1.0)
    if derived.split == "outer_train":
        return expand_manifest_condition_support((derived,), policy=policy)
    if derived.fold_id is not None:
        raise ValueError("source holdout candidate unexpectedly has a fold")
    return (derived,)


def _scan_candidate_manifest(
    path: Path,
    *,
    policy: ConditionAugmentationPolicy,
    draw_policy: DrawDerivationPolicy,
    retain_outer_candidates: bool,
) -> _BaseCandidateScan:
    items, read_tail = _stream_candidate_items(path)
    outer: list[CandidateItem] = []
    source_rows = 0
    expanded_rows = 0
    outer_base_rows = 0
    semantic = hashlib.sha256()
    eligible = hashlib.sha256()
    split_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    blocks: set[str] = set()
    target_sums: dict[str, float] = {}
    draw_sums: dict[str, float] = {}
    prior_key: tuple[str, float, str] | None = None
    for item in items:
        source_rows += 1
        key = (item.t0_utc, item.lead_hours, item.condition_signature)
        if prior_key is not None and key <= prior_key:
            raise ValueError("source candidate rows are not canonically ordered")
        prior_key = key
        expanded = _expanded_items_for_source(
            item, policy=policy, draw_policy=draw_policy
        )
        if item.split == "outer_train":
            outer_base_rows += 1
            if retain_outer_candidates:
                outer.append(_base_candidate(item))
        for output_item in expanded:
            expanded_rows += 1
            _digest_semantic(semantic, asdict(output_item))
            _digest_semantic(eligible, output_item.sample_id)
            split_counts[output_item.split] = (
                split_counts.get(output_item.split, 0) + 1
            )
            stratum_counts[output_item.stratum] = (
                stratum_counts.get(output_item.stratum, 0) + 1
            )
            blocks.add(output_item.block_id)
            target_sums[output_item.split] = target_sums.get(
                output_item.split, 0.0
            ) + output_item.p_target
            draw_sums[output_item.split] = draw_sums.get(
                output_item.split, 0.0
            ) + output_item.p_draw
            if not math.isclose(
                output_item.omega,
                output_item.p_target / output_item.p_draw,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("expanded candidate importance weight changed")
    metadata = read_tail()
    _split_intervals(metadata)
    if source_rows == 0 or outer_base_rows == 0:
        raise ValueError("source candidate manifest has no outer_train rows")
    for split in split_counts:
        if not math.isclose(
            target_sums[split], 1.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(f"expanded target probabilities drifted in {split}")
        if not math.isclose(
            draw_sums[split], 1.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(f"expanded draw probabilities drifted in {split}")
    audit = {
        "items": expanded_rows,
        "unique_blocks": len(blocks),
        "split_counts": split_counts,
        "stratum_counts": stratum_counts,
        "unassigned_eligible_item_fraction": 0.0,
    }
    return _BaseCandidateScan(
        metadata=metadata,
        outer_candidates=outer,
        source_rows=source_rows,
        expanded_rows=expanded_rows,
        outer_base_rows=outer_base_rows,
        expanded_candidate_semantic_sha256=semantic.hexdigest(),
        expanded_eligible_sha256=eligible.hexdigest(),
        candidate_audit=audit,
    )


def _expanded_item_stream(
    path: Path,
    *,
    policy: ConditionAugmentationPolicy,
    draw_policy: DrawDerivationPolicy,
) -> Iterable[ManifestItem]:
    items, read_tail = _stream_candidate_items(path)
    for item in items:
        yield from _expanded_items_for_source(
            item, policy=policy, draw_policy=draw_policy
        )
    # Force validation of the suffix/version after iterator exhaustion.
    read_tail()


def _copy_source_artifact(
    *,
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"artifact output path collision: {destination}")
    shutil.copy2(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def augment_condition_bundle(
    *,
    source_bundle_dir: Path,
    output_dir: Path,
    policy_path: Path,
    stage: BundleStage,
    draw_policy: DrawDerivationPolicy,
    training_seed: int,
    source_draw_purpose_id: str,
    signature_purpose_id: str,
    expected_base_candidate_count: int,
    planned_epochs: int,
    maximum_oof_bytes: int | None = None,
) -> str:
    """Publish and return the new bundle-metadata SHA-256."""

    source_dir = source_bundle_dir.resolve()
    output = output_dir.resolve()
    policy_file = policy_path.resolve()
    if output.exists():
        raise FileExistsError(output)
    if source_dir == output:
        raise ValueError("source and augmented bundle directories must differ")
    if not source_dir.is_dir():
        raise FileNotFoundError("source bundle directory is missing")
    if not policy_file.is_file():
        raise FileNotFoundError("condition policy JSON is missing")
    if stage not in {"pilot", "production"}:
        raise ValueError("stage must be explicitly pilot or production")
    if draw_policy not in {
        "preserve-source",
        "full-coverage-shuffled",
    }:
        raise ValueError("unsupported explicit draw derivation policy")
    training_seed = _nonnegative_integer(training_seed, name="training_seed")
    expected_base_candidate_count = _positive_integer(
        expected_base_candidate_count,
        name="expected_base_candidate_count",
    )
    planned_epochs = _positive_integer(planned_epochs, name="planned_epochs")
    if (
        not isinstance(source_draw_purpose_id, str)
        or not source_draw_purpose_id
    ):
        raise ValueError("source_draw_purpose_id must be explicit and non-empty")
    if not isinstance(signature_purpose_id, str) or not signature_purpose_id:
        raise ValueError("signature_purpose_id must be explicit and non-empty")

    metadata_path = source_dir / "bundle-metadata.json"
    source_bundle = _read_json_mapping(metadata_path, name="source bundle metadata")
    source_metadata_hash = sha256_file(metadata_path)
    if source_bundle.get("condition_augmentation") is not None:
        raise ValueError("source bundle is already condition-augmented")
    protocol = str(source_bundle.get("protocol_version", ""))
    policy = ConditionAugmentationPolicy.from_mapping(
        _read_json_mapping(policy_file, name="condition policy")
    )
    if policy.protocol_version != protocol:
        raise ValueError("condition policy/source bundle protocol mismatch")
    if stage == "pilot":
        if draw_policy != "preserve-source":
            raise ValueError("pilot stage must preserve the source bounded draw")
    elif draw_policy != "full-coverage-shuffled":
        raise ValueError("production stage uses the full-coverage-shuffled draw")

    candidate_path, candidate_record = _resolve_artifact(
        source_dir, source_bundle, name="candidate_manifest"
    )
    draw_path, draw_record = _resolve_artifact(
        source_dir, source_bundle, name="training_draw_manifest"
    )
    if tuple(source_bundle.get("condition_signatures", ())) != (
        policy.source_signature.key,
    ):
        raise ValueError("source bundle signature disagrees with condition policy")
    sampling = _mapping(source_bundle.get("sampling"), name="source sampling")
    input_source_purpose = sampling.get("purpose_id")
    if not isinstance(input_source_purpose, str) or not input_source_purpose:
        raise ValueError("source bundle has no draw purpose_id")
    input_seed = _nonnegative_integer(sampling.get("seed"), name="source seed")
    if signature_purpose_id == source_draw_purpose_id:
        raise ValueError("signature purpose must differ from source draw purpose")
    scan = _scan_candidate_manifest(
        candidate_path,
        policy=policy,
        draw_policy=draw_policy,
        retain_outer_candidates=(draw_policy == "full-coverage-shuffled"),
    )
    candidate_metadata = scan.metadata
    if scan.outer_base_rows != expected_base_candidate_count:
        raise ValueError(
            "outer_train base candidate count mismatch: "
            f"{scan.outer_base_rows} != {expected_base_candidate_count}"
        )

    if draw_policy == "preserve-source":
        if training_seed != input_seed:
            raise ValueError("explicit pilot seed disagrees with source bundle")
        if source_draw_purpose_id != input_source_purpose:
            raise ValueError(
                "explicit pilot draw purpose disagrees with source bundle"
            )
        source_rows = read_draw_manifest(draw_path)
        if draw_record.get("rows") != len(source_rows):
            raise ValueError("source draw row count mismatch")
        source_draw_origin = "preserved_input_bundle_draw"
    else:
        source_rows = full_coverage_shuffled_draw_manifest(
            scan.outer_candidates,
            seed=training_seed,
            purpose_id=source_draw_purpose_id,
        )
        if len(source_rows) != scan.outer_base_rows:
            raise AssertionError("full-coverage source draw lost candidates")
        source_draw_origin = "derived_from_complete_outer_train_candidate_support"

    result = materialize_condition_signatures(
        source_rows,
        policy=policy,
        training_seed=training_seed,
        source_draw_purpose_id=source_draw_purpose_id,
        signature_purpose_id=signature_purpose_id,
        source_draw_manifest_sha256=canonical_draw_rows_sha256(source_rows),
    )
    required_oof_bytes = oof_dense_byte_budget(result.rows)
    enforce_byte_budget(required_oof_bytes, maximum_oof_bytes)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".incomplete",
            dir=output.parent,
        )
    )
    try:
        policy_output = temporary / "condition-policy.json"
        policy_payload = (
            json.dumps(policy.to_json(), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _write_fsynced(policy_output, policy_payload)
        candidate_output = temporary / "candidate-manifest.json"
        augmented_candidate_metadata = {
            **dict(candidate_metadata),
            "draw_policy": (
                str(sampling.get("draw_policy"))
                if draw_policy == "preserve-source"
                else "full-coverage-shuffled"
            ),
            "condition_augmentation_policy_sha256": policy.semantic_sha256,
            "source_candidate_manifest_sha256": sha256_file(candidate_path),
            "eligible_universe_sample_ids_sha256": (
                scan.expanded_eligible_sha256
            ),
            "candidate_semantic_sha256": (
                scan.expanded_candidate_semantic_sha256
            ),
        }
        write_manifest(
            candidate_output,
            _expanded_item_stream(
                candidate_path,
                policy=policy,
                draw_policy=draw_policy,
            ),  # type: ignore[arg-type]
            metadata=augmented_candidate_metadata,
        )
        source_draw_output = temporary / "source-training-draw-manifest.jsonl"
        write_draw_manifest(source_draw_output, source_rows)
        augmented_draw_output = temporary / "training-draw-manifest.jsonl"
        write_draw_manifest(augmented_draw_output, result.rows)

        artifacts = dict(
            _mapping(source_bundle.get("artifacts"), name="source artifacts")
        )
        artifacts["candidate_manifest"] = _artifact_record(
            candidate_output, temporary, rows=scan.expanded_rows
        )
        artifacts["source_training_draw_manifest"] = _artifact_record(
            source_draw_output, temporary, rows=len(source_rows)
        ) | {"purpose_id": source_draw_purpose_id}
        artifacts["training_draw_manifest"] = _artifact_record(
            augmented_draw_output, temporary, rows=len(result.rows)
        )
        artifacts["condition_policy"] = _artifact_record(
            policy_output, temporary, rows=len(policy.entries)
        )
        reserved_paths = {
            candidate_output.relative_to(temporary),
            source_draw_output.relative_to(temporary),
            augmented_draw_output.relative_to(temporary),
            policy_output.relative_to(temporary),
        }
        for name in tuple(artifacts):
            if name in {
                "candidate_manifest",
                "source_training_draw_manifest",
                "training_draw_manifest",
                "condition_policy",
            }:
                continue
            source_artifact, source_record = _resolve_artifact(
                source_dir, source_bundle, name=name
            )
            relative = Path(str(source_record["path"]))
            if relative in reserved_paths:
                raise ValueError(f"source artifact path collision: {relative}")
            destination = temporary / relative
            _copy_source_artifact(
                source=source_artifact,
                destination=destination,
            )
            copied = dict(source_record)
            copied.update(_artifact_record(destination, temporary))
            artifacts[name] = copied
        if draw_policy == "full-coverage-shuffled":
            input_draw_output = temporary / "input-training-draw-manifest.jsonl"
            _copy_source_artifact(source=draw_path, destination=input_draw_output)
            artifacts["input_training_draw_manifest"] = _artifact_record(
                input_draw_output,
                temporary,
                rows=_positive_integer(
                    draw_record.get("rows"), name="input draw rows"
                ),
            ) | {"purpose_id": input_source_purpose}

        weights = np.asarray(
            [row.omega for row in result.rows], dtype=np.float64
        )
        weight_sum = float(weights.sum())
        augmented_sampling: dict[str, object] = {
            "stage": stage,
            "planned_epochs": planned_epochs,
            "draw_split": "outer_train",
            "target_policy": "eligible_items_uniform",
            "probability_scope": "within_split",
            "draw_policy": (
                str(sampling.get("draw_policy"))
                if draw_policy == "preserve-source"
                else "full-coverage-shuffled"
            ),
            "draw_count": len(source_rows),
            "seed": training_seed,
            "purpose_id": source_draw_purpose_id,
            "weight_clipping": result.provenance.weight_clipping,
            "omega_min": float(np.min(weights)),
            "omega_median": float(np.median(weights)),
            "omega_p95": float(np.percentile(weights, 95)),
            "omega_max": float(np.max(weights)),
            "weighted_sample_ess": float(
                weight_sum**2 / np.square(weights).sum()
            ),
            "condition_selection_order": (
                "base_sample_then_one_condition_signature"
            ),
            "signature_purpose_id": signature_purpose_id,
            "source_draw_origin": source_draw_origin,
        }
        if draw_policy == "full-coverage-shuffled":
            augmented_sampling.update(
                {
                    "coverage": "all_outer_train_base_candidates_once",
                    "replacement": False,
                    "base_candidate_count": scan.outer_base_rows,
                    "shuffle_key": "sha256(seed,purpose_id,sample_id)",
                }
            )
        metadata = {
            **dict(source_bundle),
            "derivation_format_version": AUGMENTED_BUNDLE_FORMAT_VERSION,
            "source_bundle_metadata_sha256": source_metadata_hash,
            "source_bundle_path": str(source_dir),
            "source_candidate_manifest_sha256": sha256_file(candidate_path),
            "source_training_draw_manifest_sha256": sha256_file(draw_path),
            "source_sampling": dict(sampling),
            "condition_signatures": list(result.provenance.supported_signatures),
            "condition_augmentation_policy_sha256": policy.semantic_sha256,
            "condition_policy_file_sha256": sha256_file(policy_file),
            "condition_augmentation_provenance_sha256": (
                result.provenance.semantic_sha256
            ),
            "condition_augmentation": result.provenance.to_json(),
            "eligible_universe_sample_ids_sha256": (
                scan.expanded_eligible_sha256
            ),
            "candidate_semantic_sha256": (
                scan.expanded_candidate_semantic_sha256
            ),
            "candidate_audit": dict(scan.candidate_audit),
            "sampling": augmented_sampling,
            "oof_preflight": {
                "unique_semantic_items": result.provenance.unique_oof_items,
                "height": 256,
                "width": 256,
                "fields": 2,
                "bytes_per_value": 4,
                "required_bytes": required_oof_bytes,
                "maximum_bytes": maximum_oof_bytes,
            },
            "artifacts": artifacts,
        }
        metadata_output = temporary / "bundle-metadata.json"
        metadata_payload = (
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _write_fsynced(metadata_output, metadata_payload)
        metadata_hash = hashlib.sha256(metadata_payload).hexdigest()
        _write_fsynced(
            temporary / "bundle-metadata.sha256",
            f"{metadata_hash}  bundle-metadata.json\n".encode("ascii"),
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return metadata_hash
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("pilot", "production"), required=True
    )
    parser.add_argument(
        "--draw-policy",
        choices=("preserve-source", "full-coverage-shuffled"),
        required=True,
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--source-draw-purpose-id", required=True)
    parser.add_argument("--signature-purpose-id", required=True)
    parser.add_argument(
        "--expected-base-candidate-count", type=int, required=True
    )
    parser.add_argument("--planned-epochs", type=int, required=True)
    parser.add_argument(
        "--maximum-oof-bytes",
        type=int,
        default=None,
        help="optional advisory dense OOF byte budget",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    digest = augment_condition_bundle(
        source_bundle_dir=arguments.source_bundle_dir,
        output_dir=arguments.output_dir,
        policy_path=arguments.policy_json,
        stage=arguments.stage,
        draw_policy=arguments.draw_policy,
        training_seed=arguments.training_seed,
        source_draw_purpose_id=arguments.source_draw_purpose_id,
        signature_purpose_id=arguments.signature_purpose_id,
        expected_base_candidate_count=(
            arguments.expected_base_candidate_count
        ),
        planned_epochs=arguments.planned_epochs,
        maximum_oof_bytes=arguments.maximum_oof_bytes,
    )
    print(json.dumps({"bundle_metadata_sha256": digest}, sort_keys=True))
    return 0


__all__ = [
    "AUGMENTED_BUNDLE_FORMAT_VERSION",
    "augment_condition_bundle",
    "main",
    "parse_args",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())

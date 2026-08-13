"""Frozen condition-signature augmentation for Stage-2 training draws.

The v1.1.3b protocol treats ``(t0, lead, condition_signature)`` as the
training item.  A base ``(t0, lead)`` draw is therefore selected first and one
configured condition signature is selected second.  This module materializes
that second selection before training; datasets and trainers must consume the
resulting row and must never redraw a dropout mask at runtime.

Only identity, split, and declared sampling probabilities enter this module.
There is deliberately no target/label argument.  Input normalization is also
orthogonal to dropout: statistics are fit once on deduplicated physical
outer-train provider values, before a source or channel is replaced by its
learned null state.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re
from typing import Literal, Mapping, Sequence, cast

from .condition_schema import parse_condition_signature
from .provider_adapter import ConditionSignature
from .sampling import CandidateItem, DrawRow, canonical_counter
from .split_manifest import ManifestItem, canonical_sample_id, parse_utc


PROTOCOL_VERSION = "v1.1.3b"
POLICY_FORMAT_VERSION = "kcorrdiff.condition-augmentation-policy.v1"
PROVENANCE_FORMAT_VERSION = "kcorrdiff.condition-augmentation-provenance.v1"
NORMALIZATION_CONTRACT = (
    "outer_train_physical_provider_values_before_condition_dropout.v1"
)
NULL_CONDITION_SIGNATURE = "null_provider:era=0:tp=0:no_era_access"
TemporalAccessStrategy = Literal[
    "matched_arm",
    "full_with_target_end_causal_augmentation",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_draw_rows_sha256(rows: Sequence[DrawRow]) -> str:
    """Hash rows exactly as :func:`sampling.write_draw_manifest` writes them."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(asdict(row), sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _positive_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


@dataclass(frozen=True, slots=True)
class ConditionMixtureEntry:
    """One explicitly configured conditional signature probability.

    ``target_probability`` is ``P_target(a | t0, tau)`` and
    ``draw_probability`` is ``P_draw(a | t0, tau)``.  Neither has a default:
    v1.1.3b leaves the dropout probabilities to preregistration.
    """

    condition_signature: str | ConditionSignature
    target_probability: float
    draw_probability: float

    def __post_init__(self) -> None:
        signature = parse_condition_signature(self.condition_signature)
        object.__setattr__(self, "condition_signature", signature.key)
        object.__setattr__(
            self,
            "target_probability",
            _positive_probability(
                self.target_probability, name="conditional target probability"
            ),
        )
        object.__setattr__(
            self,
            "draw_probability",
            _positive_probability(
                self.draw_probability, name="conditional draw probability"
            ),
        )

    @property
    def signature(self) -> ConditionSignature:
        return parse_condition_signature(self.condition_signature)

    def to_json(self) -> dict[str, str | float]:
        return {
            "condition_signature": str(self.condition_signature),
            "target_probability": self.target_probability,
            "draw_probability": self.draw_probability,
        }


_NULL_ONLY_ENTRY = ConditionMixtureEntry(
    condition_signature=NULL_CONDITION_SIGNATURE,
    target_probability=1.0,
    draw_probability=1.0,
)


@dataclass(frozen=True, slots=True)
class ConditionAugmentationPolicy:
    """Strict, explicit whole-ERA/tp/access augmentation contract.

    Every ERA-capable policy must include the original ERA+tp signature, a
    whole-source-null signature, and at least one ERA-present/tp-off
    signature.  Temporal access is either one matched official arm or an
    explicitly named full-trajectory policy that also trains causal samples.
    No provider switch is allowed through dropout.
    """

    policy_id: str
    protocol_version: str
    source_condition_signature: str | ConditionSignature
    temporal_access_strategy: TemporalAccessStrategy
    entries: tuple[ConditionMixtureEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("condition augmentation policy_id must be non-empty")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"condition augmentation requires protocol {PROTOCOL_VERSION}"
            )
        if self.temporal_access_strategy not in {
            "matched_arm",
            "full_with_target_end_causal_augmentation",
        }:
            raise ValueError("unsupported temporal-access augmentation strategy")

        source = parse_condition_signature(self.source_condition_signature)
        if not source.era_present or not source.tp_present:
            raise ValueError(
                "source condition signature must have ERA and tp present"
            )
        if source.provider_track == "null_provider":
            raise ValueError("ERA-present source cannot use null_provider")
        object.__setattr__(self, "source_condition_signature", source.key)

        raw_entries = tuple(self.entries)
        if not raw_entries:
            raise ValueError("condition augmentation entries cannot be empty")
        if any(not isinstance(entry, ConditionMixtureEntry) for entry in raw_entries):
            raise TypeError("entries must contain ConditionMixtureEntry values")
        entries = tuple(
            sorted(raw_entries, key=lambda entry: str(entry.condition_signature))
        )
        keys = tuple(str(entry.condition_signature) for entry in entries)
        if len(keys) != len(set(keys)):
            raise ValueError("condition augmentation signatures must be unique")
        object.__setattr__(self, "entries", entries)

        for name, total in (
            (
                "conditional target probabilities",
                math.fsum(entry.target_probability for entry in entries),
            ),
            (
                "conditional draw probabilities",
                math.fsum(entry.draw_probability for entry in entries),
            ),
        ):
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"{name} must sum to one")

        if source.key not in keys:
            raise ValueError("policy must include its ERA+tp source signature")
        if NULL_CONDITION_SIGNATURE not in keys:
            raise ValueError("policy must include whole-ERA source dropout")

        era_entries = tuple(
            entry.signature for entry in entries if entry.signature.era_present
        )
        if any(
            signature.provider_track != source.provider_track
            for signature in era_entries
        ):
            raise ValueError("condition dropout cannot switch provider tracks")
        if not any(not signature.tp_present for signature in era_entries):
            raise ValueError("policy must include a separate tp-channel dropout state")

        access_modes = {signature.temporal_access_mode for signature in era_entries}
        for access_mode in access_modes:
            tp_states = {
                signature.tp_present
                for signature in era_entries
                if signature.temporal_access_mode == access_mode
            }
            if tp_states != {False, True}:
                raise ValueError(
                    "each configured ERA temporal-access mode must include "
                    "both tp-present and tp-dropout signatures"
                )
        if self.temporal_access_strategy == "matched_arm":
            if access_modes != {source.temporal_access_mode}:
                raise ValueError(
                    "matched_arm cannot mix temporal-access contracts"
                )
        else:
            if source.temporal_access_mode != "full_trajectory":
                raise ValueError(
                    "causal augmentation requires a full-trajectory source arm"
                )
            if access_modes != {"full_trajectory", "target_end_causal"}:
                raise ValueError(
                    "causal augmentation must explicitly include full and "
                    "target-end-causal signatures"
                )

    @property
    def source_signature(self) -> ConditionSignature:
        return parse_condition_signature(self.source_condition_signature)

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        return {
            "format_version": POLICY_FORMAT_VERSION,
            "policy_id": self.policy_id,
            "protocol_version": self.protocol_version,
            "source_condition_signature": str(self.source_condition_signature),
            "temporal_access_strategy": self.temporal_access_strategy,
            "entries": [entry.to_json() for entry in self.entries],
            "selection_order": "base_sample_then_one_condition_signature",
            "weight_clipping": None,
        }

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, object]
    ) -> "ConditionAugmentationPolicy":
        """Parse an immutable policy artifact without supplying defaults."""

        required = {
            "format_version",
            "policy_id",
            "protocol_version",
            "source_condition_signature",
            "temporal_access_strategy",
            "entries",
            "selection_order",
            "weight_clipping",
        }
        if set(raw) != required:
            missing = sorted(required - set(raw))
            unknown = sorted(set(raw) - required)
            raise ValueError(
                "condition policy keys mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        if raw["format_version"] != POLICY_FORMAT_VERSION:
            raise ValueError("unsupported condition augmentation policy format")
        if raw["selection_order"] != (
            "base_sample_then_one_condition_signature"
        ):
            raise ValueError("condition policy selection order changed")
        if raw["weight_clipping"] is not None:
            raise ValueError("v1.1.3b condition importance weights are unclipped")
        for key in (
            "policy_id",
            "protocol_version",
            "source_condition_signature",
            "temporal_access_strategy",
        ):
            if not isinstance(raw[key], str):
                raise TypeError(f"condition policy {key} must be a string")
        raw_entries = raw["entries"]
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes)
        ):
            raise TypeError("condition policy entries must be a sequence")
        entries: list[ConditionMixtureEntry] = []
        required_entry = {
            "condition_signature",
            "target_probability",
            "draw_probability",
        }
        for index, value in enumerate(raw_entries):
            if not isinstance(value, Mapping):
                raise TypeError(f"condition policy entry {index} must be a mapping")
            if set(value) != required_entry:
                raise ValueError(
                    f"condition policy entry {index} keys mismatch"
                )
            if not isinstance(value["condition_signature"], str):
                raise TypeError(
                    f"condition policy entry {index} signature must be a string"
                )
            entries.append(
                ConditionMixtureEntry(
                    condition_signature=cast(
                        str, value["condition_signature"]
                    ),
                    target_probability=cast(
                        float, value["target_probability"]
                    ),
                    draw_probability=cast(float, value["draw_probability"]),
                )
            )
        return cls(
            policy_id=cast(str, raw["policy_id"]),
            protocol_version=cast(str, raw["protocol_version"]),
            source_condition_signature=cast(
                str, raw["source_condition_signature"]
            ),
            temporal_access_strategy=cast(
                TemporalAccessStrategy, raw["temporal_access_strategy"]
            ),
            entries=tuple(entries),
        )


@dataclass(frozen=True, slots=True)
class ConditionSupportExpansion:
    """One supported signature for a base candidate, not a flat draw item."""

    source_sample_id: str
    sample_id: str
    t0_utc: str
    lead_hours: float
    source_condition_signature: str
    condition_signature: str
    block_id: str
    stratum: str
    split: str
    fold_id: int
    base_target_probability: float
    full_target_probability: float
    conditional_target_probability: float
    conditional_draw_probability: float


@dataclass(frozen=True, slots=True)
class OOFConditionRequirement:
    """Canonical Stage-2 OOF identity implied by an augmented draw manifest."""

    sample_id: str
    t0_utc: str
    lead_hours: float
    condition_signature: str
    block_id: str
    fold_id: int

    @property
    def semantic_key(self) -> tuple[str, float, str]:
        return (self.t0_utc, self.lead_hours, self.condition_signature)


@dataclass(frozen=True, slots=True)
class ConditionAugmentationProvenance:
    format_version: str
    protocol_version: str
    policy_id: str
    policy_sha256: str
    source_draw_manifest_sha256: str
    augmented_draw_manifest_sha256: str
    oof_requirements_sha256: str
    training_seed: int
    source_draw_purpose_id: str
    signature_purpose_id: str
    draws: int
    unique_oof_items: int
    supported_signatures: tuple[str, ...]
    signature_counts: tuple[tuple[str, int], ...]
    fit_split: str = "outer_train"
    selection_order: str = "base_sample_then_one_condition_signature"
    canonical_counter_key: str = "training_seed,purpose_id,global_example_index"
    normalization_contract: str = NORMALIZATION_CONTRACT
    target_label_fields_used: tuple[str, ...] = ()
    weight_clipping: None = None

    def __post_init__(self) -> None:
        if self.format_version != PROVENANCE_FORMAT_VERSION:
            raise ValueError("unsupported condition augmentation provenance format")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("condition provenance protocol version changed")
        if not self.policy_id:
            raise ValueError("condition provenance policy_id must be non-empty")
        for name, digest in (
            ("policy_sha256", self.policy_sha256),
            ("source_draw_manifest_sha256", self.source_draw_manifest_sha256),
            (
                "augmented_draw_manifest_sha256",
                self.augmented_draw_manifest_sha256,
            ),
            ("oof_requirements_sha256", self.oof_requirements_sha256),
        ):
            _require_sha256(digest, name=name)
        canonical_counter(self.training_seed, self.signature_purpose_id, 0)
        if (
            not isinstance(self.source_draw_purpose_id, str)
            or not self.source_draw_purpose_id
        ):
            raise ValueError("source draw purpose must be non-empty")
        if (
            not isinstance(self.signature_purpose_id, str)
            or not self.signature_purpose_id
        ):
            raise ValueError("signature purpose must be non-empty")
        if self.source_draw_purpose_id == self.signature_purpose_id:
            raise ValueError("source and signature purposes must be distinct")
        if isinstance(self.draws, bool) or not isinstance(self.draws, int):
            raise TypeError("condition provenance draws must be an integer")
        if self.draws <= 0:
            raise ValueError("condition provenance draws must be positive")
        if (
            isinstance(self.unique_oof_items, bool)
            or not isinstance(self.unique_oof_items, int)
        ):
            raise TypeError("unique_oof_items must be an integer")
        if not 0 < self.unique_oof_items <= self.draws:
            raise ValueError("unique OOF item count must lie in [1, draws]")
        if not self.supported_signatures:
            raise ValueError("supported condition signatures cannot be empty")
        canonical_signatures = tuple(
            sorted(
                parse_condition_signature(value).key
                for value in self.supported_signatures
            )
        )
        if canonical_signatures != self.supported_signatures or len(
            canonical_signatures
        ) != len(set(canonical_signatures)):
            raise ValueError(
                "supported signatures must be sorted, canonical, and unique"
            )
        counts = dict(self.signature_counts)
        if len(counts) != len(self.signature_counts):
            raise ValueError("condition signature counts must be unique")
        if tuple(sorted(self.signature_counts)) != self.signature_counts:
            raise ValueError("condition signature counts must be canonically ordered")
        if set(counts) != set(canonical_signatures):
            raise ValueError("signature counts must cover exactly the policy support")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError("condition signature counts must be non-negative integers")
        if sum(counts.values()) != self.draws:
            raise ValueError("condition signature counts must sum to draws")
        if self.fit_split != "outer_train":
            raise ValueError(
                "condition augmentation provenance must be outer_train-only"
            )
        if self.selection_order != "base_sample_then_one_condition_signature":
            raise ValueError("condition selection order changed")
        if self.canonical_counter_key != (
            "training_seed,purpose_id,global_example_index"
        ):
            raise ValueError("condition counter key changed")
        if self.normalization_contract != NORMALIZATION_CONTRACT:
            raise ValueError("condition normalization contract changed")
        if self.target_label_fields_used:
            raise ValueError("condition assignment must not use target label fields")
        if self.weight_clipping is not None:
            raise ValueError("v1.1.3b condition importance weights are unclipped")

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "protocol_version": self.protocol_version,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "source_draw_manifest_sha256": self.source_draw_manifest_sha256,
            "augmented_draw_manifest_sha256": self.augmented_draw_manifest_sha256,
            "oof_requirements_sha256": self.oof_requirements_sha256,
            "training_seed": self.training_seed,
            "source_draw_purpose_id": self.source_draw_purpose_id,
            "signature_purpose_id": self.signature_purpose_id,
            "draws": self.draws,
            "unique_oof_items": self.unique_oof_items,
            "supported_signatures": list(self.supported_signatures),
            "signature_counts": dict(self.signature_counts),
            "fit_split": self.fit_split,
            "selection_order": self.selection_order,
            "canonical_counter_key": self.canonical_counter_key,
            "normalization_contract": self.normalization_contract,
            "target_label_fields_used": list(self.target_label_fields_used),
            "weight_clipping": self.weight_clipping,
        }

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, object]
    ) -> "ConditionAugmentationProvenance":
        """Parse the exact bundle sidecar representation, with no defaults."""

        required = {
            "format_version",
            "protocol_version",
            "policy_id",
            "policy_sha256",
            "source_draw_manifest_sha256",
            "augmented_draw_manifest_sha256",
            "oof_requirements_sha256",
            "training_seed",
            "source_draw_purpose_id",
            "signature_purpose_id",
            "draws",
            "unique_oof_items",
            "supported_signatures",
            "signature_counts",
            "fit_split",
            "selection_order",
            "canonical_counter_key",
            "normalization_contract",
            "target_label_fields_used",
            "weight_clipping",
        }
        if set(raw) != required:
            missing = sorted(required - set(raw))
            unknown = sorted(set(raw) - required)
            raise ValueError(
                "condition provenance keys mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        signatures = raw["supported_signatures"]
        if not isinstance(signatures, Sequence) or isinstance(
            signatures, (str, bytes)
        ):
            raise TypeError("supported_signatures must be a sequence")
        if any(not isinstance(value, str) for value in signatures):
            raise TypeError("supported_signatures must contain strings")
        raw_counts = raw["signature_counts"]
        if not isinstance(raw_counts, Mapping):
            raise TypeError("signature_counts must be a mapping")
        counts: list[tuple[str, int]] = []
        for key, value in raw_counts.items():
            if not isinstance(key, str):
                raise TypeError("signature count keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("signature count values must be integers")
            counts.append((key, value))
        labels = raw["target_label_fields_used"]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError("target_label_fields_used must be a sequence")
        if any(not isinstance(value, str) for value in labels):
            raise TypeError("target_label_fields_used must contain strings")
        return cls(
            format_version=cast(str, raw["format_version"]),
            protocol_version=cast(str, raw["protocol_version"]),
            policy_id=cast(str, raw["policy_id"]),
            policy_sha256=cast(str, raw["policy_sha256"]),
            source_draw_manifest_sha256=cast(
                str, raw["source_draw_manifest_sha256"]
            ),
            augmented_draw_manifest_sha256=cast(
                str, raw["augmented_draw_manifest_sha256"]
            ),
            oof_requirements_sha256=cast(
                str, raw["oof_requirements_sha256"]
            ),
            training_seed=cast(int, raw["training_seed"]),
            source_draw_purpose_id=cast(str, raw["source_draw_purpose_id"]),
            signature_purpose_id=cast(str, raw["signature_purpose_id"]),
            draws=cast(int, raw["draws"]),
            unique_oof_items=cast(int, raw["unique_oof_items"]),
            supported_signatures=tuple(cast(Sequence[str], signatures)),
            signature_counts=tuple(sorted(counts)),
            fit_split=cast(str, raw["fit_split"]),
            selection_order=cast(str, raw["selection_order"]),
            canonical_counter_key=cast(str, raw["canonical_counter_key"]),
            normalization_contract=cast(str, raw["normalization_contract"]),
            target_label_fields_used=tuple(cast(Sequence[str], labels)),
            weight_clipping=cast(None, raw["weight_clipping"]),
        )


@dataclass(frozen=True, slots=True)
class ConditionAugmentationResult:
    rows: tuple[DrawRow, ...]
    oof_requirements: tuple[OOFConditionRequirement, ...]
    provenance: ConditionAugmentationProvenance


def _canonical_base_signature(
    value: str, policy: ConditionAugmentationPolicy
) -> ConditionSignature:
    signature = parse_condition_signature(value)
    if signature.key == policy.source_signature.key:
        return signature
    if not signature.era_present and signature.key == NULL_CONDITION_SIGNATURE:
        # A genuinely unavailable provider supports only the canonical null
        # signature.  It is not stochastically upgraded to an ERA condition.
        return signature
    raise ValueError(
        "base row condition signature must be the declared source or an "
        "actually unavailable canonical null source"
    )


def _canonical_identity(
    *, t0_utc: str, lead_hours: float, condition_signature: str
) -> str:
    return canonical_sample_id(
        parse_utc(t0_utc), lead_hours, condition_signature
    )


def _validate_fold(value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Stage-2 condition rows require a non-negative fold_id")
    return value


def _entries_for_base(
    base: ConditionSignature, policy: ConditionAugmentationPolicy
) -> tuple[ConditionMixtureEntry, ...]:
    if base.era_present:
        return policy.entries
    return (_NULL_ONLY_ENTRY,)


def expand_condition_signatures(
    candidates: Sequence[CandidateItem],
    *,
    policy: ConditionAugmentationPolicy,
) -> tuple[ConditionSupportExpansion, ...]:
    """Expand configured support without changing base-sample draw weight.

    The returned records expose conditional probabilities and intentionally
    are not :class:`CandidateItem` values: flattening them and drawing them as
    independent candidates would violate the sample-first protocol.
    """

    if not candidates:
        raise ValueError("at least one base condition candidate is required")
    seen_sources: set[str] = set()
    seen_outputs: set[tuple[str, float, str]] = set()
    result: list[ConditionSupportExpansion] = []
    for candidate in candidates:
        if candidate.split != "outer_train":
            raise ValueError("condition augmentation is outer_train-only")
        fold_id = _validate_fold(candidate.fold_id)
        base = _canonical_base_signature(candidate.condition_signature, policy)
        expected_source_id = _canonical_identity(
            t0_utc=candidate.t0_utc,
            lead_hours=candidate.lead_hours,
            condition_signature=base.key,
        )
        if candidate.sample_id != expected_source_id:
            raise ValueError("base candidate sample_id/signature lineage mismatch")
        if candidate.sample_id in seen_sources:
            raise ValueError("base candidate sample IDs must be unique")
        seen_sources.add(candidate.sample_id)
        _positive_probability(candidate.p_target, name="base target probability")

        for entry in _entries_for_base(base, policy):
            output_id = _canonical_identity(
                t0_utc=candidate.t0_utc,
                lead_hours=candidate.lead_hours,
                condition_signature=str(entry.condition_signature),
            )
            identity = (
                candidate.t0_utc,
                candidate.lead_hours,
                str(entry.condition_signature),
            )
            if identity in seen_outputs:
                raise ValueError("expanded condition support contains a duplicate item")
            seen_outputs.add(identity)
            result.append(
                ConditionSupportExpansion(
                    source_sample_id=candidate.sample_id,
                    sample_id=output_id,
                    t0_utc=candidate.t0_utc,
                    lead_hours=candidate.lead_hours,
                    source_condition_signature=base.key,
                    condition_signature=str(entry.condition_signature),
                    block_id=candidate.block_id,
                    stratum=candidate.stratum,
                    split=candidate.split,
                    fold_id=fold_id,
                    base_target_probability=float(candidate.p_target),
                    full_target_probability=(
                        float(candidate.p_target) * entry.target_probability
                    ),
                    conditional_target_probability=entry.target_probability,
                    conditional_draw_probability=entry.draw_probability,
                )
            )
    return tuple(result)


def expand_manifest_condition_support(
    items: Sequence[ManifestItem],
    *,
    policy: ConditionAugmentationPolicy,
) -> tuple[ManifestItem, ...]:
    """Build full-item candidate support for membership/probability audits.

    This preserves every dependency, block, stratum, and fold field while
    multiplying base and conditional target/draw probabilities.  It does not
    perform random selection.  In particular, callers must not flatten this
    support into the base sampler; use :func:`materialize_condition_signatures`
    after the base draw sequence has been fixed.
    """

    if not items:
        raise ValueError("at least one base manifest item is required")
    seen_sources: set[str] = set()
    seen_outputs: set[tuple[str, float, str]] = set()
    expanded: list[ManifestItem] = []
    for item in items:
        if item.split != "outer_train":
            raise ValueError("condition augmentation is outer_train-only")
        _validate_fold(item.fold_id)
        base = _canonical_base_signature(item.condition_signature, policy)
        expected_source_id = _canonical_identity(
            t0_utc=item.t0_utc,
            lead_hours=item.lead_hours,
            condition_signature=base.key,
        )
        if item.sample_id != expected_source_id:
            raise ValueError("base manifest sample_id/signature lineage mismatch")
        if item.sample_id in seen_sources:
            raise ValueError("base manifest sample IDs must be unique")
        seen_sources.add(item.sample_id)
        for name, value in (
            ("base target probability", item.p_target),
            ("base draw probability", item.p_draw),
            ("base importance weight", item.omega),
        ):
            _positive_probability(value, name=name)
        if not math.isclose(
            item.omega,
            item.p_target / item.p_draw,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("base manifest importance weight is inconsistent")

        for entry in _entries_for_base(base, policy):
            signature = str(entry.condition_signature)
            identity = (item.t0_utc, item.lead_hours, signature)
            if identity in seen_outputs:
                raise ValueError(
                    "expanded manifest contains a duplicate condition item"
                )
            seen_outputs.add(identity)
            p_target = item.p_target * entry.target_probability
            p_draw = item.p_draw * entry.draw_probability
            _positive_probability(p_target, name="full-item target probability")
            _positive_probability(p_draw, name="full-item draw probability")
            omega = p_target / p_draw
            _positive_probability(omega, name="full-item importance weight")
            expanded.append(
                replace(
                    item,
                    sample_id=_canonical_identity(
                        t0_utc=item.t0_utc,
                        lead_hours=item.lead_hours,
                        condition_signature=signature,
                    ),
                    condition_signature=signature,
                    p_target=p_target,
                    p_draw=p_draw,
                    omega=omega,
                )
            )
    return tuple(expanded)


def _validate_base_draw_row(
    row: DrawRow,
    *,
    expected_index: int,
    policy: ConditionAugmentationPolicy,
) -> ConditionSignature:
    if row.global_example_index != expected_index:
        raise ValueError("base draw rows must have canonical monotone global indices")
    if row.split != "outer_train":
        raise ValueError("condition augmentation is outer_train-only")
    _validate_fold(row.fold_id)
    base = _canonical_base_signature(row.condition_signature, policy)
    expected_sample_id = _canonical_identity(
        t0_utc=row.t0_utc,
        lead_hours=row.lead_hours,
        condition_signature=base.key,
    )
    if row.sample_id != expected_sample_id:
        raise ValueError("base draw sample_id/signature lineage mismatch")
    for name, value in (
        ("base target probability", row.p_target),
        ("base draw probability", row.p_draw),
        ("base importance weight", row.omega),
    ):
        _positive_probability(value, name=name)
    if not math.isclose(
        row.omega,
        row.p_target / row.p_draw,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("base draw importance weight is inconsistent")
    return base


def _oof_requirements(
    rows: Sequence[DrawRow],
) -> tuple[OOFConditionRequirement, ...]:
    by_identity: dict[tuple[str, float, str], OOFConditionRequirement] = {}
    for row in rows:
        requirement = OOFConditionRequirement(
            sample_id=row.sample_id,
            t0_utc=row.t0_utc,
            lead_hours=row.lead_hours,
            condition_signature=row.condition_signature,
            block_id=row.block_id,
            fold_id=_validate_fold(row.fold_id),
        )
        prior = by_identity.setdefault(requirement.semantic_key, requirement)
        if prior != requirement:
            raise ValueError("one OOF semantic identity has conflicting lineage")
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (
                item.t0_utc,
                item.lead_hours,
                item.condition_signature,
            ),
        )
    )


def validate_materialized_condition_provenance(
    rows: Sequence[DrawRow],
    *,
    policy: ConditionAugmentationPolicy,
    provenance: ConditionAugmentationProvenance,
) -> tuple[OOFConditionRequirement, ...]:
    """Validate a published augmented draw and return its exact OOF universe."""

    if not rows:
        raise ValueError("augmented draw manifest cannot be empty")
    if provenance.protocol_version != policy.protocol_version:
        raise ValueError("condition policy/provenance protocol mismatch")
    if provenance.policy_id != policy.policy_id:
        raise ValueError("condition policy/provenance ID mismatch")
    if provenance.policy_sha256 != policy.semantic_sha256:
        raise ValueError("condition policy/provenance SHA-256 mismatch")
    supported = tuple(
        str(entry.condition_signature) for entry in policy.entries
    )
    if provenance.supported_signatures != supported:
        raise ValueError("condition provenance support disagrees with policy")

    counts = {signature: 0 for signature in supported}
    for expected_index, row in enumerate(rows):
        if row.global_example_index != expected_index:
            raise ValueError("augmented rows require canonical monotone indices")
        if row.split != "outer_train":
            raise ValueError("augmented condition rows must be outer_train")
        signature = parse_condition_signature(row.condition_signature).key
        if signature not in counts:
            raise ValueError("augmented row signature is outside policy support")
        if row.sample_id != _canonical_identity(
            t0_utc=row.t0_utc,
            lead_hours=row.lead_hours,
            condition_signature=signature,
        ):
            raise ValueError("augmented draw sample_id/signature lineage mismatch")
        _validate_fold(row.fold_id)
        for name, value in (
            ("full-item target probability", row.p_target),
            ("full-item draw probability", row.p_draw),
            ("full-item importance weight", row.omega),
        ):
            _positive_probability(value, name=name)
        if not math.isclose(
            row.omega,
            row.p_target / row.p_draw,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("augmented draw importance weight is inconsistent")
        counts[signature] += 1

    if canonical_draw_rows_sha256(rows) != (
        provenance.augmented_draw_manifest_sha256
    ):
        raise ValueError("augmented draw manifest SHA-256 mismatch")
    if provenance.draws != len(rows):
        raise ValueError("condition provenance draw count mismatch")
    if dict(provenance.signature_counts) != counts:
        raise ValueError("condition provenance signature counts mismatch")
    requirements = _oof_requirements(rows)
    if provenance.unique_oof_items != len(requirements):
        raise ValueError("condition provenance unique OOF count mismatch")
    requirement_hash = _semantic_sha256(
        [asdict(requirement) for requirement in requirements]
    )
    if provenance.oof_requirements_sha256 != requirement_hash:
        raise ValueError("condition provenance OOF requirements hash mismatch")
    return requirements


def materialize_condition_signatures(
    base_rows: Sequence[DrawRow],
    *,
    policy: ConditionAugmentationPolicy,
    training_seed: int,
    source_draw_purpose_id: str,
    signature_purpose_id: str,
    source_draw_manifest_sha256: str,
) -> ConditionAugmentationResult:
    """Materialize one signature per already selected base draw.

    Signature randomness is keyed only by ``(training_seed,
    signature_purpose_id, global_example_index)``.  Batch size, accumulation,
    rank, worker count, and traversal topology are not accepted inputs.
    """

    if not base_rows:
        raise ValueError("base draw manifest cannot be empty")
    if not isinstance(source_draw_purpose_id, str) or not source_draw_purpose_id:
        raise ValueError("source_draw_purpose_id must be non-empty")
    if not isinstance(signature_purpose_id, str) or not signature_purpose_id:
        raise ValueError("signature_purpose_id must be non-empty")
    if signature_purpose_id == source_draw_purpose_id:
        raise ValueError("signature selection requires a distinct purpose namespace")
    _require_sha256(
        source_draw_manifest_sha256, name="source_draw_manifest_sha256"
    )
    actual_source_hash = canonical_draw_rows_sha256(base_rows)
    if source_draw_manifest_sha256 != actual_source_hash:
        raise ValueError("source draw manifest hash does not match its rows")
    # Validate seed and purpose before entering the materialization loop.
    canonical_counter(training_seed, signature_purpose_id, 0)

    counts: dict[str, int] = {
        str(entry.condition_signature): 0 for entry in policy.entries
    }
    cumulative: list[float] = []
    cumulative_total = 0.0
    for entry in policy.entries:
        cumulative_total += entry.draw_probability
        cumulative.append(cumulative_total)
    cumulative[-1] = 1.0
    rows: list[DrawRow] = []
    for expected_index, row in enumerate(base_rows):
        base = _validate_base_draw_row(
            row, expected_index=expected_index, policy=policy
        )
        entries = _entries_for_base(base, policy)
        if base.era_present:
            counter = canonical_counter(
                training_seed, signature_purpose_id, row.global_example_index
            )
            uniform = (counter >> 11) * (1.0 / (1 << 53))
            selected = entries[bisect_right(cumulative, uniform)]
        else:
            selected = entries[0]

        selected_signature = str(selected.condition_signature)
        sample_id = _canonical_identity(
            t0_utc=row.t0_utc,
            lead_hours=row.lead_hours,
            condition_signature=selected_signature,
        )
        p_target = row.p_target * selected.target_probability
        p_draw = row.p_draw * selected.draw_probability
        _positive_probability(p_target, name="full-item target probability")
        _positive_probability(p_draw, name="full-item draw probability")
        omega = p_target / p_draw
        _positive_probability(omega, name="full-item importance weight")
        rows.append(
            replace(
                row,
                sample_id=sample_id,
                condition_signature=selected_signature,
                p_target=p_target,
                p_draw=p_draw,
                omega=omega,
            )
        )
        counts[selected_signature] = counts.get(selected_signature, 0) + 1

    materialized = tuple(rows)
    requirements = _oof_requirements(materialized)
    requirements_hash = _semantic_sha256(
        [asdict(requirement) for requirement in requirements]
    )
    provenance = ConditionAugmentationProvenance(
        format_version=PROVENANCE_FORMAT_VERSION,
        protocol_version=policy.protocol_version,
        policy_id=policy.policy_id,
        policy_sha256=policy.semantic_sha256,
        source_draw_manifest_sha256=source_draw_manifest_sha256,
        augmented_draw_manifest_sha256=canonical_draw_rows_sha256(materialized),
        oof_requirements_sha256=requirements_hash,
        training_seed=training_seed,
        source_draw_purpose_id=source_draw_purpose_id,
        signature_purpose_id=signature_purpose_id,
        draws=len(materialized),
        unique_oof_items=len(requirements),
        supported_signatures=tuple(
            str(entry.condition_signature) for entry in policy.entries
        ),
        signature_counts=tuple(sorted(counts.items())),
    )
    return ConditionAugmentationResult(
        rows=materialized,
        oof_requirements=requirements,
        provenance=provenance,
    )


def oof_lineage_metadata(
    result: ConditionAugmentationResult,
    *,
    normalization_artifact_sha256: str,
) -> Mapping[str, object]:
    """Return hashes an OOF publication must bind before dense inference.

    The actual OOF writer additionally binds fold-regression checkpoints and
    target-builder provenance.  This mapping supplies the condition-specific
    half of that lineage without inspecting any target, validation, or test
    label.
    """

    normalization_hash = _require_sha256(
        normalization_artifact_sha256, name="normalization_artifact_sha256"
    )
    provenance = result.provenance
    if canonical_draw_rows_sha256(result.rows) != (
        provenance.augmented_draw_manifest_sha256
    ):
        raise ValueError("augmented rows no longer match condition provenance")
    requirements_hash = _semantic_sha256(
        [asdict(requirement) for requirement in result.oof_requirements]
    )
    if requirements_hash != provenance.oof_requirements_sha256:
        raise ValueError("OOF requirements no longer match condition provenance")
    return {
        "condition_augmentation_provenance_sha256": provenance.semantic_sha256,
        "condition_policy_sha256": provenance.policy_sha256,
        "source_draw_manifest_sha256": provenance.source_draw_manifest_sha256,
        "augmented_draw_manifest_sha256": (
            provenance.augmented_draw_manifest_sha256
        ),
        "oof_requirements_sha256": provenance.oof_requirements_sha256,
        "normalization_artifact_sha256": normalization_hash,
        "normalization_fit_split": "outer_train",
        "normalization_contract": NORMALIZATION_CONTRACT,
        "condition_identity_match": (
            "sample_id,t0_utc,lead_hours,condition_signature,block_id,fold_id"
        ),
        "target_label_fields_used_for_condition_assignment": [],
    }


__all__ = [
    "NORMALIZATION_CONTRACT",
    "NULL_CONDITION_SIGNATURE",
    "POLICY_FORMAT_VERSION",
    "PROTOCOL_VERSION",
    "PROVENANCE_FORMAT_VERSION",
    "ConditionAugmentationPolicy",
    "ConditionAugmentationProvenance",
    "ConditionAugmentationResult",
    "ConditionMixtureEntry",
    "ConditionSupportExpansion",
    "OOFConditionRequirement",
    "TemporalAccessStrategy",
    "canonical_draw_rows_sha256",
    "expand_condition_signatures",
    "expand_manifest_condition_support",
    "materialize_condition_signatures",
    "oof_lineage_metadata",
    "validate_materialized_condition_provenance",
]

"""Immutable, exact-signature Stage 3 calibration artifacts.

This module is the publication boundary around :mod:`kcorrdiff.training.calibration`.
It deliberately keeps model selection out of the fitting API: architecture,
``d_enabled``, probability family, and pooling order arrive in one already-frozen
decision.  Calibration labels can therefore fit parameters, but cannot choose a
model or silently change a calibration family.

The JSON representation is canonical and self-authenticating.  Its
``semantic_sha256`` covers every semantic field other than the digest itself;
publication uses an atomic, no-overwrite hard link and validates a full typed
round trip before making the file visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import Tensor

from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.training.calibration import (
    CALIBRATION_SPLIT,
    POOLING_ORDER,
    FoldCalibrationMoments,
    IndependentBlockSupport,
    LocationScaleCalibration,
    PoolingDecision,
    ProbabilityCalibration,
    SamplerCalibration,
    apply_residual_calibration,
    fit_location_total_scale,
    fit_monotone_logit_linear_probability,
    fit_sampler_bias_and_spread,
    independent_block_support,
    monotone_logit_linear_probability,
    select_pooling_level,
)
from kcorrdiff.training.edm_sampling import (
    EnsembleSignature,
    SamplerCoreSignature,
)


CALIBRATION_ARTIFACT_FORMAT = "kcorrdiff.calibration.v1"
MONOTONE_LOGIT_LINEAR_FAMILY = "monotone_logit_linear"
OFFICIAL_LEADS_HOURS = tuple(index / 2.0 for index in range(1, 13))
OFFICIAL_ENSEMBLE_THRESHOLDS_MM = (A_WET_MM, 1.0, 5.0)
ReleaseStatus = Literal["development", "complete"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_NAME = re.compile(r"[a-z][a-z0-9_]*_sha256")


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _lead(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("lead_hours must be a real scalar")
    result = float(value)
    if result not in OFFICIAL_LEADS_HOURS:
        raise ValueError("lead_hours must be one of the 12 official half-hour leads")
    return result


def _condition(value: str) -> str:
    return parse_condition_signature(value).key


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sampler_core_dict(value: SamplerCoreSignature) -> dict[str, object]:
    return {
        "checkpoint_id": value.checkpoint_id,
        "checkpoint_kind": value.checkpoint_kind,
        "edm_steps": value.edm_steps,
        "rho": value.rho,
        "sigma_max": value.sigma_max,
        "sigma_min": value.sigma_min,
        "sigma_schedule": value.sigma_schedule,
        "solver": value.solver,
    }


def _sampler_core_from_dict(raw: Mapping[str, object]) -> SamplerCoreSignature:
    return SamplerCoreSignature(
        checkpoint_id=str(raw["checkpoint_id"]),
        checkpoint_kind=str(raw["checkpoint_kind"]),  # type: ignore[arg-type]
        edm_steps=int(raw["edm_steps"]),
        solver=str(raw["solver"]),  # type: ignore[arg-type]
        sigma_schedule=str(raw["sigma_schedule"]),  # type: ignore[arg-type]
        sigma_min=float(raw["sigma_min"]),
        sigma_max=float(raw["sigma_max"]),
        rho=float(raw["rho"]),
    )


def _ensemble_dict(value: EnsembleSignature) -> dict[str, object]:
    return {
        "member_count": value.member_count,
        "sampler_core": _sampler_core_dict(value.sampler_core),
    }


def _ensemble_from_dict(raw: Mapping[str, object]) -> EnsembleSignature:
    return EnsembleSignature(
        sampler_core=_sampler_core_from_dict(_as_mapping(raw["sampler_core"])),
        member_count=int(raw["member_count"]),
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration JSON object must be a mapping")
    return value


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


@dataclass(frozen=True, slots=True)
class FrozenModelSelectionDecision:
    """Decision fitted without access to final calibration labels."""

    decision_sha256: str
    architecture_sha256: str
    d_enabled: bool
    probability_mapping_family: str = MONOTONE_LOGIT_LINEAR_FAMILY
    pooling_order: tuple[str, ...] = POOLING_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "pooling_order", tuple(self.pooling_order))
        _require_sha256(self.decision_sha256, name="model-selection decision")
        _require_sha256(self.architecture_sha256, name="selected architecture")
        if not isinstance(self.d_enabled, bool):
            raise TypeError("d_enabled must be a frozen boolean")
        if self.probability_mapping_family != MONOTONE_LOGIT_LINEAR_FAMILY:
            raise ValueError("probability mapping family is not the frozen family")
        if tuple(self.pooling_order) != POOLING_ORDER:
            raise ValueError("pooling order is not the predeclared deterministic ladder")

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture_sha256": self.architecture_sha256,
            "d_enabled": self.d_enabled,
            "decision_sha256": self.decision_sha256,
            "pooling_order": list(self.pooling_order),
            "probability_mapping_family": self.probability_mapping_family,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FrozenModelSelectionDecision":
        return cls(
            decision_sha256=str(raw["decision_sha256"]),
            architecture_sha256=str(raw["architecture_sha256"]),
            d_enabled=raw["d_enabled"],  # type: ignore[arg-type]
            probability_mapping_family=str(raw["probability_mapping_family"]),
            pooling_order=tuple(
                str(value)
                for value in _as_list(raw["pooling_order"], name="pooling_order")
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Canonical set of immutable source/config artifact hashes."""

    hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hashes",
            tuple((name, value) for name, value in self.hashes),
        )
        if not self.hashes:
            raise ValueError("calibration provenance hashes cannot be empty")
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for name, value in self.hashes:
            if not isinstance(name, str) or _PROVENANCE_NAME.fullmatch(name) is None:
                raise ValueError(
                    "provenance names must be canonical snake_case names ending _sha256"
                )
            if name in seen:
                raise ValueError(f"duplicate provenance hash: {name}")
            seen.add(name)
            normalized.append((name, _require_sha256(value, name=name)))
        if tuple(sorted(normalized)) != tuple(normalized):
            raise ValueError("provenance hashes must be sorted by canonical name")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, str]) -> "CalibrationProvenance":
        if not isinstance(raw, Mapping):
            raise TypeError("provenance_hashes must be a mapping")
        return cls(tuple(sorted((str(name), value) for name, value in raw.items())))

    @property
    def mapping(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.hashes))

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {name: value for name, value in self.hashes}


@dataclass(frozen=True, slots=True)
class LocationScaleKey:
    """Exact ``b,c`` key: lead, condition, all fold hashes, and full hash."""

    lead_hours: float
    condition_signature: str
    fold_checkpoint_sha256s: tuple[str, ...]
    full_checkpoint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fold_checkpoint_sha256s", tuple(self.fold_checkpoint_sha256s)
        )
        object.__setattr__(self, "lead_hours", _lead(self.lead_hours))
        object.__setattr__(
            self, "condition_signature", _condition(self.condition_signature)
        )
        if len(self.fold_checkpoint_sha256s) < 2:
            raise ValueError("b/c key requires at least two fold checkpoint hashes")
        for index, value in enumerate(self.fold_checkpoint_sha256s):
            _require_sha256(value, name=f"fold checkpoint {index}")
        if len(set(self.fold_checkpoint_sha256s)) != len(
            self.fold_checkpoint_sha256s
        ):
            raise ValueError("fold checkpoint hashes must be unique")
        _require_sha256(self.full_checkpoint_sha256, name="full checkpoint")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_signature": self.condition_signature,
            "fold_checkpoint_sha256s": list(self.fold_checkpoint_sha256s),
            "full_checkpoint_sha256": self.full_checkpoint_sha256,
            "lead_hours": self.lead_hours,
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "LocationScaleKey":
        return cls(
            lead_hours=float(raw["lead_hours"]),
            condition_signature=str(raw["condition_signature"]),
            fold_checkpoint_sha256s=tuple(
                str(value)
                for value in _as_list(
                    raw["fold_checkpoint_sha256s"], name="fold_checkpoint_sha256s"
                )
            ),
            full_checkpoint_sha256=str(raw["full_checkpoint_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SamplerBiasKey:
    """Exact ``d`` key ``(tau, condition, sampler_core_signature)``."""

    lead_hours: float
    condition_signature: str
    sampler_core: SamplerCoreSignature

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_hours", _lead(self.lead_hours))
        object.__setattr__(
            self, "condition_signature", _condition(self.condition_signature)
        )
        if not isinstance(self.sampler_core, SamplerCoreSignature):
            raise TypeError("sampler_core must be a SamplerCoreSignature")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_signature": self.condition_signature,
            "lead_hours": self.lead_hours,
            "sampler_core": _sampler_core_dict(self.sampler_core),
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SamplerBiasKey":
        return cls(
            lead_hours=float(raw["lead_hours"]),
            condition_signature=str(raw["condition_signature"]),
            sampler_core=_sampler_core_from_dict(_as_mapping(raw["sampler_core"])),
        )


@dataclass(frozen=True, slots=True)
class SpreadKey:
    """Exact ``gamma`` key ``(tau, condition, ensemble_signature)``."""

    lead_hours: float
    condition_signature: str
    ensemble_signature: EnsembleSignature

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_hours", _lead(self.lead_hours))
        object.__setattr__(
            self, "condition_signature", _condition(self.condition_signature)
        )
        if not isinstance(self.ensemble_signature, EnsembleSignature):
            raise TypeError("ensemble_signature must be an EnsembleSignature")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_signature": self.condition_signature,
            "ensemble_signature": _ensemble_dict(self.ensemble_signature),
            "lead_hours": self.lead_hours,
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SpreadKey":
        return cls(
            lead_hours=float(raw["lead_hours"]),
            condition_signature=str(raw["condition_signature"]),
            ensemble_signature=_ensemble_from_dict(
                _as_mapping(raw["ensemble_signature"])
            ),
        )


@dataclass(frozen=True, slots=True)
class RegressionProbabilityKey:
    """Exact regression ``p_cal`` key; threshold is fixed to ``A_wet``."""

    lead_hours: float
    threshold_mm: float
    condition_signature: str
    regression_checkpoint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_hours", _lead(self.lead_hours))
        object.__setattr__(
            self, "condition_signature", _condition(self.condition_signature)
        )
        threshold = float(self.threshold_mm)
        if threshold != A_WET_MM:
            raise ValueError("regression p_cal threshold must be exactly A_wet")
        object.__setattr__(self, "threshold_mm", threshold)
        _require_sha256(
            self.regression_checkpoint_sha256, name="regression checkpoint"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_signature": self.condition_signature,
            "lead_hours": self.lead_hours,
            "regression_checkpoint_sha256": self.regression_checkpoint_sha256,
            "threshold_mm": self.threshold_mm,
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RegressionProbabilityKey":
        return cls(
            lead_hours=float(raw["lead_hours"]),
            threshold_mm=float(raw["threshold_mm"]),
            condition_signature=str(raw["condition_signature"]),
            regression_checkpoint_sha256=str(raw["regression_checkpoint_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class EnsembleProbabilityKey:
    """Exact ``q_cal_T`` key, including member count in the ensemble signature."""

    lead_hours: float
    threshold_mm: float
    condition_signature: str
    ensemble_signature: EnsembleSignature

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_hours", _lead(self.lead_hours))
        object.__setattr__(
            self, "condition_signature", _condition(self.condition_signature)
        )
        threshold = float(self.threshold_mm)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("ensemble probability threshold must be finite and positive")
        if threshold not in (*OFFICIAL_ENSEMBLE_THRESHOLDS_MM, 10.0):
            raise ValueError("q_cal threshold must be official or the declared 10 mm auxiliary")
        object.__setattr__(self, "threshold_mm", threshold)
        if not isinstance(self.ensemble_signature, EnsembleSignature):
            raise TypeError("ensemble_signature must be an EnsembleSignature")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_signature": self.condition_signature,
            "ensemble_signature": _ensemble_dict(self.ensemble_signature),
            "lead_hours": self.lead_hours,
            "threshold_mm": self.threshold_mm,
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "EnsembleProbabilityKey":
        return cls(
            lead_hours=float(raw["lead_hours"]),
            threshold_mm=float(raw["threshold_mm"]),
            condition_signature=str(raw["condition_signature"]),
            ensemble_signature=_ensemble_from_dict(
                _as_mapping(raw["ensemble_signature"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolingEvidence:
    """Raw independent-block evidence for one predeclared pooling level."""

    block_id: Sequence[str]
    weight: ArrayLike
    observation: ArrayLike | None = None


@dataclass(frozen=True, slots=True)
class PoolingLevelAudit:
    level: str
    record_count: int
    positive_weight_record_count: int
    support: IndependentBlockSupport

    def __post_init__(self) -> None:
        if self.level not in POOLING_ORDER:
            raise ValueError("unknown pooling level")
        if (
            isinstance(self.record_count, bool)
            or isinstance(self.positive_weight_record_count, bool)
            or self.record_count < 0
            or self.positive_weight_record_count < 0
            or self.positive_weight_record_count > self.record_count
        ):
            raise ValueError("invalid pooling record counts")
        if not isinstance(self.support, IndependentBlockSupport):
            raise TypeError("support must be IndependentBlockSupport")

    def to_dict(self) -> dict[str, object]:
        return {
            "block_count": self.support.block_count,
            "block_ess": self.support.block_ess,
            "level": self.level,
            "negative_support_blocks": self.support.negative_support_blocks,
            "positive_support_blocks": self.support.positive_support_blocks,
            "positive_weight_record_count": self.positive_weight_record_count,
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PoolingLevelAudit":
        return cls(
            level=str(raw["level"]),
            record_count=int(raw["record_count"]),
            positive_weight_record_count=int(raw["positive_weight_record_count"]),
            support=IndependentBlockSupport(
                block_count=int(raw["block_count"]),
                block_ess=float(raw["block_ess"]),
                positive_support_blocks=int(raw["positive_support_blocks"]),
                negative_support_blocks=int(raw["negative_support_blocks"]),
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolingAudit:
    """Full ladder evidence plus its deterministic first-passing decision."""

    probability_gate: bool
    ladder: tuple[PoolingLevelAudit, ...]
    decision: PoolingDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "ladder", tuple(self.ladder))
        if not isinstance(self.probability_gate, bool):
            raise TypeError("probability_gate must be boolean")
        if tuple(item.level for item in self.ladder) != POOLING_ORDER:
            raise ValueError("pooling audit must record every level in declared order")
        expected = select_pooling_level(
            {item.level: item.support for item in self.ladder},
            probability_gate=self.probability_gate,
        )
        if self.decision != expected:
            raise ValueError("pooling decision is not the deterministic first-passing level")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_level": self.decision.level,
            "ladder": [item.to_dict() for item in self.ladder],
            "probability_gate": self.probability_gate,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PoolingAudit":
        ladder = tuple(
            PoolingLevelAudit.from_dict(_as_mapping(value))
            for value in _as_list(raw["ladder"], name="pooling ladder")
        )
        probability_gate = raw["probability_gate"]
        if not isinstance(probability_gate, bool):
            raise TypeError("probability_gate must be boolean")
        decision = select_pooling_level(
            {item.level: item.support for item in ladder},
            probability_gate=probability_gate,
        )
        if str(raw["decision_level"]) != decision.level:
            raise ValueError("serialized pooling decision does not match support")
        return cls(probability_gate, ladder, decision)


def build_pooling_audit(
    evidence_by_level: Mapping[str, PoolingEvidence],
    *,
    probability_gate: bool,
) -> PoolingAudit:
    """Compute all ladder counts/support in fixed order, then select once."""

    if not isinstance(probability_gate, bool):
        raise TypeError("probability_gate must be boolean")
    missing = [level for level in POOLING_ORDER if level not in evidence_by_level]
    extras = sorted(set(evidence_by_level) - set(POOLING_ORDER))
    if missing or extras:
        raise ValueError(
            f"pooling evidence mismatch; missing={missing}, extras={extras}"
        )
    ladder: list[PoolingLevelAudit] = []
    for level in POOLING_ORDER:
        evidence = evidence_by_level[level]
        if not isinstance(evidence, PoolingEvidence):
            raise TypeError("pooling evidence entries must be PoolingEvidence")
        weights = np.asarray(evidence.weight, dtype=np.float64).reshape(-1)
        if len(evidence.block_id) != weights.size:
            raise ValueError("pooling block IDs and weights must share length")
        observation = evidence.observation
        if probability_gate and observation is None:
            raise ValueError("probability pooling requires binary observations")
        if not probability_gate and observation is not None:
            raise ValueError("residual pooling must not inspect outcome classes")
        support = independent_block_support(
            block_id=evidence.block_id,
            weight=weights,
            observation=observation,
        )
        ladder.append(
            PoolingLevelAudit(
                level=level,
                record_count=int(weights.size),
                positive_weight_record_count=int(np.count_nonzero(weights > 0.0)),
                support=support,
            )
        )
    support_by_level = {item.level: item.support for item in ladder}
    decision = select_pooling_level(
        support_by_level, probability_gate=probability_gate
    )
    return PoolingAudit(probability_gate, tuple(ladder), decision)


@dataclass(frozen=True, slots=True)
class LocationScaleRecord:
    key: LocationScaleKey
    calibration: LocationScaleCalibration
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.pooling.probability_gate:
            raise ValueError("b/c requires residual pooling support")
        _require_sha256(self.provenance_sha256, name="record provenance")

    def to_dict(self) -> dict[str, object]:
        value = self.calibration
        return {
            "key": self.key.to_dict(),
            "parameters": {
                "fold_mixture_mean": value.fold_mixture_mean,
                "fold_mixture_variance": value.fold_mixture_variance,
                "fold_weights": list(value.fold_weights),
                "full_mean": value.full_mean,
                "full_variance": value.full_variance,
                "location_b": value.location_b,
                "total_scale_c": value.total_scale_c,
            },
            "pooling": self.pooling.to_dict(),
            "provenance_sha256": self.provenance_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "LocationScaleRecord":
        params = _as_mapping(raw["parameters"])
        return cls(
            key=LocationScaleKey.from_dict(_as_mapping(raw["key"])),
            calibration=LocationScaleCalibration(
                location_b=float(params["location_b"]),
                total_scale_c=float(params["total_scale_c"]),
                fold_weights=tuple(
                    float(value)
                    for value in _as_list(params["fold_weights"], name="fold_weights")
                ),
                fold_mixture_mean=float(params["fold_mixture_mean"]),
                fold_mixture_variance=float(params["fold_mixture_variance"]),
                full_mean=float(params["full_mean"]),
                full_variance=float(params["full_variance"]),
            ),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=str(raw["provenance_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SamplerBiasRecord:
    key: SamplerBiasKey
    sampler_bias_d: float
    d_enabled: bool
    bias_fraction: float
    location_scale_key_sha256: str
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.d_enabled, bool):
            raise TypeError("d_enabled must be boolean")
        values = self.sampler_bias_d, self.bias_fraction
        if not all(math.isfinite(value) for value in values) or self.bias_fraction < 0.0:
            raise ValueError("invalid sampler-bias record")
        if not self.d_enabled and self.sampler_bias_d != 0.0:
            raise ValueError("d must be zero for the disabled frozen arm")
        if self.pooling.probability_gate:
            raise ValueError("d requires residual pooling support")
        _require_sha256(self.location_scale_key_sha256, name="location-scale key")
        _require_sha256(self.provenance_sha256, name="record provenance")

    def to_dict(self) -> dict[str, object]:
        return {
            "bias_fraction": self.bias_fraction,
            "d_enabled": self.d_enabled,
            "key": self.key.to_dict(),
            "location_scale_key_sha256": self.location_scale_key_sha256,
            "pooling": self.pooling.to_dict(),
            "provenance_sha256": self.provenance_sha256,
            "sampler_bias_d": self.sampler_bias_d,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SamplerBiasRecord":
        return cls(
            key=SamplerBiasKey.from_dict(_as_mapping(raw["key"])),
            sampler_bias_d=float(raw["sampler_bias_d"]),
            d_enabled=raw["d_enabled"],  # type: ignore[arg-type]
            bias_fraction=float(raw["bias_fraction"]),
            location_scale_key_sha256=str(raw["location_scale_key_sha256"]),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=str(raw["provenance_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SpreadRecord:
    key: SpreadKey
    spread_gamma: float
    sampler_bias_key_sha256: str
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.spread_gamma) or self.spread_gamma <= 0.0:
            raise ValueError("spread gamma must be finite and positive")
        if self.pooling.probability_gate:
            raise ValueError("gamma requires residual pooling support")
        _require_sha256(self.sampler_bias_key_sha256, name="sampler-bias key")
        _require_sha256(self.provenance_sha256, name="record provenance")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "pooling": self.pooling.to_dict(),
            "provenance_sha256": self.provenance_sha256,
            "sampler_bias_key_sha256": self.sampler_bias_key_sha256,
            "spread_gamma": self.spread_gamma,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SpreadRecord":
        return cls(
            key=SpreadKey.from_dict(_as_mapping(raw["key"])),
            spread_gamma=float(raw["spread_gamma"]),
            sampler_bias_key_sha256=str(raw["sampler_bias_key_sha256"]),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=str(raw["provenance_sha256"]),
        )


# ProbabilityCalibration retains only its selected PoolingDecision.  The
# artifact must retain the entire ladder, so probability records use a small
# dedicated representation rather than the generic class above.
@dataclass(frozen=True, slots=True)
class RegressionProbabilityRecord:
    key: RegressionProbabilityKey
    calibration: ProbabilityCalibration
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        _validate_probability_record(self.calibration, self.pooling)
        _require_sha256(self.provenance_sha256, name="record provenance")

    def to_dict(self) -> dict[str, object]:
        return _probability_record_dict(
            self.key.to_dict(), self.calibration, self.pooling, self.provenance_sha256
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RegressionProbabilityRecord":
        calibration, pooling = _probability_from_dict(raw)
        return cls(
            RegressionProbabilityKey.from_dict(_as_mapping(raw["key"])),
            calibration,
            pooling,
            str(raw["provenance_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class EnsembleProbabilityRecord:
    key: EnsembleProbabilityKey
    calibration: ProbabilityCalibration
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        _validate_probability_record(self.calibration, self.pooling)
        _require_sha256(self.provenance_sha256, name="record provenance")

    def to_dict(self) -> dict[str, object]:
        return _probability_record_dict(
            self.key.to_dict(), self.calibration, self.pooling, self.provenance_sha256
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "EnsembleProbabilityRecord":
        calibration, pooling = _probability_from_dict(raw)
        return cls(
            EnsembleProbabilityKey.from_dict(_as_mapping(raw["key"])),
            calibration,
            pooling,
            str(raw["provenance_sha256"]),
        )


def _validate_probability_record(
    calibration: ProbabilityCalibration, pooling: PoolingAudit
) -> None:
    if not pooling.probability_gate:
        raise ValueError("probability mapping requires probability pooling support")
    if calibration.pooling != pooling.decision:
        raise ValueError("probability fit and persisted pooling decision disagree")


def _probability_record_dict(
    key: Mapping[str, object],
    calibration: ProbabilityCalibration,
    pooling: PoolingAudit,
    provenance_sha256: str,
) -> dict[str, object]:
    return {
        "key": dict(key),
        "parameters": {
            "alpha": calibration.alpha,
            "beta_raw": calibration.beta_raw,
            "iterations": calibration.iterations,
            "slope": calibration.slope,
            "weighted_log_loss": calibration.weighted_log_loss,
        },
        "pooling": pooling.to_dict(),
        "provenance_sha256": provenance_sha256,
    }


def _probability_from_dict(
    raw: Mapping[str, object],
) -> tuple[ProbabilityCalibration, PoolingAudit]:
    params = _as_mapping(raw["parameters"])
    pooling = PoolingAudit.from_dict(_as_mapping(raw["pooling"]))
    return (
        ProbabilityCalibration(
            alpha=float(params["alpha"]),
            beta_raw=float(params["beta_raw"]),
            slope=float(params["slope"]),
            weighted_log_loss=float(params["weighted_log_loss"]),
            iterations=int(params["iterations"]),
            pooling=pooling.decision,
        ),
        pooling,
    )


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Typed immutable final table; tuple order is canonical key-hash order."""

    split: str
    release_status: ReleaseStatus
    model_selection: FrozenModelSelectionDecision
    provenance: CalibrationProvenance
    location_scale: tuple[LocationScaleRecord, ...] = ()
    sampler_bias: tuple[SamplerBiasRecord, ...] = ()
    spread: tuple[SpreadRecord, ...] = ()
    regression_probability: tuple[RegressionProbabilityRecord, ...] = ()
    ensemble_probability: tuple[EnsembleProbabilityRecord, ...] = ()
    calibration_absent_identity: bool = False

    def __post_init__(self) -> None:
        for name in (
            "location_scale",
            "sampler_bias",
            "spread",
            "regression_probability",
            "ensemble_probability",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.split != CALIBRATION_SPLIT:
            raise ValueError("published calibration must use split='calibration'")
        if self.release_status not in ("development", "complete"):
            raise ValueError("release_status must be development or complete")
        if not isinstance(self.model_selection, FrozenModelSelectionDecision):
            raise TypeError("model_selection must be frozen before calibration")
        if not isinstance(self.provenance, CalibrationProvenance):
            raise TypeError("provenance must be CalibrationProvenance")
        if not isinstance(self.calibration_absent_identity, bool):
            raise TypeError("calibration_absent_identity must be boolean")
        expected_provenance = self.provenance.semantic_sha256
        decision_hash = self.provenance.mapping.get(
            "model_selection_decision_sha256"
        )
        architecture_hash = self.provenance.mapping.get(
            "selected_architecture_sha256"
        )
        if decision_hash != self.model_selection.decision_sha256:
            raise ValueError("provenance does not bind the frozen decision hash")
        if architecture_hash != self.model_selection.architecture_sha256:
            raise ValueError("provenance does not bind the selected architecture hash")
        groups: tuple[tuple[object, ...], ...] = (
            self.location_scale,
            self.sampler_bias,
            self.spread,
            self.regression_probability,
            self.ensemble_probability,
        )
        all_records = tuple(record for group in groups for record in group)
        if self.calibration_absent_identity:
            if self.release_status != "development" or all_records:
                raise ValueError(
                    "absent-calibration identity is allowed only as an empty development artifact"
                )
        elif not all_records:
            if self.release_status == "complete":
                raise ValueError("a complete release cannot use absent calibration identity")
            raise ValueError(
                "an empty development artifact must explicitly declare absent-calibration identity"
            )
        for records in groups:
            digests = tuple(record.key.semantic_sha256 for record in records)  # type: ignore[attr-defined]
            if digests != tuple(sorted(digests)) or len(set(digests)) != len(digests):
                raise ValueError("calibration records must have unique canonical key order")
            for record in records:
                if record.provenance_sha256 != expected_provenance:  # type: ignore[attr-defined]
                    raise ValueError("calibration record provenance link mismatch")
        for record in self.sampler_bias:
            if record.d_enabled != self.model_selection.d_enabled:
                raise ValueError("calibration labels cannot change frozen d_enabled")
            matching_locations = [
                item
                for item in self.location_scale
                if item.key.semantic_sha256 == record.location_scale_key_sha256
            ]
            if len(matching_locations) != 1:
                raise ValueError("sampler-bias record lacks its exact b/c record")
            location = matching_locations[0]
            if (
                location.key.lead_hours != record.key.lead_hours
                or location.key.condition_signature != record.key.condition_signature
            ):
                raise ValueError("sampler-bias and b/c signatures disagree")
        for record in self.spread:
            matching_bias = [
                item
                for item in self.sampler_bias
                if item.key.semantic_sha256 == record.sampler_bias_key_sha256
            ]
            if len(matching_bias) != 1:
                raise ValueError("spread record lacks its exact sampler-bias record")
            bias = matching_bias[0]
            if (
                bias.key.lead_hours != record.key.lead_hours
                or bias.key.condition_signature != record.key.condition_signature
                or bias.key.sampler_core != record.key.ensemble_signature.sampler_core
            ):
                raise ValueError("gamma and d exact signatures disagree")

    def semantic_dict(self) -> dict[str, object]:
        return {
            "calibration_absent_identity": self.calibration_absent_identity,
            "ensemble_probability": [
                record.to_dict() for record in self.ensemble_probability
            ],
            "format_version": CALIBRATION_ARTIFACT_FORMAT,
            "location_scale": [record.to_dict() for record in self.location_scale],
            "model_selection": self.model_selection.to_dict(),
            "provenance_hashes": self.provenance.to_dict(),
            "regression_probability": [
                record.to_dict() for record in self.regression_probability
            ],
            "release_status": self.release_status,
            "sampler_bias": [record.to_dict() for record in self.sampler_bias],
            "split": self.split,
            "spread": [record.to_dict() for record in self.spread],
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.semantic_dict())

    def to_dict(self) -> dict[str, object]:
        result = self.semantic_dict()
        result["semantic_sha256"] = self.semantic_sha256
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CalibrationArtifact":
        if raw.get("format_version") != CALIBRATION_ARTIFACT_FORMAT:
            raise ValueError("unsupported calibration artifact format")
        supplied_hash = str(raw.get("semantic_sha256", ""))
        _require_sha256(supplied_hash, name="calibration semantic digest")
        semantic = dict(raw)
        semantic.pop("semantic_sha256", None)
        if _semantic_sha256(semantic) != supplied_hash:
            raise ValueError("calibration artifact semantic SHA-256 mismatch")
        provenance_raw = _as_mapping(raw["provenance_hashes"])
        result = cls(
            split=str(raw["split"]),
            release_status=str(raw["release_status"]),  # type: ignore[arg-type]
            model_selection=FrozenModelSelectionDecision.from_dict(
                _as_mapping(raw["model_selection"])
            ),
            provenance=CalibrationProvenance.from_mapping(
                {str(name): str(value) for name, value in provenance_raw.items()}
            ),
            location_scale=tuple(
                LocationScaleRecord.from_dict(_as_mapping(value))
                for value in _as_list(raw["location_scale"], name="location_scale")
            ),
            sampler_bias=tuple(
                SamplerBiasRecord.from_dict(_as_mapping(value))
                for value in _as_list(raw["sampler_bias"], name="sampler_bias")
            ),
            spread=tuple(
                SpreadRecord.from_dict(_as_mapping(value))
                for value in _as_list(raw["spread"], name="spread")
            ),
            regression_probability=tuple(
                RegressionProbabilityRecord.from_dict(_as_mapping(value))
                for value in _as_list(
                    raw["regression_probability"], name="regression_probability"
                )
            ),
            ensemble_probability=tuple(
                EnsembleProbabilityRecord.from_dict(_as_mapping(value))
                for value in _as_list(
                    raw["ensemble_probability"], name="ensemble_probability"
                )
            ),
            calibration_absent_identity=raw["calibration_absent_identity"],  # type: ignore[arg-type]
        )
        if result.to_dict() != dict(raw):
            raise ValueError("calibration artifact typed round trip is not canonical")
        return result


class CalibrationArtifactBuilder:
    """Fit a table under one immutable selection decision and provenance set."""

    def __init__(
        self,
        *,
        split: str,
        model_selection: FrozenModelSelectionDecision,
        provenance_hashes: Mapping[str, str],
    ) -> None:
        if split != CALIBRATION_SPLIT:
            raise ValueError("final calibration fitting requires split='calibration'")
        if not isinstance(model_selection, FrozenModelSelectionDecision):
            raise TypeError("model_selection must be frozen before fitting")
        if not provenance_hashes:
            raise ValueError("at least one calibration input provenance hash is required")
        hashes = dict(provenance_hashes)
        for name, expected in (
            ("model_selection_decision_sha256", model_selection.decision_sha256),
            ("selected_architecture_sha256", model_selection.architecture_sha256),
        ):
            supplied = hashes.setdefault(name, expected)
            if supplied != expected:
                raise ValueError(f"{name} conflicts with the frozen decision")
        self.split = split
        self.model_selection = model_selection
        self.provenance = CalibrationProvenance.from_mapping(hashes)
        self._location: dict[str, LocationScaleRecord] = {}
        self._bias: dict[str, SamplerBiasRecord] = {}
        self._spread: dict[str, SpreadRecord] = {}
        self._p: dict[str, RegressionProbabilityRecord] = {}
        self._q: dict[str, EnsembleProbabilityRecord] = {}

    def _audit(
        self,
        evidence: Mapping[str, PoolingEvidence],
        *,
        probability_gate: bool,
    ) -> PoolingAudit:
        return build_pooling_audit(evidence, probability_gate=probability_gate)

    @staticmethod
    def _insert(target: dict[str, object], key_sha256: str, record: object) -> None:
        if key_sha256 in target:
            raise ValueError("duplicate exact calibration key")
        target[key_sha256] = record

    def fit_location_scale(
        self,
        key: LocationScaleKey,
        *,
        folds: Sequence[FoldCalibrationMoments],
        full_residual: ArrayLike,
        calibration_weight: ArrayLike,
        pooling_evidence: Mapping[str, PoolingEvidence],
    ) -> LocationScaleRecord:
        calibration = fit_location_total_scale(
            folds=folds,
            full_residual=full_residual,
            calibration_weight=calibration_weight,
            split=self.split,
        )
        if len(calibration.fold_weights) != len(key.fold_checkpoint_sha256s):
            raise ValueError("fold moment count and b/c checkpoint key disagree")
        record = LocationScaleRecord(
            key,
            calibration,
            self._audit(pooling_evidence, probability_gate=False),
            self.provenance.semantic_sha256,
        )
        self._insert(self._location, key.semantic_sha256, record)
        return record

    def fit_sampler(
        self,
        *,
        bias_key: SamplerBiasKey,
        spread_key: SpreadKey,
        location_scale_key: LocationScaleKey,
        restored_members: ArrayLike,
        full_residual: ArrayLike,
        calibration_weight: ArrayLike,
        pooling_evidence: Mapping[str, PoolingEvidence],
    ) -> tuple[SamplerBiasRecord, SpreadRecord]:
        if (
            bias_key.lead_hours != spread_key.lead_hours
            or bias_key.condition_signature != spread_key.condition_signature
            or bias_key.sampler_core != spread_key.ensemble_signature.sampler_core
        ):
            raise ValueError("d and gamma keys must share an exact sampler signature")
        if (
            location_scale_key.lead_hours != bias_key.lead_hours
            or location_scale_key.condition_signature != bias_key.condition_signature
        ):
            raise ValueError("sampler and b/c lead/condition keys disagree")
        try:
            location_record = self._location[location_scale_key.semantic_sha256]
        except KeyError as error:
            raise KeyError("fit and freeze b/c before d/gamma") from error
        fitted: SamplerCalibration = fit_sampler_bias_and_spread(
            restored_members=restored_members,
            full_residual=full_residual,
            calibration_weight=calibration_weight,
            location_scale=location_record.calibration,
            d_enabled=self.model_selection.d_enabled,
            split=self.split,
        )
        pooling = self._audit(pooling_evidence, probability_gate=False)
        bias = SamplerBiasRecord(
            key=bias_key,
            sampler_bias_d=fitted.sampler_bias_d,
            d_enabled=fitted.d_enabled,
            bias_fraction=fitted.bias_fraction,
            location_scale_key_sha256=location_scale_key.semantic_sha256,
            pooling=pooling,
            provenance_sha256=self.provenance.semantic_sha256,
        )
        spread = SpreadRecord(
            key=spread_key,
            spread_gamma=fitted.spread_gamma,
            sampler_bias_key_sha256=bias_key.semantic_sha256,
            pooling=pooling,
            provenance_sha256=self.provenance.semantic_sha256,
        )
        self._insert(self._bias, bias_key.semantic_sha256, bias)
        self._insert(self._spread, spread_key.semantic_sha256, spread)
        return bias, spread

    def fit_regression_probability(
        self,
        key: RegressionProbabilityKey,
        *,
        probability: ArrayLike,
        observation: ArrayLike,
        weight: ArrayLike,
        pooling_evidence: Mapping[str, PoolingEvidence],
    ) -> RegressionProbabilityRecord:
        pooling = self._audit(pooling_evidence, probability_gate=True)
        fitted = fit_monotone_logit_linear_probability(
            probability,
            observation,
            weight,
            split=self.split,
            pooling=pooling.decision,
        )
        record = RegressionProbabilityRecord(
            key, fitted, pooling, self.provenance.semantic_sha256
        )
        self._insert(self._p, key.semantic_sha256, record)
        return record

    def fit_ensemble_probability(
        self,
        key: EnsembleProbabilityKey,
        *,
        probability: ArrayLike,
        observation: ArrayLike,
        weight: ArrayLike,
        pooling_evidence: Mapping[str, PoolingEvidence],
    ) -> EnsembleProbabilityRecord:
        pooling = self._audit(pooling_evidence, probability_gate=True)
        fitted = fit_monotone_logit_linear_probability(
            probability,
            observation,
            weight,
            split=self.split,
            pooling=pooling.decision,
        )
        record = EnsembleProbabilityRecord(
            key, fitted, pooling, self.provenance.semantic_sha256
        )
        self._insert(self._q, key.semantic_sha256, record)
        return record

    def build(
        self,
        *,
        release_status: ReleaseStatus,
        calibration_absent_identity: bool = False,
    ) -> CalibrationArtifact:
        def ordered(values: Mapping[str, object]) -> tuple[object, ...]:
            return tuple(values[key] for key in sorted(values))

        return CalibrationArtifact(
            split=self.split,
            release_status=release_status,
            model_selection=self.model_selection,
            provenance=self.provenance,
            location_scale=ordered(self._location),  # type: ignore[arg-type]
            sampler_bias=ordered(self._bias),  # type: ignore[arg-type]
            spread=ordered(self._spread),  # type: ignore[arg-type]
            regression_probability=ordered(self._p),  # type: ignore[arg-type]
            ensemble_probability=ordered(self._q),  # type: ignore[arg-type]
            calibration_absent_identity=calibration_absent_identity,
        )


def publish_calibration_artifact(path: Path, artifact: CalibrationArtifact) -> str:
    """Atomically publish a canonical artifact without ever replacing a path."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be a CalibrationArtifact")
    if artifact.split != CALIBRATION_SPLIT:
        raise ValueError("only independent calibration artifacts may be published")
    envelope = artifact.to_dict()
    payload = _canonical_bytes(envelope) + b"\n"

    # Validate the exact serialized representation before it can become visible.
    decoded = json.loads(payload)
    round_trip = CalibrationArtifact.from_dict(_as_mapping(decoded))
    if round_trip != artifact or round_trip.semantic_sha256 != artifact.semantic_sha256:
        raise ValueError("calibration artifact failed typed publication round trip")

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"immutable calibration artifact already exists: {destination}"
            ) from None
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

    persisted = read_calibration_artifact(destination)
    if persisted.semantic_sha256 != artifact.semantic_sha256:
        raise RuntimeError("published calibration artifact did not verify")
    return artifact.semantic_sha256


def read_calibration_artifact(path: Path) -> CalibrationArtifact:
    """Read, canonical-byte check, digest check, and typed round-trip validate."""

    source = Path(path)
    payload = source.read_bytes()
    try:
        raw = _as_mapping(json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibration artifact is not valid UTF-8 JSON") from error
    artifact = CalibrationArtifact.from_dict(raw)
    canonical = _canonical_bytes(artifact.to_dict()) + b"\n"
    if payload != canonical:
        raise ValueError("calibration artifact JSON is not canonical")
    return artifact


class CalibrationResolver:
    """Fail-closed exact lookups and application for release or development."""

    def __init__(
        self,
        artifact: CalibrationArtifact | None,
        *,
        development_mode: bool = False,
        frozen_model_selection: FrozenModelSelectionDecision | None = None,
    ) -> None:
        if artifact is None:
            if not development_mode:
                raise FileNotFoundError(
                    "calibration artifact is required outside explicit development mode"
                )
            if frozen_model_selection is None:
                raise ValueError(
                    "development identity still requires a frozen model-selection decision"
                )
        elif artifact.calibration_absent_identity:
            if not development_mode:
                raise ValueError(
                    "absent-calibration artifact requires explicit development_mode=True"
                )
            if frozen_model_selection is not None:
                raise ValueError("development artifact already carries its frozen decision")
        elif development_mode:
            raise ValueError("development identity mode requires absent calibration")
        elif frozen_model_selection is not None:
            raise ValueError("artifact already carries its frozen model-selection decision")
        self.artifact = artifact
        self.development_mode = development_mode
        self._identity_mode = artifact is None or artifact.calibration_absent_identity
        self.model_selection = (
            artifact.model_selection if artifact is not None else frozen_model_selection
        )

    @classmethod
    def for_complete_release(
        cls, artifact: CalibrationArtifact | None
    ) -> "CalibrationResolver":
        if artifact is None:
            raise FileNotFoundError("a complete release requires a calibration artifact")
        if artifact.release_status != "complete" or artifact.calibration_absent_identity:
            raise ValueError("development identity cannot be used as a complete release")
        return cls(artifact)

    @classmethod
    def development_identity(
        cls, model_selection: FrozenModelSelectionDecision
    ) -> "CalibrationResolver":
        return cls(
            None,
            development_mode=True,
            frozen_model_selection=model_selection,
        )

    @staticmethod
    def _lookup(records: Sequence[object], key_sha256: str, *, kind: str) -> object:
        matches = [
            record
            for record in records
            if record.key.semantic_sha256 == key_sha256  # type: ignore[attr-defined]
        ]
        if len(matches) != 1:
            raise KeyError(f"no exact {kind} calibration for signature {key_sha256}")
        return matches[0]

    def location(self, key: LocationScaleKey) -> LocationScaleRecord:
        if self._identity_mode:
            raise LookupError("development identity has no published b/c record")
        assert self.artifact is not None
        return self._lookup(
            self.artifact.location_scale, key.semantic_sha256, kind="b/c"
        )  # type: ignore[return-value]

    def bias(self, key: SamplerBiasKey) -> SamplerBiasRecord:
        if self._identity_mode:
            raise LookupError("development identity has no published d record")
        assert self.artifact is not None
        return self._lookup(
            self.artifact.sampler_bias, key.semantic_sha256, kind="d"
        )  # type: ignore[return-value]

    def gamma(self, key: SpreadKey) -> SpreadRecord:
        if self._identity_mode:
            raise LookupError("development identity has no published gamma record")
        assert self.artifact is not None
        return self._lookup(
            self.artifact.spread, key.semantic_sha256, kind="gamma"
        )  # type: ignore[return-value]

    def apply_residual(
        self,
        restored_members: Tensor,
        *,
        location_key: LocationScaleKey,
        bias_key: SamplerBiasKey,
        spread_key: SpreadKey,
    ) -> Tensor:
        if (
            location_key.lead_hours != bias_key.lead_hours
            or location_key.condition_signature != bias_key.condition_signature
            or bias_key.lead_hours != spread_key.lead_hours
            or bias_key.condition_signature != spread_key.condition_signature
            or bias_key.sampler_core != spread_key.ensemble_signature.sampler_core
        ):
            raise ValueError("residual calibration keys have mismatched exact signatures")
        if self._identity_mode:
            if restored_members.ndim != 5:
                raise ValueError("residual members must have shape [B,N,1,H,W]")
            if restored_members.dtype is not torch.float32:
                raise TypeError("residual members must remain float32")
            # The explicit development identity is a true pass-through.  Avoid
            # introducing rounding via otherwise equivalent mean/anomaly math.
            return restored_members
        assert self.artifact is not None
        location = self.location(location_key)
        bias = self.bias(bias_key)
        spread = self.gamma(spread_key)
        if bias.location_scale_key_sha256 != location_key.semantic_sha256:
            raise ValueError("d record is not linked to the requested b/c signature")
        if spread.sampler_bias_key_sha256 != bias_key.semantic_sha256:
            raise ValueError("gamma record is not linked to the requested d signature")
        return apply_residual_calibration(
            restored_members,
            location_b=location.calibration.location_b,
            total_scale_c=location.calibration.total_scale_c,
            sampler_bias_d=bias.sampler_bias_d,
            spread_gamma=spread.spread_gamma,
        )

    def apply_regression_probability(
        self, probability: Tensor, *, key: RegressionProbabilityKey
    ) -> Tensor:
        if self._identity_mode:
            _validate_identity_probability(probability)
            return probability
        assert self.artifact is not None
        record = self._lookup(
            self.artifact.regression_probability,
            key.semantic_sha256,
            kind="p_cal",
        )
        assert isinstance(record, RegressionProbabilityRecord)
        return monotone_logit_linear_probability(
            probability,
            alpha=record.calibration.alpha,
            beta_raw=record.calibration.beta_raw,
        )

    def apply_ensemble_probability(
        self, probability: Tensor, *, key: EnsembleProbabilityKey
    ) -> Tensor:
        if self._identity_mode:
            _validate_identity_probability(probability)
            return probability
        assert self.artifact is not None
        record = self._lookup(
            self.artifact.ensemble_probability,
            key.semantic_sha256,
            kind="q_cal",
        )
        assert isinstance(record, EnsembleProbabilityRecord)
        return monotone_logit_linear_probability(
            probability,
            alpha=record.calibration.alpha,
            beta_raw=record.calibration.beta_raw,
        )


def _validate_identity_probability(probability: Tensor) -> None:
    if probability.dtype is not torch.float32:
        raise TypeError("probability must remain float32")
    if not bool(torch.isfinite(probability).all().item()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any().item()
    ):
        raise ValueError("probability must be finite and lie in [0,1]")


# Descriptive aliases for callers that use read/write rather than publish terms.
write_calibration_artifact = publish_calibration_artifact
load_calibration_artifact = read_calibration_artifact


__all__ = [
    "CALIBRATION_ARTIFACT_FORMAT",
    "MONOTONE_LOGIT_LINEAR_FAMILY",
    "CalibrationArtifact",
    "CalibrationArtifactBuilder",
    "CalibrationProvenance",
    "CalibrationResolver",
    "EnsembleProbabilityKey",
    "EnsembleProbabilityRecord",
    "FrozenModelSelectionDecision",
    "LocationScaleKey",
    "LocationScaleRecord",
    "OFFICIAL_ENSEMBLE_THRESHOLDS_MM",
    "OFFICIAL_LEADS_HOURS",
    "PoolingAudit",
    "PoolingEvidence",
    "PoolingLevelAudit",
    "RegressionProbabilityKey",
    "RegressionProbabilityRecord",
    "SamplerBiasKey",
    "SamplerBiasRecord",
    "SpreadKey",
    "SpreadRecord",
    "build_pooling_audit",
    "load_calibration_artifact",
    "publish_calibration_artifact",
    "read_calibration_artifact",
    "write_calibration_artifact",
]

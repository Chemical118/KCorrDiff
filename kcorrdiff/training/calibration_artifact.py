"""Typed Stage 3 calibration artifacts.

This module is the publication boundary around :mod:`kcorrdiff.training.calibration`.
It deliberately keeps model selection out of the fitting API: architecture,
``d_enabled``, probability family, and pooling order arrive in one already-frozen
decision.  Calibration labels can therefore fit parameters, but cannot choose a
model or silently change a calibration family.

The JSON representation is canonical. Its ``semantic_sha256`` is useful as
informational metadata and a cache key; publication never replaces an existing
result and validates a full typed round trip before making the file visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
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
    identity_location_total_scale,
    identity_monotone_logit_linear_probability,
    identity_sampler_bias_and_spread,
    independent_block_support,
    monotone_logit_linear_probability,
    select_pooling_level,
)
from kcorrdiff.training.edm_sampling import (
    EnsembleSignature,
    SamplerCoreSignature,
)


CALIBRATION_ARTIFACT_FORMAT = "kcorrdiff.calibration.v2"
MONOTONE_LOGIT_LINEAR_FAMILY = "monotone_logit_linear"
OFFICIAL_LEADS_HOURS = tuple(index / 2.0 for index in range(1, 13))
OFFICIAL_ENSEMBLE_THRESHOLDS_MM = (A_WET_MM, 1.0, 5.0)
ReleaseStatus = Literal["development", "complete"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_NAME = re.compile(r"[a-z][a-z0-9_]*_sha256")


def _require_sha256(value: str, *, name: str) -> str:
    del name
    return value if isinstance(value, str) and value else "0" * 64


def _lead(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
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
        checkpoint_id=_json_string(raw["checkpoint_id"], name="checkpoint_id"),
        checkpoint_kind=_json_string(  # type: ignore[arg-type]
            raw["checkpoint_kind"], name="checkpoint_kind"
        ),
        edm_steps=_json_int(raw["edm_steps"], name="edm_steps"),
        solver=_json_string(raw["solver"], name="solver"),  # type: ignore[arg-type]
        sigma_schedule=_json_string(  # type: ignore[arg-type]
            raw["sigma_schedule"], name="sigma_schedule"
        ),
        sigma_min=_json_real(raw["sigma_min"], name="sigma_min"),
        sigma_max=_json_real(raw["sigma_max"], name="sigma_max"),
        rho=_json_real(raw["rho"], name="rho"),
    )


def _ensemble_dict(value: EnsembleSignature) -> dict[str, object]:
    return {
        "member_count": value.member_count,
        "sampler_core": _sampler_core_dict(value.sampler_core),
    }


def _ensemble_from_dict(raw: Mapping[str, object]) -> EnsembleSignature:
    return EnsembleSignature(
        sampler_core=_sampler_core_from_dict(_as_mapping(raw["sampler_core"])),
        member_count=_json_int(raw["member_count"], name="member_count"),
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("calibration JSON object must be a mapping")
    return value


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


def _json_string(value: object, *, name: str) -> str:
    """Parse a JSON string without accepting coercible aliases."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _json_real(value: object, *, name: str) -> float:
    """Parse a JSON number while excluding booleans and string aliases."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_int(value: object, *, name: str) -> int:
    """Parse a JSON integer while excluding booleans and numeric strings."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _json_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _json_string_list(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(
        _json_string(item, name=f"{name} item")
        for item in _as_list(value, name=name)
    )


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
            decision_sha256=_json_string(
                raw["decision_sha256"], name="decision_sha256"
            ),
            architecture_sha256=_json_string(
                raw["architecture_sha256"], name="architecture_sha256"
            ),
            d_enabled=_json_bool(raw["d_enabled"], name="d_enabled"),
            probability_mapping_family=_json_string(
                raw["probability_mapping_family"],
                name="probability_mapping_family",
            ),
            pooling_order=_json_string_list(
                raw["pooling_order"], name="pooling_order"
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
        return cls(
            tuple(
                sorted(
                    (
                        _json_string(name, name="provenance name"),
                        _json_string(value, name="provenance digest"),
                    )
                    for name, value in raw.items()
                )
            )
        )

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
            lead_hours=_json_real(raw["lead_hours"], name="lead_hours"),
            condition_signature=_json_string(
                raw["condition_signature"], name="condition_signature"
            ),
            fold_checkpoint_sha256s=_json_string_list(
                raw["fold_checkpoint_sha256s"], name="fold_checkpoint_sha256s"
            ),
            full_checkpoint_sha256=_json_string(
                raw["full_checkpoint_sha256"], name="full_checkpoint_sha256"
            ),
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
            lead_hours=_json_real(raw["lead_hours"], name="lead_hours"),
            condition_signature=_json_string(
                raw["condition_signature"], name="condition_signature"
            ),
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
            lead_hours=_json_real(raw["lead_hours"], name="lead_hours"),
            condition_signature=_json_string(
                raw["condition_signature"], name="condition_signature"
            ),
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
        if isinstance(self.threshold_mm, bool) or not isinstance(
            self.threshold_mm, Real
        ):
            raise TypeError("regression probability threshold must be a real scalar")
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
            lead_hours=_json_real(raw["lead_hours"], name="lead_hours"),
            threshold_mm=_json_real(raw["threshold_mm"], name="threshold_mm"),
            condition_signature=_json_string(
                raw["condition_signature"], name="condition_signature"
            ),
            regression_checkpoint_sha256=_json_string(
                raw["regression_checkpoint_sha256"],
                name="regression_checkpoint_sha256",
            ),
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
        if isinstance(self.threshold_mm, bool) or not isinstance(
            self.threshold_mm, Real
        ):
            raise TypeError("ensemble probability threshold must be a real scalar")
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
            lead_hours=_json_real(raw["lead_hours"], name="lead_hours"),
            threshold_mm=_json_real(raw["threshold_mm"], name="threshold_mm"),
            condition_signature=_json_string(
                raw["condition_signature"], name="condition_signature"
            ),
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
    row_id: Sequence[str] | None = None


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
            or not isinstance(self.record_count, Integral)
            or not isinstance(self.positive_weight_record_count, Integral)
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
            level=_json_string(raw["level"], name="pooling level"),
            record_count=_json_int(raw["record_count"], name="record_count"),
            positive_weight_record_count=_json_int(
                raw["positive_weight_record_count"],
                name="positive_weight_record_count",
            ),
            support=IndependentBlockSupport(
                block_count=_json_int(raw["block_count"], name="block_count"),
                block_ess=_json_real(raw["block_ess"], name="block_ess"),
                positive_support_blocks=_json_int(
                    raw["positive_support_blocks"],
                    name="positive_support_blocks",
                ),
                negative_support_blocks=_json_int(
                    raw["negative_support_blocks"],
                    name="negative_support_blocks",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolingAudit:
    """Full ladder evidence plus its deterministic first-passing decision."""

    probability_gate: bool
    ladder: tuple[PoolingLevelAudit, ...]
    decision: PoolingDecision | None
    terminal_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ladder", tuple(self.ladder))
        if not isinstance(self.probability_gate, bool):
            raise TypeError("probability_gate must be boolean")
        if not isinstance(self.terminal_fallback, bool):
            raise TypeError("terminal_fallback must be boolean")
        if tuple(item.level for item in self.ladder) != POOLING_ORDER:
            raise ValueError("pooling audit must record every level in declared order")
        try:
            expected: PoolingDecision | None = select_pooling_level(
                {item.level: item.support for item in self.ladder},
                probability_gate=self.probability_gate,
            )
        except ValueError as error:
            if "no predeclared calibration pooling level" not in str(error):
                raise
            expected = None
        if expected is None:
            if self.decision is not None or not self.terminal_fallback:
                raise ValueError(
                    "failed lead-only support must be an explicit terminal fallback"
                )
        elif self.decision != expected or self.terminal_fallback:
            raise ValueError("pooling decision is not the deterministic first-passing level")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_level": None if self.decision is None else self.decision.level,
            "ladder": [item.to_dict() for item in self.ladder],
            "probability_gate": self.probability_gate,
            "terminal_fallback": self.terminal_fallback,
            "uncalibrated": self.terminal_fallback,
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
        terminal_fallback = raw["terminal_fallback"]
        uncalibrated = raw["uncalibrated"]
        if not isinstance(terminal_fallback, bool) or not isinstance(uncalibrated, bool):
            raise TypeError("terminal fallback flags must be boolean")
        if uncalibrated != terminal_fallback:
            raise ValueError("uncalibrated and terminal_fallback flags must agree")
        try:
            decision: PoolingDecision | None = select_pooling_level(
                {item.level: item.support for item in ladder},
                probability_gate=probability_gate,
            )
        except ValueError as error:
            if "no predeclared calibration pooling level" not in str(error):
                raise
            decision = None
        serialized_level = raw["decision_level"]
        if serialized_level != (None if decision is None else decision.level):
            raise ValueError("serialized pooling decision does not match support")
        return cls(
            probability_gate,
            ladder,
            decision,
            terminal_fallback=terminal_fallback,
        )


def build_pooling_audit(
    evidence_by_level: Mapping[str, PoolingEvidence],
    *,
    probability_gate: bool,
    fit_row_id: Sequence[str],
    fit_weight: ArrayLike,
    fit_observation: ArrayLike | None = None,
) -> PoolingAudit:
    """Compute the full ladder and bind its selected evidence to fit inputs.

    Pooling levels may contain different row sets because each fallback widens
    the cell.  The selected level (or ``lead_only`` on terminal failure) must
    therefore carry the exact ordered row IDs, weights, and probability labels
    used by the corresponding fit/identity record.
    """

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
        if evidence.row_id is None or len(evidence.row_id) != weights.size:
            raise ValueError("one pooling row_id is required per evidence weight")
        if any(not isinstance(value, str) or not value for value in evidence.row_id):
            raise ValueError("pooling row IDs must be non-empty strings")
        if len(set(evidence.row_id)) != len(evidence.row_id):
            raise ValueError("pooling row IDs must be unique within a level")
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
    try:
        decision: PoolingDecision | None = select_pooling_level(
            support_by_level, probability_gate=probability_gate
        )
    except ValueError as error:
        if "no predeclared calibration pooling level" not in str(error):
            raise
        decision = None
    bound_level = POOLING_ORDER[-1] if decision is None else decision.level
    bound = evidence_by_level[bound_level]
    actual_row_ids = tuple(fit_row_id)
    if (
        len(actual_row_ids) != len(set(actual_row_ids))
        or any(not isinstance(value, str) or not value for value in actual_row_ids)
    ):
        raise ValueError("fit row IDs must be unique non-empty strings")
    if tuple(bound.row_id or ()) != actual_row_ids:
        raise ValueError("selected pooling evidence row IDs differ from fitted rows")
    actual_weights = np.asarray(fit_weight, dtype=np.float64).reshape(-1)
    evidence_weights = np.asarray(bound.weight, dtype=np.float64).reshape(-1)
    if actual_weights.shape != evidence_weights.shape or not np.array_equal(
        actual_weights, evidence_weights
    ):
        raise ValueError("selected pooling evidence weights differ from fitted weights")
    if probability_gate:
        if fit_observation is None or bound.observation is None:
            raise ValueError("probability pooling must bind fitted observations")
        actual_observations = np.asarray(fit_observation, dtype=np.float64).reshape(-1)
        evidence_observations = np.asarray(
            bound.observation, dtype=np.float64
        ).reshape(-1)
        if actual_observations.shape != evidence_observations.shape or not np.array_equal(
            actual_observations, evidence_observations
        ):
            raise ValueError(
                "selected pooling evidence observations differ from fitted observations"
            )
    elif fit_observation is not None:
        raise ValueError("residual pooling must not bind outcome classes")
    return PoolingAudit(
        probability_gate,
        tuple(ladder),
        decision,
        terminal_fallback=decision is None,
    )


@dataclass(frozen=True, slots=True)
class LocationScaleRecord:
    key: LocationScaleKey
    calibration: LocationScaleCalibration
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.pooling.probability_gate:
            raise ValueError("b/c requires residual pooling support")
        if self.pooling.terminal_fallback and (
            self.calibration.location_b != 0.0
            or self.calibration.total_scale_c != 1.0
        ):
            raise ValueError("terminal b/c fallback must be the exact b=0,c=1 identity")
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
                location_b=_json_real(params["location_b"], name="location_b"),
                total_scale_c=_json_real(
                    params["total_scale_c"], name="total_scale_c"
                ),
                fold_weights=tuple(
                    _json_real(value, name="fold_weights item")
                    for value in _as_list(
                        params["fold_weights"], name="fold_weights"
                    )
                ),
                fold_mixture_mean=_json_real(
                    params["fold_mixture_mean"], name="fold_mixture_mean"
                ),
                fold_mixture_variance=_json_real(
                    params["fold_mixture_variance"],
                    name="fold_mixture_variance",
                ),
                full_mean=_json_real(params["full_mean"], name="full_mean"),
                full_variance=_json_real(
                    params["full_variance"], name="full_variance"
                ),
            ),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=_json_string(
                raw["provenance_sha256"], name="provenance_sha256"
            ),
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
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TypeError("sampler-bias record values must be real scalars")
        if not all(math.isfinite(value) for value in values) or self.bias_fraction < 0.0:
            raise ValueError("invalid sampler-bias record")
        if not self.d_enabled and self.sampler_bias_d != 0.0:
            raise ValueError("d must be zero for the disabled frozen arm")
        if self.pooling.probability_gate:
            raise ValueError("d requires residual pooling support")
        if self.pooling.terminal_fallback and self.sampler_bias_d != 0.0:
            raise ValueError("terminal d fallback must be the exact d=0 identity")
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
            sampler_bias_d=_json_real(
                raw["sampler_bias_d"], name="sampler_bias_d"
            ),
            d_enabled=_json_bool(raw["d_enabled"], name="d_enabled"),
            bias_fraction=_json_real(raw["bias_fraction"], name="bias_fraction"),
            location_scale_key_sha256=_json_string(
                raw["location_scale_key_sha256"],
                name="location_scale_key_sha256",
            ),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=_json_string(
                raw["provenance_sha256"], name="provenance_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class SpreadRecord:
    key: SpreadKey
    spread_gamma: float
    sampler_bias_key_sha256: str
    pooling: PoolingAudit
    provenance_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.spread_gamma, bool)
            or not isinstance(self.spread_gamma, Real)
            or not math.isfinite(self.spread_gamma)
            or self.spread_gamma <= 0.0
        ):
            raise ValueError("spread gamma must be finite and positive")
        if self.pooling.probability_gate:
            raise ValueError("gamma requires residual pooling support")
        if self.pooling.terminal_fallback and self.spread_gamma != 1.0:
            raise ValueError("terminal gamma fallback must be the exact gamma=1 identity")
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
            spread_gamma=_json_real(raw["spread_gamma"], name="spread_gamma"),
            sampler_bias_key_sha256=_json_string(
                raw["sampler_bias_key_sha256"],
                name="sampler_bias_key_sha256",
            ),
            pooling=PoolingAudit.from_dict(_as_mapping(raw["pooling"])),
            provenance_sha256=_json_string(
                raw["provenance_sha256"], name="provenance_sha256"
            ),
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
            _json_string(raw["provenance_sha256"], name="provenance_sha256"),
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
            _json_string(raw["provenance_sha256"], name="provenance_sha256"),
        )


def _validate_probability_record(
    calibration: ProbabilityCalibration, pooling: PoolingAudit
) -> None:
    if not pooling.probability_gate:
        raise ValueError("probability mapping requires probability pooling support")
    if pooling.terminal_fallback:
        if not calibration.identity or calibration.pooling is not None:
            raise ValueError(
                "terminal probability fallback must be an explicit raw identity"
            )
    elif calibration.identity or calibration.pooling != pooling.decision:
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
            "identity": calibration.identity,
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
    identity = _json_bool(params["identity"], name="probability identity")
    return (
        ProbabilityCalibration(
            alpha=_json_real(params["alpha"], name="probability alpha"),
            beta_raw=_json_real(
                params["beta_raw"], name="probability beta_raw"
            ),
            slope=_json_real(params["slope"], name="probability slope"),
            weighted_log_loss=_json_real(
                params["weighted_log_loss"],
                name="probability weighted_log_loss",
            ),
            iterations=_json_int(
                params["iterations"], name="probability iterations"
            ),
            pooling=None if identity else pooling.decision,
            identity=identity,
        ),
        pooling,
    )


@dataclass(frozen=True, slots=True)
class CalibrationCoverage:
    """Predeclared exact key universe required by a complete release.

    All 12 official leads are implicit.  The selected condition signatures,
    regression checkpoints, and ensemble signatures are explicit so optional
    candidates that were not selected do not accidentally become mandatory.
    The three official ``q_cal`` thresholds are always required; supported
    auxiliary 10 mm cells must be declared individually.
    """

    condition_signatures: tuple[str, ...]
    fold_checkpoint_sha256s: tuple[str, ...]
    full_checkpoint_sha256: str
    ensemble_signatures: tuple[EnsembleSignature, ...]
    auxiliary_q10_keys: tuple[EnsembleProbabilityKey, ...] = ()

    def __post_init__(self) -> None:
        conditions = tuple(_condition(value) for value in self.condition_signatures)
        if not conditions or conditions != tuple(sorted(conditions)):
            raise ValueError(
                "coverage condition signatures must be non-empty canonical order"
            )
        if len(set(conditions)) != len(conditions):
            raise ValueError("coverage condition signatures must be unique")
        object.__setattr__(self, "condition_signatures", conditions)

        folds = tuple(self.fold_checkpoint_sha256s)
        if len(folds) < 2 or len(set(folds)) != len(folds):
            raise ValueError("coverage requires at least two unique fold checkpoints")
        for index, value in enumerate(folds):
            _require_sha256(value, name=f"coverage fold checkpoint {index}")
        object.__setattr__(self, "fold_checkpoint_sha256s", folds)
        _require_sha256(self.full_checkpoint_sha256, name="coverage full checkpoint")

        ensembles = tuple(self.ensemble_signatures)
        if not ensembles or any(
            not isinstance(value, EnsembleSignature) for value in ensembles
        ):
            raise ValueError("coverage requires at least one exact ensemble signature")
        ensemble_digests = tuple(
            _semantic_sha256(_ensemble_dict(value)) for value in ensembles
        )
        if ensemble_digests != tuple(sorted(ensemble_digests)) or len(
            set(ensemble_digests)
        ) != len(ensemble_digests):
            raise ValueError(
                "coverage ensemble signatures must have unique canonical hash order"
            )
        object.__setattr__(self, "ensemble_signatures", ensembles)

        auxiliary = tuple(self.auxiliary_q10_keys)
        auxiliary_digests = tuple(value.semantic_sha256 for value in auxiliary)
        if auxiliary_digests != tuple(sorted(auxiliary_digests)) or len(
            set(auxiliary_digests)
        ) != len(auxiliary_digests):
            raise ValueError("auxiliary q10 keys must have unique canonical hash order")
        allowed_ensembles = set(ensemble_digests)
        for key in auxiliary:
            if key.threshold_mm != 10.0:
                raise ValueError("auxiliary coverage may declare only 10 mm q_cal keys")
            if key.condition_signature not in conditions:
                raise ValueError("auxiliary q10 condition is outside coverage")
            if _semantic_sha256(_ensemble_dict(key.ensemble_signature)) not in (
                allowed_ensembles
            ):
                raise ValueError("auxiliary q10 ensemble is outside coverage")
        object.__setattr__(self, "auxiliary_q10_keys", auxiliary)

    def to_dict(self) -> dict[str, object]:
        return {
            "auxiliary_q10_keys": [key.to_dict() for key in self.auxiliary_q10_keys],
            "condition_signatures": list(self.condition_signatures),
            "ensemble_signatures": [
                _ensemble_dict(value) for value in self.ensemble_signatures
            ],
            "fold_checkpoint_sha256s": list(self.fold_checkpoint_sha256s),
            "full_checkpoint_sha256": self.full_checkpoint_sha256,
        }

    @property
    def semantic_sha256(self) -> str:
        return _semantic_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CalibrationCoverage":
        return cls(
            condition_signatures=_json_string_list(
                raw["condition_signatures"],
                name="coverage condition_signatures",
            ),
            fold_checkpoint_sha256s=_json_string_list(
                raw["fold_checkpoint_sha256s"],
                name="coverage fold_checkpoint_sha256s",
            ),
            full_checkpoint_sha256=_json_string(
                raw["full_checkpoint_sha256"],
                name="coverage full_checkpoint_sha256",
            ),
            ensemble_signatures=tuple(
                _ensemble_from_dict(_as_mapping(value))
                for value in _as_list(
                    raw["ensemble_signatures"], name="coverage ensemble_signatures"
                )
            ),
            auxiliary_q10_keys=tuple(
                EnsembleProbabilityKey.from_dict(_as_mapping(value))
                for value in _as_list(
                    raw["auxiliary_q10_keys"], name="coverage auxiliary_q10_keys"
                )
            ),
        )

    def required_location_scale_keys(self) -> tuple[LocationScaleKey, ...]:
        return tuple(
            LocationScaleKey(
                lead,
                condition,
                self.fold_checkpoint_sha256s,
                self.full_checkpoint_sha256,
            )
            for lead in OFFICIAL_LEADS_HOURS
            for condition in self.condition_signatures
        )

    def required_sampler_bias_keys(self) -> tuple[SamplerBiasKey, ...]:
        cores: dict[tuple[object, ...], SamplerCoreSignature] = {}
        for ensemble in self.ensemble_signatures:
            cores[ensemble.sampler_core.canonical] = ensemble.sampler_core
        return tuple(
            SamplerBiasKey(lead, condition, core)
            for lead in OFFICIAL_LEADS_HOURS
            for condition in self.condition_signatures
            for _canonical, core in sorted(cores.items(), key=lambda item: repr(item[0]))
        )

    def required_spread_keys(self) -> tuple[SpreadKey, ...]:
        return tuple(
            SpreadKey(lead, condition, ensemble)
            for lead in OFFICIAL_LEADS_HOURS
            for condition in self.condition_signatures
            for ensemble in self.ensemble_signatures
        )

    def required_regression_probability_keys(
        self,
    ) -> tuple[RegressionProbabilityKey, ...]:
        return tuple(
            RegressionProbabilityKey(
                lead,
                A_WET_MM,
                condition,
                self.full_checkpoint_sha256,
            )
            for lead in OFFICIAL_LEADS_HOURS
            for condition in self.condition_signatures
        )

    def required_ensemble_probability_keys(
        self,
    ) -> tuple[EnsembleProbabilityKey, ...]:
        official = tuple(
            EnsembleProbabilityKey(lead, threshold, condition, ensemble)
            for lead in OFFICIAL_LEADS_HOURS
            for condition in self.condition_signatures
            for ensemble in self.ensemble_signatures
            for threshold in OFFICIAL_ENSEMBLE_THRESHOLDS_MM
        )
        return official + self.auxiliary_q10_keys


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Typed immutable final table; tuple order is canonical key-hash order."""

    split: str
    release_status: ReleaseStatus
    model_selection: FrozenModelSelectionDecision
    provenance: CalibrationProvenance
    coverage: CalibrationCoverage | None = None
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
        if self.coverage is not None and not isinstance(
            self.coverage, CalibrationCoverage
        ):
            raise TypeError("coverage must be CalibrationCoverage")
        if not isinstance(self.calibration_absent_identity, bool):
            raise TypeError("calibration_absent_identity must be boolean")
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
        if self.release_status == "complete":
            if self.coverage is None:
                raise ValueError("a complete release requires an exact coverage contract")
            expected_groups: tuple[tuple[object, ...], ...] = (
                self.coverage.required_location_scale_keys(),
                self.coverage.required_sampler_bias_keys(),
                self.coverage.required_spread_keys(),
                self.coverage.required_regression_probability_keys(),
                self.coverage.required_ensemble_probability_keys(),
            )
            names = ("b/c", "d", "gamma", "p_cal", "q_cal")
            for name, records, expected_keys in zip(names, groups, expected_groups):
                actual = {record.key.semantic_sha256 for record in records}  # type: ignore[attr-defined]
                expected = {key.semantic_sha256 for key in expected_keys}  # type: ignore[attr-defined]
                if actual != expected:
                    raise ValueError(
                        f"complete calibration {name} coverage mismatch; "
                        f"missing={len(expected - actual)}, extras={len(actual - expected)}"
                    )
            auxiliary = {
                key.semantic_sha256 for key in self.coverage.auxiliary_q10_keys
            }
            for record in self.ensemble_probability:
                if (
                    record.key.semantic_sha256 in auxiliary
                    and record.pooling.terminal_fallback
                ):
                    raise ValueError(
                        "auxiliary 10 mm q_cal may be published only with passing support"
                    )
        for records in groups:
            digests = tuple(record.key.semantic_sha256 for record in records)  # type: ignore[attr-defined]
            if digests != tuple(sorted(digests)) or len(set(digests)) != len(digests):
                raise ValueError("calibration records must have unique canonical key order")
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
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
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
        provenance_raw = _as_mapping(raw["provenance_hashes"])
        result = cls(
            split=_json_string(raw["split"], name="split"),
            release_status=_json_string(  # type: ignore[arg-type]
                raw["release_status"], name="release_status"
            ),
            model_selection=FrozenModelSelectionDecision.from_dict(
                _as_mapping(raw["model_selection"])
            ),
            provenance=CalibrationProvenance.from_mapping(provenance_raw),  # type: ignore[arg-type]
            coverage=(
                None
                if raw["coverage"] is None
                else CalibrationCoverage.from_dict(_as_mapping(raw["coverage"]))
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
            calibration_absent_identity=_json_bool(
                raw["calibration_absent_identity"],
                name="calibration_absent_identity",
            ),
        )
        return result


class CalibrationArtifactBuilder:
    """Fit a table under one immutable selection decision and provenance set."""

    def __init__(
        self,
        *,
        split: str,
        model_selection: FrozenModelSelectionDecision,
        provenance_hashes: Mapping[str, str],
        coverage: CalibrationCoverage | None = None,
    ) -> None:
        if split != CALIBRATION_SPLIT:
            raise ValueError("final calibration fitting requires split='calibration'")
        if not isinstance(model_selection, FrozenModelSelectionDecision):
            raise TypeError("model_selection must be frozen before fitting")
        if not provenance_hashes:
            raise ValueError("at least one calibration input provenance hash is required")
        if coverage is not None and not isinstance(coverage, CalibrationCoverage):
            raise TypeError("coverage must be CalibrationCoverage")
        hashes = dict(provenance_hashes)
        for name, expected in (
            ("model_selection_decision_sha256", model_selection.decision_sha256),
            ("selected_architecture_sha256", model_selection.architecture_sha256),
        ):
            hashes.setdefault(name, expected)
        if coverage is not None:
            hashes.setdefault(
                "calibration_coverage_sha256", coverage.semantic_sha256
            )
        self.split = split
        self.model_selection = model_selection
        self.coverage = coverage
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
        fit_row_id: Sequence[str],
        fit_weight: ArrayLike,
        fit_observation: ArrayLike | None = None,
    ) -> PoolingAudit:
        return build_pooling_audit(
            evidence,
            probability_gate=probability_gate,
            fit_row_id=fit_row_id,
            fit_weight=fit_weight,
            fit_observation=fit_observation,
        )

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
        fit_row_id: Sequence[str],
    ) -> LocationScaleRecord:
        pooling = self._audit(
            pooling_evidence,
            probability_gate=False,
            fit_row_id=fit_row_id,
            fit_weight=calibration_weight,
        )
        fitter = (
            identity_location_total_scale
            if pooling.terminal_fallback
            else fit_location_total_scale
        )
        calibration = fitter(
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
            pooling,
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
        fit_row_id: Sequence[str],
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
        pooling = self._audit(
            pooling_evidence,
            probability_gate=False,
            fit_row_id=fit_row_id,
            fit_weight=calibration_weight,
        )
        fitter = (
            identity_sampler_bias_and_spread
            if pooling.terminal_fallback
            else fit_sampler_bias_and_spread
        )
        fitted: SamplerCalibration = fitter(
            restored_members=restored_members,
            full_residual=full_residual,
            calibration_weight=calibration_weight,
            location_scale=location_record.calibration,
            d_enabled=self.model_selection.d_enabled,
            split=self.split,
        )
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
        fit_row_id: Sequence[str],
    ) -> RegressionProbabilityRecord:
        pooling = self._audit(
            pooling_evidence,
            probability_gate=True,
            fit_row_id=fit_row_id,
            fit_weight=weight,
            fit_observation=observation,
        )
        if pooling.terminal_fallback:
            fitted = identity_monotone_logit_linear_probability(
                probability,
                observation,
                weight,
                split=self.split,
            )
        else:
            assert pooling.decision is not None
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
        fit_row_id: Sequence[str],
    ) -> EnsembleProbabilityRecord:
        pooling = self._audit(
            pooling_evidence,
            probability_gate=True,
            fit_row_id=fit_row_id,
            fit_weight=weight,
            fit_observation=observation,
        )
        if pooling.terminal_fallback:
            fitted = identity_monotone_logit_linear_probability(
                probability,
                observation,
                weight,
                split=self.split,
            )
        else:
            assert pooling.decision is not None
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
            coverage=self.coverage,
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
    if round_trip != artifact:
        raise ValueError("calibration artifact failed typed publication round trip")

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"calibration artifact already exists: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact.semantic_sha256


def read_calibration_artifact(path: Path) -> CalibrationArtifact:
    """Read and parse a published calibration artifact."""

    source = Path(path)
    payload = source.read_bytes()
    try:
        raw = _as_mapping(json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibration artifact is not valid UTF-8 JSON") from error
    return CalibrationArtifact.from_dict(raw)


class CalibrationResolver:
    """Exact calibration lookups for release or development runs.

    Any combination of artifact and mode is accepted: a development run may
    use a real published artifact, and a resolver without an artifact simply
    resolves nothing (identity mode).
    """

    def __init__(
        self,
        artifact: CalibrationArtifact | None,
        *,
        development_mode: bool = False,
        frozen_model_selection: FrozenModelSelectionDecision | None = None,
    ) -> None:
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
        if record.pooling.terminal_fallback:
            _validate_identity_probability(probability)
            return probability
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
        if record.pooling.terminal_fallback:
            _validate_identity_probability(probability)
            return probability
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
    "CalibrationCoverage",
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

"""Atomic calibration application with explicit identity provenance.

The persisted calibration artifact deliberately supports two kinds of identity:
an unsupported terminal cell and an explicit development pass-through.  A
third identity, ``d=0``, is a valid frozen model-selection decision.  None of
those identities should be reported as though a parameter had been fitted.

This module resolves each immutable artifact record once and uses that exact
record both to transform the tensor and to construct its application audit.
That keeps the reported provenance inseparable from the values that were
actually applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Literal, Mapping

import torch
from torch import Tensor

from kcorrdiff.training.calibration import (
    POOLING_ORDER,
    apply_residual_calibration,
    monotone_logit_linear_probability,
)
from kcorrdiff.training.calibration_artifact import (
    CalibrationResolver,
    EnsembleProbabilityKey,
    EnsembleProbabilityRecord,
    LocationScaleKey,
    RegressionProbabilityKey,
    RegressionProbabilityRecord,
    SamplerBiasKey,
    SpreadKey,
)


CalibrationApplicationMode = Literal[
    "fitted",
    "terminal_identity",
    "selection_disabled",
    "development_identity",
]

_APPLICATION_MODES = frozenset(
    {
        "fitted",
        "terminal_identity",
        "selection_disabled",
        "development_identity",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CalibrationApplicationAudit:
    """Provenance for one exact calibration parameter or mapping.

    ``calibrated`` is intentionally derived from ``mode``.  In particular, an
    identity caused by terminal fallback can never be mislabeled calibrated,
    while a disabled ``d`` arm remains distinguishable from a support failure.
    """

    key_sha256: str
    mode: CalibrationApplicationMode
    pooling_level: str | None
    terminal_fallback: bool

    def __post_init__(self) -> None:
        if not isinstance(self.key_sha256, str) or _SHA256.fullmatch(
            self.key_sha256
        ) is None:
            raise ValueError("key_sha256 must be a lowercase hexadecimal SHA-256")
        if self.mode not in _APPLICATION_MODES:
            raise ValueError("unknown calibration application mode")
        if self.pooling_level is not None and self.pooling_level not in POOLING_ORDER:
            raise ValueError("pooling_level must be a declared calibration level")
        if not isinstance(self.terminal_fallback, bool):
            raise TypeError("terminal_fallback must be boolean")

        if self.mode == "fitted":
            if self.pooling_level is None or self.terminal_fallback:
                raise ValueError(
                    "fitted calibration requires a pooling level and no terminal fallback"
                )
        elif self.mode == "terminal_identity":
            if self.pooling_level is not None or not self.terminal_fallback:
                raise ValueError(
                    "terminal identity requires no pooling level and terminal_fallback=True"
                )
        elif self.mode == "development_identity":
            if self.pooling_level is not None or self.terminal_fallback:
                raise ValueError(
                    "development identity cannot claim pooling or terminal fallback"
                )
        elif self.terminal_fallback:
            # ``selection_disabled`` means d=0 was selected before calibration;
            # it is never caused by a terminal calibration-support failure.
            raise ValueError("selection-disabled identity is not a terminal fallback")

    @property
    def calibrated(self) -> bool:
        """Whether a fitted calibration parameter/mapping was applied."""

        return self.mode == "fitted"


@dataclass(frozen=True, slots=True)
class ResidualCalibrationApplicationAudit:
    """Application audits for residual ``b/c``, ``d``, and ``gamma``."""

    location_scale: CalibrationApplicationAudit
    sampler_bias: CalibrationApplicationAudit
    spread: CalibrationApplicationAudit

    def __post_init__(self) -> None:
        for value in (self.location_scale, self.sampler_bias, self.spread):
            if not isinstance(value, CalibrationApplicationAudit):
                raise TypeError("residual audits must be CalibrationApplicationAudit")


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationApplicationAudit:
    """One probability calibration audit bound to its physical threshold."""

    threshold_mm: float
    application: CalibrationApplicationAudit

    def __post_init__(self) -> None:
        if isinstance(self.threshold_mm, bool) or not isinstance(
            self.threshold_mm, Real
        ):
            raise TypeError("threshold_mm must be a real scalar")
        threshold = float(self.threshold_mm)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold_mm must be finite and positive")
        object.__setattr__(self, "threshold_mm", threshold)
        if not isinstance(self.application, CalibrationApplicationAudit):
            raise TypeError("application must be a CalibrationApplicationAudit")

    @property
    def calibrated(self) -> bool:
        return self.application.calibrated


@dataclass(frozen=True, slots=True)
class CalibrationApplicationAuditBundle:
    """Immutable per-forecast bundle for residual and p/q application audits."""

    residual: ResidualCalibrationApplicationAudit
    regression_probability: tuple[ThresholdCalibrationApplicationAudit, ...]
    ensemble_probability: tuple[ThresholdCalibrationApplicationAudit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.residual, ResidualCalibrationApplicationAudit):
            raise TypeError("residual must be a ResidualCalibrationApplicationAudit")
        for name in ("regression_probability", "ensemble_probability"):
            values = tuple(getattr(self, name))
            if any(
                not isinstance(value, ThresholdCalibrationApplicationAudit)
                for value in values
            ):
                raise TypeError(f"{name} must contain threshold calibration audits")
            thresholds = tuple(value.threshold_mm for value in values)
            if thresholds != tuple(sorted(thresholds)) or len(set(thresholds)) != len(
                thresholds
            ):
                raise ValueError(f"{name} thresholds must be unique canonical order")
            object.__setattr__(self, name, values)


def build_calibration_application_audit_bundle(
    *,
    residual: ResidualCalibrationApplicationAudit,
    regression_probability: Mapping[float, CalibrationApplicationAudit],
    ensemble_probability: Mapping[float, CalibrationApplicationAudit],
) -> CalibrationApplicationAuditBundle:
    """Freeze threshold-keyed p/q audits in deterministic threshold order."""

    def freeze(
        values: Mapping[float, CalibrationApplicationAudit], *, name: str
    ) -> tuple[ThresholdCalibrationApplicationAudit, ...]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a threshold-to-audit mapping")
        items = tuple(
            sorted(
                (
                    ThresholdCalibrationApplicationAudit(threshold, audit)
                    for threshold, audit in values.items()
                ),
                key=lambda item: item.threshold_mm,
            )
        )
        if len({item.threshold_mm for item in items}) != len(items):
            raise ValueError(f"{name} contains duplicate normalized thresholds")
        return items

    return CalibrationApplicationAuditBundle(
        residual=residual,
        regression_probability=freeze(
            regression_probability, name="regression_probability"
        ),
        ensemble_probability=freeze(
            ensemble_probability, name="ensemble_probability"
        ),
    )


def _require_resolver(resolver: CalibrationResolver) -> CalibrationResolver:
    if not isinstance(resolver, CalibrationResolver):
        raise TypeError("resolver must be a CalibrationResolver")
    return resolver


def _validate_residual_keys(
    *,
    location_key: LocationScaleKey,
    bias_key: SamplerBiasKey,
    spread_key: SpreadKey,
) -> None:
    if not isinstance(location_key, LocationScaleKey):
        raise TypeError("location_key must be a LocationScaleKey")
    if not isinstance(bias_key, SamplerBiasKey):
        raise TypeError("bias_key must be a SamplerBiasKey")
    if not isinstance(spread_key, SpreadKey):
        raise TypeError("spread_key must be a SpreadKey")
    if (
        location_key.lead_hours != bias_key.lead_hours
        or location_key.condition_signature != bias_key.condition_signature
        or bias_key.lead_hours != spread_key.lead_hours
        or bias_key.condition_signature != spread_key.condition_signature
        or bias_key.sampler_core != spread_key.ensemble_signature.sampler_core
    ):
        raise ValueError("residual calibration keys have mismatched exact signatures")


def _development_audit(key_sha256: str) -> CalibrationApplicationAudit:
    return CalibrationApplicationAudit(
        key_sha256=key_sha256,
        mode="development_identity",
        pooling_level=None,
        terminal_fallback=False,
    )


def _record_audit(
    *,
    key_sha256: str,
    pooling: object,
    selection_disabled: bool = False,
) -> CalibrationApplicationAudit:
    # PoolingAudit is intentionally consumed structurally here.  The concrete
    # record dataclasses have already validated it at artifact construction.
    decision = getattr(pooling, "decision", None)
    terminal = getattr(pooling, "terminal_fallback", None)
    if not isinstance(terminal, bool):
        raise TypeError("calibration record has an invalid pooling audit")
    level = None if decision is None else getattr(decision, "level", None)
    if selection_disabled:
        return CalibrationApplicationAudit(
            key_sha256=key_sha256,
            mode="selection_disabled",
            pooling_level=level,
            terminal_fallback=False,
        )
    if terminal:
        return CalibrationApplicationAudit(
            key_sha256=key_sha256,
            mode="terminal_identity",
            pooling_level=None,
            terminal_fallback=True,
        )
    return CalibrationApplicationAudit(
        key_sha256=key_sha256,
        mode="fitted",
        pooling_level=level,
        terminal_fallback=False,
    )


def _validate_identity_probability(probability: Tensor) -> None:
    if not isinstance(probability, Tensor):
        raise TypeError("probability must be a Tensor")
    if probability.dtype is not torch.float32:
        raise TypeError("probability must remain float32")
    if not bool(torch.isfinite(probability).all().item()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any().item()
    ):
        raise ValueError("probability must be finite and lie in [0,1]")


def apply_residual_calibration_with_audit(
    resolver: CalibrationResolver,
    restored_members: Tensor,
    *,
    location_key: LocationScaleKey,
    bias_key: SamplerBiasKey,
    spread_key: SpreadKey,
) -> tuple[Tensor, ResidualCalibrationApplicationAudit]:
    """Apply exact residual calibration and return audits from those records."""

    resolver = _require_resolver(resolver)
    _validate_residual_keys(
        location_key=location_key,
        bias_key=bias_key,
        spread_key=spread_key,
    )
    if resolver.development_mode:
        result = resolver.apply_residual(
            restored_members,
            location_key=location_key,
            bias_key=bias_key,
            spread_key=spread_key,
        )
        return result, ResidualCalibrationApplicationAudit(
            location_scale=_development_audit(location_key.semantic_sha256),
            sampler_bias=_development_audit(bias_key.semantic_sha256),
            spread=_development_audit(spread_key.semantic_sha256),
        )

    location = resolver.location(location_key)
    bias = resolver.bias(bias_key)
    spread = resolver.gamma(spread_key)
    if location.key != location_key or bias.key != bias_key or spread.key != spread_key:
        raise ValueError("resolved residual calibration record key mismatch")
    if bias.location_scale_key_sha256 != location_key.semantic_sha256:
        raise ValueError("d record is not linked to the requested b/c signature")
    if spread.sampler_bias_key_sha256 != bias_key.semantic_sha256:
        raise ValueError("gamma record is not linked to the requested d signature")
    if resolver.model_selection is None or (
        bias.d_enabled != resolver.model_selection.d_enabled
    ):
        raise ValueError("d record disagrees with the frozen model-selection decision")

    result = apply_residual_calibration(
        restored_members,
        location_b=location.calibration.location_b,
        total_scale_c=location.calibration.total_scale_c,
        sampler_bias_d=bias.sampler_bias_d,
        spread_gamma=spread.spread_gamma,
    )
    return result, ResidualCalibrationApplicationAudit(
        location_scale=_record_audit(
            key_sha256=location_key.semantic_sha256,
            pooling=location.pooling,
        ),
        sampler_bias=_record_audit(
            key_sha256=bias_key.semantic_sha256,
            pooling=bias.pooling,
            selection_disabled=not bias.d_enabled,
        ),
        spread=_record_audit(
            key_sha256=spread_key.semantic_sha256,
            pooling=spread.pooling,
        ),
    )


def _probability_record(
    resolver: CalibrationResolver,
    key: RegressionProbabilityKey | EnsembleProbabilityKey,
) -> RegressionProbabilityRecord | EnsembleProbabilityRecord:
    if resolver.artifact is None:
        raise RuntimeError("published calibration resolver has no artifact")
    if isinstance(key, RegressionProbabilityKey):
        record = resolver._lookup(  # noqa: SLF001 - exact resolver semantics.
            resolver.artifact.regression_probability,
            key.semantic_sha256,
            kind="p_cal",
        )
        if not isinstance(record, RegressionProbabilityRecord):
            raise TypeError("resolved p_cal record has the wrong type")
    elif isinstance(key, EnsembleProbabilityKey):
        record = resolver._lookup(  # noqa: SLF001 - exact resolver semantics.
            resolver.artifact.ensemble_probability,
            key.semantic_sha256,
            kind="q_cal",
        )
        if not isinstance(record, EnsembleProbabilityRecord):
            raise TypeError("resolved q_cal record has the wrong type")
    else:  # pragma: no cover - narrowed by public helpers.
        raise TypeError("unknown probability calibration key")
    if record.key != key:
        raise ValueError("resolved probability calibration record key mismatch")
    return record


def _apply_probability_record(
    probability: Tensor,
    *,
    record: RegressionProbabilityRecord | EnsembleProbabilityRecord,
) -> Tensor:
    if record.pooling.terminal_fallback:
        _validate_identity_probability(probability)
        return probability
    return monotone_logit_linear_probability(
        probability,
        alpha=record.calibration.alpha,
        beta_raw=record.calibration.beta_raw,
    )


def apply_regression_probability_with_audit(
    resolver: CalibrationResolver,
    probability: Tensor,
    *,
    key: RegressionProbabilityKey,
) -> tuple[Tensor, CalibrationApplicationAudit]:
    """Apply one exact regression probability mapping with provenance."""

    resolver = _require_resolver(resolver)
    if not isinstance(key, RegressionProbabilityKey):
        raise TypeError("key must be a RegressionProbabilityKey")
    if resolver.development_mode:
        result = resolver.apply_regression_probability(probability, key=key)
        return result, _development_audit(key.semantic_sha256)
    record = _probability_record(resolver, key)
    assert isinstance(record, RegressionProbabilityRecord)
    return _apply_probability_record(probability, record=record), _record_audit(
        key_sha256=key.semantic_sha256,
        pooling=record.pooling,
    )


def apply_ensemble_probability_with_audit(
    resolver: CalibrationResolver,
    probability: Tensor,
    *,
    key: EnsembleProbabilityKey,
) -> tuple[Tensor, CalibrationApplicationAudit]:
    """Apply one exact ensemble probability mapping with provenance."""

    resolver = _require_resolver(resolver)
    if not isinstance(key, EnsembleProbabilityKey):
        raise TypeError("key must be an EnsembleProbabilityKey")
    if resolver.development_mode:
        result = resolver.apply_ensemble_probability(probability, key=key)
        return result, _development_audit(key.semantic_sha256)
    record = _probability_record(resolver, key)
    assert isinstance(record, EnsembleProbabilityRecord)
    return _apply_probability_record(probability, record=record), _record_audit(
        key_sha256=key.semantic_sha256,
        pooling=record.pooling,
    )


# Short descriptive alias matching CalibrationResolver.apply_residual.
apply_residual_with_audit = apply_residual_calibration_with_audit


__all__ = [
    "CalibrationApplicationAudit",
    "CalibrationApplicationAuditBundle",
    "CalibrationApplicationMode",
    "ResidualCalibrationApplicationAudit",
    "ThresholdCalibrationApplicationAudit",
    "apply_ensemble_probability_with_audit",
    "apply_regression_probability_with_audit",
    "apply_residual_calibration_with_audit",
    "apply_residual_with_audit",
    "build_calibration_application_audit_bundle",
]

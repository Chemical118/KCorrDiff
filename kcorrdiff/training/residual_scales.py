"""Train-only OOF residual scales with deterministic sparse-cell pooling.

Each exact lead/condition cell tries the documented full-cell support first,
then the three predeclared pooling fallbacks.  Exhausting ``lead_only`` never
manufactures an identity or global scale: the serialized cell is explicitly
``diffusion_scale_unsupported`` so downstream code can route it to the
regression-only forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.training.calibration import (
    POOLING_ORDER,
    IndependentBlockSupport,
)


SCALE_FORMAT_VERSION = "kcorrdiff.residual-scales.v2"
DEFAULT_MINIMUM_INDEPENDENT_BLOCKS = 30
DEFAULT_MINIMUM_BLOCK_ESS = 20.0


def _state_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _state_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _state_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError(f"{name} must be a canonical non-empty string")
    return value


@dataclass(slots=True)
class _Moments:
    weighted_square_sum: float = 0.0
    weight_sum: float = 0.0
    valid_pixels: int = 0
    items: int = 0
    block_masses: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.block_masses is None:
            self.block_masses = {}


class ResidualScaleAccumulator:
    def __init__(
        self,
        *,
        epsilon_scale: float = 1.0e-3,
        minimum_independent_blocks: int = DEFAULT_MINIMUM_INDEPENDENT_BLOCKS,
        minimum_block_ess: float = DEFAULT_MINIMUM_BLOCK_ESS,
    ) -> None:
        if not math.isfinite(epsilon_scale) or epsilon_scale <= 0.0:
            raise ValueError("epsilon_scale must be finite and positive")
        if (
            isinstance(minimum_independent_blocks, bool)
            or not isinstance(minimum_independent_blocks, int)
            or minimum_independent_blocks <= 0
        ):
            raise ValueError("minimum_independent_blocks must be a positive integer")
        if not math.isfinite(minimum_block_ess) or minimum_block_ess <= 0.0:
            raise ValueError("minimum_block_ess must be finite and positive")
        self.epsilon_scale = float(epsilon_scale)
        self.minimum_independent_blocks = minimum_independent_blocks
        self.minimum_block_ess = float(minimum_block_ess)
        self._moments: dict[tuple[float, str, str], _Moments] = {}

    @staticmethod
    def _pool_keys(
        lead_hours: float, condition_signature: str
    ) -> tuple[tuple[float, str, str], ...]:
        signature = parse_condition_signature(condition_signature)
        return (
            (lead_hours, "full_cell", signature.key),
            (
                lead_hours,
                "lead_provider_era_present",
                f"{signature.provider_track}:era={int(signature.era_present)}",
            ),
            (lead_hours, "lead_provider", signature.provider_track),
            (lead_hours, "lead_only", "*"),
        )

    @staticmethod
    def _support(moments: _Moments) -> IndependentBlockSupport:
        masses = tuple((moments.block_masses or {}).values())
        if not masses:
            return IndependentBlockSupport(0, 0.0, 0, 0)
        values = np.asarray(masses, dtype=np.float64)
        scale = float(values.max())
        normalized = values / scale
        ess = float(normalized.sum() ** 2 / np.square(normalized).sum())
        if not math.isfinite(ess):
            raise OverflowError("residual-scale block ESS is not finite")
        return IndependentBlockSupport(len(values), ess, 0, 0)

    def update(
        self,
        *,
        lead_hours: float,
        condition_signature: str,
        block_id: str,
        target_z: ArrayLike,
        mu_z_oof: ArrayLike,
        target_validity: ArrayLike,
        omega: float,
        multiplicity: int = 1,
    ) -> None:
        if lead_hours not in tuple(index / 2 for index in range(1, 13)):
            raise ValueError("scale lead must be one of the official 12 leads")
        canonical_signature = parse_condition_signature(condition_signature).key
        if (
            not isinstance(block_id, str)
            or not block_id
            or block_id.strip() != block_id
        ):
            raise ValueError(
                "residual scale update requires a canonical non-empty block_id"
            )
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("omega must be finite and positive")
        if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity <= 0:
            raise ValueError("residual scale multiplicity must be a positive integer")
        target = np.asarray(target_z, dtype=np.float64)
        prediction = np.asarray(mu_z_oof, dtype=np.float64)
        validity = np.asarray(target_validity)
        if target.shape != prediction.shape or target.shape != validity.shape:
            raise ValueError("residual scale arrays must share one shape")
        if validity.dtype != np.bool_:
            if not np.issubdtype(validity.dtype, np.number) or not np.all(
                (validity == 0) | (validity == 1)
            ):
                raise ValueError("target_validity must be boolean or 0/1")
            validity = validity.astype(np.bool_)
        if not np.all(np.isfinite(target[validity])) or not np.all(
            np.isfinite(prediction[validity])
        ):
            raise ValueError("valid residual-scale values must be finite")
        residual = target[validity] - prediction[validity]
        square_sum = float(omega * multiplicity * np.dot(residual, residual))
        weight_mass = float(omega * multiplicity * residual.size)
        if not math.isfinite(square_sum) or not math.isfinite(weight_mass):
            raise OverflowError("residual-scale sufficient statistics overflowed float64")
        for key in self._pool_keys(float(lead_hours), canonical_signature):
            moments = self._moments.setdefault(key, _Moments())
            moments.weighted_square_sum += square_sum
            moments.weight_sum += weight_mass
            moments.valid_pixels += int(multiplicity * residual.size)
            moments.items += multiplicity
            if weight_mass > 0.0:
                assert moments.block_masses is not None
                moments.block_masses[block_id] = (
                    moments.block_masses.get(block_id, 0.0) + weight_mass
                )
            if (
                not math.isfinite(moments.weighted_square_sum)
                or not math.isfinite(moments.weight_sum)
                or any(
                    not math.isfinite(value)
                    for value in (moments.block_masses or {}).values()
                )
            ):
                raise OverflowError(
                    "pooled residual-scale sufficient statistics overflowed float64"
                )

    def result(self) -> dict[str, object]:
        if not self._moments:
            raise ValueError("no residual scale observations were supplied")
        records: list[dict[str, object]] = []
        exact_cells = tuple(
            (lead, key)
            for lead, level, key in self._moments
            if level == "full_cell"
        )
        for lead, signature in sorted(exact_cells):
            keys = self._pool_keys(lead, signature)
            ladder: list[dict[str, object]] = []
            selected_level: str | None = None
            selected_moments: _Moments | None = None
            for key in keys:
                moments = self._moments[key]
                support = self._support(moments)
                ladder.append(
                    {
                        "level": key[1],
                        "block_count": support.block_count,
                        "block_ess": support.block_ess,
                        "items": moments.items,
                        "valid_pixels": moments.valid_pixels,
                        "importance_weight_sum": moments.weight_sum,
                    }
                )
                if selected_level is None and support.passes_residual_gate(
                    minimum_blocks=self.minimum_independent_blocks,
                    minimum_ess=self.minimum_block_ess,
                ):
                    selected_level = key[1]
                    selected_moments = moments
            terminal = selected_moments is None
            audit_moments = self._moments[keys[-1]] if terminal else selected_moments
            assert audit_moments is not None
            exact_items = self._moments[keys[0]].items
            if terminal:
                scale: float | None = None
                raw: float | None = None
                epsilon_applied = False
            else:
                if audit_moments.weight_sum <= 0.0:
                    raise ValueError(
                        f"selected residual scale cell has no valid pixels: {lead}, {signature}"
                    )
                raw = math.sqrt(
                    audit_moments.weighted_square_sum / audit_moments.weight_sum
                )
                if not math.isfinite(raw):
                    raise OverflowError("residual scale RMS overflowed float64")
                scale = max(raw, self.epsilon_scale)
                epsilon_applied = raw < self.epsilon_scale
            records.append(
                {
                    "lead_hours": lead,
                    "condition_signature": signature,
                    "scale": scale,
                    "raw_rms": raw,
                    "epsilon_applied": epsilon_applied,
                    "items": exact_items,
                    "valid_pixels": audit_moments.valid_pixels,
                    "importance_weight_sum": audit_moments.weight_sum,
                    "pooling_level": selected_level,
                    "pooling_ladder": ladder,
                    "terminal_fallback": terminal,
                    "diffusion_scale_unsupported": terminal,
                }
            )
        return {
            "format_version": SCALE_FORMAT_VERSION,
            "epsilon_scale": self.epsilon_scale,
            "minimum_independent_blocks": self.minimum_independent_blocks,
            "minimum_block_ess": self.minimum_block_ess,
            "pooling_order": list(POOLING_ORDER),
            "records": records,
        }

    def merge_state(self) -> dict[str, object]:
        """Return deterministic sufficient statistics for rank-local merging."""

        return {
            "epsilon_scale": self.epsilon_scale,
            "minimum_independent_blocks": self.minimum_independent_blocks,
            "minimum_block_ess": self.minimum_block_ess,
            "records": [
                {
                    "lead_hours": lead,
                    "pooling_level": level,
                    "pooling_key": key,
                    "weighted_square_sum": moments.weighted_square_sum,
                    "weight_sum": moments.weight_sum,
                    "valid_pixels": moments.valid_pixels,
                    "items": moments.items,
                    "block_masses": [
                        {"block_id": block_id, "mass": mass}
                        for block_id, mass in sorted(
                            (moments.block_masses or {}).items()
                        )
                    ],
                }
                for (lead, level, key), moments in sorted(self._moments.items())
            ],
        }

    def merge(self, other: "ResidualScaleAccumulator") -> None:
        if not isinstance(other, ResidualScaleAccumulator):
            raise TypeError("can only merge ResidualScaleAccumulator")
        if not math.isclose(
            self.epsilon_scale, other.epsilon_scale, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("cannot merge residual scales with different epsilon")
        if (
            self.minimum_independent_blocks != other.minimum_independent_blocks
            or not math.isclose(
                self.minimum_block_ess,
                other.minimum_block_ess,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("cannot merge residual scales with different support gates")
        for key, source in other._moments.items():
            target = self._moments.setdefault(key, _Moments())
            target.weighted_square_sum += source.weighted_square_sum
            target.weight_sum += source.weight_sum
            target.valid_pixels += source.valid_pixels
            target.items += source.items
            assert target.block_masses is not None
            for block_id, mass in (source.block_masses or {}).items():
                target.block_masses[block_id] = (
                    target.block_masses.get(block_id, 0.0) + mass
                )
            if (
                not math.isfinite(target.weighted_square_sum)
                or not math.isfinite(target.weight_sum)
                or any(
                    not math.isfinite(value)
                    for value in target.block_masses.values()
                )
            ):
                raise OverflowError("merged residual-scale statistics overflowed float64")

    @classmethod
    def from_merge_state(cls, raw: Mapping[str, object]) -> "ResidualScaleAccumulator":
        if set(raw) != {
            "epsilon_scale",
            "minimum_independent_blocks",
            "minimum_block_ess",
            "records",
        }:
            raise ValueError("residual-scale merge state schema mismatch")
        result = cls(
            epsilon_scale=_state_real(raw["epsilon_scale"], name="epsilon_scale"),
            minimum_independent_blocks=_state_int(
                raw["minimum_independent_blocks"],
                name="minimum_independent_blocks",
            ),
            minimum_block_ess=_state_real(
                raw["minimum_block_ess"], name="minimum_block_ess"
            ),
        )
        records = raw.get("records")
        if not isinstance(records, list):
            raise TypeError("residual-scale merge records must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("residual-scale merge record must be a mapping")
            if set(record) != {
                "lead_hours",
                "pooling_level",
                "pooling_key",
                "weighted_square_sum",
                "weight_sum",
                "valid_pixels",
                "items",
                "block_masses",
            }:
                raise ValueError("residual-scale merge record schema mismatch")
            key = (
                _state_real(record["lead_hours"], name="lead_hours"),
                _state_string(record["pooling_level"], name="pooling_level"),
                _state_string(record["pooling_key"], name="pooling_key"),
            )
            if key[0] not in tuple(index / 2 for index in range(1, 13)):
                raise ValueError("residual-scale merge lead is not official")
            if key[1] not in POOLING_ORDER or not key[2]:
                raise ValueError("residual-scale merge pooling key is invalid")
            if key in result._moments:
                raise ValueError(f"duplicate residual-scale merge cell: {key}")
            raw_block_masses = record.get("block_masses")
            if not isinstance(raw_block_masses, list):
                raise TypeError("residual-scale block_masses must be a list")
            block_masses: dict[str, float] = {}
            for raw_mass in raw_block_masses:
                if not isinstance(raw_mass, Mapping):
                    raise TypeError("residual-scale block mass must be a mapping")
                if set(raw_mass) != {"block_id", "mass"}:
                    raise ValueError("residual-scale block mass schema mismatch")
                block_id = _state_string(raw_mass["block_id"], name="block_id")
                mass = _state_real(raw_mass["mass"], name="block mass")
                if (
                    not block_id
                    or block_id in block_masses
                    or not math.isfinite(mass)
                    or mass <= 0.0
                ):
                    raise ValueError("invalid residual-scale block mass")
                block_masses[block_id] = mass
            moments = _Moments(
                weighted_square_sum=_state_real(
                    record["weighted_square_sum"], name="weighted_square_sum"
                ),
                weight_sum=_state_real(record["weight_sum"], name="weight_sum"),
                valid_pixels=_state_int(record["valid_pixels"], name="valid_pixels"),
                items=_state_int(record["items"], name="items"),
                block_masses=block_masses,
            )
            if (
                not math.isfinite(moments.weighted_square_sum)
                or not math.isfinite(moments.weight_sum)
                or moments.weighted_square_sum < 0.0
                or moments.weight_sum < 0.0
                or moments.valid_pixels < 0
                or moments.items < 0
                or not math.isclose(
                    sum(block_masses.values()),
                    moments.weight_sum,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError("invalid residual-scale merge moments")
            result._moments[key] = moments
        if not result._moments:
            return result
        full_cells = tuple(
            (lead, key)
            for lead, level, key in result._moments
            if level == "full_cell"
        )
        expected_keys = {
            pool_key
            for lead, condition in full_cells
            for pool_key in result._pool_keys(lead, condition)
        }
        if not full_cells or set(result._moments) != expected_keys:
            raise ValueError(
                "residual-scale merge state does not contain one complete pooling ladder per cell"
            )
        return result


def write_residual_scales(
    path: Path,
    accumulator: ResidualScaleAccumulator,
    *,
    oof_manifest_sha256: str,
    regression_checkpoint_set_sha256: str,
) -> str:
    if not oof_manifest_sha256 or not regression_checkpoint_set_sha256:
        raise ValueError("scale provenance hashes must be non-empty")
    payload = accumulator.result()
    payload["oof_manifest_sha256"] = oof_manifest_sha256
    payload["regression_checkpoint_set_sha256"] = regression_checkpoint_set_sha256
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def scale_lookup(raw: Mapping[str, object]) -> dict[tuple[float, str], float]:
    if raw.get("format_version") != SCALE_FORMAT_VERSION:
        raise ValueError("unsupported residual scale format")
    result: dict[tuple[float, str], float] = {}
    for record in raw["records"]:  # type: ignore[index]
        key = (float(record["lead_hours"]), str(record["condition_signature"]))
        if key in result:
            raise ValueError(f"duplicate residual scale key: {key}")
        if record.get("diffusion_scale_unsupported") is True:
            if record.get("scale") is not None:
                raise ValueError(f"unsupported residual scale is not null: {key}")
            continue
        value = float(record["scale"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid residual scale: {key}")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_MINIMUM_BLOCK_ESS",
    "DEFAULT_MINIMUM_INDEPENDENT_BLOCKS",
    "SCALE_FORMAT_VERSION",
    "ResidualScaleAccumulator",
    "scale_lookup",
    "write_residual_scales",
]

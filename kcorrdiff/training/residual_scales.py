"""Train-only lead/signature RMS scales for normalized OOF residuals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike


SCALE_FORMAT_VERSION = "kcorrdiff.residual-scales.v1"


@dataclass(slots=True)
class _Moments:
    weighted_square_sum: float = 0.0
    weight_sum: float = 0.0
    valid_pixels: int = 0
    items: int = 0


class ResidualScaleAccumulator:
    def __init__(self, *, epsilon_scale: float = 1.0e-3) -> None:
        if not math.isfinite(epsilon_scale) or epsilon_scale <= 0.0:
            raise ValueError("epsilon_scale must be finite and positive")
        self.epsilon_scale = float(epsilon_scale)
        self._moments: dict[tuple[float, str], _Moments] = {}

    def update(
        self,
        *,
        lead_hours: float,
        condition_signature: str,
        target_z: ArrayLike,
        mu_z_oof: ArrayLike,
        target_validity: ArrayLike,
        omega: float,
        multiplicity: int = 1,
    ) -> None:
        if lead_hours not in tuple(index / 2 for index in range(1, 13)):
            raise ValueError("scale lead must be one of the official 12 leads")
        if not condition_signature:
            raise ValueError("condition_signature must be non-empty")
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
        moments = self._moments.setdefault(
            (float(lead_hours), condition_signature), _Moments()
        )
        moments.weighted_square_sum += float(
            omega * multiplicity * np.dot(residual, residual)
        )
        moments.weight_sum += float(omega * multiplicity * residual.size)
        moments.valid_pixels += int(multiplicity * residual.size)
        moments.items += multiplicity

    def result(self) -> dict[str, object]:
        if not self._moments:
            raise ValueError("no residual scale observations were supplied")
        records: list[dict[str, object]] = []
        for (lead, signature), moments in sorted(self._moments.items()):
            if moments.weight_sum <= 0.0:
                raise ValueError(f"residual scale cell has no valid pixels: {lead}, {signature}")
            raw = math.sqrt(moments.weighted_square_sum / moments.weight_sum)
            records.append(
                {
                    "lead_hours": lead,
                    "condition_signature": signature,
                    "scale": max(raw, self.epsilon_scale),
                    "raw_rms": raw,
                    "epsilon_applied": raw < self.epsilon_scale,
                    "items": moments.items,
                    "valid_pixels": moments.valid_pixels,
                    "importance_weight_sum": moments.weight_sum,
                }
            )
        return {
            "format_version": SCALE_FORMAT_VERSION,
            "epsilon_scale": self.epsilon_scale,
            "records": records,
        }

    def merge_state(self) -> dict[str, object]:
        """Return deterministic sufficient statistics for rank-local merging."""

        return {
            "epsilon_scale": self.epsilon_scale,
            "records": [
                {
                    "lead_hours": lead,
                    "condition_signature": signature,
                    "weighted_square_sum": moments.weighted_square_sum,
                    "weight_sum": moments.weight_sum,
                    "valid_pixels": moments.valid_pixels,
                    "items": moments.items,
                }
                for (lead, signature), moments in sorted(self._moments.items())
            ],
        }

    def merge(self, other: "ResidualScaleAccumulator") -> None:
        if not isinstance(other, ResidualScaleAccumulator):
            raise TypeError("can only merge ResidualScaleAccumulator")
        if not math.isclose(
            self.epsilon_scale, other.epsilon_scale, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("cannot merge residual scales with different epsilon")
        for key, source in other._moments.items():
            target = self._moments.setdefault(key, _Moments())
            target.weighted_square_sum += source.weighted_square_sum
            target.weight_sum += source.weight_sum
            target.valid_pixels += source.valid_pixels
            target.items += source.items

    @classmethod
    def from_merge_state(cls, raw: Mapping[str, object]) -> "ResidualScaleAccumulator":
        result = cls(epsilon_scale=float(raw["epsilon_scale"]))
        records = raw.get("records")
        if not isinstance(records, list):
            raise TypeError("residual-scale merge records must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("residual-scale merge record must be a mapping")
            key = (float(record["lead_hours"]), str(record["condition_signature"]))
            if key in result._moments:
                raise ValueError(f"duplicate residual-scale merge cell: {key}")
            moments = _Moments(
                weighted_square_sum=float(record["weighted_square_sum"]),
                weight_sum=float(record["weight_sum"]),
                valid_pixels=int(record["valid_pixels"]),
                items=int(record["items"]),
            )
            if (
                not math.isfinite(moments.weighted_square_sum)
                or not math.isfinite(moments.weight_sum)
                or moments.weighted_square_sum < 0.0
                or moments.weight_sum < 0.0
                or moments.valid_pixels < 0
                or moments.items < 0
            ):
                raise ValueError("invalid residual-scale merge moments")
            result._moments[key] = moments
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
        value = float(record["scale"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid residual scale: {key}")
        result[key] = value
    return result


__all__ = [
    "SCALE_FORMAT_VERSION",
    "ResidualScaleAccumulator",
    "scale_lookup",
    "write_residual_scales",
]

"""KMA PUB HSR raw-format and sentinel contract.

The official 2024-03-29 format document specifies a 1024-byte header followed
by 2305 x 2881 signed 16-bit values.  Empirical header/body checks select
little-endian storage.  Reflectivity values are fixed-point ``dBZ * 100``;
``-25000`` and ``-30000`` are the two documented NULL sentinels.  ``-20000``
is a display minimum and remains a valid numeric value.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import BinaryIO

import numpy as np


HEADER_BYTES = 1024
RAW_SHAPE = (2881, 2305)
RAW_DTYPE = np.dtype("<i2")
IN_DOMAIN_UNOBSERVED = -25000
OUTSIDE_RADAR_RANGE = -30000
NULL_SENTINELS = frozenset({IN_DOMAIN_UNOBSERVED, OUTSIDE_RADAR_RANGE})
DISPLAY_MINIMUM = -20000
REFLECTIVITY_SCALE = 100.0
EXPECTED_PAYLOAD_BYTES = HEADER_BYTES + int(np.prod(RAW_SHAPE)) * RAW_DTYPE.itemsize


@dataclass(frozen=True, slots=True)
class RawHSR:
    header: bytes
    values: np.ndarray
    valid: np.ndarray


def decode_hsr_payload(payload: bytes) -> RawHSR:
    if len(payload) != EXPECTED_PAYLOAD_BYTES:
        raise ValueError(
            f"KMA HSR payload has {len(payload)} bytes; expected {EXPECTED_PAYLOAD_BYTES}"
        )
    values = np.frombuffer(payload, dtype=RAW_DTYPE, offset=HEADER_BYTES).reshape(RAW_SHAPE)
    valid = ~np.isin(values, tuple(NULL_SENTINELS))
    return RawHSR(header=payload[:HEADER_BYTES], values=values, valid=valid)


def read_hsr_gzip(path: Path) -> RawHSR:
    with gzip.open(path, "rb") as stream:
        payload = stream.read(EXPECTED_PAYLOAD_BYTES + 1)
    return decode_hsr_payload(payload)


def reflectivity_dbz(raw_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_values)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("raw values and validity must have equal shape")
    output = np.full(values.shape, np.nan, dtype=np.float32)
    output[mask] = values[mask].astype(np.float32) / REFLECTIVITY_SCALE
    return output


def sentinel_counts(values: np.ndarray) -> dict[int, int]:
    array = np.asarray(values)
    return {sentinel: int(np.count_nonzero(array == sentinel)) for sentinel in sorted(NULL_SENTINELS)}

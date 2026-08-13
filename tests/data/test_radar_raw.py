from __future__ import annotations

import numpy as np
import pytest

from kcorrdiff.data.radar_raw import (
    DISPLAY_MINIMUM,
    EXPECTED_PAYLOAD_BYTES,
    HEADER_BYTES,
    NULL_SENTINELS,
    RAW_SHAPE,
    decode_hsr_payload,
    reflectivity_dbz,
)


def test_raw_contract_keeps_display_minimum_valid() -> None:
    values = np.zeros(RAW_SHAPE, dtype="<i2")
    values[0, 0] = -25000
    values[0, 1] = -30000
    values[0, 2] = DISPLAY_MINIMUM
    raw = decode_hsr_payload(bytes(HEADER_BYTES) + values.tobytes())
    assert not raw.valid[0, 0] and not raw.valid[0, 1]
    assert raw.valid[0, 2]
    dbz = reflectivity_dbz(raw.values[:1, :3], raw.valid[:1, :3])
    assert np.isnan(dbz[0, 0]) and np.isnan(dbz[0, 1])
    assert dbz[0, 2] == -200.0


def test_raw_contract_rejects_truncated_or_extra_payload() -> None:
    with pytest.raises(ValueError, match="expected"):
        decode_hsr_payload(bytes(EXPECTED_PAYLOAD_BYTES - 1))
    with pytest.raises(ValueError, match="expected"):
        decode_hsr_payload(bytes(EXPECTED_PAYLOAD_BYTES + 1))

"""Measure the production lossless OOF codec on real FP32 prediction fields.

The input is a NumPy ``.npy`` array with exact shape ``[N,2,H,W]`` where field
0 is occurrence probability and field 1 is ``mu_z``.  The command emits a
canonical, hashed JSON report and refuses any non-FP32 input or non-bitwise
round trip.  It is intentionally independent of model inference so a held GPU
job can save one representative prediction window and benchmark multiple codec
settings without rerunning the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Sequence

import numpy as np

from kcorrdiff.training.oof import (
    OOF_COMPRESSED_ENCODING,
    _write_deterministic_npz,
    load_compressed_oof_fields,
)


REPORT_FORMAT = "kcorrdiff.oof-codec-benchmark.v1"


def benchmark_oof_codec(input_path: Path) -> dict[str, object]:
    selected = input_path.resolve()
    values = np.load(selected, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("OOF codec input must be float32 [N,2,H,W]")
    logical = np.ascontiguousarray(values, dtype=np.float32)
    logical_bytes = logical.nbytes
    if logical_bytes <= 0:
        raise ValueError("OOF codec input cannot be empty")
    input_sha = hashlib.sha256(logical.tobytes(order="C")).hexdigest()
    descriptor, name = tempfile.mkstemp(prefix="oof-codec-", suffix=".npz")
    os.close(descriptor)
    temporary = Path(name)
    try:
        encode_started = time.perf_counter()
        _write_deterministic_npz(temporary, logical)
        encode_seconds = time.perf_counter() - encode_started
        archive_bytes = temporary.stat().st_size
        archive_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        decode_started = time.perf_counter()
        restored = load_compressed_oof_fields(temporary)
        decode_seconds = time.perf_counter() - decode_started
        restored_sha = hashlib.sha256(restored.tobytes(order="C")).hexdigest()
        if input_sha != restored_sha or not np.array_equal(
            logical.view(np.uint32), restored.view(np.uint32)
        ):
            raise ValueError("OOF codec round trip is not bitwise lossless")
        return {
            "format_version": REPORT_FORMAT,
            "codec": OOF_COMPRESSED_ENCODING,
            "input_path": str(selected),
            "shape": list(logical.shape),
            "dtype": "float32",
            "logical_bytes": logical_bytes,
            "compressed_bytes": archive_bytes,
            "compression_ratio": archive_bytes / logical_bytes,
            "space_saving_fraction": 1.0 - archive_bytes / logical_bytes,
            "encode_seconds": encode_seconds,
            "encode_mib_per_second": logical_bytes / 1024**2 / encode_seconds,
            "decode_seconds": decode_seconds,
            "decode_mib_per_second": logical_bytes / 1024**2 / decode_seconds,
            "input_array_sha256": input_sha,
            "restored_array_sha256": restored_sha,
            "compressed_sha256": archive_sha,
            "bitwise_equal": True,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_report(path: Path, report: dict[str, object]) -> str:
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = benchmark_oof_codec(arguments.input_npy)
    digest = _atomic_report(arguments.output_json, report)
    print(json.dumps({**report, "report_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate raw ERA5 GRIBs and build the yearly float32 mmap cache.

Examples::

    python -m kcorrdiff.data.era5_cache_cli validate \
        --raw-root /workspace/data/raw/era5

    python -m kcorrdiff.data.era5_cache_cli build \
        --raw-root /workspace/data/raw/era5 \
        --output /workspace/data/cache/era5-v1

Both commands default to the complete 2020--2025 archive.  ``build`` always
runs a metadata-only preflight before allocating cache shards and uses an
exclusive sibling lock, so two builders cannot target the same output.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
from typing import Sequence
import uuid

from .eccodes_decoder import EcCodesDecoder, discover_era5_archive
from .era5_reader import build_era5_mmap_cache, estimate_era5_cache_bytes
from .provider_adapter import Era5ProviderAdapter


DEFAULT_RAW_ROOT = Path("/workspace/data/raw/era5")


class _ExclusiveBuildLock(AbstractContextManager["_ExclusiveBuildLock"]):
    """Own a non-destructive, inspectable lock next to one cache output."""

    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.path = self.output.parent / f".{self.output.name}.build.lock"
        self.token = uuid.uuid4().hex
        self._owned = False

    def __enter__(self) -> "_ExclusiveBuildLock":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": self.token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_utc": datetime.now(UTC).isoformat(),
            "output": str(self.output),
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"ERA5 build lock already exists; inspect the owner before removing it: "
                f"{self.path}"
            ) from error
        try:
            os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._owned = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self._owned:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if payload.get("token") == self.token:
            self.path.unlink()
        self._owned = False


def _add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="metadata-audit every expected GRIB message without writing a cache"
    )
    _add_archive_arguments(validate)

    build = commands.add_parser(
        "build", help="validate and build separate yearly instantaneous/tp mmap shards"
    )
    _add_archive_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--reserve-gib",
        type=float,
        default=20.0,
        help="free space to retain after the estimated cache payload (default: 20)",
    )
    build.add_argument(
        "--no-sha256",
        action="store_true",
        help="skip final shard SHA-256 hashes (faster, less verification)",
    )
    return parser.parse_args(argv)


def _years(args: argparse.Namespace) -> tuple[int, ...]:
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    return tuple(range(args.start_year, args.end_year + 1))


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    years = _years(args)
    sources = discover_era5_archive(args.raw_root, years=years)

    if args.command == "validate":
        audit = EcCodesDecoder().validate_sources(sources)
        _print_json({"status": "valid", "archive": audit.to_json()})
        return 0

    if args.reserve_gib < 0.0:
        raise ValueError("--reserve-gib must be non-negative")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"ERA5 cache destination already exists: {output}")
    with _ExclusiveBuildLock(output):
        # Fail on archive semantics before creating multi-GiB mmap files.
        audit = EcCodesDecoder().validate_sources(sources)
        decoder = EcCodesDecoder()
        manifest = build_era5_mmap_cache(
            decoder=decoder,
            sources=sources,
            output_dir=output,
            years=years,
            adapter=Era5ProviderAdapter(),
            require_complete_instantaneous=True,
            require_complete_tp=True,
            reserve_bytes=int(args.reserve_gib * 1024**3),
            verify_sha256=not args.no_sha256,
        )
    _print_json(
        {
            "status": "built",
            "output": str(output),
            "estimated_payload_bytes": estimate_era5_cache_bytes(years),
            "archive": audit.to_json(),
            "cache_manifest": manifest.to_json(),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_RAW_ROOT", "main"]

from __future__ import annotations

from pathlib import Path

import pytest

from kcorrdiff.data.era5_cache_cli import _ExclusiveBuildLock, _parse_args, _years


def test_cli_defaults_to_complete_2020_2025_archive() -> None:
    args = _parse_args(["validate"])
    assert _years(args) == tuple(range(2020, 2026))
    assert args.raw_root == Path("/workspace/data/raw/era5")


def test_build_cli_requires_explicit_output() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["build"])


def test_exclusive_build_lock_refuses_concurrent_same_output(tmp_path: Path) -> None:
    output = tmp_path / "era5-v1"
    with _ExclusiveBuildLock(output) as owner:
        assert owner.path.is_file()
        with pytest.raises(FileExistsError, match="holds the lock"):
            with _ExclusiveBuildLock(output):
                pass
    assert owner.path.exists()

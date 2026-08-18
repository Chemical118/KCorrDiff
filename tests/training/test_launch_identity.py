from __future__ import annotations

import json
from pathlib import Path

from kcorrdiff.training.launch_identity import (
    PLACEHOLDER_SHA256,
    development_identity,
    load_stage2_launch_identity,
    source_file_hashes,
    source_tree_semantic_sha256,
    write_stage2_launch_identity,
)


IMAGE = "sha256:" + "a" * 64


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    (root / "kcorrdiff/training").mkdir(parents=True)
    (root / "configs").mkdir()
    files = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "configs/data-contract-v1.1.3b.json": '{"protocol_version":"v1.1.3b"}\n',
        "kcorrdiff/__init__.py": "",
        "kcorrdiff/training/worker.py": "VALUE = 1\n",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    report = tmp_path / "pip-report.json"
    report.write_text(json.dumps({"version": "1", "install": []}), encoding="utf-8")
    return root, report, root / "configs/data-contract-v1.1.3b.json"


def test_launch_identity_roundtrip_records_source_runtime_and_contract(
    tmp_path: Path,
) -> None:
    root, report, contract = _source_tree(tmp_path)
    path = tmp_path / "run" / "launch-identity.json"
    write_stage2_launch_identity(
        path,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    identity = load_stage2_launch_identity(path, source_root=root)
    assert identity.source_tree_sha256 == source_tree_semantic_sha256(
        source_file_hashes(root)
    )
    assert identity.container_image_sha256 == "a" * 64
    assert set(identity.provenance()) == {
        "artifact_sha256",
        "source_tree_sha256",
        "container_image_sha256",
        "runtime_report_sha256",
        "data_contract_sha256",
    }


def test_launch_identity_is_informational_and_never_verified(tmp_path: Path) -> None:
    root, report, contract = _source_tree(tmp_path)
    path = tmp_path / "launch-identity.json"
    write_stage2_launch_identity(
        path,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    # Editing the source tree after publication must not block loading.
    (root / "kcorrdiff/training/worker.py").write_text("VALUE = 2\n")
    identity = load_stage2_launch_identity(path, source_root=root)
    assert identity.artifact_path == path.resolve()


def test_development_identity_needs_no_artifacts(tmp_path: Path) -> None:
    identity = development_identity(tmp_path)
    assert identity.artifact_path is None
    assert identity.source_tree_sha256 == PLACEHOLDER_SHA256
    assert set(identity.provenance()) == {
        "artifact_sha256",
        "source_tree_sha256",
        "container_image_sha256",
        "runtime_report_sha256",
        "data_contract_sha256",
    }

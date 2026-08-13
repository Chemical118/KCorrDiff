from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcorrdiff.training.launch_identity import (
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
        "configs/stage2-full-width.yaml": "protocol_version: v1.1.3b\n",
        "configs/stage2-loader-selection.json": '{"selected":true}\n',
        "configs/stage2-runtime-requirements.txt": "wandb==0.28.2\n",
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


def test_launch_identity_roundtrip_reproduces_source_runtime_and_contract(
    tmp_path: Path,
) -> None:
    root, report, contract = _source_tree(tmp_path)
    path = tmp_path / "run" / "launch-identity.json"
    digest = write_stage2_launch_identity(
        path,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    identity = load_stage2_launch_identity(
        path,
        expected_sha256=digest,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    assert identity.artifact_sha256 == digest
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
    identity.verify_current_files()


@pytest.mark.parametrize("target", ("source", "runtime", "contract"))
def test_launch_identity_rejects_any_live_input_mutation(
    tmp_path: Path, target: str
) -> None:
    root, report, contract = _source_tree(tmp_path)
    path = tmp_path / "launch-identity.json"
    digest = write_stage2_launch_identity(
        path,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    if target == "source":
        (root / "kcorrdiff/training/worker.py").write_text("VALUE = 2\n")
    elif target == "runtime":
        report.write_text('{"install":[{"name":"changed"}]}\n')
    else:
        contract.write_text('{"protocol_version":"changed"}\n')
    with pytest.raises(ValueError, match="source tree|runtime dependency|data-contract"):
        load_stage2_launch_identity(
            path,
            expected_sha256=digest,
            source_root=root,
            container_image_digest=IMAGE,
            runtime_report=report,
            data_contract=contract,
        )


def test_launch_identity_rejects_digest_or_container_substitution(
    tmp_path: Path,
) -> None:
    root, report, contract = _source_tree(tmp_path)
    path = tmp_path / "launch-identity.json"
    digest = write_stage2_launch_identity(
        path,
        source_root=root,
        container_image_digest=IMAGE,
        runtime_report=report,
        data_contract=contract,
    )
    with pytest.raises(ValueError, match="artifact SHA-256"):
        load_stage2_launch_identity(
            path,
            expected_sha256="0" * 64,
            source_root=root,
            container_image_digest=IMAGE,
            runtime_report=report,
            data_contract=contract,
        )
    with pytest.raises(ValueError, match="container image digest mismatch"):
        load_stage2_launch_identity(
            path,
            expected_sha256=digest,
            source_root=root,
            container_image_digest="sha256:" + "b" * 64,
            runtime_report=report,
            data_contract=contract,
        )

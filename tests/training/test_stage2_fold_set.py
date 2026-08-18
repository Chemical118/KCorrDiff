from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest
import torch
from torch import nn

from kcorrdiff.training.checkpoints import load_training_checkpoint
from kcorrdiff.training.stage2_fold_set import (
    load_verified_fold_model,
    verify_stage2_fold_set,
)
from tests.training.test_collect_stage2_folds import (
    _assemble_valid_fold_set,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allow_manifest_fixture_rewrite(manifest: Path) -> None:
    manifest.chmod(manifest.stat().st_mode | stat.S_IWUSR)
    sidecar = manifest.with_name("fold-set-manifest.sha256")
    sidecar.chmod(sidecar.stat().st_mode | stat.S_IWUSR)


def _verify_assembled(root: Path):
    manifest, fixture, expected_states = _assemble_valid_fold_set(root)
    verified = verify_stage2_fold_set(
        manifest,
        expected_manifest_sha256=_sha256(manifest),
        expected_policy_sha256=fixture.policy_sha256,
        rows=fixture.rows,
        expected_config_sha256=fixture.config_sha256,
        expected_artifact_hashes=fixture.artifact_hashes,
        expected_source_tree_sha256=fixture.launch_identity["source_tree_sha256"],
    )
    return manifest, fixture, expected_states, verified






def test_model_only_load_uses_strict_state_dict_compatibility(
    tmp_path: Path,
) -> None:
    _, _, _, fold_set = _verify_assembled(tmp_path)
    verified = fold_set.by_fold()[2]

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_verified_fold_model(
            verified, nn.Linear(3, 1, dtype=torch.float32)
        )




def test_live_legacy_v1_set_without_strict_receipts_is_deeply_reverified(
    tmp_path: Path,
) -> None:
    manifest, fixture, expected_states, _ = _verify_assembled(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("verification_format")
    payload.pop("lineage")
    for record in payload["records"]:
        record.pop("verification")
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _allow_manifest_fixture_rewrite(manifest)
    manifest.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    manifest.with_name("fold-set-manifest.sha256").write_text(
        digest + "\n", encoding="ascii"
    )

    verified = verify_stage2_fold_set(
        manifest,
        expected_manifest_sha256=digest,
        expected_policy_sha256=fixture.policy_sha256,
        rows=fixture.rows,
        expected_config_sha256=fixture.config_sha256,
        expected_artifact_hashes=fixture.artifact_hashes,
        expected_source_tree_sha256=fixture.launch_identity["source_tree_sha256"],
    )
    model = nn.Linear(2, 1, dtype=torch.float32)
    load_verified_fold_model(verified.by_fold()[0], model)
    assert all(
        torch.equal(value, expected_states[0][name])
        for name, value in model.state_dict().items()
    )

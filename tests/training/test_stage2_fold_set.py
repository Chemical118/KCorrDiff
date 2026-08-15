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


def test_b12_fold_loads_model_only_but_legacy_world2_resume_stays_forbidden(
    tmp_path: Path,
) -> None:
    _, _, expected_states, fold_set = _verify_assembled(tmp_path)
    verified = fold_set.by_fold()[1]
    assert verified.provenance.world_size == 1
    assert verified.provenance.per_rank_microbatch_size == 12
    assert verified.provenance.gradient_accumulation_steps == 1

    model = nn.Linear(2, 1, dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    rng_before = torch.get_rng_state().clone()
    load_verified_fold_model(verified, model)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_states[1][name])

    world2_b8 = replace(
        verified.provenance,
        world_size=2,
        per_rank_microbatch_size=8,
        gradient_accumulation_steps=1,
    )
    with pytest.raises(ValueError, match="provenance/topology mismatch"):
        load_training_checkpoint(
            verified.checkpoint_path,
            model=nn.Linear(2, 1, dtype=torch.float32),
            optimizer=None,
            scheduler=None,
            expected_provenance=world2_b8,
            restore_rng=False,
        )


def test_model_only_load_rejects_checkpoint_changed_after_verification(
    tmp_path: Path,
) -> None:
    _, _, _, fold_set = _verify_assembled(tmp_path)
    verified = fold_set.by_fold()[0]
    verified.checkpoint_path.chmod(
        verified.checkpoint_path.stat().st_mode | stat.S_IWUSR
    )
    with verified.checkpoint_path.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="bytes changed"):
        load_verified_fold_model(
            verified, nn.Linear(2, 1, dtype=torch.float32)
        )


def test_model_only_load_uses_strict_state_dict_compatibility(
    tmp_path: Path,
) -> None:
    _, _, _, fold_set = _verify_assembled(tmp_path)
    verified = fold_set.by_fold()[2]

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_verified_fold_model(
            verified, nn.Linear(3, 1, dtype=torch.float32)
        )


def test_fold_set_rejects_tampered_embedded_verification_receipt(
    tmp_path: Path,
) -> None:
    manifest, fixture, _, _ = _verify_assembled(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][1]["verification"]["plan_optimizer_steps"] += 1
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _allow_manifest_fixture_rewrite(manifest)
    manifest.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    manifest.with_name("fold-set-manifest.sha256").write_text(
        digest + "\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="embedded verification receipt mismatch"):
        verify_stage2_fold_set(
            manifest,
            expected_manifest_sha256=digest,
            expected_policy_sha256=fixture.policy_sha256,
            rows=fixture.rows,
            expected_config_sha256=fixture.config_sha256,
            expected_artifact_hashes=fixture.artifact_hashes,
            expected_source_tree_sha256=fixture.launch_identity[
                "source_tree_sha256"
            ],
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


def test_fold_set_rejects_model_code_source_tree_mismatch(tmp_path: Path) -> None:
    manifest, fixture, _ = _assemble_valid_fold_set(tmp_path)

    with pytest.raises(ValueError, match="source tree is incompatible"):
        verify_stage2_fold_set(
            manifest,
            expected_manifest_sha256=_sha256(manifest),
            expected_policy_sha256=fixture.policy_sha256,
            rows=fixture.rows,
            expected_config_sha256=fixture.config_sha256,
            expected_artifact_hashes=fixture.artifact_hashes,
            expected_source_tree_sha256="9" * 64,
        )

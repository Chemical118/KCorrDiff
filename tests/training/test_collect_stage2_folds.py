from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.collect_stage2_folds import collect, mark_complete


def test_single_node_markers_are_verified_and_assembled(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    policy = "a" * 64
    config = "b" * 64
    source_inodes: list[int] = []
    for fold_id in range(3):
        worker = run_root / "workers" / f"fold-{fold_id}"
        checkpoint = worker / f"fold-{fold_id}" / "final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{fold_id}".encode())
        source_inodes.append(checkpoint.stat().st_ino)
        (worker / "partial-manifest.json").write_text(
            json.dumps(
                {
                    "config_sha256": config,
                    "selection": {
                        "roles": [f"fold:{fold_id}"],
                        "single_node_fold_worker": True,
                        "node_name": "porsche",
                    },
                    "topology": {
                        "world_size": 1,
                        "per_rank_microbatch_size": 12,
                        "gradient_accumulation_steps": 1,
                        "single_node_fold_policy_sha256": policy,
                    },
                }
            ),
            encoding="utf-8",
        )
        mark_complete(
            argparse.Namespace(
                run_root=run_root,
                worker_root=worker,
                fold_id=fold_id,
                policy_sha256=policy,
            )
        )

    collect(
        argparse.Namespace(
            run_root=run_root,
            poll_seconds=0.01,
            timeout_hours=0.01,
        )
    )
    assembled = run_root / "assembled" / "fold-set-v1"
    manifest = json.loads((assembled / "fold-set-manifest.json").read_text())
    assert manifest["node_name"] == "porsche"
    assert manifest["per_rank_microbatch_size"] == 12
    assert [record["fold_id"] for record in manifest["records"]] == [0, 1, 2]
    assert [
        (assembled / f"fold-{fold_id}" / "final.pt").stat().st_ino
        for fold_id in range(3)
    ] == source_inodes

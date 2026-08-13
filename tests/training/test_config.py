from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcorrdiff.training.config import load_stage2_config


CONFIG = Path(__file__).parents[2] / "configs" / "stage2-full-width.yaml"


def test_production_stage2_config_is_strict_and_topology_bound() -> None:
    config = load_stage2_config(CONFIG)
    assert config.runtime.target_widths == (64, 128, 256, 384, 512)
    assert config.runtime.context_widths == (32, 64, 128, 256, 384)
    assert config.crossfit.folds == 3
    assert config.data.normalization.as_posix().endswith(
        "/event-pretrain-production-v1/normalization.json"
    )
    assert config.data.condition_augmentation is not None
    assert config.data.condition_augmentation.policy_id == (
        "era5-full-dropout-50-25-25-v1"
    )
    assert config.data.condition_augmentation.temporal_access_strategy == (
        "matched_arm"
    )
    assert config.data.condition_augmentation_sha256 == (
        "813d7395ac59897a7a258ed07205fff289c262b7e8df93cd2cef4deeace9c9f8"
    )
    assert len(config.sha256) == 64
    config.validate_topology(world_size=2)
    with pytest.raises(ValueError, match="global effective batch"):
        config.validate_topology(world_size=1)


def test_config_rejects_fallback_unknown_keys_and_weight_clipping(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    changed = deepcopy(raw)
    changed["runtime"]["target_widths"][0] = 32
    path = tmp_path / "fallback.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="target widths"):
        load_stage2_config(path)

    changed = deepcopy(raw)
    changed["loss"]["importance_weight_clipping"] = 10.0
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="clipping"):
        load_stage2_config(path)

    changed = deepcopy(raw)
    changed["mystery"] = True
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_stage2_config(path)


def test_config_accepts_only_explicit_hashed_condition_augmentation(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    full = raw["data"]["condition_signature"]
    tp_off = "era5_oracle:era=1:tp=0:full_trajectory"
    null = "null_provider:era=0:tp=0:no_era_access"
    raw["data"]["condition_augmentation"] = {
        "format_version": "kcorrdiff.condition-augmentation-policy.v1",
        "policy_id": "test-full-dropout-v1",
        "protocol_version": "v1.1.3b",
        "source_condition_signature": full,
        "temporal_access_strategy": "matched_arm",
        "entries": [
            {
                "condition_signature": full,
                "target_probability": 0.7,
                "draw_probability": 0.5,
            },
            {
                "condition_signature": tp_off,
                "target_probability": 0.2,
                "draw_probability": 0.25,
            },
            {
                "condition_signature": null,
                "target_probability": 0.1,
                "draw_probability": 0.25,
            },
        ],
        "selection_order": "base_sample_then_one_condition_signature",
        "weight_clipping": None,
    }
    path = tmp_path / "augmented.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_stage2_config(path)

    assert config.data.condition_augmentation is not None
    assert config.data.condition_signatures == (tp_off, full, null)
    assert len(config.data.condition_augmentation_sha256 or "") == 64

    raw["data"]["condition_augmentation"]["source_condition_signature"] = (
        "era5_oracle:era=1:tp=1:target_end_causal"
    )
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="source signature"):
        load_stage2_config(path)

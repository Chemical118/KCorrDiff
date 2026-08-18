from __future__ import annotations

import pytest
import torch

from kcorrdiff.training.runtime import (
    RuntimeContract,
    assert_float32_tree,
    configure_strict_float32,
    contract_from_mapping,
)


def full_mapping() -> dict[str, object]:
    return {
        "target_widths": [64, 128, 256, 384, 512],
        "context_widths": [32, 64, 128, 256, 384],
        "era_latent_channels": 128,
        "era_grid_size": 33,
        "precision": "float32",
        "tf32": False,
        "allow_cpu_fallback": False,
        "allow_model_width_fallback": False,
        "allow_precision_fallback": False,
        "allow_era_grid_fallback": False,
    }


def test_runtime_contract_accepts_tuning_and_defaults_optional_flags() -> None:
    assert contract_from_mapping(full_mapping()) == RuntimeContract()
    tuned = full_mapping()
    tuned["target_widths"] = [48, 96, 192, 288, 384]
    tuned["allow_precision_fallback"] = True
    assert contract_from_mapping(tuned).target_widths == (48, 96, 192, 288, 384)
    optional = {
        key: value
        for key, value in full_mapping().items()
        if not key.startswith("allow_") and key != "tf32"
    }
    assert contract_from_mapping(optional).allow_precision_fallback is False

    for key, alias in (
        ("era_grid_size", 33.0),
        ("era_latent_channels", "128"),
        ("precision", 32),
    ):
        invalid = full_mapping()
        invalid[key] = alias
        with pytest.raises(TypeError):
            contract_from_mapping(invalid)
    invalid = full_mapping()
    invalid["target_widths"] = [64.0, 128, 256, 384, 512]
    with pytest.raises(TypeError, match="integer"):
        contract_from_mapping(invalid)
    # Unknown keys are tolerated (research code).
    extra = full_mapping()
    extra["unknown"] = True
    assert contract_from_mapping(extra) == RuntimeContract()


def test_float32_guard_and_cpu_diagnostic_path() -> None:
    module = torch.nn.Conv2d(2, 3, 3)
    assert_float32_tree(module)
    assert configure_strict_float32(require_cuda=False).type in {"cpu", "cuda"}
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert_float32_tree(module.float())
    with pytest.raises(TypeError, match="non-float32"):
        assert_float32_tree(module.double())

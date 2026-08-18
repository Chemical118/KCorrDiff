from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from kcorrdiff.inference.model_binding import (
    bind_development_regression_model,
    bind_development_residual_edm_model,
)
from kcorrdiff.models.regression_system import (
    RegressionSystem,
    RegressionSystemConfig,
)
from kcorrdiff.models.residual_edm import ResidualEDM, ResidualEDMConfig


TARGET_WIDTHS = (4, 8, 12, 16, 20)
CONTEXT_WIDTHS = (4, 8, 12, 16, 20)


def _regression() -> RegressionSystem:
    return RegressionSystem(
        RegressionSystemConfig(
            target_widths=TARGET_WIDTHS,
            context_widths=CONTEXT_WIDTHS,
            input_size=16,
            era_stem_channels=8,
            era_spatial_blocks=0,
            regression_query_chunk_size=2,
            allow_test_override=True,
        )
    )


def _residual_edm(*, variant: str = "edm_a") -> ResidualEDM:
    return ResidualEDM(
        ResidualEDMConfig(
            variant=variant,  # type: ignore[arg-type]
            target_widths=TARGET_WIDTHS,
            context_widths=CONTEXT_WIDTHS,
            input_size=16,
            query_chunk_size=2,
            allow_test_override=True,
        )
    )


def test_development_regression_binding_is_state_derived_and_immutable() -> None:
    torch.manual_seed(1201)
    model = _regression().train()

    verified = bind_development_regression_model(model)
    rebound = bind_development_regression_model(model)

    assert verified.model is model
    assert verified.validate() is model
    assert verified.is_production is False
    assert len(verified.checkpoint_sha256) == 64
    assert verified.checkpoint_sha256 == rebound.checkpoint_sha256
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(FrozenInstanceError):
        verified.checkpoint_sha256 = "0" * 64  # type: ignore[misc]

    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    changed = bind_development_regression_model(model)
    assert changed.checkpoint_sha256 != verified.checkpoint_sha256
    assert changed.validate() is model


def test_development_residual_binding_captures_variant_and_state_id() -> None:
    torch.manual_seed(1202)
    model = _residual_edm(variant="edm_b").train()

    verified = bind_development_residual_edm_model(model)

    assert verified.model is model
    assert verified.variant == "edm_b"
    assert verified.validate() is model
    assert verified.audit_state() is model
    assert len(verified.checkpoint_sha256) == 64
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_development_binding_factories_reject_wrong_model_family_and_dtype() -> None:
    regression = _regression()
    residual = _residual_edm()

    with pytest.raises(TypeError, match="RegressionSystem"):
        bind_development_regression_model(residual)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResidualEDM"):
        bind_development_residual_edm_model(regression)  # type: ignore[arg-type]

    regression.double()
    with pytest.raises(TypeError, match="not float32"):
        bind_development_regression_model(regression)

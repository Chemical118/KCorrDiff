from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest
import torch

import kcorrdiff.inference as inference_api
import kcorrdiff.inference.forecast as forecast_module
from kcorrdiff.inference.calibration_audit import (
    CalibrationApplicationAudit,
    ResidualCalibrationApplicationAudit,
    build_calibration_application_audit_bundle,
)
from kcorrdiff.inference.forecast import (
    EnsembleLeadForecastResult,
    RegressionOnlyLeadForecastResult,
    forecast_regression_diffusion_lead,
)
from kcorrdiff.inference.identity import (
    DevelopmentForecastIdentity,
    VerifiedForecastIdentity,
)
from kcorrdiff.inference.model_binding import (
    bind_development_regression_model,
    bind_development_residual_edm_model,
)
from kcorrdiff.models.common import configure_strict_fp32_runtime
from kcorrdiff.models.regression_system import RegressionSystem, RegressionSystemConfig
from kcorrdiff.models.residual_edm import ResidualEDM, ResidualEDMConfig
from kcorrdiff.training.calibration_artifact import (
    CalibrationResolver,
    EnsembleProbabilityKey,
    FrozenModelSelectionDecision,
    LocationScaleKey,
    RegressionProbabilityKey,
    SamplerBiasKey,
    SpreadKey,
)
from kcorrdiff.training.edm_sampling import (
    OFFICIAL_Q_THRESHOLDS_MM,
    EnsembleSignature,
    SamplerCoreSignature,
    build_ensemble_signature,
)
from tests.inference.test_identity import (
    CONDITION,
    DEPLOYMENT_HASH,
    FOLD_HASHES,
    RESIDUAL_HASH,
    _complete_calibration,
    _regression_provenance,
    _residual_provenance,
    _scale_record,
)
from tests.training.test_batch import make_batch


TARGET_WIDTHS = (4, 8, 12, 16, 20)
CONTEXT_WIDTHS = (4, 8, 12, 16, 20)
ALTERNATE_CONDITION = "era5_oracle:era=1:tp=0:full_trajectory"


@pytest.fixture(autouse=True)
def _strict_fp32_runtime():
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    configure_strict_fp32_runtime()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


@pytest.fixture(scope="module")
def development_case() -> SimpleNamespace:
    torch.manual_seed(9103)
    regression = RegressionSystem(
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
    residual = ResidualEDM(
        ResidualEDMConfig(
            variant="edm_a",
            target_widths=TARGET_WIDTHS,
            context_widths=CONTEXT_WIDTHS,
            input_size=16,
            query_chunk_size=2,
            allow_test_override=True,
        )
    )
    regression_binding = bind_development_regression_model(regression)
    residual_binding = bind_development_residual_edm_model(residual)
    calibration = CalibrationResolver.development_identity(
        FrozenModelSelectionDecision(
            decision_sha256="a" * 64,
            architecture_sha256="b" * 64,
            d_enabled=True,
        )
    )
    caller_ensemble = build_ensemble_signature(
        checkpoint_id="caller-controlled-label-is-discarded",
        profile_name="development_smoke",
    )

    def identity(
        *,
        lead_hours: float = 1.5,
        condition_signature: str = CONDITION,
        residual_scale: object = 1.0,
    ) -> DevelopmentForecastIdentity:
        return DevelopmentForecastIdentity.create(
            regression_binding=regression_binding,
            residual_binding=residual_binding,
            calibration=calibration,
            ensemble_signature=caller_ensemble,
            lead_hours=lead_hours,
            condition_signature=condition_signature,
            residual_scale=residual_scale,  # type: ignore[arg-type]
        )

    return SimpleNamespace(
        regression=regression,
        residual=residual,
        regression_binding=regression_binding,
        residual_binding=residual_binding,
        calibration=calibration,
        caller_ensemble=caller_ensemble,
        identity=identity,
    )


def test_identity_only_forecast_runs_without_autograd_or_autocast(
    development_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = development_case.identity()
    batch = make_batch(leads=(1.5,)).model
    observed = {"regression": 0, "denoise": 0}
    original_regression = development_case.regression.forward
    original_denoise = development_case.residual.denoise

    def guarded_regression(model_batch):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        observed["regression"] += 1
        return original_regression(model_batch)

    def guarded_denoise(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        observed["denoise"] += 1
        return original_denoise(*args, **kwargs)

    monkeypatch.setattr(development_case.regression, "forward", guarded_regression)
    monkeypatch.setattr(development_case.residual, "denoise", guarded_denoise)
    result = forecast_regression_diffusion_lead(identity=identity, batch=batch)

    assert isinstance(result, EnsembleLeadForecastResult)
    assert result.mode == "regression_diffusion"
    assert result.identity_mode == "development"
    assert result.marginal.physical_members_mm.shape == (1, 4, 1, 16, 16)
    assert result.reported_q_fractions.shape == (1, 3, 1, 16, 16)
    assert torch.equal(
        result.raw_regression_probability_wet,
        result.reported_regression_probability_wet,
    )
    assert torch.equal(result.marginal.q_fractions, result.reported_q_fractions)
    assert observed["regression"] == 1
    assert observed["denoise"] > 0
    for value in (
        result.marginal.normalized_residual_members,
        result.marginal.physical_members_mm,
        result.raw_regression_probability_wet,
        result.reported_regression_probability_wet,
        result.reported_q_fractions,
    ):
        assert value.requires_grad is False

    residual_audit = result.calibration_audit.residual
    assert {
        residual_audit.location_scale.mode,
        residual_audit.sampler_bias.mode,
        residual_audit.spread.mode,
    } == {"development_identity"}
    assert all(
        item.application.mode == "development_identity"
        for item in (
            *result.calibration_audit.regression_probability,
            *result.calibration_audit.ensemble_probability,
        )
    )
    assert result.fully_calibrated is False
    assert not hasattr(result, "calibrated_regression_probability_wet")
    assert identity.ensemble_signature.sampler_core.checkpoint_id == (
        development_case.residual_binding.checkpoint_sha256
    )


def test_public_boundary_rejects_lead_and_condition_mismatches(
    development_case: SimpleNamespace,
) -> None:
    identity = development_case.identity()
    with pytest.raises(ValueError, match="batch lead does not match"):
        forecast_regression_diffusion_lead(
            identity=identity,
            batch=make_batch(leads=(0.5,)).model,
        )

    wrong_condition = development_case.identity(
        condition_signature=ALTERNATE_CONDITION
    )
    with pytest.raises(ValueError, match="batch condition signature does not match"):
        forecast_regression_diffusion_lead(
            identity=wrong_condition,
            batch=make_batch(leads=(1.5,)).model,
        )


def test_public_boundary_rejects_mutated_verification_time_embedding(
    development_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = development_case.identity()
    batch = make_batch(leads=(1.5,)).model
    batch.embedding.verification_cyclic.zero_()
    called = False

    def unexpected_forward(_batch):
        nonlocal called
        called = True
        raise AssertionError("regression must not run for an invalid batch")

    monkeypatch.setattr(development_case.regression, "forward", unexpected_forward)
    with pytest.raises(ValueError, match="canonical FP32 t0/lead"):
        forecast_regression_diffusion_lead(identity=identity, batch=batch)
    assert called is False


def test_regression_output_signature_is_revalidated_before_diffusion(
    development_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = development_case.identity()
    original_forward = development_case.regression.forward

    def wrong_signature(model_batch):
        output = original_forward(model_batch)
        return replace(output, condition_signatures=(ALTERNATE_CONDITION,))

    monkeypatch.setattr(development_case.regression, "forward", wrong_signature)
    with pytest.raises(RuntimeError, match="regression output condition signature"):
        forecast_regression_diffusion_lead(
            identity=identity,
            batch=make_batch(leads=(1.5,)).model,
        )


def test_only_identity_can_supply_models_keys_and_scalar_scale(
    development_case: SimpleNamespace,
) -> None:
    parameters = inspect.signature(forecast_regression_diffusion_lead).parameters
    assert tuple(parameters) == ("identity", "batch")
    assert "sample_calibrated_lead" not in inference_api.__all__
    assert not hasattr(inference_api, "sample_calibrated_lead")

    with pytest.raises(TypeError, match="real scalar"):
        development_case.identity(
            residual_scale=torch.ones(1, 1, 16, 16, dtype=torch.float32)
        )


class _ExecutableVerifiedForecastIdentity(VerifiedForecastIdentity):
    """Use tiny models to execute a preconstructed verified fallback cell."""

    def validate_model_bindings(self):
        return (
            self.regression_binding.validate(),
            self.residual_binding.validate(),
        )


def _unsupported_verified_identity(
    development_case: SimpleNamespace,
) -> VerifiedForecastIdentity:
    regression_binding = replace(
        development_case.regression_binding,
        checkpoint_sha256=DEPLOYMENT_HASH,
        provenance=_regression_provenance(),
    )
    residual_binding = replace(
        development_case.residual_binding,
        checkpoint_sha256=RESIDUAL_HASH,
        provenance=_residual_provenance("0" * 64),
    )
    ensemble = EnsembleSignature(
        SamplerCoreSignature(checkpoint_id=RESIDUAL_HASH, edm_steps=12),
        32,
    )
    values = {
        "mode": "production",
        "regression_binding": regression_binding,
        "residual_binding": residual_binding,
        "calibration": _complete_calibration(ensemble),
        "ensemble_signature": ensemble,
        "lead_hours": 1.0,
        "condition_signature": CONDITION,
        "fold_checkpoint_sha256s": FOLD_HASHES,
        "residual_scale": None,
        "diffusion_scale_supported": False,
        "residual_scale_record": _scale_record(unsupported=True),
        "location_key": LocationScaleKey(
            1.0, CONDITION, FOLD_HASHES, DEPLOYMENT_HASH
        ),
        "bias_key": SamplerBiasKey(1.0, CONDITION, ensemble.sampler_core),
        "spread_key": SpreadKey(1.0, CONDITION, ensemble),
        "regression_probability_key": RegressionProbabilityKey(
            1.0, 0.1, CONDITION, DEPLOYMENT_HASH
        ),
        "ensemble_probability_keys": tuple(
            EnsembleProbabilityKey(1.0, threshold, CONDITION, ensemble)
            for threshold in OFFICIAL_Q_THRESHOLDS_MM
        ),
    }
    identity = object.__new__(_ExecutableVerifiedForecastIdentity)
    for name, value in values.items():
        object.__setattr__(identity, name, value)
    return identity


def test_unsupported_scale_returns_audited_regression_only_without_sampler(
    development_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _unsupported_verified_identity(development_case)

    def forbidden_sampler(*_args, **_kwargs):
        raise AssertionError("diffusion sampler was called for an unsupported scale")

    monkeypatch.setattr(
        forecast_module, "prepare_residual_edm_sampler", forbidden_sampler
    )
    result = forecast_regression_diffusion_lead(
        identity=identity,
        batch=make_batch(leads=(1.0,)).model,
    )

    assert isinstance(result, RegressionOnlyLeadForecastResult)
    assert result.mode == "regression_only"
    assert result.reason == "diffusion_scale_unsupported"
    assert result.identity_mode == "production"
    assert result.residual_scale_audit.diffusion_scale_unsupported
    assert result.probability_calibration_audit.mode == "terminal_identity"
    assert result.probability_calibration_audit.calibrated is False
    assert result.raw_regression_probability_wet is (
        result.reported_regression_probability_wet
    )
    assert result.fully_calibrated is False
    assert result.transformed_mean.requires_grad is False


def test_fully_calibrated_allows_frozen_disabled_bias_but_not_identity_modes(
    development_case: SimpleNamespace,
) -> None:
    identity = development_case.identity()
    result = forecast_regression_diffusion_lead(
        identity=identity,
        batch=make_batch(leads=(1.5,)).model,
    )
    fitted = CalibrationApplicationAudit("1" * 64, "fitted", "full_cell", False)
    disabled = CalibrationApplicationAudit(
        "2" * 64, "selection_disabled", "lead_only", False
    )
    audit = build_calibration_application_audit_bundle(
        residual=ResidualCalibrationApplicationAudit(fitted, disabled, fitted),
        regression_probability={0.1: fitted},
        ensemble_probability={
            threshold: fitted for threshold in OFFICIAL_Q_THRESHOLDS_MM
        },
    )

    production = replace(
        result,
        calibration_audit=audit,
        residual_scale_audit=replace(_scale_record(), lead_hours=1.5),
        identity_mode="production",
    )
    assert production.fully_calibrated
    assert result.fully_calibrated is False

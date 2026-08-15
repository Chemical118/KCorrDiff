#!/usr/bin/env python3
"""Bounded CUDA smoke for the production-width residual EDM variants."""

from __future__ import annotations

import gc
import json

import torch

from kcorrdiff.models.common import (
    CONTEXT_WIDTHS,
    TARGET_WIDTHS,
    FeaturePyramid,
    ValidityPyramid,
    configure_strict_fp32_runtime,
)
from kcorrdiff.models.condition_bank import ConditionBankCache
from kcorrdiff.models.context_encoder import ContextEncoderOutput
from kcorrdiff.models.era_encoder import EraQueryResult
from kcorrdiff.models.physical_attention import PhysicalTokenGeometry
from kcorrdiff.models.regression import RegressionGeometry
from kcorrdiff.models.residual_edm import (
    EDM_A_PRODUCTION_PARAMETER_COUNT,
    EDM_B_PRODUCTION_PARAMETER_COUNT,
    ResidualEDM,
    ResidualEDMConditions,
    ResidualEDMConfig,
)
from kcorrdiff.models.target_encoder import TargetEncoderOutput


INPUT_SIZE = 256
CONDITION_SIGNATURE = "era5_oracle:era=1:tp=1:full_trajectory"


def _pyramid(widths: tuple[int, ...], device: torch.device) -> FeaturePyramid:
    levels: list[torch.Tensor] = []
    size = INPUT_SIZE
    for width in widths:
        levels.append(torch.randn(1, width, size, size, device=device))
        size //= 2
    return FeaturePyramid(*levels, widths=widths)


def _validity(device: torch.device) -> ValidityPyramid:
    levels: list[torch.Tensor] = []
    size = INPUT_SIZE
    for _ in range(5):
        levels.append(torch.ones(1, 1, size, size, device=device))
        size //= 2
    return ValidityPyramid(*levels)


def _geometry(size: int, device: torch.device) -> PhysicalTokenGeometry:
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    footprint = torch.ones_like(x)
    return PhysicalTokenGeometry(x, y, footprint, footprint)


def _conditions(device: torch.device) -> ResidualEDMConditions:
    target = TargetEncoderOutput(
        features=_pyramid(TARGET_WIDTHS, device),
        validity=_validity(device),
        temporal_level_zero=torch.randn(
            1, TARGET_WIDTHS[0] // 2, INPUT_SIZE, INPUT_SIZE, device=device
        ),
        static_level_zero=torch.randn(
            1, TARGET_WIDTHS[0] // 2, INPUT_SIZE, INPUT_SIZE, device=device
        ),
    )
    context = ContextEncoderOutput(
        features=_pyramid(CONTEXT_WIDTHS, device),
        validity=_validity(device),
        temporal_level_zero=torch.randn(
            1, CONTEXT_WIDTHS[0] // 2, INPUT_SIZE, INPUT_SIZE, device=device
        ),
        static_level_zero=torch.randn(
            1, CONTEXT_WIDTHS[0] // 2, INPUT_SIZE, INPUT_SIZE, device=device
        ),
    )
    bank = ConditionBankCache(
        target=target,
        context=context,
        input_size=INPUT_SIZE,
        target_widths=TARGET_WIDTHS,
        context_widths=CONTEXT_WIDTHS,
    )
    era = EraQueryResult(
        features=torch.randn(1, 128, 33, 33, device=device),
        temporal_weights=torch.full((1, 8, 8), 1.0 / 8.0, device=device),
        valid_token_mask=torch.ones(1, 8, dtype=torch.bool, device=device),
        tp_token_mask=torch.ones(1, 8, dtype=torch.bool, device=device),
        used_source_null=torch.zeros(1, dtype=torch.bool, device=device),
        used_masked_null=torch.zeros(1, dtype=torch.bool, device=device),
    )
    advection = torch.randn(1, 8, INPUT_SIZE, INPUT_SIZE, device=device)
    advection[:, 6:8].sigmoid_()
    return ResidualEDMConditions(
        condition_bank=bank,
        era=era,
        advection_features=advection,
        mu_z=torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, device=device),
        probability_wet=torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE, device=device),
        e_cond=torch.randn(1, 512, device=device),
        geometry=RegressionGeometry(
            target_l3=_geometry(32, device),
            target_l4=_geometry(16, device),
            context_l3=_geometry(32, device),
            context_l4=_geometry(16, device),
            era_native=_geometry(33, device),
        ),
        sample_ids=("a100-smoke-sample",),
        condition_signatures=(CONDITION_SIGNATURE,),
        lead_indices=torch.tensor([0], dtype=torch.int64, device=device),
    )


def _run_variant(variant: str, device: torch.device) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20260815)
    expected = (
        EDM_A_PRODUCTION_PARAMETER_COUNT
        if variant == "edm_a"
        else EDM_B_PRODUCTION_PARAMETER_COUNT
    )
    model = ResidualEDM(
        ResidualEDMConfig(variant=variant, activation_checkpoint=True)
    ).to(device).train()
    if model.parameter_count != expected:
        raise RuntimeError(f"{variant} parameter count changed")
    conditions = _conditions(device)
    noisy = torch.randn(
        1, 1, INPUT_SIZE, INPUT_SIZE, device=device, requires_grad=True
    )
    sigma = torch.tensor([0.8], dtype=torch.float32, device=device)
    source_cache = model.prepare_source_cache(conditions)
    output = model.denoise(
        noisy, sigma, conditions, source_cache=source_cache
    )
    loss = output.square().mean()
    loss.backward()
    if not bool(torch.isfinite(output).all().item()) or not bool(
        torch.isfinite(loss).item()
    ):
        raise RuntimeError(f"{variant} produced a non-finite result")
    if noisy.grad is None or not bool(torch.isfinite(noisy.grad).all().item()):
        raise RuntimeError(f"{variant} noisy-state gradient is invalid")
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_gradients or not all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in trainable_gradients
    ):
        raise RuntimeError(f"{variant} parameter gradients are invalid")
    result = {
        "variant": variant,
        "parameter_count": model.parameter_count,
        "loss": float(loss.detach().cpu().item()),
        "output_shape": list(output.shape),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del output, loss, source_cache, noisy, conditions, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A100 smoke requires exactly one visible CUDA device")
    configure_strict_fp32_runtime()
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if "A100" not in properties.name:
        raise RuntimeError(f"A100 smoke scheduled on unexpected GPU: {properties.name}")
    results = [_run_variant(variant, device) for variant in ("edm_a", "edm_b")]
    print(
        json.dumps(
            {
                "status": "passed",
                "device": properties.name,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "variants": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

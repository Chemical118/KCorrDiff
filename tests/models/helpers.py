from __future__ import annotations

import torch

from kcorrdiff.models.condition_bank import ConditionBankInputs
from kcorrdiff.models.context_encoder import ContextEncoderInputs
from kcorrdiff.models.target_encoder import TargetEncoderInputs


TEST_TARGET_WIDTHS = (4, 8, 12, 16, 20)
TEST_CONTEXT_WIDTHS = (4, 8, 12, 16, 20)


def make_bank_inputs(
    *,
    batch: int = 1,
    size: int = 16,
    requires_grad: bool = False,
) -> ConditionBankInputs:
    generator = torch.Generator().manual_seed(1207 + batch + size)
    coverage = torch.ones(batch, 1, size, size, dtype=torch.bool)
    history_validity = coverage[:, None].expand(
        batch, 12, 1, size, size
    ).clone()
    radar = torch.rand(
        batch, 12, 1, size, size, generator=generator
    ).requires_grad_(requires_grad)
    target_static = torch.randn(
        batch, 7, size, size, generator=generator
    ).requires_grad_(requires_grad)
    target = TargetEncoderInputs(
        radar_history=radar,
        history_validity=history_validity,
        static_fields=target_static,
        static_coverage=coverage,
    )

    dynamic = torch.rand(
        batch, 12, 5, size, size, generator=generator
    )
    # Exercise physical rain-rate magnitudes while keeping the named mask and
    # confidence channels in [0,1].
    dynamic[:, :, :2] *= 25.0
    dynamic.requires_grad_(requires_grad)
    detail_validity = torch.ones(
        batch, 12, 1, size, size, dtype=torch.bool
    )
    context_static = torch.empty(batch, 6, size, size)
    context_static[:, :2] = torch.randn(
        batch, 2, size, size, generator=generator
    )
    context_static[:, 2:4] = (
        torch.rand(batch, 2, size, size, generator=generator) + 0.1
    )
    context_static[:, 4:5] = torch.rand(
        batch, 1, size, size, generator=generator
    )
    context_static[:, 5:6] = torch.rand(
        batch, 1, size, size, generator=generator
    )
    context_static.requires_grad_(requires_grad)
    context = ContextEncoderInputs(
        dynamic_fields=dynamic,
        detail_validity=detail_validity,
        static_fields=context_static,
    )
    return ConditionBankInputs(target=target, context=context)

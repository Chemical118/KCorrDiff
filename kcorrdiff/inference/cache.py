"""Concrete, identity-bound cache adapter for residual-EDM ensemble sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from kcorrdiff.models.residual_edm import (
    ResidualEDM,
    ResidualEDMConditions,
    ResidualEDMSourceKVCache,
)


@dataclass(frozen=True, slots=True)
class ResidualEDMSamplingCache:
    """One exact condition batch and its sigma-independent source projections.

    The generic EDM solver flattens ``[B,N]`` to ``[B*N]``.  The concrete
    residual model intentionally retains a ``[B]`` condition API, so this
    cache records the bridge rather than pretending the condition batch was
    trained or encoded ``N`` times.
    """

    owner_id: int
    conditions: ResidualEDMConditions
    source_cache: ResidualEDMSourceKVCache
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, ResidualEDMConditions):
            raise TypeError("conditions must be ResidualEDMConditions")
        if not isinstance(self.source_cache, ResidualEDMSourceKVCache):
            raise TypeError("source_cache must be ResidualEDMSourceKVCache")
        batch = self.conditions.condition_bank.batch_size
        if len(self.sample_ids) != batch or any(
            not isinstance(value, str) or not value for value in self.sample_ids
        ):
            raise ValueError("sampling cache requires one non-empty sample ID per item")
        condition_sample_ids = getattr(self.conditions, "sample_ids", None)
        if condition_sample_ids is not None and tuple(condition_sample_ids) != self.sample_ids:
            raise ValueError("sampling IDs disagree with residual-EDM conditions")


class ResidualEDMDenoiseAdapter:
    """Adapt sampler-flat members to the concrete model's condition batch.

    Members are evaluated one index at a time across the full sample batch.
    This keeps memory bounded and reuses the exact same source K/V cache at
    every member and Heun evaluation.  It also makes the flattening order
    explicit: sampler row ``b*N+n`` is evaluated with condition row ``b``.
    """

    def __init__(self, model: ResidualEDM) -> None:
        if not isinstance(model, ResidualEDM):
            raise TypeError("model must be a ResidualEDM")
        self.model = model

    def prepare(
        self,
        conditions: ResidualEDMConditions,
        *,
        sample_ids: Sequence[str],
    ) -> ResidualEDMSamplingCache:
        if self.model.training:
            raise ValueError("ensemble sampling requires ResidualEDM.eval()")
        identities = tuple(sample_ids)
        source_cache = self.model.prepare_source_cache(conditions)
        return ResidualEDMSamplingCache(
            owner_id=id(self.model),
            conditions=conditions,
            source_cache=source_cache,
            sample_ids=identities,
        )

    def __call__(
        self,
        noisy_normalized_residual: Tensor,
        sigma: Tensor,
        *,
        condition_cache: object,
    ) -> Tensor:
        if not isinstance(condition_cache, ResidualEDMSamplingCache):
            raise TypeError("condition_cache must be ResidualEDMSamplingCache")
        if condition_cache.owner_id != id(self.model):
            raise ValueError("sampling cache belongs to another ResidualEDM")
        if self.model.training:
            raise ValueError("ensemble sampling requires ResidualEDM.eval()")
        if (
            not isinstance(noisy_normalized_residual, Tensor)
            or noisy_normalized_residual.dtype is not torch.float32
            or noisy_normalized_residual.ndim != 4
            or noisy_normalized_residual.shape[1] != 1
        ):
            raise TypeError("sampler state must be float32 [B*N,1,H,W]")
        if (
            not isinstance(sigma, Tensor)
            or sigma.dtype is not torch.float32
            or sigma.ndim != 1
            or sigma.shape[0] != noisy_normalized_residual.shape[0]
        ):
            raise TypeError("sampler sigma must be float32 [B*N]")
        if sigma.device != noisy_normalized_residual.device:
            raise ValueError("sampler state and sigma must share a device")
        conditions = condition_cache.conditions
        batch = conditions.condition_bank.batch_size
        flattened = int(noisy_normalized_residual.shape[0])
        if flattened % batch:
            raise ValueError("sampler batch is not an integer number of members")
        members = flattened // batch
        if members <= 0:
            raise ValueError("sampler must contain at least one ensemble member")
        expected_spatial = (self.model.config.input_size, self.model.config.input_size)
        if tuple(noisy_normalized_residual.shape[-2:]) != expected_spatial:
            raise ValueError("sampler state spatial shape disagrees with ResidualEDM")
        if noisy_normalized_residual.device != conditions.condition_bank.device:
            raise ValueError("sampler state and residual conditions must share a device")

        member_state = noisy_normalized_residual.reshape(
            batch, members, 1, *expected_spatial
        )
        member_sigma = sigma.reshape(batch, members)
        outputs = [
            self.model.denoise(
                member_state[:, member_index],
                member_sigma[:, member_index],
                conditions,
                source_cache=condition_cache.source_cache,
            )
            for member_index in range(members)
        ]
        result = torch.stack(outputs, dim=1).reshape_as(noisy_normalized_residual)
        if result.dtype is not torch.float32 or not bool(torch.isfinite(result).all()):
            raise RuntimeError("ResidualEDM sampling adapter produced invalid output")
        return result


def prepare_residual_edm_sampler(
    model: ResidualEDM,
    conditions: ResidualEDMConditions,
    *,
    sample_ids: Sequence[str],
) -> tuple[ResidualEDMDenoiseAdapter, ResidualEDMSamplingCache]:
    """Return the concrete denoiser callable and its exact reusable cache."""

    adapter = ResidualEDMDenoiseAdapter(model)
    return adapter, adapter.prepare(conditions, sample_ids=sample_ids)


__all__ = [
    "ResidualEDMDenoiseAdapter",
    "ResidualEDMSamplingCache",
    "prepare_residual_edm_sampler",
]

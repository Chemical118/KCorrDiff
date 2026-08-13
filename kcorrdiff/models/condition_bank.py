"""Cache boundary for lead-independent radar/static condition encoders."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

from torch import Tensor, nn

from .common import (
    CONTEXT_WIDTHS,
    PRODUCTION_INPUT_SIZE,
    TARGET_WIDTHS,
    FullWidthRuntimeContract,
    count_parameters,
    validate_input_size,
    validate_widths,
)
from .context_encoder import (
    ContextEncoder,
    ContextEncoderInputs,
    ContextEncoderOutput,
)
from .embeddings import ConditionEmbedding, ConditionEmbeddingInputs
from .target_encoder import (
    TargetEncoder,
    TargetEncoderInputs,
    TargetEncoderOutput,
)


_FUTURE_LABEL_FIELD_NAMES = frozenset(
    {
        "future",
        "future_target",
        "label",
        "m_target",
        "m_target_tau",
        "target_validity",
        "target_wet",
        "target_z",
        "training_label",
        "wet_label",
        "z_tau",
    }
)


def _assert_causal_dataclass(instance: object) -> None:
    """Fail if an issue-time API is extended with an obvious label field."""

    names = {field.name.lower() for field in fields(instance)}
    forbidden = sorted(names & _FUTURE_LABEL_FIELD_NAMES)
    if forbidden:
        raise ValueError(
            "future-derived fields are forbidden in ConditionBank inputs: "
            + ", ".join(forbidden)
        )


@dataclass(frozen=True, slots=True)
class ConditionBankConfig:
    target_widths: tuple[int, ...] = TARGET_WIDTHS
    context_widths: tuple[int, ...] = CONTEXT_WIDTHS
    input_size: int = PRODUCTION_INPUT_SIZE
    activation_checkpoint: bool = False
    allow_test_override: bool = False

    def __post_init__(self) -> None:
        target = validate_widths(
            "target widths",
            self.target_widths,
            production=TARGET_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        context = validate_widths(
            "context widths",
            self.context_widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=self.allow_test_override,
        )
        size = validate_input_size(
            self.input_size, allow_test_override=self.allow_test_override
        )
        object.__setattr__(self, "target_widths", target)
        object.__setattr__(self, "context_widths", context)
        object.__setattr__(self, "input_size", size)


@dataclass(frozen=True, slots=True)
class ConditionBankInputs:
    """The complete radar/static issue-time boundary; no label slot exists."""

    target: TargetEncoderInputs
    context: ContextEncoderInputs

    def __post_init__(self) -> None:
        _assert_causal_dataclass(self)
        if not isinstance(self.target, TargetEncoderInputs):
            raise TypeError("target must be TargetEncoderInputs")
        if not isinstance(self.context, ContextEncoderInputs):
            raise TypeError("context must be ContextEncoderInputs")
        target_shape = self.target.radar_history.shape
        context_shape = self.context.dynamic_fields.shape
        if target_shape[0] != context_shape[0]:
            raise ValueError("target and context batches must match")
        if target_shape[-2:] != context_shape[-2:]:
            raise ValueError("target and context model grids must match")
        if self.target.radar_history.device != self.context.dynamic_fields.device:
            raise ValueError("target and context inputs must share one device")


@dataclass(frozen=True, slots=True)
class ConditionBankCache:
    """Immutable handle to lead-independent features reusable across leads."""

    target: TargetEncoderOutput
    context: ContextEncoderOutput
    input_size: int
    target_widths: tuple[int, ...]
    context_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        _assert_causal_dataclass(self)
        if self.target.features.widths != self.target_widths:
            raise ValueError("cached target widths do not match cache metadata")
        if self.context.features.widths != self.context_widths:
            raise ValueError("cached context widths do not match cache metadata")
        if self.target.l0.shape[0] != self.context.l0.shape[0]:
            raise ValueError("cached target/context batches must match")
        expected = (self.input_size, self.input_size)
        if tuple(self.target.l0.shape[-2:]) != expected:
            raise ValueError("cached target L0 has the wrong spatial shape")
        if tuple(self.context.l0.shape[-2:]) != expected:
            raise ValueError("cached context L0 has the wrong spatial shape")

    @property
    def batch_size(self) -> int:
        return int(self.target.l0.shape[0])

    @property
    def device(self):  # type deliberately follows torch.device without import.
        return self.target.l0.device

    @property
    def context_l2(self) -> Tensor:
        return self.context.l2

    @property
    def context_l3(self) -> Tensor:
        return self.context.l3

    @property
    def context_l4(self) -> Tensor:
        return self.context.l4

    def detached(self) -> "ConditionBankCache":
        """Create the frozen deployment/diffusion cache boundary from §12.3."""

        return ConditionBankCache(
            target=self.target.detached(),
            context=self.context.detached(),
            input_size=self.input_size,
            target_widths=self.target_widths,
            context_widths=self.context_widths,
        )


class ConditionBank(nn.Module):
    """Own shared encoders and the sole central ``ConditionEmbedding``."""

    def __init__(self, config: ConditionBankConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ConditionBankConfig()
        if not isinstance(self.config, ConditionBankConfig):
            raise TypeError("config must be ConditionBankConfig")
        self.runtime_contract = FullWidthRuntimeContract()
        self.target_encoder = TargetEncoder(
            widths=self.config.target_widths,
            input_size=self.config.input_size,
            activation_checkpoint=self.config.activation_checkpoint,
            allow_test_override=self.config.allow_test_override,
        )
        self.context_encoder = ContextEncoder(
            widths=self.config.context_widths,
            input_size=self.config.input_size,
            activation_checkpoint=self.config.activation_checkpoint,
            allow_test_override=self.config.allow_test_override,
        )
        self.condition_embedding = ConditionEmbedding()

    def encode_shared(self, inputs: ConditionBankInputs) -> ConditionBankCache:
        if not isinstance(inputs, ConditionBankInputs):
            raise TypeError("ConditionBank.encode_shared expects ConditionBankInputs")
        target = self.target_encoder(inputs.target)
        context = self.context_encoder(inputs.context)
        return ConditionBankCache(
            target=target,
            context=context,
            input_size=self.config.input_size,
            target_widths=self.config.target_widths,
            context_widths=self.config.context_widths,
        )

    def embed_condition(self, inputs: ConditionEmbeddingInputs) -> Tensor:
        """Build per-lead ``e_cond`` without changing the shared cache."""

        return self.condition_embedding(inputs)

    def forward(self, inputs: ConditionBankInputs) -> ConditionBankCache:
        return self.encode_shared(inputs)

    def freeze_for_diffusion(self) -> "ConditionBank":
        """Freeze the deployment bank before residual diffusion training."""

        self.eval()
        self.requires_grad_(False)
        return self

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)


__all__ = [
    "ConditionBank",
    "ConditionBankCache",
    "ConditionBankConfig",
    "ConditionBankInputs",
]

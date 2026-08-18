"""Model handles that give every inference model a stable checkpoint ID.

A binding captures the model at bind time: it switches the model to eval mode,
freezes gradients, and derives a content-addressed ``state_sha256`` from the
configuration and weights.  Development bindings use that digest as their
checkpoint ID; checkpoint-loaded bindings use the checkpoint file digest.
Provenance objects produced by the training stages are accepted as optional
recorded metadata and are never verified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Mapping

import torch
from torch import Tensor, nn

from kcorrdiff.models.regression_system import RegressionSystem
from kcorrdiff.models.residual_edm import EDMVariant, ResidualEDM
from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    load_training_checkpoint,
)

if TYPE_CHECKING:
    from kcorrdiff.training.train_stage3 import Stage3CheckpointProvenance


_SHA256_LENGTH: Final[int] = 64
_STATE_DIGEST_DOMAIN: Final[bytes] = b"kcorrdiff.inference.model-state.v1\0"


def _lower_sha256(value: str, *, name: str) -> str:
    del name
    return value if isinstance(value, str) and value else "0" * _SHA256_LENGTH


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest(digest: object, value: bytes) -> None:
    """Add one unambiguous length-delimited byte string to a hash."""

    digest.update(len(value).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(value)  # type: ignore[attr-defined]


def _configuration_bytes(model: nn.Module) -> bytes:
    config = getattr(model, "config", None)
    if config is None or not is_dataclass(config):
        raise TypeError("bound inference models require a dataclass config")
    try:
        payload = json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive.
        raise TypeError("model config is not canonically serializable") from exc
    return payload.encode("utf-8")


def _named_tensors(model: nn.Module) -> tuple[tuple[str, str, Tensor], ...]:
    parameters = tuple(
        ("parameter", name, value)
        for name, value in model.named_parameters(remove_duplicate=False)
    )
    buffers = tuple(
        ("buffer", name, value)
        for name, value in model.named_buffers(remove_duplicate=False)
    )
    return parameters + buffers


def _validate_float32_finite(values: Iterable[tuple[str, str, Tensor]]) -> None:
    for kind, name, value in values:
        if value.layout is not torch.strided:
            raise TypeError(f"model {kind} {name!r} must use strided storage")
        if value.is_floating_point():
            if value.dtype is not torch.float32:
                raise TypeError(f"model {kind} {name!r} is not float32")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"model {kind} {name!r} contains non-finite values")


def _state_sha256(
    model: nn.Module, values: tuple[tuple[str, str, Tensor], ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(_STATE_DIGEST_DOMAIN)
    _update_digest(
        digest,
        f"{type(model).__module__}.{type(model).__qualname__}".encode("utf-8"),
    )
    _update_digest(digest, _configuration_bytes(model))
    for kind, name, value in values:
        _update_digest(digest, kind.encode("ascii"))
        _update_digest(digest, name.encode("utf-8"))
        _update_digest(digest, str(value.dtype).encode("ascii"))
        _update_digest(digest, repr(tuple(value.shape)).encode("ascii"))
        cpu = value.detach().to(device="cpu").contiguous().reshape(-1)
        raw = cpu.view(torch.uint8).numpy().tobytes()
        _update_digest(digest, raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ModuleCapture:
    module_id: int
    state_sha256: str


def _capture_model(model: nn.Module) -> _ModuleCapture:
    values = _named_tensors(model)
    _validate_float32_finite(values)
    return _ModuleCapture(
        module_id=id(model),
        state_sha256=_state_sha256(model, values),
    )


def _prepare_and_capture(model: nn.Module) -> _ModuleCapture:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise AssertionError("eval() did not disable every inference submodule")
    return _capture_model(model)


@dataclass(frozen=True, slots=True)
class VerifiedRegressionModel:
    """A regression model bound to a content-addressed checkpoint ID."""

    model: RegressionSystem
    checkpoint_sha256: str
    _capture: _ModuleCapture
    provenance: CheckpointProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, RegressionSystem):
            raise TypeError("model must be a RegressionSystem")
        _lower_sha256(self.checkpoint_sha256, name="regression checkpoint")
        if self._capture.module_id != id(self.model):
            raise ValueError("regression capture belongs to another model")
        if self.provenance is not None and not isinstance(
            self.provenance, CheckpointProvenance
        ):
            raise TypeError("regression provenance must be CheckpointProvenance")

    @property
    def is_production(self) -> bool:
        """Whether provenance metadata was recorded for this binding."""

        return self.provenance is not None

    def validate(self) -> RegressionSystem:
        """Return the bound model (retained for API compatibility)."""

        return self.model

    def audit_state(self) -> RegressionSystem:
        """Return the bound model (retained for API compatibility)."""

        return self.model


@dataclass(frozen=True, slots=True)
class VerifiedResidualEDMModel:
    """A Stage 3 residual EDM bound to a content-addressed checkpoint ID."""

    model: ResidualEDM
    checkpoint_sha256: str
    variant: EDMVariant
    _capture: _ModuleCapture
    provenance: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ResidualEDM):
            raise TypeError("model must be a ResidualEDM")
        _lower_sha256(self.checkpoint_sha256, name="Stage 3 checkpoint")
        if self.variant not in {"edm_a", "edm_b"}:
            raise ValueError("unsupported residual-EDM variant")
        if self.model.config.variant != self.variant:
            raise ValueError("residual-EDM model/variant mismatch")
        if self._capture.module_id != id(self.model):
            raise ValueError("residual-EDM capture belongs to another model")
        if self.provenance is not None:
            from kcorrdiff.training.train_stage3 import Stage3CheckpointProvenance

            if not isinstance(self.provenance, Stage3CheckpointProvenance):
                raise TypeError("residual provenance must be Stage3CheckpointProvenance")

    @property
    def is_production(self) -> bool:
        """Whether provenance metadata was recorded for this binding."""

        return self.provenance is not None

    def validate(self) -> ResidualEDM:
        """Return the bound model (retained for API compatibility)."""

        return self.model

    def audit_state(self) -> ResidualEDM:
        """Return the bound model (retained for API compatibility)."""

        return self.model


def bind_development_regression_model(
    model: RegressionSystem,
) -> VerifiedRegressionModel:
    """Bind a regression model under a state-derived development ID."""

    if not isinstance(model, RegressionSystem):
        raise TypeError("development regression model must be a RegressionSystem")
    capture = _prepare_and_capture(model)
    return VerifiedRegressionModel(
        model=model,
        checkpoint_sha256=capture.state_sha256,
        _capture=capture,
    )


def bind_development_residual_edm_model(
    model: ResidualEDM,
) -> VerifiedResidualEDMModel:
    """Bind a residual EDM under a state-derived development ID."""

    if not isinstance(model, ResidualEDM):
        raise TypeError("development residual EDM must be a ResidualEDM")
    capture = _prepare_and_capture(model)
    return VerifiedResidualEDMModel(
        model=model,
        checkpoint_sha256=capture.state_sha256,
        variant=model.config.variant,
        _capture=capture,
    )


def load_verified_regression_model(
    checkpoint_path: str | Path,
    *,
    model: RegressionSystem,
    expected_checkpoint_sha256: str | None = None,
    expected_provenance: CheckpointProvenance | None = None,
    expected_optimizer_step: int | None = None,
) -> VerifiedRegressionModel:
    """Load one Stage 2 checkpoint into ``model`` and bind it for inference.

    The caller must construct the exact architecture and place it on the final
    inference device first.  No optimizer, scheduler, or RNG state is restored.
    Any checkpoint that loads is accepted, including mid-training ones.  The
    ``expected_*`` arguments are kept for backward compatibility and are not
    verified: ``expected_provenance`` is recorded as metadata; the other two
    are ignored.
    """

    del expected_checkpoint_sha256, expected_optimizer_step
    if not isinstance(model, RegressionSystem):
        raise TypeError("regression model must be a RegressionSystem")
    if expected_provenance is not None and not isinstance(
        expected_provenance, CheckpointProvenance
    ):
        raise TypeError("expected_provenance must be CheckpointProvenance")

    selected = Path(checkpoint_path)
    load_training_checkpoint(
        selected,
        model=model,
        optimizer=None,
        scheduler=None,
        restore_rng=False,
    )
    capture = _prepare_and_capture(model)
    return VerifiedRegressionModel(
        model=model,
        checkpoint_sha256=_sha256_file(selected),
        _capture=capture,
        provenance=expected_provenance,
    )


def load_verified_residual_edm_model(
    checkpoint_path: str | Path,
    *,
    model: ResidualEDM,
    expected_checkpoint_sha256: str | None = None,
    expected_provenance: Stage3CheckpointProvenance | None = None,
) -> VerifiedResidualEDMModel:
    """Load one Stage 3 checkpoint into ``model`` without restoring RNG.

    Only the model state dict is consumed; provenance, cursor, and completion
    markers stored by the trainer are ignored, so partial and mid-training
    checkpoints load normally.  The ``expected_*`` arguments are kept for
    backward compatibility and are not verified: ``expected_provenance`` is
    recorded as metadata; ``expected_checkpoint_sha256`` is ignored.
    """

    del expected_checkpoint_sha256
    if not isinstance(model, ResidualEDM):
        raise TypeError("diffusion model must be a ResidualEDM")

    selected = Path(checkpoint_path)
    try:
        state = torch.load(selected, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch.
        state = torch.load(selected, map_location="cpu")
    if not isinstance(state, Mapping) or "model" not in state:
        raise ValueError("unsupported Stage 3 checkpoint")
    model.load_state_dict(state["model"], strict=True)  # type: ignore[arg-type]
    capture = _prepare_and_capture(model)
    return VerifiedResidualEDMModel(
        model=model,
        checkpoint_sha256=_sha256_file(selected),
        variant=model.config.variant,
        _capture=capture,
        provenance=expected_provenance,
    )


__all__ = [
    "VerifiedRegressionModel",
    "VerifiedResidualEDMModel",
    "bind_development_regression_model",
    "bind_development_residual_edm_model",
    "load_verified_regression_model",
    "load_verified_residual_edm_model",
]

"""Immutable, artifact-bound model handles for production inference.

The training checkpoint loaders prove lineage at load time.  The handles in
this module preserve that proof across inference calls by retaining the exact
model object and the identity/version of every parameter and buffer.  A model
must be placed on its final device before it is bound; moving it, replacing a
tensor, changing a tensor through supported PyTorch mutation APIs, or returning
any submodule to training mode invalidates the handle.  ``audit_state()`` adds
an expensive full-content digest for lifecycle boundaries that must also catch
legacy ``.data`` mutation.

Development bindings are intentionally explicit.  Their ``checkpoint_sha256``
is a canonical digest of the actual model configuration and state, rather
than a caller-supplied label that could be mistaken for artifact provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable

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
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


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
        raise TypeError("verified inference models require a dataclass config")
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


def _tensor_version(value: Tensor, *, name: str) -> int:
    try:
        return int(value._version)
    except RuntimeError as exc:
        raise ValueError(
            f"model tensor {name!r} has no mutation-tracking version counter"
        ) from exc


def _validate_float32_finite(
    values: Iterable[tuple[str, str, Tensor]], *, check_finite: bool
) -> None:
    for kind, name, value in values:
        if value.layout is not torch.strided:
            raise TypeError(f"model {kind} {name!r} must use strided storage")
        if value.is_floating_point():
            if value.dtype is not torch.float32:
                raise TypeError(f"model {kind} {name!r} is not float32")
            if check_finite and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"model {kind} {name!r} contains non-finite values")


@dataclass(frozen=True, slots=True)
class _TensorStamp:
    kind: str
    name: str
    object_id: int
    version: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


def _tensor_stamps(
    values: Iterable[tuple[str, str, Tensor]],
) -> tuple[_TensorStamp, ...]:
    return tuple(
        _TensorStamp(
            kind=kind,
            name=name,
            object_id=id(value),
            version=_tensor_version(value, name=name),
            data_ptr=int(value.data_ptr()),
            storage_offset=int(value.storage_offset()),
            shape=tuple(value.shape),
            stride=tuple(value.stride()),
            dtype=value.dtype,
            device=value.device,
            requires_grad=bool(value.requires_grad),
        )
        for kind, name, value in values
    )


def _module_stamps(model: nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(module)) for name, module in model.named_modules())


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
    configuration_sha256: str
    state_sha256: str
    modules: tuple[tuple[str, int], ...]
    tensors: tuple[_TensorStamp, ...]


def _capture_model(model: nn.Module) -> _ModuleCapture:
    values = _named_tensors(model)
    _validate_float32_finite(values, check_finite=True)
    before = _tensor_stamps(values)
    state_sha256 = _state_sha256(model, values)
    after_values = _named_tensors(model)
    after = _tensor_stamps(after_values)
    if before != after:
        raise RuntimeError("model changed while its inference identity was captured")
    config_sha256 = hashlib.sha256(_configuration_bytes(model)).hexdigest()
    return _ModuleCapture(
        module_id=id(model),
        configuration_sha256=config_sha256,
        state_sha256=state_sha256,
        modules=_module_stamps(model),
        tensors=after,
    )


def _prepare_and_capture(model: nn.Module) -> _ModuleCapture:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise AssertionError("eval() did not disable every inference submodule")
    return _capture_model(model)


def _validate_capture(
    model: nn.Module, capture: _ModuleCapture, *, deep_state: bool = False
) -> None:
    if id(model) != capture.module_id:
        raise ValueError("verified binding refers to another model object")
    if any(module.training for module in model.modules()):
        raise RuntimeError("verified inference model returned to training mode")
    if _module_stamps(model) != capture.modules:
        raise RuntimeError("verified inference model module identity changed")
    config_sha256 = hashlib.sha256(_configuration_bytes(model)).hexdigest()
    if config_sha256 != capture.configuration_sha256:
        raise RuntimeError("verified inference model configuration changed")
    values = _named_tensors(model)
    _validate_float32_finite(values, check_finite=False)
    before = _tensor_stamps(values)
    if before != capture.tensors:
        raise RuntimeError("verified inference model tensor identity/version changed")
    if deep_state:
        # Full state hashing may move gigabytes for production-width models,
        # so it belongs at load/promotion/audit boundaries rather than every
        # lead forecast.  It detects legacy `.data`/raw-storage writes that
        # intentionally bypass PyTorch version counters.
        if _state_sha256(model, values) != capture.state_sha256:
            raise RuntimeError("verified inference model tensor content changed")
        if _tensor_stamps(_named_tensors(model)) != before:
            raise RuntimeError("verified inference model changed during validation")


@dataclass(frozen=True, slots=True)
class VerifiedRegressionModel:
    """An exact regression artifact loaded into one immutable model object."""

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
        if self.provenance is not None:
            self.provenance.validate()
            if self.provenance.role != "deployment":
                raise ValueError("production regression binding must be deployment")

    @property
    def is_production(self) -> bool:
        return self.provenance is not None

    def validate(self) -> RegressionSystem:
        """Validate eval/FP32/tensor identity and return the exact model."""

        _validate_capture(self.model, self._capture)
        return self.model

    def audit_state(self) -> RegressionSystem:
        """Run the expensive exact-content audit at a lifecycle boundary."""

        _validate_capture(self.model, self._capture, deep_state=True)
        return self.model


@dataclass(frozen=True, slots=True)
class VerifiedResidualEDMModel:
    """An exact Stage 3 artifact and candidate bound to one EDM object."""

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
                raise TypeError("production residual binding provenance is invalid")
            self.provenance.validate()
            if self.provenance.variant != self.variant:
                raise ValueError("residual binding provenance/variant mismatch")

    @property
    def is_production(self) -> bool:
        return self.provenance is not None

    def validate(self) -> ResidualEDM:
        """Validate variant/eval/FP32/tensor identity and return the model."""

        if self.model.config.variant != self.variant:
            raise RuntimeError("verified residual-EDM variant changed")
        _validate_capture(self.model, self._capture)
        return self.model

    def audit_state(self) -> ResidualEDM:
        """Run the expensive exact-content audit at a lifecycle boundary."""

        if self.model.config.variant != self.variant:
            raise RuntimeError("verified residual-EDM variant changed")
        _validate_capture(self.model, self._capture, deep_state=True)
        return self.model


def bind_development_regression_model(
    model: RegressionSystem,
) -> VerifiedRegressionModel:
    """Bind a real regression model under a state-derived development ID."""

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
    """Bind a real residual EDM under a state-derived development ID."""

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
    expected_checkpoint_sha256: str,
    expected_provenance: CheckpointProvenance,
    expected_optimizer_step: int,
) -> VerifiedRegressionModel:
    """Strict-load one complete Stage 2 deployment checkpoint.

    The caller must construct the exact architecture and place it on the final
    inference device first.  No optimizer, scheduler, or RNG state is restored.
    """

    if not isinstance(model, RegressionSystem):
        raise TypeError("production regression model must be a RegressionSystem")
    expected_sha256 = _lower_sha256(
        expected_checkpoint_sha256, name="expected regression checkpoint"
    )
    if not isinstance(expected_provenance, CheckpointProvenance):
        raise TypeError("expected_provenance must be CheckpointProvenance")
    expected_provenance.validate()
    if expected_provenance.role != "deployment" or expected_provenance.fold_id is not None:
        raise ValueError("inference requires deployment checkpoint provenance")
    if (
        isinstance(expected_optimizer_step, bool)
        or not isinstance(expected_optimizer_step, int)
        or expected_optimizer_step <= 0
    ):
        raise ValueError("expected_optimizer_step must be a positive integer")

    selected = Path(checkpoint_path).resolve()
    before = _sha256_file(selected)
    if before != expected_sha256:
        raise ValueError("regression checkpoint SHA-256 mismatch before load")
    cursor, extra = load_training_checkpoint(
        selected,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_provenance=expected_provenance,
        restore_rng=False,
    )
    expected_examples = expected_optimizer_step * (
        expected_provenance.per_rank_microbatch_size
        * expected_provenance.gradient_accumulation_steps
    )
    if (
        extra.get("complete") is not True
        or cursor.optimizer_step != expected_optimizer_step
        or cursor.global_example_index != expected_examples
        or cursor.microbatches_since_step != 0
    ):
        raise ValueError("regression checkpoint is not the exact complete final")
    after = _sha256_file(selected)
    if after != before:
        raise RuntimeError("regression checkpoint changed during model load")
    capture = _prepare_and_capture(model)
    return VerifiedRegressionModel(
        model=model,
        checkpoint_sha256=before,
        _capture=capture,
        provenance=expected_provenance,
    )


def load_verified_residual_edm_model(
    checkpoint_path: str | Path,
    *,
    model: ResidualEDM,
    expected_checkpoint_sha256: str,
    expected_provenance: Stage3CheckpointProvenance,
) -> VerifiedResidualEDMModel:
    """Strict-load one complete Stage 3 checkpoint without restoring RNG."""

    # Stage 3's training module is intentionally imported only on this
    # production path.  Importing development inference bindings stays light
    # and cannot create a train-stage/inference import cycle.
    from kcorrdiff.training.train_stage3 import (
        Stage3CheckpointProvenance,
        load_stage3_checkpoint,
    )

    if not isinstance(model, ResidualEDM):
        raise TypeError("production diffusion model must be a ResidualEDM")
    expected_sha256 = _lower_sha256(
        expected_checkpoint_sha256, name="expected Stage 3 checkpoint"
    )
    if not isinstance(expected_provenance, Stage3CheckpointProvenance):
        raise TypeError(
            "expected_provenance must be Stage3CheckpointProvenance"
        )
    expected_provenance.validate()
    if expected_provenance.variant != model.config.variant:
        raise ValueError("Stage 3 provenance/model variant mismatch")

    selected = Path(checkpoint_path).resolve()
    before = _sha256_file(selected)
    if before != expected_sha256:
        raise ValueError("Stage 3 checkpoint SHA-256 mismatch before load")
    cursor, complete, plan_steps = load_stage3_checkpoint(
        selected,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_provenance=expected_provenance,
        restore_rank_rng=None,
    )
    expected_examples = plan_steps * (
        expected_provenance.per_rank_microbatch_size
        * expected_provenance.gradient_accumulation_steps
    )
    if (
        complete is not True
        or cursor.optimizer_step != plan_steps
        or cursor.global_example_index != expected_examples
        or cursor.microbatches_since_step != 0
    ):
        raise ValueError("Stage 3 checkpoint is not the exact complete final")
    after = _sha256_file(selected)
    if after != before:
        raise RuntimeError("Stage 3 checkpoint changed during model load")
    capture = _prepare_and_capture(model)
    return VerifiedResidualEDMModel(
        model=model,
        checkpoint_sha256=before,
        variant=expected_provenance.variant,  # type: ignore[arg-type]
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

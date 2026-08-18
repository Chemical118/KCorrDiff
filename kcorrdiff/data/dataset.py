"""Manifest-driven, mmap-backed samples with shared-history batching.

The dataset is a deterministic consumer of a prebuilt draw manifest.  It never
substitutes another row for missing data and exposes target validity only in
the label structure, never among model conditions.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler

from .accumulation import build_accumulation_target
from .availability import format_radar_timestamp, history_times, target_times
from .cache import RadarCache
from .condition_schema import (
    ContextRadarConditions,
    Era5Conditions,
    StaticConditions,
    build_context_radar_conditions,
    parse_condition_signature,
)
from .context_regrid import ContextRegridOperator, DetailMode
from .era5_reader import Era5MmapCache
from .sampling import DrawRow
from .time_features import build_verification_time_features


# Draw plans repeat one t0 across leads/signatures; regridded context depends
# only on t0 (histories and regrid parameters are fixed per dataset), so a few
# retained entries deduplicate the per-sample regrid without growing memory.
_CONTEXT_CACHE_ENTRIES = 4


@dataclass(frozen=True, slots=True)
class RadarConditions:
    """Issue-time model inputs with future label state structurally excluded."""

    target_history: np.ndarray
    history_validity: np.ndarray
    context: ContextRadarConditions
    era5: Era5Conditions
    static: StaticConditions
    t0_utc: datetime
    lead_hours: float
    condition_signature: str
    time_features: tuple[float, float, float, float]

    @property
    def context_history(self) -> np.ndarray:
        """Return model-facing ``[12,5,H,W]`` named context channels."""

        return self.context.as_array()

    @property
    def static_channels(self) -> np.ndarray:
        """Compatibility view of explicitly named target static fields."""

        return self.static.target.values

    @property
    def static_coverage(self) -> np.ndarray:
        return self.static.target_static_coverage


@dataclass(frozen=True, slots=True)
class TrainingLabel:
    z: np.ndarray
    wet: np.ndarray
    raw_accumulation_mm: np.ndarray
    model_accumulation_mm: np.ndarray
    target_validity: np.ndarray
    omega: float


@dataclass(frozen=True, slots=True)
class TrainingSample:
    sample_id: str
    block_id: str
    fold_id: int | None
    conditions: RadarConditions
    label: TrainingLabel


class KCorrDiffDataset(Dataset[TrainingSample]):
    def __init__(
        self,
        *,
        cache: RadarCache,
        rows: Sequence[DrawRow],
        context_regrid_operator: ContextRegridOperator,
        era5_cache: Era5MmapCache,
        static_conditions: StaticConditions | None = None,
        static_channels: np.ndarray | None = None,
        static_coverage: np.ndarray | None = None,
        static_channel_names: Sequence[str] | None = None,
        fold_ids: Mapping[object, int | None] | None = None,
        context_detail_mode: DetailMode = "local_max",
        context_wet_threshold_mm_per_hour: float = 0.0,
        strict_era5_instantaneous: bool = True,
        radar_timezone: str = "Asia/Seoul",
    ) -> None:
        self.cache = cache
        self.rows = tuple(rows)
        self.context_regrid_operator = context_regrid_operator
        self.era5_cache = era5_cache
        self.fold_ids = dict(fold_ids or {})
        self.context_detail_mode = context_detail_mode
        self.context_wet_threshold_mm_per_hour = float(
            context_wet_threshold_mm_per_hour
        )
        self.strict_era5_instantaneous = bool(strict_era5_instantaneous)
        self.radar_timezone = radar_timezone
        self._context_cache: OrderedDict[str, ContextRadarConditions] = OrderedDict()
        if context_detail_mode not in ("local_max", "nearest"):
            raise ValueError("context_detail_mode must be 'local_max' or 'nearest'")
        if (
            not np.isfinite(self.context_wet_threshold_mm_per_hour)
            or self.context_wet_threshold_mm_per_hour < 0.0
        ):
            raise ValueError(
                "context_wet_threshold_mm_per_hour must be finite and non-negative"
            )

        if static_conditions is None:
            if static_channels is None or static_coverage is None:
                raise ValueError(
                    "provide static_conditions or both static_channels and "
                    "static_coverage"
                )
            static_conditions = StaticConditions.from_arrays(
                target_values=static_channels,
                target_static_coverage=static_coverage,
                target_channel_names=static_channel_names,
                context_regrid_operator=context_regrid_operator,
            )
        elif static_channels is not None or static_coverage is not None:
            raise ValueError(
                "static_conditions cannot be combined with legacy static arrays"
            )
        elif static_channel_names is not None:
            raise ValueError(
                "static_channel_names belongs to legacy static_channels input"
            )
        if static_conditions.context.source_dx_km.shape != (
            context_regrid_operator.destination_shape
        ):
            raise ValueError(
                "static context geometry does not match context regrid destination"
            )
        for actual, expected, name in (
            (
                static_conditions.context.source_dx_km,
                context_regrid_operator.source_dx_km,
                "source_dx_km",
            ),
            (
                static_conditions.context.source_dy_km,
                context_regrid_operator.source_dy_km,
                "source_dy_km",
            ),
            (
                static_conditions.context.nearest_source_distance_km,
                context_regrid_operator.nearest_source_distance_km,
                "nearest_source_distance_km",
            ),
            (
                static_conditions.context.geometry_confidence,
                context_regrid_operator.geometry_confidence,
                "geometry_confidence",
            ),
        ):
            if not np.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError(
                    f"static context {name} does not match the regrid operator"
                )
        self.static_conditions = static_conditions

    @property
    def static_channels(self) -> np.ndarray:
        return self.static_conditions.target.values

    @property
    def static_coverage(self) -> np.ndarray:
        return self.static_conditions.target_static_coverage

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingSample:
        row = self.rows[index]
        t0 = datetime.fromisoformat(row.t0_utc)
        history_datetimes = history_times(t0)
        target_datetimes = target_times(t0, row.lead_hours)
        history_keys = [
            format_radar_timestamp(value, timezone=self.radar_timezone)
            for value in history_datetimes
        ]
        target_history = np.asarray(
            self.cache.read_many("target", history_keys), dtype=np.float32
        )
        expected_target_shape = (
            len(history_keys),
            *self.static_conditions.target_static_coverage.shape,
        )
        if target_history.shape != expected_target_shape:
            raise ValueError(
                f"target history has shape {target_history.shape}, "
                f"expected {expected_target_shape}"
            )
        if not np.all(np.isfinite(target_history)) or np.any(
            (target_history < 0.0) | (target_history > 1.0)
        ):
            raise ValueError(
                "CPrecNet target history must contain finite normalized values "
                "in [0, 1]"
            )

        context = self._context_cache.get(row.t0_utc)
        if context is None:
            raw_context_history = self.cache.read_many("condition", history_keys)
            context = build_context_radar_conditions(
                raw_context_history,
                operator=self.context_regrid_operator,
                detail_mode=self.context_detail_mode,
                wet_threshold_mm_per_hour=self.context_wet_threshold_mm_per_hour,
            )
            self._context_cache[row.t0_utc] = context
            while len(self._context_cache) > _CONTEXT_CACHE_ENTRIES:
                self._context_cache.popitem(last=False)
        else:
            self._context_cache.move_to_end(row.t0_utc)
        signature = parse_condition_signature(row.condition_signature)
        era_window = self.era5_cache.read_window(
            t0,
            lead_hours=float(row.lead_hours),
            signature=signature,
            strict_instantaneous=self.strict_era5_instantaneous,
        )
        era5 = Era5Conditions.from_window(
            era_window,
            requested_signature=signature,
            lead_hours=float(row.lead_hours),
        )
        time = build_verification_time_features(t0, row.lead_hours)
        conditions = RadarConditions(
            target_history=target_history,
            # CPrecNet has no dynamic missingness; static coverage is known at t0.
            history_validity=np.broadcast_to(
                self.static_coverage, target_history.shape
            ).copy(),
            context=context,
            era5=era5,
            static=self.static_conditions,
            t0_utc=t0,
            lead_hours=row.lead_hours,
            condition_signature=signature.key,
            time_features=time.cyclic_features,
        )

        # Future target frames are deliberately read only after the complete
        # issue-time condition object has been assembled.  No future-derived
        # value or validity field exists in any condition schema above.
        target_keys = [
            format_radar_timestamp(value, timezone=self.radar_timezone)
            for value in target_datetimes
        ]
        future = self.cache.read_many("target", target_keys)
        accumulation = build_accumulation_target(
            future,
            input_encoding="cprecnet_normalized",
            static_coverage=self.static_coverage,
        )
        label = TrainingLabel(
            z=accumulation.z,
            wet=accumulation.wet,
            raw_accumulation_mm=accumulation.raw_accumulation_mm,
            model_accumulation_mm=accumulation.model_accumulation_mm,
            target_validity=accumulation.valid_mask,
            omega=row.omega,
        )
        return TrainingSample(
            sample_id=row.sample_id,
            block_id=row.block_id,
            fold_id=self._fold_id(row),
            conditions=conditions,
            label=label,
        )

    def _fold_id(self, row: DrawRow) -> int | None:
        """Read an optional duck-typed row fold, then fall back to a mapping."""

        row_fold = getattr(row, "fold_id", None)
        identity = (row.sample_id, row.condition_signature)
        mapping_fold = self.fold_ids.get(
            identity, self.fold_ids.get(row.sample_id)
        )
        if (
            row_fold is not None
            and mapping_fold is not None
            and row_fold != mapping_fold
        ):
            raise ValueError(
                f"conflicting fold_id for {row.sample_id}: "
                f"row={row_fold}, mapping={mapping_fold}"
            )
        value = row_fold if row_fold is not None else mapping_fold
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("fold_id must be an integer or None")
        fold_id = int(value)
        if fold_id < 0:
            raise ValueError("fold_id must be non-negative")
        return fold_id


class SharedT0BatchSampler(Sampler[list[int]]):
    """Group lead rows for the same t0/signature to reuse condition pages/cache."""

    def __init__(self, rows: Sequence[DrawRow], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        first_index: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            key = (row.t0_utc, row.condition_signature)
            groups[key].append(index)
            first_index.setdefault(key, index)
        # Maintain deterministic draw-order at the group level and lead order
        # inside a group; worker/GPU topology cannot reshuffle it.
        self._batches: list[list[int]] = []
        for key in sorted(groups, key=lambda value: first_index[value]):
            indices = sorted(groups[key], key=lambda i: rows[i].lead_hours)
            self._batches.extend(
                indices[start : start + batch_size]
                for start in range(0, len(indices), batch_size)
            )

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def dataloader_worker_options(num_workers: int, prefetch_factor: int = 2) -> dict[str, object]:
    """Canonical throughput options without invalid zero-worker arguments."""

    if num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("invalid DataLoader worker settings")
    options: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return options

"""Actual clock geometry, explicit eligibility, runs, windows and origin burden."""

from dataclasses import dataclass
from math import fsum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from ._owned import identity, owned


def positive_duration(value, name):
    if isinstance(value, (bool, int, float)):
        raise ValueError(f"{name} requires an explicit duration unit")
    try:
        duration = pd.Timedelta(value).as_unit("ns", round_ok=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a nanosecond duration") from error
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive fixed duration")
    return duration


def _aware_timestamp(value, name):
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or pd.isna(timestamp):
            raise ValueError("timestamp is naive or NaT")
        return timestamp.as_unit("ns", round_ok=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be an aware timestamp representable at nanosecond resolution") from error


def positions_array(positions, size):
    array = np.asarray(positions)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("positions must be a one-dimensional integer sequence")
    result = array.astype(int)
    if (result < 0).any() or (result >= size).any() or (np.diff(result) <= 0).any():
        raise ValueError("positions must be ordered, unique and within the input")
    return result


@dataclass(frozen=True)
class ObservationPolicy:
    values: str = "provided"
    min_points: int = 1
    max_imputed_fraction: float | None = None
    max_reconstructed_duration: str | None = None

    def __post_init__(self):
        if self.values not in {"provided", "observed_only", "trusted_imputation"}:
            raise ValueError("values must be provided, observed_only or trusted_imputation")
        if type(self.min_points) is not int or self.min_points < 1:
            raise ValueError("min_points must be a positive integer")
        if self.max_imputed_fraction is not None:
            value = self.max_imputed_fraction
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("max_imputed_fraction must be finite and within [0,1]")
        if self.max_reconstructed_duration is not None:
            positive_duration(self.max_reconstructed_duration, "max_reconstructed_duration")


class TimeAxis:
    """A declared regular grid (holes retained) or an actual irregular clock.

    Regular coverage is [timestamp, timestamp+cadence). Irregular coverage is
    unknown unless bounded, justified per-sample seconds are supplied explicitly.
    """

    def __init__(self, index, *, mode="irregular", cadence=None, phase=None,
                 storage_timezone=None, civil_timezone="UTC", clock_resolution="resolved",
                 coverage_seconds=None, max_coverage_seconds=None):
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("time must be a timezone-aware DatetimeIndex")
        if index.tz is None or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("time must be ordered, unique, aware and free of NaT")
        if clock_resolution != "resolved":
            raise ValueError("unresolved civil/DST/fold clock; supply explicitly resolved timestamps")
        if storage_timezone is not None and storage_timezone != str(index.tz):
            raise ValueError("storage_timezone must match the supplied clock")
        try:
            ZoneInfo(civil_timezone)
        except (ZoneInfoNotFoundError, TypeError) as error:
            raise ValueError("civil_timezone must be a known timezone") from error
        if mode not in {"regular", "irregular"}:
            raise ValueError("mode must be regular or irregular")
        try:
            index = index.as_unit("ns", round_ok=False).copy(deep=True)
        except (ValueError, OverflowError) as error:
            raise ValueError("time must be representable at nanosecond resolution") from error
        self._index = index
        self._mode = mode
        self._cadence = None
        self._phase = None
        self._coverage = None
        if mode == "regular":
            self._cadence = positive_duration(cadence, "cadence")
            self._phase = index[0] if phase is None and len(index) else phase
            if self._phase is not None:
                self._phase = _aware_timestamp(self._phase, "phase")
                # Python integer subtraction cannot overflow across distant phases.
                if any((int(tick) - self._phase.value) % self._cadence.value for tick in index.asi8):
                    raise ValueError("regular samples must lie exactly on the declared cadence/phase")
            if len(index):
                try:
                    index[-1] + self._cadence
                except (ValueError, OverflowError) as error:
                    raise ValueError("regular cell endpoint must be representable at nanosecond resolution") from error
            if coverage_seconds is not None or max_coverage_seconds is not None:
                raise ValueError("regular cell coverage is determined by cadence")
        elif cadence is not None or phase is not None:
            raise ValueError("irregular clocks cannot declare regular cadence/phase")
        if coverage_seconds is not None:
            weights = np.asarray(coverage_seconds)
            if weights.shape != (len(index),) or weights.dtype.kind not in "iuf":
                raise ValueError("coverage_seconds must align with actual timestamps")
            if (type(max_coverage_seconds) not in (int, float)
                    or not np.isfinite(max_coverage_seconds) or max_coverage_seconds <= 0):
                raise ValueError("irregular coverage needs an explicit positive maximum gap coverage")
            weights = weights.astype(float)
            if (not np.isfinite(weights).all() or (weights < 0).any()
                    or (weights > max_coverage_seconds).any()):
                raise ValueError("coverage must be finite, nonnegative and bounded by maximum coverage")
            ticks = index.asi8
            if any(weight > (int(right) - int(left)) / 1e9
                   for weight, left, right in zip(weights[:-1], ticks[:-1], ticks[1:])):
                raise ValueError("sample coverage must not overlap the next timestamp")
            self._coverage = weights.copy()
        elif max_coverage_seconds is not None:
            raise ValueError("maximum coverage requires explicit coverage_seconds")
        self._record = {"mode": mode, "cadence": self._cadence, "phase": self._phase,
                        "storage_timezone": str(index.tz), "civil_timezone": civil_timezone,
                        "clock_resolution": clock_resolution, "coverage_seconds": self._coverage,
                        "max_coverage_seconds": max_coverage_seconds,
                        "duration_convention": "cells_left_closed" if mode == "regular"
                        else "explicit_coverage" if self._coverage is not None else "unknown"}

    @property
    def index(self):
        return self._index.copy(deep=True)

    @property
    def record(self):
        return owned(self._record)

    @property
    def mode(self):
        return self._mode

    @property
    def cadence(self):
        return self._cadence

    def duration(self, positions):
        positions = positions_array(positions, len(self._index))
        if self._cadence is not None:
            return len(positions) * self._cadence.value / 1e9
        if self._coverage is not None:
            return float(self._coverage[positions].sum())
        return None

    def resolve_lag(self, requested):
        if self._mode != "regular":
            raise ValueError("sample lags require a regular clock; choose an explicit irregular estimator")
        labels = {"one_hour": "1h", "twenty_four_hours": "24h"}
        duration = self._cadence if requested == "one_step" else positive_duration(labels.get(requested, requested), "lag")
        samples, remainder = divmod(duration.value, self._cadence.value)
        return {"requested": requested, "compute_status": "unsupported_exact_lag" if remainder else "ok",
                "samples": None if remainder else int(samples),
                "realized_seconds": None if remainder else duration.value / 1e9,
                "reason": "duration is not an exact multiple of cadence" if remainder else None}


def _runs(axis, mask):
    if axis.mode != "regular":
        raise ValueError("contiguous runs require a declared regular cadence")
    runs = []
    start = None
    index = axis.index
    for i, valid in enumerate(mask):
        if start is not None and (not valid or index[i].value - index[i - 1].value != axis.cadence.value):
            runs.append((start, i))
            start = None
        if valid and start is None:
            start = i
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _interval(axis, start, stop):
    index = axis.index
    return {"start_position": start, "stop_position": stop,
            "start": index[start].isoformat(),
            "end": (index[stop - 1] + axis.cadence).isoformat(), "closure": "left",
            "count": stop - start, "duration_seconds": (stop - start) * axis.cadence.value / 1e9,
            "timestamp_span_seconds": (index[stop - 1].value - index[start].value) / 1e9}


def reconstruction_burden(series, positions=None, *, denominator=None):
    axis = series.time
    index = axis.index
    full = positions is None
    positions = np.arange(len(index)) if full else positions_array(positions, len(index))
    if denominator is None:
        denominator = len(positions)
        if full and axis.mode == "regular" and len(index):
            denominator = (index[-1].value - index[0].value) // axis.cadence.value + 1
    if type(denominator) is not int or denominator < len(positions):
        raise ValueError("denominator must include every selected cell")
    origin = series.origin
    records = {}
    contribution = series.contribution
    for j, variable in enumerate(series.values.columns):
        touched = np.zeros(len(index), dtype=bool)
        touched[positions] = origin["touched"][positions, j]
        if axis.mode == "regular":
            blocks = _runs(axis, touched)
            longest = max((stop - start for start, stop in blocks), default=0)
            longest_seconds = longest * axis.cadence.value / 1e9
        else:
            # Without declared adjacency no contiguous block duration is invented.
            longest, longest_seconds = None, None
        nonlinear = origin["fraction_kind"][positions, j] == "coverage"
        amounts = origin["reconstructed"][positions, j]
        nonlinear_count = int(nonlinear.sum())
        linear_numerator = fsum(amounts[~nonlinear])
        coverage_numerator = fsum(amounts[nonlinear])
        numerator = linear_numerator + coverage_numerator
        touched_count = int(touched.sum())
        gate_numerator = linear_numerator + int(origin["touched"][positions, j][nonlinear].sum())
        fraction_kind = ("contributor_coverage" if nonlinear_count == denominator and denominator
                         else "mixed_influence_and_coverage" if nonlinear_count else "linear_influence")
        operations, counts = np.unique(origin["operation"][positions, j], return_counts=True)
        unknown = float(origin["unknown"][positions, j].sum())
        records[variable] = {"numerator": numerator, "denominator": denominator,
                             "fraction": numerator / denominator if denominator else None,
                             "fraction_kind": fraction_kind,
                             "operation_counts": dict(zip(operations.tolist(), counts.tolist())),
                             "linear_influence": _fraction(linear_numerator, denominator - nonlinear_count),
                             "contributor_coverage": _fraction(coverage_numerator, nonlinear_count),
                             "touched_cell_coverage": _fraction(touched_count, denominator),
                             "primary_gate": {**_fraction(gate_numerator, denominator),
                                              "fraction_kind": "linear_influence_with_nonlinear_touched_cells"},
                             "unknown_numerator": unknown,
                             "unknown_fraction": unknown / denominator if denominator else None,
                             "unknown_touched_count": int(np.sum(origin["unknown_touched"][positions, j])),
                             "unavailable_count": int(np.sum(contribution[positions, j] == "unavailable"))
                             + denominator - len(positions),
                             "touched_count": touched_count,
                             "longest_reconstructed_cells": longest,
                             "longest_reconstructed_seconds": longest_seconds}
    return {"support_id": identity({"series": series.identity, "positions": positions,
                                    "denominator": denominator}),
            "positions": positions.tolist(), "denominator": denominator,
            "duration_convention": axis.record["duration_convention"], "weighting": "equal_cells",
            "variables": records}


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def use_status(series, positions, observations, *, metric_status="primary_with_support"):
    burden = reconstruction_burden(series, positions)
    if not len(positions):
        return "insufficient_support"
    if any(item["unknown_touched_count"] > 0 for item in burden["variables"].values()):
        return "unknown_provenance"
    if metric_status != "primary_with_support":
        return "sensitivity_only"
    for item in burden["variables"].values():
        if (observations.max_imputed_fraction is not None
                and item["primary_gate"]["fraction"] > observations.max_imputed_fraction):
            return "sensitivity_only"
        if observations.max_reconstructed_duration is not None:
            seconds = item["longest_reconstructed_seconds"]
            if seconds is None:
                return "unverified_gap_geometry"
            if seconds > positive_duration(observations.max_reconstructed_duration, "maximum").value / 1e9:
                return "sensitivity_only"
    return "primary_eligible"


class SupportSelection:
    def __init__(self, record):
        self._record = owned(record)

    @property
    def record(self):
        return owned(self._record)

    @property
    def positions(self):
        return np.asarray(self._record["positions"], dtype=int)


def select_support(series, *, observations=None, policy="all_runs", estimator="regular",
                   window=None, step=None, anchor=None, complete=True, label="start", closure="left",
                   within=None):
    observations = ObservationPolicy() if observations is None else observations
    if not isinstance(observations, ObservationPolicy):
        raise ValueError("observations must be an ObservationPolicy")
    if policy not in {"longest_run", "all_runs", "windows", "irregular", "complete_case"}:
        raise ValueError("unknown support selection")
    if estimator not in {"regular", "irregular", "point", "complete_case"}:
        raise ValueError("unknown estimator support contract")
    axis = series.time
    if estimator == "regular" and (axis.mode != "regular" or policy in {"irregular", "complete_case"}):
        raise ValueError("regular estimators require regular contiguous run/window support")
    if policy in {"longest_run", "all_runs"} and axis.mode != "regular":
        raise ValueError("run policies require regular time; use irregular or complete_case")
    if policy == "irregular" and estimator not in {"irregular", "point"}:
        raise ValueError("irregular support is incompatible with this estimator")
    if label not in {"start", "end"} or closure != "left" or type(complete) is not bool:
        raise ValueError("windows require left closure, start/end label and boolean complete")
    if policy != "windows" and (any(value is not None for value in (window, step, anchor))
                                or not complete or label != "start"):
        raise ValueError("window parameters require the windows policy")
    eligible = series.eligible(observations).all(axis=1)
    if within is not None:
        allowed = np.zeros(len(eligible), dtype=bool)
        allowed[positions_array(within, len(eligible))] = True
        eligible &= allowed
    runs = _runs(axis, eligible) if axis.mode == "regular" else []
    windows = []
    window_settings = None
    selected = np.flatnonzero(eligible)
    if policy == "longest_run":
        run = max(runs, key=lambda item: item[1] - item[0], default=None)
        selected = np.arange(*run) if run is not None else np.array([], dtype=int)
    elif policy == "windows":
        duration = positive_duration(window, "window")
        stride = positive_duration(step, "step")
        index = axis.index
        if axis.mode == "regular" and (duration.value % axis.cadence.value or stride.value % axis.cadence.value):
            raise ValueError("window/step must be exact cadence multiples")
        selected_set = set()
        origin = (index[0] if len(index) else None) if anchor is None else _aware_timestamp(anchor, "anchor")
        window_settings = {"window": duration, "step": stride, "anchor": origin,
                           "complete": complete, "label": label, "closure": closure}
        if len(index):
            start = origin
            if start > index[0]:
                raise ValueError("anchor must be aware and at or before the first timestamp")
            if axis.mode == "regular" and (start.value - index[0].value) % axis.cadence.value:
                raise ValueError("anchor must match the declared grid phase")
            limit = index[-1] + axis.cadence if axis.mode == "regular" else index[-1]
            while start < limit or (axis.mode == "irregular" and start == limit):
                end = start + duration
                positions = np.flatnonzero((index >= start) & (index < end))
                valid = positions[eligible[positions]]
                expected = duration.value // axis.cadence.value if axis.mode == "regular" else None
                reason = None
                if len(valid) < observations.min_points:
                    reason = "insufficient_support"
                if complete and (len(valid) != len(positions) or (expected is not None and len(valid) != expected)):
                    reason = "gap_or_incomplete_window"
                record = {"start": start.isoformat(), "end": end.isoformat(),
                          "label": (start if label == "start" else end).isoformat(),
                          "anchor": origin.isoformat(),
                          "closure": closure, "positions": valid.tolist(), "expected_cells": expected,
                          "available_cells": len(positions), "eligible_cells": len(valid),
                          "compute_status": "ok" if reason is None else "insufficient_support",
                          "reason": reason,
                          "window_cells": reconstruction_burden(series, positions,
                                                               denominator=expected if expected is not None else len(positions)),
                          "burden": reconstruction_burden(series, valid),
                          "eligible_duration_seconds": axis.duration(valid),
                          "selected_duration_seconds": axis.duration(valid) if reason is None else axis.duration([]),
                          "use_status": use_status(series, valid, observations)}
                record["support_id"] = identity({"series": series.identity, "start": start,
                                                 "end": end, "window_settings": window_settings, "positions": valid})
                windows.append(record)
                if reason is None:
                    selected_set.update(valid.tolist())
                start += stride
        selected = np.asarray(sorted(selected_set), dtype=int)
    status = "ok" if len(selected) >= observations.min_points else "insufficient_support"
    full_positions = np.arange(len(eligible))
    excluded = np.setdiff1d(full_positions, selected)
    full_burden = reconstruction_burden(series)
    full_duration = (full_burden["denominator"] * axis.cadence.value / 1e9
                     if axis.mode == "regular" else axis.duration(full_positions))
    selected_duration = axis.duration(selected)
    record = {"policy": policy, "estimator": estimator, "positions": selected.tolist(),
              "compute_status": status, "reason": None if status == "ok" else "too few eligible points",
              "runs": [_interval(axis, start, stop) for start, stop in runs], "windows": windows,
              "window_settings": window_settings,
              "full_duration_seconds": full_duration,
              "supplied_duration_seconds": axis.duration(full_positions),
              "selected_duration_seconds": selected_duration,
              "excluded_duration_seconds": full_duration - selected_duration if full_duration is not None else None,
              "missing_grid_cells": full_burden["denominator"] - len(eligible) if axis.mode == "regular" else None,
              "duration_convention": axis.record["duration_convention"],
              "timestamp_span_seconds": (axis.index[-1].value - axis.index[0].value) / 1e9 if len(eligible) else 0,
              "full_segment": full_burden,
              "selected": reconstruction_burden(series, selected),
              "excluded_positions": excluded.tolist()}
    record["support_id"] = identity({"series": series.identity, "policy": policy,
                                     "positions": selected, "windows": windows,
                                     "window_settings": window_settings,
                                     "observations": observations.__dict__})
    return SupportSelection(record)


def pair_support(first, second, *, variable_a, variable_b, observations=None):
    observations = ObservationPolicy() if observations is None else observations
    if (not first.time.index.equals(second.time.index)
            or identity(first.time.record) != identity(second.time.record)):
        raise ValueError("pair clocks must be explicitly aligned, including cadence/phase/coverage")
    a = list(first.values.columns).index(variable_a)
    b = list(second.values.columns).index(variable_b)
    positions = np.flatnonzero(first.eligible(observations)[:, a] & second.eligible(observations)[:, b])
    origin_a, origin_b = first.origin, second.origin
    denominator = len(positions)
    # Across variables, fractions describe coverage; overlapping contributor influence is not inferred.
    union = int(np.sum(origin_a["touched"][positions, a] | origin_b["touched"][positions, b]))
    record = {"positions": positions.tolist(), "denominator": denominator,
              "per_variable": {},
              "union": {"numerator": union, "denominator": denominator,
                        "fraction": union / denominator if denominator else None,
                        "fraction_kind": "touched_cell_coverage"}}
    record["support_id"] = identity({"first": first.identity, "second": second.identity,
                                     "variables": [variable_a, variable_b], "positions": positions})
    for role, data, variable in (("first", first, variable_a), ("second", second, variable_b)):
        burden = reconstruction_burden(data, positions)
        record["per_variable"][role] = {"variable": variable, **burden["variables"][variable]}
        record["per_variable"][role]["support_id"] = burden["support_id"]
    record["union"]["support_id"] = record["support_id"]
    record["union"]["unknown_touched_count"] = int(np.sum(
        origin_a["unknown_touched"][positions, a] | origin_b["unknown_touched"][positions, b]))
    return owned(record)

"""Ordered discrete mass roughness; independent partitions and complete windows."""

import numpy as np
import pandas as pd

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations, _training_signal
from .reference import FrozenReference, OutsideReferenceError, ReferenceSpec
from .support import (SupportSelection, positions_array, positive_duration,
                      reconstruction_burden, select_support)


METRIC = "historical_sensitivity_index"
METHOD = "fi1.ordered_mass_roughness_v1"


def discrete_information(probabilities):
    """FI1 and entropy of normalized ordered masses (no ghost boundary bins)."""
    p = np.asarray(probabilities)
    if (p.ndim != 1 or not len(p) or p.dtype.kind not in "iuf"
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("probabilities must be finite nonnegative normalized masses")
    positive = p[p > 0]
    return dict(fi1=float(4*np.sum(np.diff(np.sqrt(p))**2)),
                entropy_nats=float(-np.sum(positive*np.log(positive))))


class FI1Spec:
    def __init__(self, edges=None, *, partition="explicit", pseudocount=None,
                 outside="raise", min_points=2):
        if partition not in {"explicit", "local_fd_v1", "sensor_pinned"}:
            raise ValueError("unknown FI1 partition")
        if partition == "explicit":
            edges = ReferenceSpec(edges, scale=False).record["edges"]
        elif edges is not None:
            raise ValueError("automatic/pinned partitions cannot take explicit edges")
        if (type(min_points) is not int or min_points < 2
                or outside not in {"raise", "exclude"}):
            raise ValueError("min_points must be >=2; outside must be raise or exclude")
        requested = dict(partition=partition, edges=edges, pseudocount=pseudocount,
                         outside=outside, min_points=min_points)
        if pseudocount is None:
            pseudocount = 1e-12 if partition == "local_fd_v1" else 0.0
        if type(pseudocount) not in (int, float) or not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be a finite nonnegative count per bin")
        self._record = dict(requested=requested, partition=partition, edges=edges,
            pseudocount=pseudocount, outside=outside, min_points=min_points,
            entropy_base="e", units="1", last_edge_included=True,
            boundary="adjacent_bins_only", parity="unverified",
            bin_rule="ceil(range/(2*IQR*N^(-1/3))), bounded [10,40]; zero IQR -> 10"
                     if partition == "local_fd_v1" else None)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class FI1Reference(FrozenReference):
    pass


def fit_fi1_reference(data, *, variable, spec, training_positions, observations=None,
                      fit_scope="training_only"):
    if not isinstance(spec, FI1Spec) or fit_scope not in {"training_only", "global_self", "window_self"}:
        raise ValueError("FI1 fitting requires FI1Spec and an explicit fit scope")
    original = _ColumnSignal(data, variable)
    positions = positions_array(training_positions, len(original.values))
    signal = _training_signal(data, variable, positions)
    observations = _observations(observations)
    selection = select_support(signal, observations=observations, estimator="point",
                               policy="all_runs" if signal.time.mode == "regular" else "irregular")
    used = selection.positions
    x = signal.values.iloc[used, 0].to_numpy(dtype=float, na_value=np.nan)
    settings = spec.record
    status, reason, edges = "ok", None, settings["edges"]
    geometry = dict(n=len(x), iqr=None, fd_width=None, bins=None)
    if settings["partition"] == "sensor_pinned":
        status, reason = "missing_backend", "pinned sensor FI1 helper unavailable; historical numerical parity unresolved"
    elif len(x) < max(settings["min_points"], observations.min_points):
        status, reason = "insufficient_support", "too few eligible training samples"
    elif np.ptp(x) == 0:
        status, reason = "degenerate_input", "constant training samples; no invented partition support"
    elif settings["partition"] == "local_fd_v1":
        iqr = float(np.diff(np.quantile(x, [.25, .75]))[0])
        width = 2*iqr*len(x)**(-1/3)
        count = 10 if width == 0 else int(min(40, max(10, np.ceil(np.ptp(x)/width))))
        edges = np.linspace(x.min(), x.max(), count+1)
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            status, reason = "degenerate_input", "partition is not representable with positive finite widths"
            edges = None
        geometry.update(iqr=iqr, fd_width=width, bins=count)
    if edges is not None:
        geometry["bins"] = len(edges)-1
    return FI1Reference(dict(method_id=METHOD, spec=settings, spec_id=spec.identity,
        edges=edges, edges_id=identity(edges), geometry=geometry, variable=variable,
        unit=signal.metadata[variable]["unit"], basis=signal.basis,
        training_positions=positions.tolist(), eligible_training_positions=positions[used].tolist(),
        support=signal.support_record(selection), use_status=signal.use(used, observations, METRIC),
        observations=observations.__dict__, fit_scope=fit_scope,
        semantics="training_only" if fit_scope == "training_only" else "retrospective",
        compute_status=status, reason=reason))


def _compatible(signal, reference):
    if not isinstance(reference, FI1Reference):
        raise ValueError("reference must be FI1Reference")
    fit = reference.describe()
    if (signal.variable != fit["variable"] or signal.basis != fit["basis"]
            or signal.metadata[signal.variable]["unit"] != fit["unit"]):
        raise ValueError("FI1 reference variable, unit and basis must match")
    return fit


def _evaluate(signal, selection, observations, reference):
    fit = _compatible(signal, reference)
    settings = fit["spec"]
    positions = selection.positions
    x = signal.values.iloc[positions, 0].to_numpy(dtype=float, na_value=np.nan)
    support = signal.support_record(selection)
    support.update(fit=fit, edge_id=fit["edges_id"], measurement_unit=fit["unit"],
                   requested=settings["requested"], resolved=settings, observations=owned(observations.__dict__))
    status, reason = fit["compute_status"], fit["reason"]
    if status == "ok" and selection.record["compute_status"] != "ok":
        status, reason = "insufficient_support", selection.record["reason"]
    if status == "ok" and len(x) < max(settings["min_points"], observations.min_points):
        status, reason = "insufficient_support", "too few eligible evaluation samples"
    value = None
    if status == "ok":
        edges = fit["edges"]
        outside = (x < edges[0]) | (x > edges[-1])
        diagnostics = dict(below=int(np.sum(x < edges[0])), above=int(np.sum(x > edges[-1])),
                           positions=positions[outside].tolist())
        if outside.any() and settings["outside"] == "raise":
            raise OutsideReferenceError({signal.variable: diagnostics})
        support.update(outside=diagnostics, excluded_mass=float(outside.mean()),
                       normalization="conditional_on_in_range" if outside.any() else "all_eligible",
                       evaluation_positions=positions[~outside].tolist(),
                       evaluation_burden=reconstruction_burden(signal, positions[~outside]))
        positions, x = positions[~outside], x[~outside]
        if len(x) < max(settings["min_points"], observations.min_points):
            status, reason = "outside_reference", "too few evaluation samples inside fixed edges"
        elif np.ptp(x) == 0:
            status, reason = "degenerate_input", "constant evaluation samples"
        else:
            counts = np.histogram(x, bins=edges)[0]
            smooth = counts.astype(float)+settings["pseudocount"]
            if not np.isfinite(smooth.sum()):
                raise ValueError("pseudocount normalization is not representable")
            p = smooth/smooth.sum()
            value = dict(**discrete_information(p), probabilities=p, counts=counts, edges=edges,
                         occupied_bins=int(np.count_nonzero(counts)), eligible_count=len(x),
                         pseudocount=settings["pseudocount"], entropy_base="e")
    use = signal.use(positions, observations, METRIC)
    if fit["use_status"] == "unknown_provenance":
        use = "unknown_provenance"
    return ResultRecord(value, compute_status=status, reason=reason, metric_status=METRIC,
        use_status=use, support=support, units="1", representation=signal.basis,
        method_id=METHOD, config_id=identity(dict(spec=fit["spec_id"], observations=observations.__dict__)),
        reference_id=reference.identity)


def fi1_discrete(data, *, variable, spec=None, reference=None, observations=None):
    """Evaluate masses anew with exactly one explicit self spec or frozen reference."""
    if (spec is None) == (reference is None):
        raise ValueError("supply exactly one of spec or reference")
    signal = _ColumnSignal(data, variable)
    observations = _observations(observations)
    if spec is not None:
        reference = fit_fi1_reference(data, variable=variable, spec=spec,
            training_positions=np.arange(len(signal.values)), observations=observations, fit_scope="global_self")
    selection = select_support(signal, observations=observations, estimator="point",
                               policy="all_runs" if signal.time.mode == "regular" else "irregular")
    return _evaluate(signal, selection, observations, reference)


def adjacent_flags(times, values, *, step="6h"):
    """Strict median + 3 unscaled MAD on consecutive available stride differences."""
    stride = positive_duration(step, "step")
    times = pd.DatetimeIndex(times)
    raw = np.asarray(values)
    if (times.tz is None or times.hasnans or not times.is_monotonic_increasing or times.has_duplicates
            or raw.ndim != 1 or len(raw) != len(times) or raw.dtype.kind not in "iuf"):
        raise ValueError("flags require ordered unique aware times and real numeric FI1 values")
    values = raw.astype(float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("FI1 must be nonnegative finite or missing")
    deltas = np.full(len(values), np.nan)
    for i in range(1, len(values)):
        if times[i]-times[i-1] == stride and np.isfinite(values[i-1:i+1]).all():
            deltas[i] = abs(values[i]-values[i-1])
    finite = deltas[np.isfinite(deltas)]
    median = float(np.median(finite)) if len(finite) else None
    mad = float(np.median(abs(finite-median))) if len(finite) else None
    threshold = median+3*mad if len(finite) else None
    flags = [bool(x > threshold) if np.isfinite(x) and threshold is not None else None for x in deltas]
    return dict(abs_delta=deltas, flags=flags, delta_median=median, delta_mad=mad,
                threshold=threshold, delta_count=len(finite), rule="strict_greater_than_median_plus_3_unscaled_MAD")


class FI1Trajectory:
    def __init__(self, table, record):
        self._table, self._record = owned(table), owned(record)

    @property
    def table(self):
        return owned(self._table)

    def describe(self):
        return owned(self._record)


def fi1_windows(data, *, variable, anchor, spec=None, reference=None, observations=None,
                window="72h", step="6h", label="start", fit_scope="global_self"):
    """Schedule full intervals inside the record extent; retain gaps as missing rows."""
    if (spec is None) == (reference is None):
        raise ValueError("supply exactly one of spec or reference")
    if fit_scope not in {"global_self", "window_self"} or (reference is not None and fit_scope != "global_self"):
        raise ValueError("self fitting must be global_self or window_self; reference forbids self-window fitting")
    signal = _ColumnSignal(data, variable)
    if signal.time.mode != "regular":
        raise ValueError("FI1 windows require a declared regular clock")
    if anchor is None:
        raise ValueError("FI1 windows require an explicit aware anchor")
    observations = _observations(observations)
    selection = select_support(signal, observations=observations, policy="windows", estimator="regular",
                               window=window, step=step, anchor=anchor, complete=True, label=label)
    settings = selection.record["window_settings"]
    extent = signal.time.index[-1]+signal.time.cadence if len(signal.values) else None
    scheduled = [w for w in selection.record["windows"] if pd.Timestamp(w["end"]) <= extent]
    if reference is None and fit_scope == "global_self":
        reference = fit_fi1_reference(data, variable=variable, spec=spec, observations=observations,
            training_positions=np.arange(len(signal.values)), fit_scope=fit_scope)
    if reference is not None:
        _compatible(signal, reference)
    elif not isinstance(spec, FI1Spec):
        raise ValueError("self-window fitting requires FI1Spec")
    rows = []
    for w in scheduled:
        w["window_settings"] = settings
        ref = reference if reference is not None else fit_fi1_reference(data, variable=variable,
            spec=spec, observations=observations, training_positions=w["positions"], fit_scope="window_self")
        result = _evaluate(signal, SupportSelection(w), observations, ref)
        record, value = result.summary(), result.value
        rows.append(dict(start=w["start"], end=w["end"], label=w["label"],
            fi1=value["fi1"] if value else np.nan, entropy_nats=value["entropy_nats"] if value else np.nan,
            eligible_count=w["eligible_cells"],
            evaluation_count=len(record["support"].get("evaluation_positions", w["positions"])),
            compute_status=record["compute_status"], metric_status=record["metric_status"],
            use_status=record["use_status"], reason=record["reason"], basis=signal.basis,
            edge_id=record["support"]["edge_id"], reference_id=record["reference_id"],
            method_id=record["method_id"], config_id=record["config_id"], units="1",
            distribution=value, support=record["support"]))
    columns = ["start", "end", "label", "fi1", "entropy_nats", "eligible_count", "evaluation_count",
               "compute_status", "metric_status", "use_status", "reason", "basis", "edge_id", "reference_id",
               "method_id", "config_id", "units", "distribution", "support"]
    table = pd.DataFrame(rows, columns=columns)
    times = pd.DatetimeIndex(table["start"]) if rows else pd.DatetimeIndex([], tz="UTC")
    flags = adjacent_flags(times, table["fi1"].to_numpy(dtype=float), step=step)
    table["abs_delta"] = flags.pop("abs_delta")
    table["abrupt_flag"] = pd.array(flags.pop("flags"), dtype="boolean")
    valid = table.loc[table["compute_status"] == "ok"]
    status = "ok" if len(valid) else "insufficient_support"
    summary = dict(eligible_windows=len(valid), scheduled_windows=len(table),
        median_fi1=float(valid.fi1.median()) if len(valid) else None,
        iqr_fi1=float(valid.fi1.quantile(.75)-valid.fi1.quantile(.25)) if len(valid) else None,
        median_entropy_nats=float(valid.entropy_nats.median()) if len(valid) else None,
        flagged_count=int(table.abrupt_flag.sum()), available_flags=int(table.abrupt_flag.notna().sum()),
        **flags)
    support = signal.support_record(selection)
    support["schedule"] = "full_intervals_with_end_at_or_before_record_cell_extent"
    support["windows"] = scheduled
    support["observations"] = owned(observations.__dict__)
    trajectory_use = signal.use(selection.positions, observations, METRIC)
    if any(row["use_status"] == "unknown_provenance" for row in rows):
        trajectory_use = "unknown_provenance"
    record = dict(summary=summary, window_settings=settings, support=support, method_id=METHOD,
        config_id=identity(dict(window=settings, spec=spec.identity if spec else None,
                               reference=reference.identity if reference else None, fit_scope=fit_scope,
                               observations=observations.__dict__)),
        reference_id=reference.identity if reference else None,
        fit_scope="frozen_reference" if reference is not None and spec is None else fit_scope,
        compute_status=status, reason=None if len(valid) else "no eligible complete nondegenerate windows",
        metric_status=METRIC, use_status=trajectory_use,
        basis=signal.basis, units="1", measurement_unit=signal.metadata[variable]["unit"])
    record["identity"] = identity(dict(record=record, rows=rows))
    return FI1Trajectory(table, record)

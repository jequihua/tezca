"""Independent actual-time LS projection-density WOSA; no historical parity."""

import numpy as np
import pandas as pd

from ._owned import identity, owned
from .individual import _ColumnSignal, _observations, _result
from .spectral_summary import ModelSpec, _positive, _real_array, _summary, fit_log_models
from .support import _aware_timestamp, positive_duration, reconstruction_burden, select_support


METHOD = "spectral.ls_projection_density_wosa_v1"


class IrregularSpec:
    def __init__(self, *, interval, segment, hop, anchor, frequency_hz, f_ref_hz,
                 requested_band_hz, taper="boxcar", detrend="none", min_points=3,
                 min_span="1ns", min_cycles=0, rank_tolerance=1e-10, weights=None):
        if len(interval) != 2:
            raise ValueError("interval requires start/end")
        start, end = [_aware_timestamp(t, "interval") for t in interval]
        anchor = _aware_timestamp(anchor, "anchor")
        duration, stride = positive_duration(segment, "segment"), positive_duration(hop, "hop")
        span = positive_duration(min_span, "min_span")
        if end <= start or anchor > start or span > duration:
            raise ValueError("increasing interval, anchor <= start, min_span <= segment required")
        frequency = _real_array(frequency_hz)
        band = _real_array(requested_band_hz)
        if (frequency.ndim != 1 or not len(frequency) or not np.isfinite(frequency).all()
                or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0)):
            raise ValueError("frequencies must be strictly increasing, positive and finite")
        if (band.shape != (2,) or not np.isfinite(band).all() or band[0] <= 0
                or band[1] < band[0] or frequency[0] < band[0] or frequency[-1] > band[1]):
            raise ValueError("requested positive band must contain the frequency grid")
        if taper not in {"boxcar", "hann"} or detrend not in {"none", "constant", "linear"}:
            raise ValueError("unsupported taper/detrend")
        if type(min_points) is not int or min_points < 3:
            raise ValueError("min_points must be an integer >=3")
        tol = _positive(rank_tolerance, "rank_tolerance")
        if tol >= 1:
            raise ValueError("rank_tolerance must be <1")
        weight = None if weights is None else _real_array(weights)
        if weight is not None and (weight.ndim != 1 or not np.isfinite(weight).all() or np.any(weight <= 0)):
            raise ValueError("weights must be a positive finite vector in chronological segment order")
        self._record = dict(interval=[start, end], segment=duration, hop=stride, anchor=anchor,
            frequency_hz=frequency.copy(), f_ref_hz=_positive(f_ref_hz, "f_ref_hz"),
            requested_band_hz=band.copy(), taper=taper, detrend=detrend, min_points=min_points,
            min_span=span, min_cycles=_positive(min_cycles, "min_cycles", zero=True),
            rank_tolerance=tol, weights=None if weight is None else weight.copy(), closure="left")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def _segment_starts(cfg):
    start, end = (t.value for t in cfg["interval"])
    anchor, hop = cfg["anchor"].value, cfg["hop"].value
    first = anchor + ((start-anchor+hop-1)//hop)*hop
    # Request all full anchor-aligned windows. A shorter analysis interval still
    # retains its sole partial-window diagnostic instead of disappearing.
    stop = end-cfg["segment"].value+1
    return range(first, max(stop, first+1) if first < end else first, hop)


def _projection(t, frequency, cfg):
    """Return L with B=L L.T, using only O(n) two-column storage."""
    phase = 2*np.pi*frequency*t
    if not np.isfinite(phase).all():
        return None, dict(reason="nonfinite_phase", rank=None, trace=None)
    design = np.column_stack((np.cos(phase), np.sin(phase)))
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*cfg["rank_tolerance"]))
    info = dict(rank=rank, singular_values=singular, trace=None, reason=None)
    if rank != 2:
        return None, dict(info, reason="rank_deficient_frequency")
    taper = np.ones(len(t)) if cfg["taper"] == "boxcar" else .5-.5*np.cos(2*np.pi*t/(cfg["segment"].value/1e9))
    factor = taper[:, None]*u
    if cfg["detrend"] != "none":
        d = np.ones((len(t), 1))
        if cfg["detrend"] == "linear":
            d = np.column_stack((d, (t-t.mean())/(cfg["segment"].value/1e9)))
        q, _ = np.linalg.qr(d, mode="reduced")
        factor -= q@(q.T@factor)
    trace = float(np.sum(factor*factor))
    info["trace"] = trace
    if trace <= np.finfo(float).eps**2:
        return None, dict(info, reason="zero_projection_trace")
    return factor/np.sqrt(trace), info


def _pool(entries):
    """Canonical pooling and shared-row quadratic moments, independent of traversal."""
    entries = sorted(entries, key=lambda e: e["segment_id"])
    weights = np.array([e["weight"] for e in entries], dtype=float)
    weights = weights/weights.max()
    weights /= weights.sum()
    trace_square = 0.
    for i, left in enumerate(entries):
        for j in range(i, len(entries)):
            right = entries[j]
            _, a, b = np.intersect1d(left["positions"], right["positions"], return_indices=True)
            cross = left["factor"][a].T@right["factor"][b]
            trace_square += (1 if i == j else 2)*weights[i]*weights[j]*float(np.sum(cross*cross))
    return dict(psd=float(weights@np.array([e["psd"] for e in entries])),
        normalized_weights=weights, segment_ids=[e["segment_id"] for e in entries],
        n_eff=1/(2*trace_square), approximate_dof=1/trace_square,
        relative_variance=2*trace_square, trace_square=trace_square)


def irregular_spectrum(data, *, variable, spec, observations=None):
    """Windowed LSFT-WOSA on actual eligible instants, with per-ordinate support."""
    if not isinstance(spec, IrregularSpec):
        raise ValueError("irregular_spectrum requires IrregularSpec")
    obs = _observations(observations)
    signal = _ColumnSignal(data, variable)
    cfg = spec.record
    starts = list(_segment_starts(cfg))
    weights = np.ones(len(starts)) if cfg["weights"] is None else cfg["weights"]
    if len(weights) != len(starts):
        raise ValueError("one positive weight is required for every requested segment, including tails")
    ticks = signal.time.index.asi8
    values = signal.values.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
    eligible = signal.eligible(obs)[:, 0]
    frequencies = cfg["frequency_hz"]
    entries = [[] for _ in frequencies]
    segments, used = [], set()
    interval_start, interval_end = (t.value for t in cfg["interval"])
    for k, start in enumerate(starts):
        end = start + cfg["segment"].value
        positions = np.flatnonzero((ticks >= start) & (ticks < min(end, interval_end)))
        selected = positions[eligible[positions]]
        t = np.array([(int(ticks[p])-start)/1e9 for p in selected])
        span = float(t[-1]-t[0]) if len(t) else 0.
        reason = ("partial_analysis_window" if end > interval_end else
                  "insufficient_points" if len(t) < max(cfg["min_points"], obs.min_points) else
                  "insufficient_occupied_span" if span < cfg["min_span"].value/1e9 else None)
        segment_id = identity(dict(input=signal.identity, start=start, end=end, settings=cfg))
        row = dict(segment_id=segment_id, start=pd.Timestamp(start, tz="UTC").isoformat(),
            end=pd.Timestamp(end, tz="UTC").isoformat(), positions=selected.tolist(),
            excluded_positions=positions[~eligible[positions]].tolist(),
            exclusions=[dict(position=int(p), reason="ineligible_observation_or_missing_value")
                        for p in positions[~eligible[positions]]],
            occupied_span_seconds=span, weight=float(weights[k]), reason=reason,
            compute_status="ok" if reason is None else "insufficient_support",
            burden=reconstruction_burden(signal, selected), ordinates=[])
        for j, frequency in enumerate(frequencies):
            why = reason or ("insufficient_cycles" if frequency*span < cfg["min_cycles"] else None)
            info = dict(reason=why, rank=None, trace=None)
            factor = None
            if why is None:
                factor, info = _projection(t, frequency, cfg)
            density = None
            if factor is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    projected = factor.T@values[selected]
                    density = float(2*np.sum(projected*projected)/cfg["f_ref_hz"])
                if not np.isfinite(density):
                    info["reason"], density = "nonfinite_projection_density", None
                else:
                    entries[j].append(dict(segment_id=segment_id, positions=selected,
                        factor=factor, weight=weights[k], psd=density, span=span))
            row["ordinates"].append(dict(info, frequency_hz=float(frequency), psd=density,
                cycles=frequency*span, compute_status="ok" if density is not None else "unavailable"))
        segments.append(row)
    psd = np.full(len(frequencies), np.nan)
    pooled = []
    for j, group in enumerate(entries):
        row = dict(frequency_hz=float(frequencies[j]), raw_segments=len(group),
            compute_status="ok" if len(group) >= 2 else "insufficient_segments",
            reason=None if len(group) >= 2 else "averaging requires at least two supported segments",
            segment_ids=[], normalized_weights=[], n_eff=None, approximate_dof=None,
            relative_variance=None, trace_square=None, psd=None,
            min_occupied_span_seconds=min((e["span"] for e in group), default=0))
        if group:
            diagnostics = _pool(group)
            row.update(diagnostics)
            if len(group) >= 2:
                psd[j] = row["psd"]
                for entry in group:
                    used.update(entry["positions"].tolist())
            else:
                row["psd"] = None
        pooled.append(row)
    selection = select_support(signal, observations=obs, policy="irregular", estimator="irregular", within=sorted(used))
    good = np.flatnonzero(np.isfinite(psd))
    resolution = 1e9/cfg["segment"].value
    resolved = dict(cfg, physical_resolution_hz=resolution, frequency_units="Hz",
        normalization="2*x.T*B*x/f_ref; B=R*W*Q*Q.T*W*R/trace(A)",
        density_scale_role="declared scale, not inferred irregular Nyquist",
        theoretical_nyquist_hz=None, numpy_version=np.__version__,
        supported_positive_range_hz=[frequencies[good[0]], frequencies[good[-1]]] if len(good) else None)
    extra = dict(requested=cfg, resolved=resolved, segments=segments, frequency_support=pooled,
        contributing_positions=sorted(used), outside_interval_positions=np.flatnonzero(
            (ticks < interval_start) | (ticks >= interval_end)).tolist(),
        constant_input=bool(used and np.ptp(values[sorted(used)]) == 0),
        historical_parity="unavailable_Q001", effective_averages=dict(
            method="shared_observation_gaussian_quadratic_moments_v1", approximate=True,
            assumptions="iid equal-variance Gaussian observations; moment matching, not generic chi-square inference",
            values=[r["n_eff"] for r in pooled], dof=[r["approximate_dof"] for r in pooled]),
        raw_segments=len(segments), variable_order=[variable])
    return _result(signal, selection, obs, metric=METHOD,
        value=dict(frequency_hz=frequencies, psd=psd) if len(good) else None,
        status="ok" if len(good) else "insufficient_support",
        reason=None if len(good) else "no frequency has two supported segments",
        units=signal.metadata[variable]["unit"]+"^2/Hz", config=dict(requested=cfg, resolved=resolved),
        metric_status="descriptive_irregular_projection_density", extra=extra)


def irregular_spectral_models(spectrum, *, spec, min_ordinates=3, min_cycles=2):
    """Explicit adapter to the same log-fit definition, without extrapolation.

    Greedily retain actual supported ordinates separated by >=1/T, starting at
    the lowest requested frequency. Dense display grids cannot earn resolution.
    """
    if not isinstance(spec, ModelSpec) or type(min_ordinates) is not int or min_ordinates < 3:
        raise ValueError("ModelSpec and integer min_ordinates >=3 required")
    min_cycles = _positive(min_cycles, "min_cycles", zero=True)
    record = spectrum.summary()
    if record["method_id"] != METHOD:
        raise ValueError("adapter requires independent LS projection-density WOSA")
    if spec.record["basis"] is not None and spec.record["basis"] != record["representation"]:
        raise ValueError("requested basis does not match spectrum")
    settings = dict(min_ordinates=min_ordinates, min_cycles=min_cycles,
                    selection="ascending_actual_frequencies_separated_by_1/T")
    # Include adapter gates in the same summary identity as the log-fit settings.
    class SummarySpec:
        record = dict(spec.record, adapter=settings)
    def result(value, status, reason, extra=None):
        return _summary(record, SummarySpec(), value, "spectral.irregular_log_models_v1",
                        status=status, reason=reason, extra=extra)
    if record["compute_status"] != "ok":
        return result(None, record["compute_status"], record["reason"])
    f, p = record["value"]["frequency_hz"], record["value"]["psd"]
    interval = spec.record["interval_hz"]
    resolution = record["support"]["resolved"]["physical_resolution_hz"]
    selected, exclusions = [], []
    for j, (frequency, power) in enumerate(zip(f, p)):
        span = record["support"]["frequency_support"][j]["min_occupied_span_seconds"]
        why = ("outside_requested_interval" if interval is not None and not interval[0] <= frequency <= interval[1] else
               "unsupported_or_nonpositive_power" if not np.isfinite(power) or power <= 0 else
               "insufficient_cycles" if frequency*span < min_cycles else
               "unresolved_dense_ordinate" if selected and frequency-f[selected[-1]] < resolution*(1-1e-12) else None)
        if why:
            exclusions.append(dict(position=j, reason=why))
        else:
            selected.append(j)
    extra = dict(adapter_selection=dict(positions=selected, exclusions=exclusions, **settings))
    if len(selected) < min_ordinates:
        status = "insufficient_cycles" if any(r["reason"] == "insufficient_cycles" for r in exclusions) else "insufficient_frequency_resolution"
        return result(None, status, "too few cycle-qualified resolved positive ordinates", extra)
    fit = fit_log_models(f[selected], p[selected], spec=spec)
    extra["fit_selection"] = fit["selection"]
    extra["fit_grid"] = "actual_supported_resolution_thinned"
    return result(fit if fit["compute_status"] == "ok" else None, fit["compute_status"], fit["reason"], extra)

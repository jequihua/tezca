"""Descriptive log models, ranked peaks and supported density integrals."""

from numbers import Real

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ._owned import identity, owned
from .data import ResultRecord


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _real_array(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("ordinates and frequency/period geometry must be real numeric arrays")
    return raw.astype(float)


class ModelSpec:
    def __init__(self, *, interval_hz=None, candidates_hz=None, min_arm=3,
                 log_base=10, tie_tolerance=1e-12, exact_rss_tolerance=1e-24, basis=None):
        if log_base not in (np.e, 10) or isinstance(log_base, bool):
            raise ValueError("log_base must be e or 10")
        if type(min_arm) is not int or min_arm < 3:
            raise ValueError("min_arm must be an integer >=3")
        interval = None if interval_hz is None else _interval(interval_hz)
        candidates = None if candidates_hz is None else _real_array(candidates_hz)
        if candidates is not None and (candidates.ndim != 1 or not np.isfinite(candidates).all() or
                np.any(candidates <= 0) or np.any(np.diff(candidates) <= 0)):
            raise ValueError("candidate breaks must be increasing positive finite frequencies")
        _basis(basis)
        self._record = dict(interval_hz=interval, candidates_hz=candidates, min_arm=min_arm,
            log_base=float(log_base), tie_tolerance=_positive(tie_tolerance, "tie_tolerance", zero=True),
            exact_rss_tolerance=_positive(exact_rss_tolerance, "exact_rss_tolerance", zero=True),
            basis=basis, interval_closure="both", positive_floor=None, continuous=True,
            tie_policy="lower_frequency", single_k=3, broken_k=5,
            parameter_count="including residual variance and estimated breakpoint")

    @property
    def record(self):
        return owned(self._record)


def _basis(basis):
    if basis is not None and (not isinstance(basis, str) or not basis.strip()):
        raise ValueError("basis must be a nonempty representation or None")


def _interval(interval):
    a = _real_array(interval)
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] <= 0 or a[1] <= a[0]:
        raise ValueError("interval must have increasing positive finite bounds")
    return a.copy()


def _arrays(frequency_hz, density):
    f, p = _real_array(frequency_hz), _real_array(density)
    if f.ndim != 1 or p.shape != f.shape or not np.isfinite(f).all() or np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be a finite increasing nonnegative vector with matching ordinates")
    return f, p


def _frequency_mask(f, supported_range_hz):
    if supported_range_hz is None:
        return np.ones(len(f), dtype=bool)
    low, high = _interval(supported_range_hz)
    return (f >= low) & (f <= high)


def fit_log_models(frequency_hz, power, *, spec, supported_range_hz=None):
    """Array primitive; deterministic ordinate controls have no source provenance."""
    if not isinstance(spec, ModelSpec):
        raise ValueError("fit requires ModelSpec")
    f, p = _arrays(frequency_hz, power)
    cfg = spec.record
    interval = cfg["interval_hz"]
    inside = np.ones(len(f), dtype=bool) if interval is None else (f >= interval[0]) & (f <= interval[1])
    supported = _frequency_mask(f, supported_range_hz)
    eligible = inside & supported & (f > 0) & np.isfinite(p) & (p > 0)
    selection = dict(positions=np.flatnonzero(eligible).tolist(),
        inherited_range_hz=owned(supported_range_hz),
        frequency_exclusions=[dict(position=int(j), reason="outside_frequency_support") for j in np.flatnonzero(~supported)],
        resolved_interval_hz=[float(f[eligible][0]), float(f[eligible][-1])] if eligible.any() else None,
        outside_interval=np.flatnonzero(~inside).tolist(),
        nonpositive_frequency=np.flatnonzero(f <= 0).tolist(),
        nonpositive_or_nonfinite_power=np.flatnonzero(~np.isfinite(p) | (p <= 0)).tolist())
    x, y = np.log(f[eligible])/np.log(cfg["log_base"]), np.log(p[eligible])/np.log(cfg["log_base"])
    n = len(x)
    def fit(design, k, breakpoint=None):
        coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        rss = float(np.sum((y-design@coefficient)**2))
        sst = float(np.sum((y-y.mean())**2))
        exact = rss <= cfg["exact_rss_tolerance"]
        status = "exact_fit" if exact else "insufficient_aicc_points" if n <= k+1 else "ok"
        aicc = None if status != "ok" else n*np.log(rss/n)+2*k+2*k*(k+1)/(n-k-1)
        row = dict(n=n, k=k, rss=rss, aicc=aicc, comparison_status=status,
            r2=None if sst == 0 else 1-rss/sst, r2_reason="constant log ordinates" if sst == 0 else None,
            intercept=float(coefficient[0]), beta_low=float(-coefficient[1]),
            beta_high=float(-coefficient[1]-(coefficient[2] if len(coefficient)==3 else 0)), break_hz=breakpoint,
            break_period_seconds=None if breakpoint is None else 1/breakpoint)
        return row
    if n < 3:
        return owned(dict(compute_status="insufficient_support", reason="fewer than three positive ordinates",
                          selection=selection, single=None, broken=None, candidates=[], preferred=None, delta_aicc=None))
    single = fit(np.column_stack([np.ones(n), x]), 3)
    candidates = f[eligible] if cfg["candidates_hz"] is None else cfg["candidates_hz"]
    rows, best = [], None
    for candidate in candidates:
        low, high = int(np.sum(f[eligible] <= candidate)), int(np.sum(f[eligible] > candidate))
        if low < cfg["min_arm"] or high < cfg["min_arm"]:
            rows.append(dict(break_hz=float(candidate), comparison_status="insufficient_arm_points", low_n=low, high_n=high))
            continue
        b = np.log(candidate)/np.log(cfg["log_base"])
        row = fit(np.column_stack([np.ones(n), x, np.maximum(x-b, 0)]), 5, float(candidate))
        row.update(low_n=low, high_n=high)
        rows.append(row)
        if best is None or row["rss"] < best["rss"]-cfg["tie_tolerance"]:
            best = row
    delta = None if best is None or best["aicc"] is None or single["aicc"] is None else best["aicc"]-single["aicc"]
    return owned(dict(compute_status="ok", reason=None, selection=selection, single=single, broken=best,
        candidates=rows, resolved_candidates_hz=candidates.copy(), delta_aicc=delta,
        preferred=None if delta is None else "broken" if delta < -cfg["tie_tolerance"] else "single",
        comparison_status="unavailable" if delta is None else "descriptive_working_aicc",
        semantics="retrospective descriptive log-ordinate least squares; no process calibration"))


class PeakSpec:
    def __init__(self, *, period_seconds=None, min_cycles=2, max_peaks=5, prominence="linear", basis=None):
        if type(max_peaks) is not int or max_peaks < 1 or prominence not in {"linear", "log10"}:
            raise ValueError("positive integer max_peaks and linear/log10 prominence required")
        _basis(basis)
        self._record = dict(period_seconds=None if period_seconds is None else _interval(period_seconds),
            min_cycles=_positive(min_cycles, "min_cycles", zero=True), max_peaks=max_peaks,
            prominence=prominence, basis=basis, tie_policy="descending_prominence_then_lower_frequency")

    @property
    def record(self):
        return owned(self._record)


def rank_peaks(frequency_hz, power, *, duration_seconds, spec, supported_range_hz=None):
    if not isinstance(spec, PeakSpec):
        raise ValueError("peaks require PeakSpec")
    duration = _positive(duration_seconds, "duration_seconds", zero=True)
    f, p = _arrays(frequency_hz, power)
    cfg = spec.record
    supported = _frequency_mask(f, supported_range_hz)
    eligible = supported & (f > 0) & np.isfinite(p) & (p >= 0 if cfg["prominence"] == "linear" else p > 0)
    # Exclusions split runs: no prominence is inferred across a missing ordinate.
    positions = np.flatnonzero(eligible)
    runs = np.split(positions, np.flatnonzero(np.diff(positions) != 1)+1)
    rows, rejected = [], []
    for run in runs:
        y = p[run] if cfg["prominence"] == "linear" else np.log10(p[run])
        peaks, properties = find_peaks(y, prominence=0)
        for local, prominence in zip(peaks, properties["prominences"]):
            j = run[local]
            period, cycles = 1/f[j], f[j]*duration
            row = dict(position=int(j), frequency_hz=float(f[j]), period_seconds=float(period), power=float(p[j]),
                       prominence=float(prominence), cycles=float(cycles))
            limits = cfg["period_seconds"]
            reason = "outside_period_range" if limits is not None and not limits[0] <= period <= limits[1] else "insufficient_cycles" if cycles < cfg["min_cycles"] else None
            if reason:
                rejected.append(dict(**row, reason=reason))
            else:
                rows.append(row)
    rows.sort(key=lambda r: (-r["prominence"], r["frequency_hz"]))
    rejected.extend(dict(**r, reason="max_peaks") for r in rows[cfg["max_peaks"]:])
    rows = rows[:cfg["max_peaks"]]
    columns = ["frequency_hz", "period_seconds", "power", "prominence", "cycles"]
    table = pd.DataFrame(rows, columns=columns).astype({c: "float64" for c in columns})
    table.insert(0, "position", pd.Series([r["position"] for r in rows], dtype="int64"))
    table.insert(0, "rank", np.arange(1, len(table)+1, dtype=np.int64))
    return owned(dict(table=table, rejected=rejected, excluded_positions=np.flatnonzero(~eligible).tolist(),
                      inherited_range_hz=supported_range_hz, eligible_positions=positions.tolist(),
                      resolved_interval_hz=[float(f[positions[0]]), float(f[positions[-1]])] if len(positions) else None,
                      frequency_exclusions=[dict(position=int(j), reason="outside_frequency_support") for j in np.flatnonzero(~supported)],
                      empty_reason="no eligible local peaks" if not rows else None))


class BandSpec:
    def __init__(self, bounds, *, coordinates="hours", min_cycles=2, min_ordinates=3, basis=None):
        interval = _interval(bounds)
        if coordinates not in {"hours", "hz"} or type(min_ordinates) is not int or min_ordinates < 3:
            raise ValueError("hours/hz coordinates and integer min_ordinates >=3 required")
        _basis(basis)
        self._record = dict(requested_bounds=interval, coordinates=coordinates,
            frequency_bounds_hz=interval if coordinates == "hz" else 1/(interval[::-1]*3600),
            min_cycles=_positive(min_cycles, "min_cycles", zero=True), min_ordinates=min_ordinates, basis=basis,
            quadrature="piecewise_linear_trapezoid_with_interpolated_boundaries",
            denominator="entire_supported_positive_frequency_range_DC_excluded")

    @property
    def record(self):
        return owned(self._record)


def integrate_band(frequency_hz, density, *, physical_frequency_hz, duration_seconds, spec, supported_range_hz=None):
    if not isinstance(spec, BandSpec):
        raise ValueError("band requires BandSpec")
    f, p = _arrays(frequency_hz, density)
    physical = _real_array(physical_frequency_hz)
    if physical.ndim != 1 or not np.isfinite(physical).all() or np.any(physical < 0) or np.any(np.diff(physical) <= 0):
        raise ValueError("physical frequencies must be finite increasing nonnegative")
    duration = _positive(duration_seconds, "duration_seconds", zero=True)
    positive = f > 0
    f, p = f[positive], p[positive]
    cfg = spec.record
    low, high = cfg["frequency_bounds_hz"]
    count = int(np.sum((physical > 0) & (physical >= low) & (physical <= high)))
    support = dict(physical_ordinates=count, cycles_at_low_frequency=low*duration, denominator_bounds_hz=None)
    def missing(status, reason):
        return dict(compute_status=status, reason=reason, value=None, support=support)
    if not len(f) or not np.isfinite(p).all() or np.any(p < 0):
        return missing("degenerate_input", "no finite nonnegative positive-frequency density")
    available = [f[0], f[-1]] if supported_range_hz is None else _interval(supported_range_hz)
    support["denominator_bounds_hz"] = list(available)
    if low < available[0] or high > available[1] or available[0] < f[0] or available[1] > f[-1]:
        return missing("outside_frequency_support", "band/denominator would require extrapolation")
    if count < cfg["min_ordinates"]:
        return missing("insufficient_frequency_resolution", "too few unpadded physical positive ordinates")
    if low*duration < cfg["min_cycles"]:
        return missing("insufficient_cycles", "longest contributing run has too few cycles")
    def integral(a, b):
        grid = np.r_[a, f[(f > a) & (f < b)], b]
        return float(np.trapezoid(np.interp(grid, f, p), grid))
    energy, total = integral(low, high), integral(*available)
    return dict(compute_status="ok", reason=None, support=support,
        value=dict(power=energy, positive_frequency_power=total, fraction=None if total == 0 else energy/total,
                   fraction_status="undefined_zero_denominator" if total == 0 else "ok"))


def _input(spectrum, spec):
    if not isinstance(spectrum, ResultRecord) or spectrum.summary()["method_id"] != "spectral.welch_mean_v1":
        raise ValueError("summary requires a tezca Welch result")
    record = spectrum.summary()
    if spec.record["basis"] is not None and spec.record["basis"] != record["representation"]:
        raise ValueError("requested basis does not match spectrum")
    return record


def _summary(record, spec, value, method, *, status="ok", reason=None, units="1", extra=None):
    support = dict(record["support"], spectral_input_id=identity(record), summary_settings=spec.record)
    if extra is not None:
        support.update(extra)
    return ResultRecord(value, compute_status=status, reason=reason,
        metric_status="descriptive_spectral_summary",
        use_status="sensitivity_only" if record["use_status"] == "primary_eligible" else record["use_status"], support=support,
        units=units, method_id=method, config_id=identity(dict(input=identity(record), settings=spec.record)),
        reference_id=record["reference_id"], representation=record["representation"])


def spectral_models(spectrum, *, spec):
    if not isinstance(spec, ModelSpec):
        raise ValueError("models require ModelSpec")
    record = _input(spectrum, spec)
    if record["compute_status"] != "ok":
        return _summary(record, spec, None, "spectral.log_models_v1", status=record["compute_status"], reason=record["reason"])
    value = record["value"]
    available = record["support"]["resolved"]["supported_positive_range_hz"]
    if available is None or available[0] == available[1]:
        return _summary(record, spec, None, "spectral.log_models_v1", status="insufficient_frequency_resolution",
            reason="no positive frequency interval", extra=dict(inherited_range_hz=available,
                excluded_frequency_positions=list(range(len(value["physical_frequency_hz"])))))
    fit = fit_log_models(value["physical_frequency_hz"], value["physical_psd"], spec=spec, supported_range_hz=available)
    return _summary(record, spec, fit if fit["compute_status"] == "ok" else None, "spectral.log_models_v1",
        status=fit["compute_status"], reason=fit["reason"], extra=dict(fit_selection=fit["selection"], fit_grid="unpadded"))


def spectral_peaks(spectrum, *, spec):
    if not isinstance(spec, PeakSpec):
        raise ValueError("peaks require PeakSpec")
    record = _input(spectrum, spec)
    if record["compute_status"] != "ok":
        return _summary(record, spec, None, "spectral.peaks_v1", status=record["compute_status"], reason=record["reason"], units=record["units"])
    value = record["value"]
    available = record["support"]["resolved"]["supported_positive_range_hz"]
    if available is None or available[0] == available[1]:
        return _summary(record, spec, None, "spectral.peaks_v1", status="insufficient_frequency_resolution",
            reason="no positive frequency interval", units=record["units"], extra=dict(inherited_range_hz=available,
                excluded_frequency_positions=list(range(len(value["frequency_hz"])))))
    peaks = rank_peaks(value["frequency_hz"], value["psd"], duration_seconds=record["support"]["longest_run_seconds"],
                       spec=spec, supported_range_hz=available)
    peaks["table"]["basis"] = record["representation"]
    return _summary(record, spec, peaks, "spectral.peaks_v1", units=record["units"], extra=dict(prominence_units=record["units"] if spec.record["prominence"] == "linear" else "log10(power ordinate)"))


def band_power(spectrum, *, spec):
    if not isinstance(spec, BandSpec):
        raise ValueError("band requires BandSpec")
    record = _input(spectrum, spec)
    if record["support"].get("requested", {}).get("scaling") != "density":
        raise ValueError("band integration requires density scaling")
    units = record["units"].removesuffix("/Hz")
    if record["compute_status"] != "ok":
        return _summary(record, spec, None, "spectral.band_power_v1", status=record["compute_status"], reason=record["reason"], units=units)
    available = record["support"]["resolved"]["supported_positive_range_hz"]
    if available is None or available[0] == available[1]:
        return _summary(record, spec, None, "spectral.band_power_v1", status="insufficient_frequency_resolution", reason="no positive frequency interval", units=units)
    value = record["value"]
    result = integrate_band(value["frequency_hz"], value["psd"], physical_frequency_hz=value["physical_frequency_hz"],
        duration_seconds=record["support"]["longest_run_seconds"], spec=spec, supported_range_hz=available)
    return _summary(record, spec, result["value"], "spectral.band_power_v1", status=result["compute_status"], reason=result["reason"], units=units, extra=result["support"])

"""Independently specified report-style mean of contiguous-run coherences."""

import numpy as np
import scipy
from scipy.signal import csd, get_window, welch

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations
from .pair_audit import audit_pair
from .support import ObservationPolicy, TimeAxis, _runs, positive_duration, select_support


class ReportCoherenceSpec:
    def __init__(self, *, basis, period_band, taper, detrend, nperseg=None,
                 min_nperseg=64, max_nperseg=256, min_segments=4, periodic=True,
                 route="independent_report"):
        if basis not in {"raw", "daily_residual"} or taper not in {"boxcar", "hann"} or detrend not in {"none", "constant", "linear"}:
            raise ValueError("explicit basis, boxcar/hann taper and none/constant/linear detrend required")
        if type(periodic) is not bool or route not in {"independent_report", "exact_report_compatibility"}:
            raise ValueError("invalid taper convention or route")
        if type(min_segments) is not int or min_segments < 4:
            raise ValueError("report coherence requires at least four raw segments")
        for value in (min_nperseg, max_nperseg):
            if type(value) is not int or value < 4 or value % 2:
                raise ValueError("segment bounds must be even integers >=4")
        if min_nperseg > max_nperseg or nperseg is not None and (type(nperseg) is not int or nperseg < 4 or nperseg % 2):
            raise ValueError("invalid segment length/bounds")
        if not isinstance(period_band, (list, tuple)) or len(period_band) != 2:
            raise ValueError("period band must contain short and long durations")
        short, long = [positive_duration(v, "period") for v in period_band]
        if short >= long:
            raise ValueError("period band must increase")
        self._record = dict(basis=basis, period_band=[short, long], taper=taper, detrend=detrend,
                            periodic=periodic, nperseg=nperseg, min_nperseg=min_nperseg,
                            max_nperseg=max_nperseg, min_segments=min_segments, route=route,
                            overlap_fraction=.5, scaling="density", aggregation="equal_run_coherence",
                            csd_definition="conj(X)*Y", dc="excluded_from_summary", nyquist="excluded_from_summary")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def _common_pair(first, second, variable_a, variable_b, basis, observations):
    a, b = _ColumnSignal(first, variable_a), _ColumnSignal(second, variable_b)
    if a.basis != basis or b.basis != basis:
        raise ValueError("pair basis must match declared preprocessing")
    if a.time.mode != "regular" or b.time.mode != "regular":
        raise ValueError("coherence requires explicit regular clocks")
    if a.time.cadence != b.time.cadence:
        raise ValueError("pair cadence mismatch; explicit prior alignment required")
    pa, pb = a.time.record["phase"], b.time.record["phase"]
    if pa is not None and pb is not None and (pa.value - pb.value) % a.time.cadence.value:
        raise ValueError("pair phase mismatch")
    common = a.time.index.tz_convert("UTC").intersection(b.time.index.tz_convert("UTC"), sort=True)
    ia, ib = a.time.index.get_indexer(common), b.time.index.get_indexer(common)
    eligible = a.eligible(observations)[ia, 0] & b.eligible(observations)[ib, 0]
    axis = TimeAxis(common, mode="regular", cadence=a.time.cadence, phase=pa)
    return a, b, axis, ia, ib, eligible


def _length(n, settings):
    length = settings["nperseg"]
    if length is None:
        candidates = range(settings["max_nperseg"], settings["min_nperseg"] - 1, -2)
        length = next((v for v in candidates if n >= v and 1 + (n - v) // (v // 2) >= settings["min_segments"]), None)
    return length


def _run_spectra(x, y, *, length, fs, settings):
    window = get_window(settings["taper"], length, fftbins=settings["periodic"])
    if np.dot(window, window) <= 0:
        raise ValueError("taper must have positive energy")
    hop, nfft = settings.get("hop", length // 2), settings.get("nfft", length)
    used_stop = ((len(x) - length) // hop) * hop + length
    x, y = x[:used_stop], y[:used_stop]
    options = dict(fs=fs, window=window, nperseg=length, noverlap=length - hop,
                   nfft=nfft, detrend=False if settings["detrend"] == "none" else settings["detrend"],
                   scaling="density", return_onesided=True, average="mean")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        frequency, xx = welch(x, **options)
        _, yy = welch(y, **options)
        _, xy = csd(x, y, **options)
        # Ratios avoid an unnecessary overflow in Sxx*Syy.
        coherence = np.abs(xy / np.sqrt(xx) / np.sqrt(yy)) ** 2
    constant = bool(np.ptp(x) == 0 or np.ptp(y) == 0)
    valid = np.isfinite(coherence) & np.isfinite(xx) & np.isfinite(yy) & (xx > 0) & (yy > 0) & ~np.bool_(constant)
    coherence[~valid] = np.nan
    if np.any(coherence[valid] > 1 + 1e-10):
        raise ArithmeticError("coherence outside numerical range")
    coherence[valid] = np.clip(coherence[valid], 0, 1)
    return dict(frequency_hz=frequency, sxx=xx, syy=yy, sxy_real=xy.real, sxy_imag=xy.imag,
                raw_coherence=coherence, physical_spacing_hz=fs / length,
                supported_positive_range_hz=[frequency[1], frequency[-2]],
                sampling_hz=fs, nperseg=length, hop=hop, nfft=nfft,
                taper=window, detrend=settings["detrend"], scaling="density", constant_input=constant)


def aggregate_report_runs(runs, *, period_band):
    """Equal-run interpolation on a zero-anchored coarsest physical grid."""
    if not isinstance(period_band, (tuple, list)) or len(period_band) != 2:
        raise ValueError("period_band requires short and long durations")
    short, long = [positive_duration(v, "period") for v in period_band]
    if short >= long:
        raise ValueError("period band must increase")
    runs = sorted(owned(list(runs)), key=lambda r: r["run_id"])
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("run identities must be unique")
    record = dict(method_id="coherence.report_equal_run_grid_v1", run_ids=[r["run_id"] for r in runs],
                  weights=[1 / len(runs)] * len(runs) if runs else [], interpolation="linear_no_extrapolation",
                  requested_band_hz=[1e9 / long.value, 1e9 / short.value], frequency_hz=np.array([]),
                  raw_coherence=np.array([]), band_mean=None, common_range_hz=None,
                  compute_status="insufficient_support", reason="no accepted runs")
    if runs:
        for run in runs:
            f, c = np.asarray(run["frequency_hz"]), np.asarray(run["raw_coherence"])
            if f.ndim != 1 or c.shape != f.shape or len(f) < 3 or not np.isfinite(f).all() or np.any(np.diff(f) <= 0):
                raise ValueError("run frequency/coherence arrays must align on an ordered physical grid")
            if run["physical_spacing_hz"] <= 0 or not np.isfinite(run["physical_spacing_hz"]):
                raise ValueError("physical spacing must be positive")
            spacing = run["physical_spacing_hz"]
            if f[0] != 0 or not np.allclose(np.diff(f), spacing, rtol=1e-12, atol=spacing * 1e-12):
                raise ValueError("run grid must be physical, uniform and DC anchored")
            if np.isinf(c).any() or np.any(np.isfinite(c) & ((c < 0) | (c > 1))):
                raise ValueError("coherence must be within [0,1] or explicitly missing")
        lo = max(record["requested_band_hz"][0], max(r["frequency_hz"][1] for r in runs))
        hi = min(record["requested_band_hz"][1], min(r["frequency_hz"][-2] for r in runs))
        spacing = max(r["physical_spacing_hz"] for r in runs)
        grid = spacing * np.arange(max(1, int(np.ceil(lo / spacing - 1e-12))), int(np.floor(hi / spacing + 1e-12)) + 1)
        grid = grid[(grid >= lo - spacing * 1e-12) & (grid <= hi + spacing * 1e-12)] if hi >= lo else np.array([])
        # Roundoff equality must never send an interpolation request outside support.
        grid = np.clip(grid, lo, hi) if len(grid) else grid
        record.update(frequency_hz=grid, common_range_hz=[lo, hi] if hi >= lo else None, physical_spacing_hz=spacing)
        if len(grid) < 3:
            record.update(compute_status="insufficient_band_support", reason="fewer than three common interior physical ordinates")
        else:
            means = [np.interp(grid, r["frequency_hz"][1:-1], r["raw_coherence"][1:-1]) for r in runs]
            aggregate = np.mean(means, axis=0)
            record["raw_coherence"] = aggregate
            if not np.isfinite(aggregate).all():
                record.update(compute_status="undefined_spectrum", reason="constant input or zero/nonfinite auto power in the common band")
            else:
                record.update(compute_status="ok", reason=None, band_mean=float(np.mean(aggregate)))
    record["aggregation_id"] = identity(record)
    return owned(record)


def _pair_runs(a, b, axis, ia, ib, eligible, settings):
    fs = 1e9 / axis.cadence.value
    x, y = a.values.iloc[:, 0].to_numpy(dtype=float), b.values.iloc[:, 0].to_numpy(dtype=float)
    runs, used_a, used_b = [], set(), set()
    for start, stop in _runs(axis, eligible):
        n = stop - start
        length = _length(n, settings)
        hop = settings.get("hop", None if length is None else length // 2)
        k = 0 if length is None or n < length else 1 + (n - length) // hop
        run = dict(start=axis.index[start].isoformat(), end=(axis.index[stop-1] + axis.cadence).isoformat(),
                   eligible_positions_a=ia[start:stop].tolist(), eligible_positions_b=ib[start:stop].tolist(),
                   points=n, nperseg=length, raw_segments=k, compute_status="insufficient_support",
                   reason="fewer than required raw segments")
        run["run_id"] = identity(dict(pair=[a.identity, b.identity], start=run["start"], end=run["end"], settings=settings))
        if length is not None and k >= settings["min_segments"]:
            used_stop = start + (k - 1) * hop + length
            run.update(_run_spectra(x[ia[start:used_stop]], y[ib[start:used_stop]], length=length, fs=fs, settings=settings))
            run["units"] = dict(frequency_hz="Hz", sxx=f"({a.metadata[a.variable]['unit']})^2/Hz",
                                syy=f"({b.metadata[b.variable]['unit']})^2/Hz",
                                sxy=f"({a.metadata[a.variable]['unit']})*({b.metadata[b.variable]['unit']})/Hz", raw_coherence="1")
            run.update(compute_status="ok", reason=None, used_positions_a=ia[start:used_stop].tolist(),
                       used_positions_b=ib[start:used_stop].tolist(), segment_starts=list(range(0, n - length + 1, hop)))
            used_a.update(ia[start:used_stop].tolist())
            used_b.update(ib[start:used_stop].tolist())
        runs.append(run)
    return runs, used_a, used_b


def _pair_qualification(a, b, used_a, used_b, observations, audit, status, reason):
    supports, uses = {}, []
    for name, signal, used in (("first", a, used_a), ("second", b, used_b)):
        selected = select_support(signal, observations=observations, policy="all_runs", within=sorted(used))
        supports[name] = signal.support_record(selected)
        uses.append(signal.use(sorted(used), observations, "exploratory_coherence"))
    use = min(uses, key=["unknown_provenance", "insufficient_support", "sensitivity_only", "unverified_gap_geometry", "primary_eligible"].index)
    if min(len(used_a), len(used_b)) < observations.min_points:
        status, reason = "insufficient_support", "too few contributing points for the observation policy"
        if use != "unknown_provenance":
            use = "insufficient_support"
    if "unknown_lineage" in audit["reasons"] or use == "unknown_provenance":
        use = "unknown_provenance"
        if status == "ok":
            status, reason = "unknown_provenance", "pair ancestry or contributing provenance is unknown"
        audit["ranking_eligible"] = False
        audit["ranking_status"] = "unknown_provenance"
        audit["audit_id"] = identity({k: v for k, v in audit.items() if k != "audit_id"})
    return supports, use, status, reason


def _report(first, second, *, variable_a, variable_b, spec, observations, lineage, nodes, target):
    settings = spec.record
    audit = audit_pair(first, second, variable_a=variable_a, variable_b=variable_b, lineage=lineage, nodes=nodes, target=target)
    a, b, axis, ia, ib, eligible = _common_pair(first, second, variable_a, variable_b, settings["basis"], observations)
    runs, used_a, used_b = _pair_runs(a, b, axis, ia, ib, eligible, settings)
    aggregate = aggregate_report_runs([r for r in runs if r["compute_status"] == "ok"], period_band=settings["period_band"])
    supports, use, status, reason = _pair_qualification(a, b, used_a, used_b, observations, audit,
                                                      aggregate["compute_status"], aggregate["reason"])
    if settings["route"] == "exact_report_compatibility":
        status, reason = "unresolved_compatibility", "Q001 inputs, helpers and interpolation behavior are not recovered"
    support = dict(pair=audit["pair"], audit=audit, basis=settings["basis"], variables=[variable_a, variable_b],
                   units=dict(first=a.metadata[variable_a]["unit"], second=b.metadata[variable_b]["unit"]),
                   common_instants=[t.isoformat() for t in axis.index], common_eligible_positions=np.flatnonzero(eligible).tolist(),
                   first=supports["first"], second=supports["second"], runs=runs, aggregation=aggregate,
                   requested=settings, resolved=settings, observations=observations.__dict__,
                   backend="scipy.signal.welch/csd", backend_version=scipy.__version__,
                   exact_report_parity="unresolved_Q001", reference_ids=[a.reference_id, b.reference_id])
    value = dict(frequency_hz=aggregate["frequency_hz"], raw_coherence=aggregate["raw_coherence"], band_mean=aggregate["band_mean"]) if status == "ok" else None
    return ResultRecord(value, compute_status=status, reason=reason, metric_status="exploratory_coherence",
                        use_status=use, support=support, units=dict(frequency_hz="Hz", raw_coherence="1", band_mean="1"),
                        method_id="coherence.report_equal_run_v1" if settings["route"] == "independent_report" else "coherence.report_exact_unresolved",
                        config_id=identity(dict(settings=settings, observations=observations.__dict__, audit_id=audit["audit_id"])),
                        reference_id=identity([a.reference_id, b.reference_id]) if a.reference_id or b.reference_id else None,
                        representation=settings["basis"])


def report_coherence(first, second, *, variable_a, variable_b, spec, observations=None, lineage=None, nodes=None, target=None):
    if not isinstance(spec, ReportCoherenceSpec):
        raise ValueError("spec must be ReportCoherenceSpec")
    observations = _observations(observations)
    result = _report(first, second, variable_a=variable_a, variable_b=variable_b, spec=spec,
                     observations=observations, lineage=lineage, nodes=nodes, target=target)
    record = result.summary()
    if record["support"]["audit"]["observed_only_required"]:
        counterpart = _report(first, second, variable_a=variable_a, variable_b=variable_b, spec=spec,
                              observations=ObservationPolicy("observed_only", min_points=observations.min_points),
                              lineage=lineage, nodes=nodes, target=target)
        record["support"]["observed_only_counterpart"] = counterpart.summary()
        return ResultRecord(**record)
    return result

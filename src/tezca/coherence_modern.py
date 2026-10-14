"""Independent pooled-spectra coherence and qualified white-noise inference."""

import numpy as np
import scipy
from scipy.signal import get_window

from ._owned import identity, owned, text_field
from .coherence import ReportCoherenceSpec, _common_pair, _pair_runs, _pair_qualification
from .data import ResultRecord
from .individual import _observations
from .pair_audit import audit_pair
from .spectral import _overlap_square_sum
from .support import ObservationPolicy


class ModernCoherenceSpec:
    def __init__(self, *, basis, period_band, nperseg, hop, taper, detrend,
                 periodic=True, nfft=None, alpha=.05, null="gaussian_white",
                 weighting="equal_segments", route="independent_modern"):
        common = ReportCoherenceSpec(basis=basis, period_band=period_band, taper=taper,
                                     detrend=detrend, nperseg=nperseg, periodic=periodic).record
        if nperseg is None or type(hop) is not int or not 1 <= hop <= nperseg:
            raise ValueError("fixed segment length and integer hop in [1,L] required")
        nfft = nperseg if nfft is None else nfft
        if type(nfft) is not int or nfft < nperseg or nfft % nperseg:
            raise ValueError("FFT length must be a positive integer multiple of segment length")
        if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if null not in {"gaussian_white", "colored", "unspecified"}:
            raise ValueError("unknown null declaration")
        if weighting not in {"equal_segments", "run_weights"} or route not in {"independent_modern", "historical_modern"}:
            raise ValueError("unresolved weighting or estimator route")
        self._record = {key: common[key] for key in ("basis", "period_band", "taper", "detrend", "periodic", "nperseg")}
        self._record.update(hop=hop, nfft=nfft, alpha=float(alpha), null=null, weighting=weighting,
                            route=route, min_segments=1, scaling="density", csd_definition="conj(X)*Y",
                            phase="arg(Sxy); delayed second signal has negative phase",
                            correction="positive_part_neff_v1", multiplicity="none_pointwise_only")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def effective_averages(run_segment_weights, *, window, hop):
    """Overlap approximation; each array describes a separate contiguous run."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 1 or len(window) < 2 or not np.isfinite(window).all() or not float(window @ window) > 0:
        raise ValueError("finite positive-energy taper required")
    if type(hop) is not int or not 1 <= hop <= len(window):
        raise ValueError("invalid hop")
    weights = [np.asarray(w, dtype=float) for w in run_segment_weights]
    if not weights or any(w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w <= 0) for w in weights):
        raise ValueError("positive finite segment weights required for every run")
    scale = max(float(w.max()) for w in weights)
    weights = [w / scale for w in weights]
    total = sum(float(w.sum()) for w in weights)
    weights = [w / total for w in weights]
    square_sum = sum(float(w @ w) for w in weights)
    overlap = sum(_overlap_square_sum(window, hop, w) for w in weights)
    neff = 1 / (square_sum + 2*overlap)
    return dict(raw_segments=sum(map(len, weights)), n_eff=neff, approximate_dof=2*neff,
                normalized_segment_weights=weights, overlap_pair_square_sum=overlap,
                method_id="gaussian_white_weighted_overlap_v1", approximate=True,
                assumptions="independent stationary Gaussian white-noise channels; independent stationary runs; overlap approximation, not ecological replication")


def coherence_inference(raw, *, n_eff, alpha, null="gaussian_white"):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or np.any(np.isinf(raw)) or np.any((raw < 0) | (raw > 1)):
        raise ValueError("raw coherence must be in [0,1] or NaN")
    if not np.isfinite(alpha) or not 0 < alpha < 1 or isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must lie strictly between zero and one")
    if not np.isfinite(n_eff) or n_eff <= 0 or null not in {"gaussian_white", "colored", "unspecified"}:
        raise ValueError("invalid effective averages or null")
    record = dict(untruncated_corrected=None, corrected=None, raw_threshold=None,
                  corrected_threshold=None, significant=None, decision_scale="raw_strict_greater",
                  alpha=float(alpha), null=null, multiplicity="none_pointwise_only",
                  inference_status="unavailable", reason=None)
    if n_eff <= 1:
        record["reason"] = "effective averages must exceed one; single-periodogram certainty is refused"
    else:
        untruncated = (n_eff*raw - 1)/(n_eff-1)
        record.update(untruncated_corrected=untruncated, corrected=np.maximum(0, untruncated))
        if null != "gaussian_white":
            record["reason"] = "unsupported null; colored or unspecified noise has no admitted threshold"
        else:
            threshold = 1 - alpha**(1/(n_eff-1))
            record.update(raw_threshold=threshold, corrected_threshold=max(0, (n_eff*threshold-1)/(n_eff-1)),
                          significant=[bool(c > threshold) if np.isfinite(c) else None for c in raw],
                          inference_status="approximate_white_noise", reason=None)
    return owned(record)


def pool_coherence_runs(runs, *, spec, run_weights=None):
    """Numerical primitive for compatible, density-normalized run spectra.

    This does not certify observation provenance; modern_coherence supplies that.
    Run weights here are keyed by stable run_id, and split equally over segments.
    """
    if not isinstance(spec, ModernCoherenceSpec):
        raise ValueError("spec must be ModernCoherenceSpec")
    settings = spec.record
    runs = sorted(owned(list(runs)), key=lambda r: r["run_id"])
    ids = [text_field(r["run_id"], "run id") for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("run identities must be unique")
    if settings["weighting"] == "equal_segments":
        if run_weights is not None:
            raise ValueError("run weights require explicit run_weights weighting")
    elif not isinstance(run_weights, dict) or set(run_weights) != set(ids) or any(
            isinstance(w, (bool, np.bool_)) or not np.isfinite(w) or w <= 0 for w in run_weights.values()):
        raise ValueError("positive run weights must name exactly the accepted run IDs")
    result = dict(method_id="coherence.pooled_spectra_v1", run_ids=ids, settings=settings,
                  compute_status="insufficient_support", reason="no accepted runs", spectra=None,
                  effective=None, inference=None, band_mean=None, significant_fraction=None)
    if runs:
        first = runs[0]
        window = get_window(settings["taper"], settings["nperseg"], fftbins=settings["periodic"])
        frequency = np.asarray(first["frequency_hz"], dtype=float)
        expected = np.fft.rfftfreq(settings["nfft"], d=1/first["sampling_hz"])
        if not np.isfinite(first["sampling_hz"]) or first["sampling_hz"] <= 0 or not np.array_equal(frequency, expected):
            raise ValueError("invalid common physical frequency grid")
        segment_weights = []
        for run in runs:
            if any(run[key] != settings[key] for key in ("nperseg", "hop", "nfft", "detrend", "scaling")) or run["sampling_hz"] != first["sampling_hz"] or not np.array_equal(run["frequency_hz"], frequency) or not np.array_equal(run["taper"], window):
                raise ValueError("pooling requires identical cadence, L, taper, detrend, hop, FFT length and grid")
            k = run["raw_segments"]
            if type(k) is not int or k < 1 or run["segment_starts"] != list(range(0, k*settings["hop"], settings["hop"])):
                raise ValueError("invalid run segment support")
            for field in ("sxx", "syy", "sxy_real", "sxy_imag"):
                if np.asarray(run[field]).shape != frequency.shape:
                    raise ValueError("spectra must align with frequency grid")
            if np.any(np.asarray(run["sxx"]) < 0) or np.any(np.asarray(run["syy"]) < 0):
                raise ValueError("auto spectra cannot be negative")
            segment_weights.append(np.ones(k) if run_weights is None else np.full(k, run_weights[run["run_id"]]/k))
        effective = effective_averages(segment_weights, window=window, hop=settings["hop"])
        totals = [float(w.sum()) for w in effective["normalized_segment_weights"]]
        spectra = {key: sum(w*np.asarray(r[key]) for w, r in zip(totals, runs)) for key in ("sxx", "syy", "sxy_real", "sxy_imag")}
        xy = spectra["sxy_real"] + 1j*spectra["sxy_imag"]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw = np.abs(xy / np.sqrt(spectra["sxx"]) / np.sqrt(spectra["syy"]))**2
        valid = np.isfinite(raw) & np.isfinite(spectra["sxx"]) & np.isfinite(spectra["syy"]) & (spectra["sxx"] > 0) & (spectra["syy"] > 0)
        if any(r["constant_input"] for r in runs):
            valid[:] = False
        if np.any(raw[valid] > 1 + 1e-10):
            raise ArithmeticError("pooled coherence outside numerical range")
        raw = np.where(valid, np.clip(raw, 0, 1), np.nan)
        spectra.update(frequency_hz=frequency, raw_coherence=raw,
                       phase_rad=np.where(valid & (np.abs(xy) > 0), np.angle(xy), np.nan))
        positive = np.ones(len(frequency), dtype=bool)
        positive[[0, -1]] = False
        physical_bins = positive & (np.arange(len(frequency)) % (settings["nfft"]//settings["nperseg"]) == 0)
        inference = coherence_inference(np.where(physical_bins, raw, np.nan), n_eff=effective["n_eff"],
                                        alpha=settings["alpha"], null=settings["null"])
        short, long = settings["period_band"]
        low, high = 1e9/long.value, 1e9/short.value
        band = positive & (frequency >= low) & (frequency <= high)
        physical = np.fft.rfftfreq(settings["nperseg"], d=1/first["sampling_hz"])[1:-1]
        physical_count = int(np.count_nonzero((physical >= low) & (physical <= high)))
        result.update(spectra=spectra, effective=effective, inference=inference, run_weights=totals,
                      band_positions=np.flatnonzero(band).tolist(), inference_positions=np.flatnonzero(band & physical_bins).tolist(),
                      physical_band_ordinates=physical_count)
        if physical_count < 3:
            result.update(compute_status="insufficient_band_support", reason="fewer than three physical interior band ordinates")
        elif not np.isfinite(raw[band]).all():
            result.update(compute_status="undefined_spectrum", reason="constant contributing run or zero/nonfinite band auto power")
        else:
            result.update(compute_status="ok", reason=None, band_mean=float(np.mean(raw[band])))
            if inference["significant"] is not None:
                result["significant_fraction"] = float(np.mean([inference["significant"][i] for i in np.flatnonzero(band & physical_bins)]))
    result["pool_id"] = identity(dict(result=result, input_spectra=runs, requested_weights=run_weights))
    return owned(result)


def _modern(first, second, *, variable_a, variable_b, spec, observations, lineage, nodes, target, run_weights, counterpart=False):
    settings = spec.record
    audit = audit_pair(first, second, variable_a=variable_a, variable_b=variable_b, lineage=lineage, nodes=nodes, target=target)
    a, b, axis, ia, ib, eligible = _common_pair(first, second, variable_a, variable_b, settings["basis"], observations)
    runs, used_a, used_b = _pair_runs(a, b, axis, ia, ib, eligible, settings)
    accepted = [r for r in runs if r["compute_status"] == "ok"]
    weights = run_weights
    if settings["weighting"] == "run_weights":
        if not isinstance(weights, dict) or set(weights) != {r["start"] for r in accepted}:
            if not counterpart:
                raise ValueError("run_weights must name exactly the accepted run start timestamps")
            pooled = dict(compute_status="unresolved_weights", reason="observed-only run support changed; explicit replacement weights required", pool_id=None)
        else:
            pooled = pool_coherence_runs(accepted, spec=spec, run_weights={r["run_id"]: weights[r["start"]] for r in accepted})
    else:
        pooled = pool_coherence_runs(accepted, spec=spec, run_weights=weights)
    supports, use, status, reason = _pair_qualification(a, b, used_a, used_b, observations, audit,
                                                      pooled["compute_status"], pooled["reason"])
    if settings["route"] == "historical_modern":
        status, reason = "unavailable_backend", "historical modern helper has not been recovered or verified"
    support = dict(pair=audit["pair"], audit=audit, basis=settings["basis"], variables=[variable_a, variable_b],
                   units=dict(first=a.metadata[variable_a]["unit"], second=b.metadata[variable_b]["unit"]),
                   common_instants=[t.isoformat() for t in axis.index], common_eligible_positions=np.flatnonzero(eligible).tolist(),
                   first=supports["first"], second=supports["second"], runs=runs, pooling=pooled,
                   requested=settings, resolved=settings, requested_run_weights=weights, observations=observations.__dict__,
                   backend="scipy.signal.welch/csd", backend_version=scipy.__version__,
                   reference_ids=[a.reference_id, b.reference_id], historical_parity="unresolved_Q001")
    value = dict(**pooled["spectra"], **pooled["inference"], band_mean=pooled["band_mean"],
                 significant_fraction=pooled["significant_fraction"]) if status == "ok" else None
    units = dict(frequency_hz="Hz", sxx=f"({a.metadata[variable_a]['unit']})^2/Hz",
                 syy=f"({b.metadata[variable_b]['unit']})^2/Hz", sxy_real=f"({a.metadata[variable_a]['unit']})*({b.metadata[variable_b]['unit']})/Hz",
                 sxy_imag=f"({a.metadata[variable_a]['unit']})*({b.metadata[variable_b]['unit']})/Hz", phase_rad="rad",
                 raw_coherence="1", corrected="1", untruncated_corrected="1", raw_threshold="1", corrected_threshold="1",
                 band_mean="1", significant_fraction="1")
    return ResultRecord(value, compute_status=status, reason=reason, metric_status="exploratory_coherence", use_status=use,
                        support=support, units=units, method_id="coherence.pooled_spectra_v1" if settings["route"] == "independent_modern" else "coherence.historical_modern_unavailable",
                        config_id=identity(dict(settings=settings, observations=observations.__dict__, weights=weights,
                                                pool_id=pooled["pool_id"], audit_id=audit["audit_id"])),
                        reference_id=identity([a.reference_id, b.reference_id]) if a.reference_id or b.reference_id else None,
                        representation=settings["basis"])


def modern_coherence(first, second, *, variable_a, variable_b, spec, observations=None, lineage=None, nodes=None, target=None, run_weights=None):
    if not isinstance(spec, ModernCoherenceSpec):
        raise ValueError("spec must be ModernCoherenceSpec")
    observations = _observations(observations)
    kwargs = dict(variable_a=variable_a, variable_b=variable_b, spec=spec, lineage=lineage, nodes=nodes, target=target, run_weights=run_weights)
    result = _modern(first, second, observations=observations, **kwargs)
    record = result.summary()
    if record["support"]["audit"]["observed_only_required"]:
        record["support"]["observed_only_counterpart"] = _modern(first, second,
            observations=ObservationPolicy("observed_only", min_points=observations.min_points), counterpart=True, **kwargs).summary()
        return ResultRecord(**record)
    return result

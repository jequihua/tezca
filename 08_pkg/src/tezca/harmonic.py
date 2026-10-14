"""Independent Fisher-g and Monte Carlo Siegel diagnostics under an assumed null."""

from math import comb, floor
from numbers import Real

import numpy as np
from scipy.optimize import brentq

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations
from .reference import FrozenReference
from .support import select_support


FISHER = "harmonic.fisher_g_exact_v1"
SIEGEL = "harmonic.siegel_dirichlet_mc_v1"
NULL = "iid_gaussian_white_noise"
METRIC = "assumed_white_noise_diagnostic"


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)


def _m(m):
    if type(m) is not int or m < 2:
        raise ValueError("m must be an integer >=2")
    return m


def fisher_tail(g, m):
    """Exact rational inclusion-exclusion for the supplied binary float, then float."""
    _m(m)
    if isinstance(g, (bool, np.bool_)) or not isinstance(g, Real) or not np.isfinite(g) or not 0 <= g <= 1:
        raise ValueError("g must be finite in [0,1]")
    g = float(g)
    if g <= 1/m:
        return 1.
    if g == 1:
        return 0.
    numerator, denominator = g.as_integer_ratio()
    # Integer powers/summation avoid cancellation entirely; no probability clipping.
    last = min(m, (denominator-1)//numerator)
    total = sum((-1)**(k-1)*comb(m, k)*(denominator-k*numerator)**(m-1)
                for k in range(1, last+1))
    divisor = denominator**(m-1)
    if not 0 <= total <= divisor:
        raise ArithmeticError("Fisher tail left its exact probability domain")
    return total/divisor


def fisher_critical(m, alpha=.05):
    _m(m)
    alpha = _probability(alpha, "alpha")
    return float(brentq(lambda g: fisher_tail(g, m)-alpha, 1/m, 1, xtol=1e-14, rtol=1e-14))


def siegel_statistic(proportions, *, alpha=.05, lambda_=.6):
    q = np.asarray(proportions)
    if (q.ndim not in {1, 2} or q.dtype.kind not in "iuf" or q.shape[-1] < 2
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=-1), 1, rtol=0, atol=1e-12)):
        raise ValueError("q must contain normalized finite nonnegative Fourier proportions")
    alpha, lambda_ = _probability(alpha, "alpha"), _probability(lambda_, "lambda")
    return np.maximum(q-lambda_*fisher_critical(q.shape[-1], alpha), 0).sum(axis=-1)


class HarmonicSpec:
    def __init__(self, *, alpha=.05, lambda_=.6, center=True, backend="independent",
                 estimator="rectangular_periodogram", taper="boxcar", nfft=None,
                 null=NULL):
        alpha, lambda_ = _probability(alpha, "alpha"), _probability(lambda_, "lambda")
        if type(center) is not bool or backend not in {"independent", "sensor_pinned"}:
            raise ValueError("center must be boolean; backend must be independent or sensor_pinned")
        if estimator not in {"rectangular_periodogram", "welch"} or taper not in {"boxcar", "hann"}:
            raise ValueError("unknown estimator or taper")
        if nfft is not None and (type(nfft) is not int or nfft < 2):
            raise ValueError("nfft must be an integer >=2 or None for the input length")
        if null not in {NULL, "colored_noise"}:
            raise ValueError("unknown null declaration")
        self._record = dict(alpha=alpha, lambda_=lambda_, center=center, backend=backend,
            estimator=estimator, taper=taper, nfft=nfft, null=null, null_status="assumed_not_tested",
            bins="ordinary_positive_excluding_DC_and_even_Nyquist", averaging="none",
            normalization="squared_DFT_modulus_over_N_then_unit_sum", input_route="explicit_raw_series")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def _degenerate_reason(x):
    if np.any(np.ptp(x, axis=-1) == 0):
        return "constant input has no admitted harmonic power"
    if x.shape[-1] % 2 == 0:
        even_constant = np.all(x[..., ::2] == x[..., :1], axis=-1)
        odd_constant = np.all(x[..., 1::2] == x[..., 1:2], axis=-1)
        if np.any(even_constant & odd_constant):
            return "power is confined to excluded DC/Nyquist ordinates"
    return None


def fourier_proportions(samples, *, center=True):
    """Array primitive: complete finite rows, unit-sum ordinary Fourier powers."""
    x = np.asarray(samples)
    if (type(center) is not bool or x.ndim not in {1, 2} or x.dtype.kind not in "iuf"
            or not np.isfinite(x).all() or x.shape[-1] < 5):
        raise ValueError("complete real finite rows with at least five samples and boolean center required")
    x = x.astype(float)
    reason = _degenerate_reason(x)
    if reason is not None:
        raise ValueError(reason)
    removed_mean = x.mean(axis=-1) if center else np.zeros(x.shape[:-1])
    analyzed = x-removed_mean[..., None]
    n = x.shape[-1]
    bins = np.arange(1, (n-1)//2+1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        power = np.abs(np.fft.rfft(analyzed, axis=-1)[..., bins])**2/n
        total = power.sum(axis=-1)
        if np.any(total <= 0) or not np.isfinite(total).all():
            raise ValueError("non-DC power is zero or unrepresentable")
        q = power/total[..., None]
    return dict(proportions=q, powers=power, bins=bins, n=n, m=len(bins), removed_mean=removed_mean)


class SiegelCalibration(FrozenReference):
    """Construct only from simulation parameters; importing arbitrary records is unsupported."""
    def __init__(self, *, m, alpha=.05, lambda_=.6, replications=19999, seed=2026101101):
        _m(m)
        alpha, lambda_ = _probability(alpha, "alpha"), _probability(lambda_, "lambda")
        if type(replications) is not int or replications < 1 or type(seed) is not int or seed < 0:
            raise ValueError("positive integer replications and nonnegative integer seed required")
        draws = np.random.Generator(np.random.PCG64(seed)).exponential(size=(replications, m))
        q = draws/draws.sum(axis=1, keepdims=True)
        sorted_T = np.sort(siegel_statistic(q, alpha=alpha, lambda_=lambda_))
        rank_count = floor(alpha*(replications+1))
        # Match the actual float p<=alpha rule even at representational boundaries.
        while rank_count/(replications+1) > alpha:
            rank_count -= 1
        while (rank_count+1)/(replications+1) <= alpha:
            rank_count += 1
        critical_index = replications-rank_count if rank_count else None
        super().__init__(dict(method_id=SIEGEL, null=NULL, null_status="assumed_not_tested",
            m=m, alpha=alpha, lambda_=lambda_, replications=replications, seed=seed,
            rng="PCG64", numpy_version=np.__version__, simulation="normalized_Exponential_1",
            sorted_T=sorted_T, fisher_critical=fisher_critical(m, alpha),
            critical_T=float(sorted_T[critical_index]) if critical_index is not None else None,
            critical_index_zero_based=critical_index, critical_order_one_based=critical_index+1 if critical_index is not None else None,
            attainable_rejection_ranks=rank_count, minimum_p=1/(replications+1),
            p_convention="(1+count(T_sim>=T))/(B+1)", ties="greater_or_equal",
            critical_rule="T>critical_T; equality cannot reject", fit_scope="null_simulation",
            compute_status="ok" if rank_count else "insufficient_calibration_resolution"))


def calibrate_siegel(*, m, alpha=.05, lambda_=.6, replications=19999, seed=2026101101):
    return SiegelCalibration(m=m, alpha=alpha, lambda_=lambda_, replications=replications, seed=seed)


def siegel_tail(statistic, calibration):
    if not isinstance(calibration, SiegelCalibration):
        raise ValueError("Siegel tail requires SiegelCalibration")
    t = np.asarray(statistic)
    if t.dtype.kind not in "iuf" or not np.isfinite(t).all() or (t < 0).any() or (t > 1).any():
        raise ValueError("T must be finite in [0,1]")
    ref = calibration.describe()
    count = ref["replications"]-np.searchsorted(ref["sorted_T"], t, side="left")
    return (1+count)/(ref["replications"]+1)


def harmonic_diagnostics(data, *, variable, spec, calibration=None, observations=None):
    """Return separate owned Fisher and Siegel results; no spectrum fallback."""
    if not isinstance(spec, HarmonicSpec):
        raise ValueError("harmonic diagnostics require HarmonicSpec")
    if calibration is not None and not isinstance(calibration, SiegelCalibration):
        raise ValueError("calibration must be SiegelCalibration")
    settings = spec.record
    cal = calibration.describe() if calibration is not None else None
    if cal is not None and (cal["method_id"] != SIEGEL or cal["alpha"] != settings["alpha"] or cal["lambda_"] != settings["lambda_"]
                            or cal["null"] != settings["null"] or settings["backend"] != "independent"):
        raise ValueError("calibration alpha/lambda/null/backend must match requested settings")
    if isinstance(data, ResultRecord):
        previous = data.summary()
        if variable != previous["support"].get("variable"):
            raise ValueError("spectrum variable does not match")
        support = dict(previous["support"], supplied_result_id=identity(previous),
                       requested=settings, resolved=None, input_route="supplied_spectrum_refused")
        basis, use = previous["representation"], previous["use_status"]
        use = "unknown_provenance" if use == "unknown_provenance" else "sensitivity_only"
        reference_id = previous["reference_id"]
        status, reason = "unsupported_estimator", "spectrum inputs are not admitted; explicitly call with raw series"
        signal = None
    else:
        signal = _ColumnSignal(data, variable)
        observations = _observations(observations)
        selection = select_support(signal, observations=observations, estimator="point",
            policy="all_runs" if signal.time.mode == "regular" else "irregular")
        positions = selection.positions
        support = signal.support_record(selection)
        n = len(signal.values)
        resolved = dict(settings, nfft=n if settings["nfft"] is None else settings["nfft"])
        support.update(requested=settings, resolved=resolved, measurement_unit=signal.metadata[variable]["unit"],
                       frequency_unit="Hz", power_unit=signal.metadata[variable]["unit"]+"^2",
                       observations=observations.__dict__)
        basis, use, reference_id = signal.basis, signal.use(positions, observations, METRIC), signal.reference_id
        status, reason = "ok", None
        if settings["backend"] == "sensor_pinned":
            status, reason = "missing_backend", "pinned sensor helper and exact parity remain unavailable"
        elif settings["null"] != NULL:
            status, reason = "unsupported_null", "only the assumed iid Gaussian white-noise null is admitted"
        elif settings["estimator"] != "rectangular_periodogram" or settings["taper"] != "boxcar" or resolved["nfft"] != n:
            status, reason = "unsupported_estimator", "classical thresholds require unaveraged rectangular ordinary Fourier bins"
        elif signal.basis != "raw":
            status, reason = "unsupported_basis", "fitted residuals/transformed bases do not satisfy this raw null contract"
        elif len(positions) < max(5, observations.min_points):
            status, reason = "insufficient_support", "at least two eligible positive Fourier bins are required"
        elif signal.time.mode != "regular":
            status, reason = "unsupported_time_geometry", "regular complete input is required"
        elif len(positions) != n or len(selection.record["runs"]) != 1:
            status, reason = "unsupported_support", "missing/excluded cells or timestamp holes cannot be compressed"
    config_id = identity(dict(spec=spec.identity, observations=observations.__dict__ if observations else None))
    support.update(methods=dict(fisher=FISHER, siegel=SIEGEL), backend=dict(requested=settings["backend"],
                   numpy=np.__version__, implementation="independent" if settings["backend"] == "independent" else None),
                   null=settings["null"], null_status="assumed_not_tested")
    values = dict(fisher=None, siegel=None)
    statuses = {key: (status, reason) for key in values}
    if status == "ok":
        x = signal.values.iloc[:, 0].to_numpy(dtype=float)
        degenerate = _degenerate_reason(x)
        if degenerate is not None:
            statuses = {key: ("degenerate_input", degenerate) for key in values}
        else:
            geometry = fourier_proportions(x, center=settings["center"])
            m = geometry["m"]
            if cal is not None and cal["m"] != m:
                raise ValueError("calibration m must match the included Fourier bins")
            support.update(geometry, frequency_hz=geometry["bins"]*(1e9/signal.time.cadence.value)/len(x),
                           estimator="rectangular_periodogram", nfft=len(x), dc_excluded=True,
                           nyquist_excluded=len(x) % 2 == 0)
            g = float(geometry["proportions"].max())
            p = fisher_tail(g, m)
            values["fisher"] = dict(g=g, p_value=p, critical=fisher_critical(m, settings["alpha"]),
                critical_scale="g", rejected=p <= settings["alpha"], alpha=settings["alpha"])
            if cal is None:
                statuses["siegel"] = ("not_computed", "explicit Siegel null calibration is required")
            elif cal["compute_status"] != "ok":
                statuses["siegel"] = (cal["compute_status"], "minimum Monte Carlo p exceeds alpha")
            else:
                T = float(siegel_statistic(geometry["proportions"], alpha=settings["alpha"], lambda_=settings["lambda_"]))
                p = float(siegel_tail(T, calibration))
                values["siegel"] = dict(T=T, p_value=p, critical=cal["critical_T"], critical_scale="T",
                    rejected=p <= settings["alpha"], alpha=settings["alpha"], lambda_=settings["lambda_"])
    results = {}
    for key, method in (("fisher", FISHER), ("siegel", SIEGEL)):
        details = owned(support)
        if key == "siegel" and cal is not None:
            details["calibration"] = {k: owned(v) for k, v in cal.items() if k != "sorted_T"}
            details["calibration_id"] = calibration.identity
        compute, reason = statuses[key]
        results[key] = ResultRecord(values[key], compute_status=compute, reason=reason,
            metric_status=METRIC, use_status=use, units="1", method_id=method, config_id=config_id,
            reference_id=calibration.identity if key == "siegel" and cal is not None else reference_id,
            representation=basis, support=details)
    return results


def compatibility_fields(results):
    """Six historical column names plus mandatory semantic/status sidecar."""
    fields, sidecar = {}, {}
    for key, symbol in (("fisher", "g"), ("siegel", "T")):
        result = results[key]
        record, value = result.summary(), result.value
        fields[f"{key}_harmonic_rejected"] = value["rejected"] if value is not None else None
        fields[f"{key}_{symbol}"] = value[symbol] if value is not None else None
        fields[f"{key}_{symbol}_critical"] = value["critical"] if value is not None else None
        sidecar[key] = dict(record, formula="max(q)" if key == "fisher" else "sum(max(q-lambda*a_m(alpha),0))",
                           compatibility="field_names_only_not_pinned_numerical_parity")
        sidecar[key].pop("value")
    return dict(fields=fields, sidecar=sidecar)

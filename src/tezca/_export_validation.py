"""Passive checks for the finite result adapters, independent of bundle hashes.

Training support belongs to frozen references. Evaluation support must bind to
the declared input snapshots; never confuse the two by recursively matching IDs.
"""

import numpy as np
import pandas as pd
from scipy.signal import get_window

from .data import ResultRecord
from ._result_codec import encode, dumps
from ._owned import identity
from .spectral import _overlap_square_sum


_NATIVE = set("value compute_status metric_status use_status support units method_id config_id reference_id reason representation".split())
_EXTRA = dict(metric=(), peaks=(), spectrum=(), pca=(), coherence=(), density=(),
              matrix=("rows",), fi1=("rows",), network=("rows", "edges"), graph=("rows",),
              regimes=("rows", "timeline", "reference"), sensitivity=("routes", "contrasts"))
_COMPOSITE = {
    "matrix": "identity source settings method_id backend reference_id variables metadata support compute_status metric_status use_status reason real_F01_admission",
    "fi1": "summary window_settings support method_id config_id reference_id fit_scope compute_status reason metric_status use_status basis units measurement_unit identity",
    "network": "method_id input_id source features units basis clock requested resolved observations config_id reference_ids start end backend backend_version window_count compute_status reason metric_status use_status",
    "graph": "rows configurations median minimum maximum interpretation grid_id",
    "regimes": "method_id compute_status reason metric_status use_status source schema reference_id training support cluster_count cluster_sizes denominator assigned noise excluded failed display_mapping membership soft_meaning warning fractions identity",
    "sensitivity": "method design interval variable_order frequency_hz geometry references completion_provenance natural_gaps qualification identity",
}
_METHODS = {
    "peaks": {"spectral.peaks_v1"},
    "spectrum": {"spectral.welch_mean_v1", "spectral.ls_projection_density_wosa_v1"},
    "pca": {"pca.frozen_projection_v1"},
    "density": {"density_fisher.histogram_interior_v1", "density_fisher.gaussian_kde_location_v1"},
    "coherence": {"coherence.pooled_spectra_v1", "coherence.historical_modern_unavailable",
                  "coherence.report_equal_run_v1", "coherence.report_exact_unresolved"},
}
_COHERENCE_SPECTRA = "frequency_hz sxx syy sxy_real sxy_imag raw_coherence".split()
_COHERENCE_INFERENCE = "untruncated_corrected corrected raw_threshold corrected_threshold significant decision_scale alpha null multiplicity inference_status reason".split()
_COHERENCE_REPORT = "frequency_hz raw_coherence band_mean".split()
_USE = ("unknown_provenance","insufficient_support","sensitivity_only","unverified_gap_geometry","primary_eligible")
_CLOCK = "mode cadence phase storage_timezone civil_timezone clock_resolution coverage_seconds max_coverage_seconds duration_convention".split()
_SOURCE = {"identity","source","metadata","clock","signal_identities","mask_hash","source_hash","details_hash"}
# Every native ResultRecord producer. Reserved prefixes cannot fall back to
# caller-defined typed records; each signal method has a concrete validator.
_WELCH, _IRREGULAR = "spectral.welch_mean_v1", "spectral.ls_projection_density_wosa_v1"
_SUMMARIES = {"spectral.log_models_v1":_WELCH, "spectral.peaks_v1":_WELCH, "spectral.band_power_v1":_WELCH,
              "spectral.irregular_log_models_v1":_IRREGULAR}
_FISHER, _SIEGEL = "harmonic.fisher_g_exact_v1", "harmonic.siegel_dirichlet_mc_v1"
_INDIVIDUAL = {"individual."+m+"_v1" for m in "mean median sample_sd iqr minimum maximum population_variance elapsed_day_ols exact_lagged_pearson".split()}
_DENSITY = {"density_fisher.histogram_interior_v1", "density_fisher.gaussian_kde_location_v1"}
_SIGNAL_METHODS = _INDIVIDUAL | _DENSITY | set(_SUMMARIES) | {_WELCH, _IRREGULAR, _FISHER, _SIEGEL, "fi1.ordered_mass_roughness_v1"}
_REFERENCE_SUMMARY = "fixed_edges_and_population_scaling_v1"  # foundation apply_reference; no input identity
_NATIVE_METHODS = _SIGNAL_METHODS | _METHODS["pca"] | _METHODS["coherence"] | {"graph.state_mass_v1", "graph.state_density_v1",
                                                                               _REFERENCE_SUMMARY}
_RESERVED = ("individual.", "spectral.", "harmonic.", "fi1.", "density_fisher.", "pca.", "graph.", "network.",
             "coherence.", "regimes.", "collective.", "daily.")
_MODEL_SETTINGS = set("interval_hz candidates_hz min_arm log_base tie_tolerance exact_rss_tolerance basis interval_closure positive_floor continuous tie_policy single_k broken_k parameter_count".split())
_SUMMARY_SETTINGS = {"spectral.log_models_v1":_MODEL_SETTINGS, "spectral.irregular_log_models_v1":_MODEL_SETTINGS|{"adapter"},
    "spectral.peaks_v1":set("period_seconds min_cycles max_peaks prominence basis tie_policy".split()),
    "spectral.band_power_v1":set("requested_bounds coordinates frequency_bounds_hz min_cycles min_ordinates basis quadrature denominator".split())}
# Fixed declarations written by ModelSpec/PeakSpec/BandSpec and the irregular adapter.
_SUMMARY_CONSTANTS = {
    "spectral.log_models_v1": dict(interval_closure="both", positive_floor=None, continuous=True, tie_policy="lower_frequency",
                                   single_k=3, broken_k=5, parameter_count="including residual variance and estimated breakpoint"),
    "spectral.peaks_v1": dict(tie_policy="descending_prominence_then_lower_frequency"),
    "spectral.band_power_v1": dict(quadrature="piecewise_linear_trapezoid_with_interpolated_boundaries",
                                   denominator="entire_supported_positive_frequency_range_DC_excluded")}
_SUMMARY_CONSTANTS["spectral.irregular_log_models_v1"] = _SUMMARY_CONSTANTS["spectral.log_models_v1"]
_ADAPTER_SELECTION = "ascending_actual_frequencies_separated_by_1/T"
# Producer-branch support added to the inherited spectrum support.
_SUMMARY_EXTRA = {"fit_selection", "fit_grid", "adapter_selection", "inherited_range_hz", "excluded_frequency_positions",
                  "prominence_units", "physical_ordinates", "cycles_at_low_frequency", "denominator_bounds_hz"}
_WELCH_SETTINGS = set("nperseg recipe overlap nfft taper periodic detrend scaling onesided average support min_segments".split())
# Regular Welch run support and successful-only diagnostics (power_spectrum).
_WELCH_RUNS = "selected_positions candidate_runs selected_runs used_runs rejected_runs segments contributing_positions duration_seconds longest_run_seconds effective_averages".split()
_WELCH_SUCCESS = {"constant_input", "zero_spectrum", "cycles_longest_run"}
_WELCH_AVERAGES = dict(method="gaussian_white_noise_window_overlap_v1", approximate=True,
                       assumptions="equal-weight mean periodograms; Gaussian white noise")
# Sums of nonnegative taper products accumulate at most (terms x 1.1e-16)
# relative rounding; 1e-9 admits ~10^6 terms across library versions while
# refusing any material change. Single products keep the existing 1e-12.
_WELCH_RTOL = 1e-9
_IRREGULAR_SETTINGS = set("interval segment hop anchor frequency_hz f_ref_hz requested_band_hz taper detrend min_points min_span min_cycles rank_tolerance weights closure".split())
_HARMONIC_SETTINGS = set("alpha lambda_ center backend estimator taper nfft null null_status bins averaging normalization input_route".split())


def _require(ok, message):
    if not ok:
        raise ValueError("invalid result schema: " + message)


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _clock(clock, length=None):
    """The native TimeAxis record; coverage length and phase presence are checked when positions are known."""
    _require(isinstance(clock, dict) and set(clock) == set(_CLOCK), "clock schema fields")
    _require(clock["mode"] in ("regular","irregular") and clock["clock_resolution"] == "resolved" and
             all(_text(clock[k]) for k in ("storage_timezone","civil_timezone")), "clock declarations")
    if clock["mode"] == "regular":
        _require(isinstance(clock["cadence"],pd.Timedelta) and clock["cadence"] > pd.Timedelta(0) and
                 clock["coverage_seconds"] is None and clock["max_coverage_seconds"] is None and
                 clock["duration_convention"] == "cells_left_closed", "regular clock declarations")
        # TimeAxis takes the first sample as phase, so only an empty axis can lack one.
        _require(clock["phase"] is not None or not length, "regular clock phase")
        if clock["phase"] is not None:
            _ns(_time(clock["phase"]))
    else:
        _require(clock["cadence"] is None and clock["phase"] is None, "irregular clock declarations")
        coverage = clock["coverage_seconds"]
        if coverage is None:
            _require(clock["max_coverage_seconds"] is None and clock["duration_convention"] == "unknown", "irregular coverage schema")
        else:
            _array(coverage,finite=True)
            _require((length is None or len(coverage) == length) and _number(clock["max_coverage_seconds"]) and
                     clock["max_coverage_seconds"] > 0 and np.all(coverage >= 0) and
                     np.all(coverage <= clock["max_coverage_seconds"]) and
                     clock["duration_convention"] == "explicit_coverage", "irregular coverage schema")


def validate_source(source):
    """Snapshot written by exports._source: compact declarations and fingerprints only."""
    residual = isinstance(source, dict) and "representation" in source
    _require(isinstance(source, dict) and set(source) == _SOURCE | ({"base_identity","representation"} if residual else set()),
             "source snapshot fields")
    _require(_text(source["identity"]) and isinstance(source["source"],dict) and isinstance(source["signal_identities"],dict) and
             all(_digest(source[k]) for k in ("mask_hash","source_hash")) and
             (source["details_hash"] is None or _digest(source["details_hash"])), "source snapshot identity/fingerprint types")
    metadata = source["metadata"]
    _require(isinstance(metadata,dict) and metadata and all(_text(v) and isinstance(m,dict) and _text(m.get("representation")) and
             type(m.get("unit")) is str and m["unit"] and not any(c.isspace() or ord(c) < 32 for c in m["unit"])
             for v,m in metadata.items()),
             "source variable metadata")
    _clock(source["clock"])
    if residual:
        representation = source["representation"]
        _require(_text(source["base_identity"]) and isinstance(representation,dict) and
                 set(representation) == {"basis","source_id","reference_id","fit","evaluation"} and
                 isinstance(representation["fit"],dict) and isinstance(representation["evaluation"],dict) and
                 set(representation["evaluation"]) == set(metadata), "residual source representation schema")


def validate_backend(backend, version):
    _require(isinstance(backend,dict) and set(backend) == {"software","python","tezca","declared"} and
             isinstance(backend["software"],dict) and set(backend["software"]) == {"numpy","pandas","scipy"} and
             all(_text(v) for v in backend["software"].values()) and _text(backend["python"]) and
             backend["tezca"] == version and isinstance(backend["declared"],dict), "software/backend identity schema")


def _fields(record, fields):
    _require(isinstance(record, dict) and set(fields) <= record.keys(), "required record fields")


def _text(value):
    return type(value) is str and bool(value.strip())


def _status(record):
    _fields(record, ("compute_status", "metric_status", "use_status", "reason"))
    _require(all(_text(record[k]) for k in ("compute_status", "metric_status", "use_status")), "status must be text")
    _require(record["reason"] is None or _text(record["reason"]), "reason must be text or null")
    _require(record["compute_status"] == "ok" or _text(record["reason"]), "unavailable status requires reason")


def _native(record, references):
    _fields(record, _NATIVE)
    _status(record)
    # The existing passive record constructor owns the native status/value/unit
    # contract. This constructs no estimator, reference, model or analysis.
    try:
        ResultRecord(**{k: record[k] for k in _NATIVE})
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid result schema: native record: " + str(exc)) from exc
    support, method = record["support"], record["method_id"]
    # Native signal methods require their provenance even if every identifying
    # field was removed. The explicit refused-spectrum harmonic route may copy
    # a caller-created record with no signal; it must remain unavailable.
    refused_spectrum = method.startswith("harmonic.") and support.get("input_route") == "supplied_spectrum_refused"
    if refused_spectrum:
        _fields(support, ("requested","resolved","supplied_result_id"))
        _require(record["compute_status"] == "unsupported_estimator" and record["value"] is None and
                 support.get("resolved") is None and _text(support.get("supplied_result_id")) and
                 isinstance(support.get("requested"),dict), "harmonic refused spectrum schema")
    if method.startswith(("individual.", "spectral.", "harmonic.", "fi1.", "density_fisher.")) and (
            not refused_spectrum or "input_id" in support):
        _fields(support, ("input_id", "variable", "basis", "source", "clock"))
    if "basis" in support and method.startswith(("individual.", "spectral.", "harmonic.", "fi1.", "density_fisher.", "pca.", "network.", "coherence.")):
        _require(record["representation"] == support["basis"], "native representation/support binding mismatch")
    elif "candidate" in support and "evaluation" in support:
        _require(all(record["representation"] == support[k]["basis"] for k in ("candidate","evaluation")),
                 "native representation/support binding mismatch")
    if method.startswith(("fi1.", "density_fisher.")):
        expected = identity(support["fit"])
    elif "calibration_id" in support:
        expected = support["calibration_id"]
    elif "reference_ids" in support:
        refs = support["reference_ids"]
        values = refs.values() if isinstance(refs, dict) else refs
        expected = identity(refs) if any(values) else None
    elif "reference_id" in support or method.startswith(("individual.", "spectral.", "harmonic.", "pca.", "graph.")):
        expected = support.get("reference_id")
    else:
        expected = record["reference_id"]  # User-declared generic ResultRecord.
    _require(record["reference_id"] == expected, "native reference binding mismatch")
    if record["value"] is not None:
        _native_value(method, record["value"])
    if method.startswith(("fi1.", "density_fisher.")):
        positions = _evaluation_support(support)
        if record["value"] is not None:
            key = "eligible_count" if method.startswith("fi1.") else "evaluation_count"
            _require(record["value"][key] == len(positions), "native evaluation count binding")
            if "counts" in record["value"]:
                _require(record["value"]["counts"].sum() == len(positions), "native histogram count binding")
    _reference_fields(record, references)


def _ns(value):
    """Exact integer nanoseconds of an aware instant, as TimeAxis stores it."""
    try:
        return int(value.as_unit("ns", round_ok=False).value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid result schema: time not representable at nanosecond resolution") from exc


def _on_phase(times, clock, message):
    """TimeAxis grid rule (t_ns - phase_ns) % cadence_ns == 0 in Python integers.

    Exact for equivalent offsets, nanoseconds and phases whose distance exceeds
    int64; one subtraction per supplied instant, never a grid between them.
    """
    if clock["mode"] == "regular" and times:
        _require(clock["phase"] is not None, message)
        phase, cadence = _ns(clock["phase"]), int(clock["cadence"].value)
        _require(all((_ns(t)-phase) % cadence == 0 for t in times), message)


def _run_intervals(runs, size, clock, instants, name):
    """Saved maximal regular runs (support._runs/_interval) within `size` positions.

    Integer bounds and order are checked before each run's instants are
    traversed, so total work is bounded by `size`. Runs touching at a position
    need a clock gap; every skipped row occupies at least one cadence. Each
    start lies on the declared phase grid; the exact end and instant offsets
    from it then keep every endpoint and bound instant on that grid.
    """
    cadence = clock["cadence"]
    _require(isinstance(runs,list), name+" run geometry")
    stop = end = None
    for run in runs:
        _fields(run,"start_position stop_position start end closure count duration_seconds timestamp_span_seconds".split())
        _counts(run,("start_position","stop_position","count"))
        a,b = run["start_position"],run["stop_position"]
        _require(type(a) is int and type(b) is int and type(run["count"]) is int and 0 <= a < b <= size and
                 (stop is None or a >= stop) and
                 run["count"] == b-a and run["closure"] == "left" and
                 _time(run["end"],strings=True)-_time(run["start"],strings=True) == (b-a)*cadence and
                 run["duration_seconds"] == (b-a)*cadence.value/1e9 and
                 run["timestamp_span_seconds"] == (b-a-1)*cadence.value/1e9, name+" run geometry")
        start = _time(run["start"],strings=True)
        _require(end is None or start-end >= max(a-stop,1)*cadence, name+" run geometry")
        _on_phase([start], clock, name+" run clock phase")
        _require(all(instants[p] == start+(p-a)*cadence for p in range(a,b) if p in instants), name+" run instant binding")
        stop, end = b, _time(run["end"],strings=True)


def _signal_selection(s, observations, *, estimator="point"):
    """Check the shared saved signal selection without recreating eligibility."""
    _fields(s, "policy estimator positions selected_instants compute_status reason runs windows window_settings full_duration_seconds supplied_duration_seconds selected_duration_seconds excluded_duration_seconds missing_grid_cells duration_convention timestamp_span_seconds full_segment selected excluded_positions support_id".split())
    clock = s["clock"]
    _observation_policy(observations)
    positions = _positions(s["positions"])
    selected_set = set(positions)
    full = _positions(s["full_segment"]["positions"])
    _clock(clock, len(full))
    _require(full == list(range(len(full))) and selected_set <= set(full) and
             _positions(s["excluded_positions"]) == [p for p in full if p not in selected_set] and
             s["policy"] in ("all_runs","longest_run","irregular","complete_case") and
             (s["policy"] not in ("all_runs","longest_run") or clock["mode"] == "regular") and
             s["estimator"] == estimator and s["windows"] == [] and s["window_settings"] is None,
             "signal selection schema")
    times = [_time(t,strings=True) for t in s["selected_instants"]]
    _require(isinstance(s["selected_instants"],list) and len(times) == len(positions) and
             all(a < b for a,b in zip(times,times[1:])), "signal selected instants")
    instants = dict(zip(positions,times))
    for key, selected in (("selected",positions),("full_segment",full)):
        burden = s[key]
        _fields(burden,("positions","denominator","variables","support_id","weighting","duration_convention"))
        _require(_positions(burden["positions"]) == selected and type(burden["denominator"]) is int and
                 burden["denominator"] >= len(selected) and list(burden["variables"]) == [s["variable"]] and
                 burden["weighting"] == "equal_cells" and burden["duration_convention"] == clock["duration_convention"] and
                 burden["support_id"] == identity(dict(series=s["input_id"],positions=np.asarray(selected,dtype=int),
                                                       denominator=burden["denominator"])), "signal burden binding")
        detail = burden["variables"][s["variable"]]
        _require(detail["denominator"] == burden["denominator"], "signal burden denominator")
        _training_burden(detail,clock,len(selected))
    _require(s["support_id"] == identity(dict(series=s["input_id"],policy=s["policy"],positions=np.asarray(positions,dtype=int),
                 windows=[],window_settings=None,observations=observations)), "signal support identity")
    selected_ok = len(positions) >= observations["min_points"]
    _require(s["compute_status"] == ("ok" if selected_ok else "insufficient_support") and
             (s["reason"] is None if selected_ok else _text(s["reason"])), "signal selection availability")
    _require(isinstance(s["runs"],list) and clock["mode"] in ("regular","irregular") and
             s["duration_convention"] == clock["duration_convention"], "signal clock support")
    denominator = s["full_segment"]["denominator"]
    if clock["mode"] == "regular":
        cadence = clock["cadence"]
        _counts(s,("missing_grid_cells",))
        _require(isinstance(cadence,pd.Timedelta) and cadence > pd.Timedelta(0) and
                 s["missing_grid_cells"] == denominator-len(full), "signal regular duration")
        duration = [denominator*cadence.value/1e9,len(full)*cadence.value/1e9,len(positions)*cadence.value/1e9]
        # Every regular selected position lies in a saved run (checked below), so
        # phase-bound run starts and exact instant offsets bind every instant.
        _run_intervals(s["runs"], len(full), clock, instants, "signal")
        run_positions = [p for run in s["runs"] for p in range(run["start_position"],run["stop_position"])]
        _require(run_positions == sorted(set(run_positions)) and selected_set <= set(run_positions) and
                 (s["policy"] == "longest_run" or run_positions == positions), "signal run selection")
    else:
        _require(s["runs"] == [] and s["missing_grid_cells"] is None and denominator == len(full), "signal irregular selection")
        coverage = clock["coverage_seconds"]
        duration = [None]*3 if coverage is None else [float(np.sum(coverage)),float(np.sum(coverage)),float(np.sum(coverage[positions]))]
    for key, expected in zip(("full_duration_seconds","supplied_duration_seconds","selected_duration_seconds"),duration):
        _require(s[key] is None if expected is None else _number(s[key]) and s[key] == expected, "signal duration binding")
    _require(s["excluded_duration_seconds"] == (None if duration[0] is None else duration[0]-duration[2]) and
             _number(s["timestamp_span_seconds"]) and s["timestamp_span_seconds"] >= 0, "signal duration schema")
    return positions, times


def _individual_support(record, unit):
    """Validate saved selection and qualification, without evaluating a metric."""
    s, method = record["support"], record["method_id"]
    _fields(s, ("metric","requested","resolved","observations"))
    observations, settings, clock = s["observations"], s["requested"], s["clock"]
    positions, times = _signal_selection(s,observations)
    instants = dict(zip(positions,times))
    _fields(settings,("support",))
    _require(s["metric"] == method and encode(settings) == encode(s["resolved"]) and
             record["config_id"] == identity(dict(settings=settings,observations=observations)) and
             record["metric_status"] == "primary_with_support", "individual method/settings binding")
    _require(record["use_status"] == _saved_signal_use(s,observations), "individual provenance qualification")
    units = ("1" if method == "individual.exact_lagged_pearson_v1" else unit+"/day" if
             method == "individual.elapsed_day_ols_v1" else unit+"^2" if
             method == "individual.population_variance_v1" else unit)
    _require(record["units"] == units, "individual output unit binding")
    minimum = observations["min_points"]
    if method == "individual.exact_lagged_pearson_v1":
        _fields(s,("lag","pairs","pair_count"))
        _fields(settings,("lag",))
        lag, pairs = s["lag"],s["pairs"]
        _fields(lag,("requested","samples","realized_seconds"))
        _require(encode(lag["requested"]) == encode(settings["lag"]) and isinstance(pairs,list) and
                 type(s["pair_count"]) is int and s["pair_count"] == len(pairs) and
                 all(isinstance(p,tuple) and len(p) == 2 and all(type(i) is int for i in p) and p[0] < p[1] for p in pairs) and
                 len(set(pairs)) == len(pairs) and sorted({i for pair in pairs for i in pair}) == positions,
                 "individual ACF pair schema")
        _require(s["policy"] == ("all_runs" if clock["mode"] == "regular" else "irregular"), "individual ACF selection policy")
        if lag["samples"] is None:
            _require(lag["realized_seconds"] is None and not pairs and record["compute_status"] == "unsupported_exact_lag", "individual unsupported lag")
        else:
            _require(clock["mode"] == "regular" and type(lag["samples"]) is int and lag["samples"] > 0 and
                     lag["realized_seconds"] == lag["samples"]*clock["cadence"].value/1e9 and
                     all(b-a == lag["samples"] and instants[b]-instants[a] == lag["samples"]*clock["cadence"] for a,b in pairs),
                     "individual ACF lag binding")
        minimum = max(2,minimum)
        count = len(pairs)
    else:
        _require(s["policy"] == settings["support"], "individual selection policy binding")
        count = len(positions)
        if method == "individual.elapsed_day_ols_v1":
            _fields(settings,("time_unit","uncertainty"))
            _fields(s,("uncertainty",))
            _require(settings["time_unit"] == "elapsed_days" and settings["uncertainty"] in ("unavailable","iid_ols_compatibility"), "individual trend settings")
            _fields(s["uncertainty"],("method","p_value","reason","assumptions"))
            uncertainty = s["uncertainty"]
            minimum = max(2,minimum)
            diagnostic = settings["uncertainty"] == "iid_ols_compatibility" and count >= minimum
            _require(uncertainty["method"] == settings["uncertainty"] and _text(uncertainty["reason"]) and
                     set(uncertainty) == {"method","p_value","reason","assumptions"}|({"trusted_temporal_inference"} if diagnostic else set()),
                     "individual trend uncertainty")
            _require((_number(uncertainty["p_value"]) and 0 <= uncertainty["p_value"] <= 1 or uncertainty["p_value"] is None)
                     if diagnostic else uncertainty["p_value"] is None, "individual trend p-value schema")
            _require((_text(uncertainty["assumptions"]) and uncertainty.get("trusted_temporal_inference") is False)
                     if diagnostic else uncertainty["assumptions"] is None, "individual trend inference qualification")
            if count >= minimum:
                _fields(s,("intercept","elapsed_days"))
                _require(isinstance(s["intercept"],(float,np.floating)) and
                         (record["compute_status"] != "ok" or np.isfinite(s["intercept"])), "individual trend numeric intercept")
                _array(s["elapsed_days"],finite=True)
                _require(len(s["elapsed_days"]) == count and np.array_equal(s["elapsed_days"],[(t-times[0]).value/86400e9 for t in times]), "individual trend time geometry")
            else:
                _require("intercept" not in s and "elapsed_days" not in s, "individual unavailable trend diagnostics")
        else:
            _fields(settings,("ddof","quantile_method"))
            _require(type(settings["ddof"]) is int and settings["ddof"] == (1 if method == "individual.sample_sd_v1" else 0) and
                     settings["quantile_method"] == "linear" and type(s["constant_input"]) is bool, "individual descriptive settings")
            if method == "individual.sample_sd_v1": minimum = max(2,minimum)
    statuses = ("ok","insufficient_support","degenerate_input") + (("unsupported_exact_lag",) if method == "individual.exact_lagged_pearson_v1" else ())
    status = record["compute_status"]
    _require(status in statuses and (status == "unsupported_exact_lag" or (status == "insufficient_support") == (count < minimum)),
             "individual result availability")


def _reference_summary(record):
    """Foundation frozen-reference evaluation: multi-variable support without an input binding."""
    s = record["support"]
    _declared(record, "exploratory_reference_summary", "1")
    _fields(s, ("policy", "estimator", "positions", "selected", "fit_scope", "semantics"))
    evaluated = "evaluation_positions" in s
    _require(("evaluation_burden" in s) == evaluated, "evaluation support availability")
    burden = s["evaluation_burden"] if evaluated else s["selected"]
    positions = _positions(s["evaluation_positions"] if evaluated else s["positions"])
    _require(record["representation"] == "frozen_reference" and _text(record["reference_id"]) and s["estimator"] == "point" and
             s["policy"] in ("longest_run", "all_runs", "complete_case", "irregular") and isinstance(burden.get("variables"), dict) and
             (s["fit_scope"], s["semantics"]) in (("training_only", "training_only"), ("self", "retrospective")),
             "foundation reference summary declarations")
    unknown = any(_number(v.get("unknown_touched_count")) and v["unknown_touched_count"] > 0 for v in burden["variables"].values())
    _use(record, "insufficient_support" if not positions else "unknown_provenance" if unknown else "sensitivity_only")


def _signal_method(record, unit):
    """Finite dispatch for native single-signal producers after source binding.

    `unit` is the bound source variable's measurement unit (None only for a
    source-free refused harmonic). Saved selections, settings, units and
    qualifications are compared; no estimate, fit or calibration is rerun.
    """
    method = record["method_id"]
    if method in _INDIVIDUAL:
        _individual_support(record, unit)
    elif method == _WELCH:
        _welch_record(record, unit)
    elif method == _IRREGULAR:
        _irregular_record(record, unit)
    elif method in _SUMMARIES:
        _summary_record(record, unit)
    elif method in (_FISHER, _SIEGEL):
        _harmonic_record(record, unit)
    elif method == "fi1.ordered_mass_roughness_v1":
        s = record["support"]
        _fields(s, ("observations",))
        _signal_selection(s, s["observations"], estimator="point")
        _require(s["policy"] == ("all_runs" if s["clock"]["mode"] == "regular" else "irregular"), "FI1 selection policy")
        _fi1_evaluation(record, s, record["value"], unit, "selected")
    elif method in _DENSITY:
        _density_record(record, unit)
    else:
        _require(False, "unknown native signal method")


def _declared(record, metric_status, units, observations=None, settings=None):
    """Constant producer qualification, unit and configuration declarations."""
    _require(record["metric_status"] == metric_status, "native metric qualification declaration")
    _require(encode(record["units"]) == encode(units), "native output unit binding")
    if settings is not None:
        _require(record["config_id"] == identity(dict(settings=settings, observations=observations)), "native configuration identity")


def _use(record, expected):
    _require(record["use_status"] == expected, "native provenance/use qualification")


def _summary_use(use):
    # Spectral summaries never promote an upstream primary qualification.
    return "sensitivity_only" if use == "primary_eligible" else use


def _welch_requested(requested):
    """Saved WelchSpec record: exact types, domains and recipe declarations."""
    sensor = isinstance(requested,dict) and requested.get("recipe") == "sensor_length"
    _require(isinstance(requested,dict) and set(requested) == _WELCH_SETTINGS | (
             {"min_nperseg","max_nperseg","length_divisor"} if sensor else set()), "Welch requested settings")
    def size(key, minimum):
        return requested[key] is None or type(requested[key]) is int and requested[key] >= minimum
    def choice(key, *values):
        return type(requested[key]) is str and requested[key] in values
    length, overlap, fft = requested["nperseg"], requested["overlap"], requested["nfft"]
    _require(choice("recipe","research_explicit","sensor_length") and (length is None) == sensor and
             size("nperseg",2) and size("overlap",0) and size("nfft",2) and
             (length is None or (overlap is None or overlap < length) and (fft is None or fft >= length)) and
             type(requested["min_segments"]) is int and requested["min_segments"] >= 1 and
             type(requested["periodic"]) is bool and requested["onesided"] is True and choice("average","mean") and
             choice("taper","boxcar","hann") and (requested["detrend"] is False or choice("detrend","constant","linear")) and
             choice("scaling","density","spectrum") and choice("support","all_runs","longest_run") and
             (not sensor or all(type(requested[k]) is int for k in ("min_nperseg","max_nperseg","length_divisor")) and
              2 <= requested["min_nperseg"] <= requested["max_nperseg"] and requested["length_divisor"] >= 1),
             "Welch requested settings")


def _welch_runs(s, instants):
    """Bind regular Welch run, segment, duration and overlap support to saved geometry.

    Follows power_spectrum: candidate runs -> support policy (first longest run
    on ties) -> resolved length (sensor rule from the pre-segment selected count)
    -> used/rejected partition -> segments -> contributed runs and durations ->
    effective averages. Only integer positions, cadence and the declared taper
    enter; each generated sequence is first bounded by a supplied list. No
    periodogram is evaluated. Returns the derived segment count.
    """
    requested, resolved = s["requested"], s["resolved"]
    candidates = s["candidate_runs"]
    _run_intervals(candidates, len(s["full_segment"]["positions"]), s["clock"], instants, "Welch candidate")
    bounds = [(run["start_position"],run["stop_position"]) for run in candidates]
    chosen = bounds if requested["support"] == "all_runs" else [max(bounds, key=lambda r: r[1]-r[0])] if bounds else []
    starts = {a for a,_ in chosen}
    selected = _positions(s["selected_positions"])
    _require(len(selected) == sum(b-a for a,b in chosen) and selected == [p for a,b in chosen for p in range(a,b)] and
             encode(s["selected_runs"]) == encode([run for run in candidates if run["start_position"] in starts]),
             "Welch support selection")
    length, hop, sampling = resolved["nperseg"], resolved["hop"], resolved["sampling_hz"]
    if requested["recipe"] == "sensor_length":
        _require(length == max(requested["min_nperseg"], min(requested["max_nperseg"], len(selected)//requested["length_divisor"])),
                 "Welch resolved segment length")
    used, rejected = [], []
    for run in candidates:
        a, b = run["start_position"], run["stop_position"]
        if a not in starts or b-a < length:
            rejected.append(dict(run, reason="not_selected_by_support" if a not in starts else "shorter_than_requested_segment"))
        else:
            k = (b-a-length)//hop+1
            stop = a+(k-1)*hop+length  # a trailing incomplete hop is unused
            used.append(dict(run, raw_segments=k, used_stop_position=stop, used_duration_seconds=(stop-a)/sampling))
    _require(encode(s["used_runs"]) == encode(used) and encode(s["rejected_runs"]) == encode(rejected),
             "Welch used/rejected run partition")
    count, segments = sum(run["raw_segments"] for run in used), s["segments"]
    _require(isinstance(segments,list) and type(s["raw_segments"]) is int and s["raw_segments"] == len(segments) == count,
             "Welch segment count")
    expected = (dict(start_position=run["start_position"]+j*hop, stop_position=run["start_position"]+j*hop+length,
                     run_start_position=run["start_position"]) for run in used for j in range(run["raw_segments"]))
    _require(all(encode(saved) == encode(row) for saved,row in zip(segments,expected)), "Welch segment geometry")
    # The validated contributing selection has exactly one run per used prefix.
    _require(len(s["runs"]) == len(used) and all(
             r["start_position"] == u["start_position"] and r["stop_position"] == u["used_stop_position"] and
             _time(r["start"],strings=True) == _time(u["start"],strings=True) for r,u in zip(s["runs"],used)),
             "Welch contributed run binding")
    _require(s["longest_run_seconds"] == max((run["used_duration_seconds"] for run in used), default=0),
             "Welch longest used run duration")
    # `_welch_geometry` bounded `length` by the saved physical grid.
    window = get_window(requested["taper"], length, fftbins=requested["periodic"])
    energy, total = float(window @ window), float(window.sum())
    _numbers(resolved, ("window_energy","window_sum","density_to_spectrum_factor"))
    _require(energy > 0 and total > 0 and all(np.isclose(resolved[k],v,rtol=_WELCH_RTOL,atol=0) for k,v in (
             ("window_energy",energy),("window_sum",total),("density_to_spectrum_factor",energy*sampling/total**2))),
             "Welch taper declarations")
    # Equal weights within each run; a gap or rejected run creates no overlap pair.
    penalty = sum(_overlap_square_sum(window, hop, np.ones(run["raw_segments"])) for run in used)
    averages = s["effective_averages"]
    _require(isinstance(averages,dict) and set(averages) == {*_WELCH_AVERAGES,"value","reason","overlap_pair_square_sum"} and
             all(type(averages[k]) is type(v) and averages[k] == v for k,v in _WELCH_AVERAGES.items()) and
             _number(averages["overlap_pair_square_sum"]) and
             np.isclose(averages["overlap_pair_square_sum"],penalty,rtol=_WELCH_RTOL,atol=0) and
             (_number(averages["value"]) and averages["reason"] is None and
              np.isclose(averages["value"],count**2/(count+2*penalty),rtol=_WELCH_RTOL,atol=0) if count else
              averages["value"] is None and type(averages["reason"]) is str and averages["reason"] == "no eligible segments"),
             "Welch effective averages")
    return count


def _welch_cycles(s):
    """Successful cycles per display ordinate; the saved vector bounds the claimed FFT grid."""
    cycles, resolved = s["cycles_longest_run"], s["resolved"]
    _require(isinstance(cycles,np.ndarray) and cycles.dtype == np.float64 and cycles.shape == (resolved["nfft"]//2+1,) and
             bool(np.isfinite(cycles).all()), "Welch cycles schema")
    display = np.arange(len(cycles))*(1/(resolved["nfft"]*(1/resolved["sampling_hz"])))
    _require(np.allclose(cycles,display*s["longest_run_seconds"],rtol=1e-12,atol=0), "Welch cycles/longest run binding")


def _welch_inherited(s, unit, spectrum=None):
    """Welch support retained by a spectrum or its summaries; returns status/units/use.

    Every saved run, segment, duration and overlap binding precedes the derived
    status; callers bind successful cycles before returning.
    """
    _fields(s, ("metric","requested","resolved","observations","raw_segments"))
    requested, resolved, observations = s["requested"], s["resolved"], s["observations"]
    _welch_requested(requested)
    regular = resolved is not None
    positions, times = _signal_selection(s, observations, estimator="regular" if regular else "point")
    _require(s["metric"] == _WELCH and s["policy"] == ("all_runs" if regular else "irregular") and
             (s["clock"]["mode"] == "regular") == regular, "Welch selection/clock binding")
    _welch_geometry(s, spectrum)
    diagnostics = set(s) & _WELCH_SUCCESS
    if not regular:
        _require(not diagnostics and not set(s) & set(_WELCH_RUNS), "Welch unavailable geometry")
        status = "unsupported_time_geometry"
    else:
        _require(isinstance(resolved,dict) and all(encode(resolved[k]) == encode(v) for k,v in requested.items()
                 if k not in ("nperseg","overlap","nfft")), "Welch requested/resolved settings")
        _fields(s, _WELCH_RUNS)
        _numbers(s, ("duration_seconds","longest_run_seconds"))
        _require(s["contributing_positions"] == positions and isinstance(s["segments"],list) and
                 s["raw_segments"] == len(s["segments"]) and s["duration_seconds"] == len(positions)/resolved["sampling_hz"] and
                 0 <= s["longest_run_seconds"] <= s["duration_seconds"], "Welch contributing segment support")
        count = _welch_runs(s, dict(zip(positions,times)))
        short = count < requested["min_segments"] or len(positions) < observations["min_points"]
        # Only a finite successful spectrum records its constant/zero/cycle diagnostics.
        _require(diagnostics in (set(),_WELCH_SUCCESS) and not (short and diagnostics), "Welch spectrum diagnostics")
        status = "insufficient_support" if short else "ok" if diagnostics else "degenerate_input"
        if status == "ok":
            _require(type(s["constant_input"]) is bool and type(s["zero_spectrum"]) is bool and
                     (spectrum is None or s["zero_spectrum"] == bool(np.all(spectrum["psd"] == 0))), "Welch spectrum diagnostics")
    units = unit + ("^2/Hz" if requested["scaling"] == "density" else "^2")
    return status, units, _saved_signal_use(s, observations)


def _welch_record(record, unit):
    s = record["support"]
    status, units, use = _welch_inherited(s, unit, record["value"])
    if status == "ok":
        _welch_cycles(s)
    config = s["requested"] if s["resolved"] is None else dict(requested=s["requested"], resolved=s["resolved"])
    _declared(record, "primary_with_support", units, s["observations"], config)
    _use(record, use)
    _require(record["compute_status"] == status, "Welch availability/support binding")


def _irregular_inherited(s, unit, spectrum=None):
    _fields(s, "metric requested resolved observations frequency_support segments contributing_positions raw_segments effective_averages".split())
    requested, resolved, observations = s["requested"], s["resolved"], s["observations"]
    _require(isinstance(requested,dict) and set(requested) == _IRREGULAR_SETTINGS and isinstance(resolved,dict) and
             all(encode(resolved[k]) == encode(v) for k,v in requested.items()) and
             isinstance(requested["segment"],pd.Timedelta) and requested["segment"] > pd.Timedelta(0) and
             resolved["physical_resolution_hz"] == 1e9/requested["segment"].value, "irregular requested/resolved settings")
    positions, _ = _signal_selection(s, observations, estimator="irregular")
    _require(s["metric"] == _IRREGULAR and s["policy"] == "irregular" and s["contributing_positions"] == positions and
             isinstance(s["segments"],list) and s["raw_segments"] == len(s["segments"]), "irregular selection binding")
    available = _irregular_frequencies(s, spectrum)
    use = _saved_signal_use(s, observations, metric_status="descriptive_irregular_projection_density")
    return "ok" if available else "insufficient_support", unit+"^2/Hz", use


def _irregular_record(record, unit):
    s = record["support"]
    status, units, use = _irregular_inherited(s, unit, record["value"])
    _declared(record, "descriptive_irregular_projection_density", units, s["observations"],
              dict(requested=s["requested"], resolved=s["resolved"]))
    _use(record, use)
    _require(record["compute_status"] == status and (record["value"] is not None) == (status == "ok"), "irregular overall availability")


def _spec_real(value):
    # Spec `_positive(..., zero=True)` saves a finite nonnegative Python float.
    return type(value) is float and bool(np.isfinite(value)) and value >= 0


def _spec_bounds(value):
    # Spec `_interval` saves increasing positive finite float64 bounds.
    return (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.shape == (2,) and
            bool(np.isfinite(value).all()) and 0 < value[0] < value[1])


def _summary_settings(method, settings):
    """Saved spec record: types, domains, enums and derived declarations."""
    _require(all(encode(settings[k]) == encode(v) for k,v in _SUMMARY_CONSTANTS[method].items()),
             "spectral summary fixed declarations")
    if method == "spectral.peaks_v1":
        _require((settings["period_seconds"] is None or _spec_bounds(settings["period_seconds"])) and
                 _spec_real(settings["min_cycles"]) and type(settings["max_peaks"]) is int and settings["max_peaks"] >= 1 and
                 type(settings["prominence"]) is str and settings["prominence"] in ("linear","log10"), "spectral peak settings domain")
    elif method == "spectral.band_power_v1":
        bounds, coordinates = settings["requested_bounds"], settings["coordinates"]
        _require(_spec_bounds(bounds) and type(coordinates) is str and coordinates in ("hours","hz") and
                 _spec_real(settings["min_cycles"]) and type(settings["min_ordinates"]) is int and settings["min_ordinates"] >= 3,
                 "band power settings domain")
        frequency = bounds if coordinates == "hz" else 1/(bounds[::-1]*3600)
        _require(encode(settings["frequency_bounds_hz"]) == encode(frequency), "band power frequency bounds declaration")
    else:
        candidates = settings["candidates_hz"]
        _require((settings["interval_hz"] is None or _spec_bounds(settings["interval_hz"])) and
                 (candidates is None or isinstance(candidates,np.ndarray) and candidates.dtype == np.float64 and candidates.ndim == 1 and
                  bool(np.isfinite(candidates).all()) and bool(np.all(candidates > 0)) and bool(np.all(np.diff(candidates) > 0))) and
                 type(settings["min_arm"]) is int and settings["min_arm"] >= 3 and
                 type(settings["log_base"]) is float and settings["log_base"] in (np.e, 10.0) and
                 _spec_real(settings["tie_tolerance"]) and _spec_real(settings["exact_rss_tolerance"]), "spectral model settings domain")
        if method == "spectral.irregular_log_models_v1":
            adapter = settings["adapter"]
            _require(isinstance(adapter,dict) and set(adapter) == {"min_ordinates","min_cycles","selection"} and
                     type(adapter["min_ordinates"]) is int and adapter["min_ordinates"] >= 3 and _spec_real(adapter["min_cycles"]) and
                     type(adapter["selection"]) is str and adapter["selection"] == _ADAPTER_SELECTION,
                     "irregular summary adapter settings domain")


def _adapter_selection(s, settings):
    """Replay the adapter's per-ordinate gates on saved support; returns selected positions.

    Uses only saved frequency, pooled power, occupied span, resolution and
    settings; no spectrum, fit or estimate is recomputed.
    """
    adapter, selection = settings["adapter"], s["adapter_selection"]
    _require(isinstance(selection,dict) and set(selection) == {"positions","exclusions"} | set(adapter) and
             all(encode(selection[k]) == encode(adapter[k]) for k in adapter), "irregular summary adapter selection")
    interval, resolution, rows = settings["interval_hz"], s["resolved"]["physical_resolution_hz"], s["frequency_support"]
    selected, exclusions = [], []
    for j, row in enumerate(rows):
        _numbers(row, ("min_occupied_span_seconds",))
        frequency, power, span = row["frequency_hz"], row["psd"], row["min_occupied_span_seconds"]
        _require(span >= 0, "irregular ordinate occupied span")
        why = ("outside_requested_interval" if interval is not None and not interval[0] <= frequency <= interval[1] else
               "unsupported_or_nonpositive_power" if power is None or power <= 0 else
               "insufficient_cycles" if frequency*span < adapter["min_cycles"] else
               "unresolved_dense_ordinate" if selected and frequency-rows[selected[-1]]["frequency_hz"] < resolution*(1-1e-12) else None)
        if why:
            exclusions.append(dict(position=j, reason=why))
        else:
            selected.append(j)
    _require(encode(selection["positions"]) == encode(selected) and encode(selection["exclusions"]) == encode(exclusions),
             "irregular summary adapter gates")
    return selected


def _fit_support(record, settings, frequency, inherited):
    """Saved log-fit selection, availability and spec gates; no refit."""
    s, value = record["support"], record["value"]
    selection = s["fit_selection"]
    _require(value is None or encode(value["selection"]) == encode(selection), "spectral fit selection binding")
    _fit_selection(selection)
    _require(_same(selection["inherited_range_hz"], inherited) and
             all(p < len(frequency) for key in _FIT_POSITIONS for p in selection[key]) and
             all(row["position"] < len(frequency) for row in selection["frequency_exclusions"]) and
             record["compute_status"] == ("ok" if len(selection["positions"]) >= 3 else "insufficient_support"),
             "spectral fit selection support/availability")
    # Frequency gates are saved, even when the PSD itself is omitted. Power
    # exclusions cannot exempt positions from the independent frequency checks.
    interval = settings["interval_hz"]
    outside = [i for i,f in enumerate(frequency) if interval is not None and not interval[0] <= f <= interval[1]]
    unsupported = [dict(position=i, reason="outside_frequency_support") for i,f in enumerate(frequency)
                   if inherited is not None and not inherited[0] <= f <= inherited[1]]
    nonpositive = [i for i,f in enumerate(frequency) if f <= 0]
    excluded = set(outside + nonpositive + selection["nonpositive_or_nonfinite_power"] +
                   [row["position"] for row in unsupported])
    selected = [i for i in range(len(frequency)) if i not in excluded]
    bounds = [float(frequency[selected[0]]), float(frequency[selected[-1]])] if selected else None
    _require(selection["outside_interval"] == outside and selection["frequency_exclusions"] == unsupported and
             selection["nonpositive_frequency"] == nonpositive and selection["positions"] == selected and
             _same(selection["resolved_interval_hz"], bounds), "spectral fit frequency gates")
    if value is not None:
        _require(settings["candidates_hz"] is None or encode(value["resolved_candidates_hz"]) == encode(settings["candidates_hz"]),
                 "spectral fit candidate settings")
        for row in value["candidates"]:
            _counts(row, ("low_n","high_n"))
            _require((row["comparison_status"] == "insufficient_arm_points") ==
                     (row["low_n"] < settings["min_arm"] or row["high_n"] < settings["min_arm"]), "spectral fit arm settings")


def _peak_gates(s, settings, value):
    """Accepted/rejected peaks agree with the saved period, cycle and count gates."""
    period, minimum, duration = settings["period_seconds"], settings["min_cycles"], s["longest_run_seconds"]
    rows = value["table"].to_dict("records")
    def gate(row):
        if period is not None and not period[0] <= row["period_seconds"] <= period[1]:
            return "outside_period_range"
        return "insufficient_cycles" if row["cycles"] < minimum else None
    _require(len(rows) <= settings["max_peaks"] and all(gate(row) is None for row in rows) and
             all(gate(row) == (None if row["reason"] == "max_peaks" else row["reason"]) for row in value["rejected"]) and
             (len(rows) == settings["max_peaks"] or all(row["reason"] != "max_peaks" for row in value["rejected"])) and
             all(np.isclose(row["cycles"],row["frequency_hz"]*duration,rtol=1e-12,atol=0) for row in rows+value["rejected"]),
             "spectral peak settings gates")


def _band_support(record, settings, supported):
    """Saved band support/denominator and the producer's availability gates."""
    s, actual = record["support"], record["compute_status"]
    _require(actual in ("ok","degenerate_input","outside_frequency_support","insufficient_frequency_resolution","insufficient_cycles"),
             "band power availability")
    _counts(s, ("physical_ordinates",))
    _numbers(s, ("cycles_at_low_frequency",))
    low, high = settings["frequency_bounds_hz"]
    count, cycles, denominator = s["physical_ordinates"], s["cycles_at_low_frequency"], s["denominator_bounds_hz"]
    physical = s["resolved"]["physical_positive_frequencies_hz"]
    _require(count == int(np.sum((physical >= low) & (physical <= high))) and cycles >= 0 and
             np.isclose(cycles,low*s["longest_run_seconds"],rtol=1e-12,atol=0), "band power ordinate/cycle support")
    # Null denominator is the retained pre-integration density diagnostic. The
    # summary omits its input PSD, so its truth cannot be recovered on reload.
    # Derive every decidable gate from saved support, never from the status claim.
    _require(denominator is None or _same(denominator, supported), "band power denominator support")
    display = s["resolved"]["display_positive_range_hz"]
    expected = ("degenerate_input" if denominator is None else
                "outside_frequency_support" if low < supported[0] or high > supported[1] or
                    supported[0] < display[0] or supported[1] > display[1] else
                "insufficient_frequency_resolution" if count < settings["min_ordinates"] else
                "insufficient_cycles" if cycles < settings["min_cycles"] else "ok")
    _require((actual == "degenerate_input") == (denominator is None), "band power denominator support")
    _require(actual == expected, "band power settings gates")


def _summary_record(record, unit):
    """Summaries inherit their spectrum's support, metric and qualification."""
    s, method, value = record["support"], record["method_id"], record["value"]
    upstream = _SUMMARIES[method]
    _fields(s, ("spectral_input_id","summary_settings"))
    settings = s["summary_settings"]
    _require(_text(s["spectral_input_id"]) and isinstance(settings,dict) and set(settings) == _SUMMARY_SETTINGS[method] and
             record["config_id"] == identity(dict(input=s["spectral_input_id"], settings=settings)) and
             (settings["basis"] is None or settings["basis"] == record["representation"]), "spectral summary input/settings binding")
    # A recomputed identity does not make a setting legal; check every saved
    # setting before any availability branch returns.
    _summary_settings(method, settings)
    if upstream == _WELCH:
        status, spectrum_units, use = _welch_inherited(s, unit)
    else:
        status, spectrum_units, use = _irregular_inherited(s, unit)
    units = ("1" if method in ("spectral.log_models_v1","spectral.irregular_log_models_v1") else
             spectrum_units if method == "spectral.peaks_v1" else spectrum_units.removesuffix("/Hz"))
    _declared(record, "descriptive_spectral_summary", units)
    _use(record, _summary_use(use))
    actual = record["compute_status"]
    _require((value is not None) == (actual == "ok"), "spectral summary value availability")
    if method == "spectral.band_power_v1":
        _require(s["requested"]["scaling"] == "density", "band power density scaling")
    extra = set(s) & _SUMMARY_EXTRA  # producer-branch support beyond the inherited spectrum
    if status != "ok":
        _require(actual == status and not extra, "spectral summary inherited availability")
        return
    if upstream == _IRREGULAR:
        _fields(s, ("adapter_selection",))
        positions = _adapter_selection(s, settings)
        if len(positions) < settings["adapter"]["min_ordinates"]:
            cycles = any(row["reason"] == "insufficient_cycles" for row in s["adapter_selection"]["exclusions"])
            _require(actual == ("insufficient_cycles" if cycles else "insufficient_frequency_resolution") and
                     extra == {"adapter_selection"}, "irregular summary adapter availability")
            return
        _require(extra == {"adapter_selection","fit_selection","fit_grid"} and
                 s["fit_grid"] == "actual_supported_resolution_thinned", "irregular summary fit availability")
        _fit_support(record, settings, [s["frequency_support"][p]["frequency_hz"] for p in positions], None)
        # Every adapter-selected ordinate is an eligible log-fit input.
        _require(s["fit_selection"]["positions"] == list(range(len(positions))), "irregular summary fit selection binding")
        return
    resolved = s["resolved"]
    supported = resolved["supported_positive_range_hz"]
    if supported is None or supported[0] == supported[1]:
        excluded = s.get("excluded_frequency_positions")
        grid = resolved["nperseg"]//2+1 if method == "spectral.log_models_v1" else resolved["nfft"]//2+1
        _require(actual == "insufficient_frequency_resolution" and (not extra if method == "spectral.band_power_v1" else
                 extra == {"inherited_range_hz","excluded_frequency_positions"} and _same(s["inherited_range_hz"], supported) and
                 isinstance(excluded,list) and len(excluded) == grid and
                 all(type(p) is int and p == i for i,p in enumerate(excluded))), "spectral summary frequency resolution")
    elif method == "spectral.log_models_v1":
        _require(extra == {"fit_selection","fit_grid"} and s["fit_grid"] == "unpadded", "spectral fit availability")
        _fit_support(record, settings, [0., *resolved["physical_positive_frequencies_hz"]], supported)
    elif method == "spectral.peaks_v1":
        _require(actual == "ok", "spectral peak availability")
        _fields(value, ("inherited_range_hz",))
        _require(actual == "ok" and extra == {"prominence_units"} and s["prominence_units"] == (
                 record["units"] if settings["prominence"] == "linear" else "log10(power ordinate)") and
                 list(value["table"]["basis"]) == [record["representation"]]*len(value["table"]) and
                 encode(value["inherited_range_hz"]) == encode(supported), "spectral peak selection binding")
        _peak_gates(s, settings, value)
    else:
        _require(extra == {"physical_ordinates","cycles_at_low_frequency","denominator_bounds_hz"}, "band power branch support")
        _band_support(record, settings, supported)
    # Gates read the bound longest run, never this vector; binding it after the
    # unresolved exclusion check keeps that list's pre-expansion refusal first.
    _welch_cycles(s)


def _harmonic_record(record, unit):
    s, method, value, status = record["support"], record["method_id"], record["value"], record["compute_status"]
    _declared(record, "assumed_white_noise_diagnostic", "1")
    _fields(s, ("requested","methods","backend","null","null_status"))
    settings, backend = s["requested"], s["backend"]
    _require(isinstance(settings,dict) and set(settings) == _HARMONIC_SETTINGS and
             s["methods"] == dict(fisher=_FISHER, siegel=_SIEGEL) and s["null"] == settings["null"] and
             s["null_status"] == "assumed_not_tested" and isinstance(backend,dict) and
             set(backend) == {"requested","numpy","implementation"} and backend["requested"] == settings["backend"] and
             _text(backend["numpy"]) and backend["implementation"] == ("independent" if settings["backend"] == "independent" else None),
             "harmonic settings/null/backend declarations")
    calibrated = "calibration" in s
    _require(calibrated == ("calibration_id" in s) and (not calibrated or method == _SIEGEL), "harmonic calibration declaration")
    if calibrated:
        calibration = s["calibration"]
        _fields(calibration, ("method_id","alpha","lambda_","null","m","critical_T","compute_status"))
        _require(calibration["method_id"] == _SIEGEL and calibration["alpha"] == settings["alpha"] and
                 calibration["lambda_"] == settings["lambda_"] and calibration["null"] == settings["null"] and
                 calibration["compute_status"] in ("ok","insufficient_calibration_resolution"), "harmonic calibration settings binding")
    if s.get("input_route") == "supplied_spectrum_refused":
        _require(record["use_status"] in ("unknown_provenance","sensitivity_only"), "native provenance/use qualification")
        return
    _fields(s, ("resolved","measurement_unit","frequency_unit","power_unit","observations"))
    resolved, observations, clock = s["resolved"], s["observations"], s["clock"]
    # Only a successful raw route with representable power retains Fourier geometry.
    geometric = status == "ok" if method == _FISHER else status in ("ok","not_computed","insufficient_calibration_resolution")
    positions, _ = _signal_selection(s, observations, estimator="rectangular_periodogram" if geometric else "point")
    n = len(s["full_segment"]["positions"])
    _require(s["policy"] == ("all_runs" if clock["mode"] == "regular" else "irregular") and
             encode(resolved) == encode(dict(settings, nfft=n if settings["nfft"] is None else settings["nfft"])) and
             s["measurement_unit"] == unit and s["frequency_unit"] == "Hz" and s["power_unit"] == unit+"^2" and
             record["config_id"] == identity(dict(spec=identity(settings), observations=observations)),
             "harmonic selection/settings/unit binding")
    _use(record, _saved_signal_use(s, observations, metric_status="assumed_white_noise_diagnostic"))
    base = ("missing_backend" if settings["backend"] == "sensor_pinned" else
            "unsupported_null" if settings["null"] != "iid_gaussian_white_noise" else
            "unsupported_estimator" if settings["estimator"] != "rectangular_periodogram" or settings["taper"] != "boxcar" or resolved["nfft"] != n else
            "unsupported_basis" if s["basis"] != "raw" else
            "insufficient_support" if len(positions) < max(5, observations["min_points"]) else
            "unsupported_time_geometry" if clock["mode"] != "regular" else
            "unsupported_support" if len(positions) != n or len(s["runs"]) != 1 else "ok")
    geometry = "proportions powers bins n m removed_mean frequency_hz nfft dc_excluded nyquist_excluded".split()
    _require(all((k in s) == geometric for k in geometry), "harmonic Fourier geometry availability")
    if base != "ok" or not geometric:
        _require(status == (base if base != "ok" else "degenerate_input"), "harmonic availability/support binding")
        return
    bins = np.arange(1, (n-1)//2+1)
    _array(s["bins"], integer=True)
    _masses(s["proportions"])
    _array(s["powers"], finite=True)
    mean = s["removed_mean"]
    _require(_number(mean) or isinstance(mean,np.ndarray) and mean.shape == () and mean.dtype.kind == "f" and np.isfinite(mean),
             "harmonic removed mean")
    _require(np.array_equal(s["bins"],bins) and s["m"] == len(bins) == len(s["proportions"]) == len(s["powers"]) and
             s["n"] == s["nfft"] == n and s["dc_excluded"] is True and s["nyquist_excluded"] == (n % 2 == 0) and
             np.all(s["powers"] >= 0) and encode(s["frequency_hz"]) == encode(bins*(1e9/clock["cadence"].value)/n),
             "harmonic Fourier geometry")
    if method == _FISHER:
        expected = "ok"
    elif not calibrated:
        expected = "not_computed"
    else:
        expected = calibration["compute_status"]
        _require(calibration["m"] == s["m"], "harmonic calibration geometry binding")
    _require(status == expected, "harmonic calibration availability")
    if value is not None:
        _require(value["alpha"] == settings["alpha"] and bool(value["rejected"]) == (value["p_value"] <= value["alpha"]) and
                 (value["g"] == float(np.max(s["proportions"])) if method == _FISHER else
                  value["lambda_"] == settings["lambda_"] and value["critical"] == calibration["critical_T"]),
                 "harmonic saved decision/statistic binding")


def _fi1_evaluation(record, s, value, unit, burden_key):
    """FI1 metric records and trajectory rows; `burden_key` names the selected burden."""
    _fields(s, ("fit","edge_id","measurement_unit","requested","resolved","observations"))
    fit, observations = s["fit"], s["observations"]
    _observation_policy(observations)
    _fields(fit, "method_id spec spec_id edges edges_id variable unit basis use_status compute_status reason".split())
    _require(fit["method_id"] == "fi1.ordered_mass_roughness_v1" and isinstance(fit["spec"],dict) and
             fit["spec_id"] == identity(fit["spec"]) and fit["edges_id"] == identity(fit["edges"]) and
             encode(s["requested"]) == encode(fit["spec"].get("requested")) and encode(s["resolved"]) == encode(fit["spec"]) and
             s["edge_id"] == fit["edges_id"] and fit["variable"] == s["variable"] and fit["basis"] == s["basis"] and
             s["measurement_unit"] == fit["unit"] == unit and fit["use_status"] in _USE, "FI1 reference/settings/unit binding")
    _declared(record, "historical_sensitivity_index", "1")
    _require(record["config_id"] == identity(dict(spec=fit["spec_id"], observations=observations)), "native configuration identity")
    _evaluation_use(record, s, value, observations, fit, burden_key, "historical_sensitivity_index")
    if fit["compute_status"] != "ok":
        _require(record["compute_status"] == fit["compute_status"] and value is None, "FI1 fitted reference availability")


def _evaluation_use(record, s, value, observations, fit, burden_key, metric_status):
    """Post-exclusion positions and burden, with unknown fitted provenance dominant."""
    evaluated = "evaluation_positions" in s
    _require(("evaluation_burden" in s) == evaluated and ("outside" not in s or evaluated) and
             (value is None or evaluated), "evaluation support availability")
    variable = dict(s, positions=s["evaluation_positions"], selected=s["evaluation_burden"]) if evaluated else dict(s, selected=s[burden_key])
    use = _saved_signal_use(variable, observations, metric_status=metric_status)
    _use(record, "unknown_provenance" if fit["use_status"] == "unknown_provenance" else use)


def _density_record(record, unit):
    s, value, method = record["support"], record["value"], record["method_id"]
    _fields(s, "fit requested resolved geometry_id measurement_unit backend evaluation_positions evaluation_burden observations".split())
    fit, observations = s["fit"], s["observations"]
    _observation_policy(observations)
    _fields(fit, "method_id spec spec_id variable unit basis center scale edges bandwidth geometry_id use_status compute_status reason".split())
    settings = fit["spec"]
    _require(isinstance(settings,dict) and fit["method_id"] == method == settings.get("method_id") and
             fit["spec_id"] == identity(settings) and encode(s["requested"]) == encode(settings) and
             encode(s["resolved"]) == encode(dict(settings, bandwidth=fit["bandwidth"], edges=fit["edges"])) and
             s["geometry_id"] == fit["geometry_id"] == identity(dict(edges=fit["edges"], bandwidth=fit["bandwidth"],
                 center=fit["center"], scale=fit["scale"], bounds=settings.get("bounds"))) and
             fit["variable"] == s["variable"] and fit["basis"] == s["basis"] and s["measurement_unit"] == fit["unit"] == unit and
             fit["use_status"] in _USE and isinstance(s["backend"],dict) and set(s["backend"]) == {"numpy","scipy"} and
             all(_text(v) for v in s["backend"].values()), "density reference/settings/unit/backend binding")
    _signal_selection(s, observations, estimator="point")
    _require(s["policy"] == ("all_runs" if s["clock"]["mode"] == "regular" else "irregular"), "density selection policy")
    _declared(record, "density_fisher_approximation", "1/"+unit+"^2")
    _require(record["config_id"] == identity(dict(spec=fit["spec_id"], observations=observations)), "native configuration identity")
    _evaluation_use(record, s, value, observations, fit, "selected", "density_fisher_approximation")
    if fit["compute_status"] != "ok":
        _require(record["compute_status"] == fit["compute_status"] and value is None, "density fitted reference availability")
    if value is not None:
        coordinate = "1" if settings["coordinates"] == "standardized" else unit
        _require(value["center"] == fit["center"] and value["scale"] == fit["scale"] and value["coordinate_units"] == coordinate and
                 value["raw_units"] == "1/"+unit+"^2" and value["coordinate_J_units"] == ("1" if coordinate == "1" else "1/"+unit+"^2"),
                 "density value reference/unit binding")


def _reference_fields(record, references):
    support, method = record["support"], record["method_id"]
    pca = method == "pca.frozen_projection_v1"
    graph = method in {"graph.state_mass_v1","graph.state_density_v1"}
    if pca or graph:
        training = support["training_support" if pca else "training"]
        _training_support(training)
        _require(_text(support["training_id"]) and training["input_id"] == support["training_id"] and
                 support["fit_scope"] in ("training_only","self") and
                 all(_text(support[k]) for k in ("backend","backend_version")), "native training support schema/binding")
        _require(support["refitted"] is False and isinstance(support["requested"],dict) and
                 encode(support["requested"]) == encode(support["resolved"]), "native frozen settings binding")
        evaluation = support if pca else support["candidate"]
        _complete_support(evaluation)
        if graph:
            _complete_support(support["evaluation"])
            _graph_evaluation(record, references.get(record["reference_id"]))
        elif record["value"] is not None:
            _require(len(evaluation["eligible_positions"]) >= evaluation["observations"]["min_points"],
                     "PCA minimum evaluation support")
            value = record["value"]
            _fields(value, ("eigenvalues", "fractions", "eigenvectors", "scaled_loadings", "eigenvector_definition", "scaled_loading_definition"))
            n, p = len(value["eigenvalues"]), len(support["requested"]["features"])
            for key, shape in (("eigenvalues", (n,)), ("fractions", (n,)), ("eigenvectors", (p, n)), ("scaled_loadings", (p, n))):
                _require(isinstance(value[key], np.ndarray) and value[key].shape == shape, "PCA array dimensions")
            _require(n == p and value["covariance"].shape == (p,p) and value["scores"].shape == (len(support["eligible_positions"]),p), "PCA covariance/score dimensions")
            _require(list(value["scores"].index) == [_time(t,strings=True) for t in support["eligible_instants"]] and
                     list(value["scores"].columns) == ["PC"+str(i+1) for i in range(p)], "PCA score support identity")
        if record["value"] is not None:
            settings = support["requested"]
            minimum = max(2, training["observations"]["min_points"])
            if pca or settings["pca_components"] is not None or settings["scaling"] == "zscore":
                minimum = max(minimum, settings["ddof"]+1)
            _require(len(training["eligible_positions"]) >= minimum, "native minimum training support")
        _require(all(encode(training[k]) == encode(evaluation[k]) for k in ("features","units","basis")) and
                 all(encode(training[k]) == encode(support["requested"][k]) for k in ("features","basis")),
                 "native training/evaluation feature binding")
        qualifications = [training["use_status"],(support if pca else support["evaluation"])["use_status"]]
        if graph:
            qualifications.append("sensitivity_only")
            if len(support["contributing_positions"]) < support["observations"]["min_points"]:
                qualifications.append("insufficient_support")
        _require(all(q in _USE for q in qualifications), "native evaluation qualification")
        expected_use = min(qualifications,key=_USE.index)
        if pca and expected_use == "primary_eligible" and support["fit_scope"] == "training_only" and any(
                v["selected"]["variables"][name]["touched_count"] for name,v in training["variables"].items()):
            expected_use = "sensitivity_only"
        _require(record["use_status"] == expected_use and
                 (not graph or expected_use != "unknown_provenance" or record["compute_status"] != "ok"),
                 "native training/result qualification binding")
        if graph and "observed_only_counterpart" in support:
            counterpart = support["observed_only_counterpart"]
            _require(isinstance(counterpart,dict) and set(counterpart) == _NATIVE, "graph counterpart schema")
            nested = counterpart["support"]
            _require(isinstance(nested,dict) and "observed_only_counterpart" not in nested and
                     counterpart["method_id"] == method and counterpart["reference_id"] == record["reference_id"] and
                     encode(nested["training"]) == encode(training) and
                     encode(nested["observations"]) == encode(dict(values="observed_only",min_points=support["observations"]["min_points"],
                         max_imputed_fraction=None,max_reconstructed_duration=None)) and
                     all(encode(nested["candidate"][k]) == encode(support["candidate"][k])
                         for k in ("input_id","features","units","basis","source","clock")), "graph counterpart binding")
            _native(counterpart,references)
    frozen = references.get(record["reference_id"])
    # A native reference identity need not have a supplied snapshot.
    if frozen is not None:
        _frozen_reference(record, frozen, pca, graph)


def _frozen_reference(record, frozen, pca, graph):
    support, method = record["support"], record["method_id"]
    if pca or graph:
        expected_method = "pca.complete_covariance_v1" if pca else "graph.state_reference_v1"
        _require(frozen["method_id"] == expected_method, "native reference method binding")
        copies = dict(training_id="training_id",fit_scope="fit_scope",requested="spec",resolved="spec")
        copies["training_support" if pca else "training"] = "support"
        if pca:
            copies.update(backend="backend",backend_version="backend_version")
        else:
            copies.update(pca_reference_id="pca_reference_id",requested_states="requested_states",rank="rank",
                          coordinate_units="coordinate_units",partition_edges="partition_edges")
        _require(all(encode(support[k]) == encode(frozen[v]) for k,v in copies.items()) and
                 encode(record["units"]) == encode(frozen["output_units"]) and
                 record["representation"] == frozen["spec"]["basis"], "native result/reference declaration binding")
        if pca and record["value"] is not None:
            _require(isinstance(frozen["eigensystem"],dict) and
                     encode({k:v for k,v in record["value"].items() if k != "scores"}) == encode(frozen["eigensystem"]),
                     "PCA eigensystem/reference binding")
        elif graph and record["value"] is not None:
            value, geometry = record["value"], frozen["geometry"]
            _require(all(encode(value[k]) == encode(geometry[v]) for k,v in
                         dict(nodes="nodes",edges="edges",weights="weights",cell_volumes="volumes").items()) and
                     all(encode(value[k]) == encode(frozen[k]) for k in ("partition_edges","realized_bins")) and
                     value["node_policy"] == frozen["spec"]["node_policy"], "graph geometry/reference binding")
    elif method.startswith(("fi1.","density_fisher.")):
        _require(encode(support["fit"]) == encode(frozen), "native fitted reference binding")
    elif "calibration_id" in support:
        _require(encode(support["calibration"]) == encode({k:v for k,v in frozen.items() if k != "sorted_T"}),
                 "native calibration reference binding")


def _pca_graph_declarations(record, training):
    """Constant qualification, configuration identity and units derived from saved settings."""
    support, settings = record["support"], record["support"]["requested"]
    units = training["units"]
    first = next(iter(units.values()))
    if record["method_id"] == "pca.frozen_projection_v1":
        metric = "primary_with_support"
        config = identity(dict(spec=settings, observations=support["observations"]))
        score = "1" if settings["scaling"] == "zscore" else first
        variance = "1" if score == "1" else f"({score})^2"
        expected = {**{k: variance for k in ("covariance", "eigenvalues", "leading_value", "rank_tolerance")},
                    **{k: score for k in ("scores", "scaled_loadings")},
                    **{k: "1" for k in ("fractions", "leading_fraction", "participation_ratio", "eigenvectors", "pc90",
                                        "rank", "roundoff_negative_eigenvalues")}}
    else:
        metric = "exploratory_graph_roughness"
        config = identity(dict(reference=record["reference_id"], settings=settings, observations=support["observations"],
                               support=support["evaluation"]["support_id"]))
        d = settings["pca_components"] or len(settings["features"])
        coordinates = support["coordinate_units"]
        _require(type(d) is int and d >= 1 and isinstance(coordinates,list) and len(coordinates) == d and
                 (all(u == "1" for u in coordinates) if settings["scaling"] == "zscore" else
                  all(u == first for u in coordinates) if settings["pca_components"] is not None else
                  coordinates == list(units.values())), "graph coordinate unit binding")
        mass = settings["measure"] == "mass"
        volume = "*".join(f"({u})" for u in coordinates) if mass else settings["volume_units"]
        energy = "1" if mass else settings["conductance_units"]+"/("+volume+")"
        expected = dict(energy=energy, per_edge=energy, cell_volumes=volume, weights="1" if mass else settings["conductance_units"],
                        node_signal="1" if mass else "1/sqrt("+volume+")")
        expected.update({k: "1" for k in ("probabilities", "observed_masses", "smoothed_masses", "counts", "connected_components",
                         "isolated_mass", "occupied_count", "empty_count", "observed_occupied_count", "observed_empty_count")})
    _require(record["metric_status"] == metric, "native metric qualification declaration")
    _require(record["config_id"] == config, "native configuration identity")
    _require(encode(record["units"]) == encode(expected), "native output unit binding")


def _graph_evaluation(record, frozen):
    support, value = record["support"], record["value"]
    candidate, evaluation = support["candidate"], support["evaluation"]
    eligible = _positions(support["eligible_positions"])
    contributing = _positions(support["contributing_positions"])
    excluded = _positions(support["excluded_positions"])
    outside = _positions(support["outside_positions"])
    _require(eligible == candidate["eligible_positions"] and contributing == evaluation["eligible_positions"] and
             evaluation["requested_positions"] == contributing and set(contributing) <= set(eligible),
             "graph evaluation position binding")
    _require(all(encode(candidate[k]) == encode(evaluation[k]) for k in
                 ("input_id", "features", "units", "basis", "source", "clock", "observations")) and
             encode(support["observations"]) == encode(candidate["observations"]), "graph evaluation declaration binding")
    _counts(support, ("excluded_count", "outside_count", "unrepresented_count"))
    remaining = [p for p in eligible if p not in set(contributing)]
    _require(set(outside) <= set(excluded) <= set(remaining) and
             support["excluded_count"] == len(excluded) and support["outside_count"] == len(outside) and
             support["unrepresented_count"] == len(excluded)-len(outside), "graph exclusion accounting")
    # Failed fitting/projection never reaches partition membership. Otherwise
    # nonempty selections prove membership was evaluated; a fitted full grid
    # also distinguishes a legitimately all-excluded evaluation from no fit.
    before_partition = (support["partition_edges"] is None or record["compute_status"] == "invalid_projection" or
                        frozen is not None and frozen["compute_status"] != "ok")
    if before_partition:
        _require(not contributing and not excluded and not outside and value is None, "graph unavailable partition support")
    elif value is not None or contributing or excluded or support["requested"]["node_policy"] == "full_grid":
        _require(excluded == remaining, "graph exclusion accounting")
    if support["requested"]["outside"] == "raise":
        _require(not excluded, "graph outside refusal policy")
    if support["requested"]["node_policy"] == "full_grid":
        _require(outside == excluded, "graph full-grid exclusion accounting")
    if value is not None:
        _require(len(contributing) >= support["observations"]["min_points"], "graph minimum evaluation support")
        _require(sum(int(v) for v in value["counts"]) == len(contributing), "graph contributing count binding")
        _counts(value, ("observed_occupied_count", "observed_empty_count"))
        _require(value["observed_occupied_count"] == np.count_nonzero(value["counts"]) and
                 value["observed_empty_count"] == np.count_nonzero(value["counts"] == 0), "graph observed occupancy binding")
        settings = support["requested"]
        _require(encode(value["partition_edges"]) == encode(support["partition_edges"]) and
                 value["realized_bins"] == [len(edge)-1 for edge in support["partition_edges"]] and
                 value["node_policy"] == settings["node_policy"] and value["pseudocount_universe"] == "declared_nodes" and
                 support["requested_states"] == len(value["nodes"]), "graph retained geometry binding")
        measure = settings["measure"]
        _require(measure in ("mass", "density") and value["measure"] == measure and
                 record["method_id"] == "graph.state_"+measure+"_v1" and
                 value["method_id"] == "graph.sqrt_"+measure+"_energy_v1", "graph measure/method binding")
        units = "1" if measure == "mass" else settings["conductance_units"]+"/("+settings["volume_units"]+")"
        _require(value["units"] == record["units"]["energy"] == record["units"]["per_edge"] == units and
                 (value["volumes"] is None if measure == "mass" else
                  np.array_equal(value["volumes"],value["cell_volumes"])), "graph measure/unit/volume binding")
        _require(np.array_equal(value["probabilities"],value["smoothed_masses"]), "graph saved probability binding")


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and np.isfinite(value)


def _numbers(record, keys, *, nullable=False):
    _fields(record, keys)
    for key in keys:
        _require(_number(record[key]) or nullable and _missing(record[key]), str(key) + " must be numeric")


def _counts(record, keys):
    _numbers(record, keys)
    _require(all(record[k] >= 0 and int(record[k]) == record[k] for k in keys), "nonnegative integer counts")


def _positions(values):
    _require(isinstance(values,list) and all(type(p) is int and p >= 0 for p in values) and
             values == sorted(set(values)), "ordered unique nonnegative positions")
    return values


def _training_support(training):
    _complete_support(training)
    _fields(training,("position_scope","original_positions","original_eligible_positions"))
    original = _positions(training["original_positions"])
    eligible = training["eligible_positions"]
    _require(training["position_scope"] == "training_subset" and
             training["requested_positions"] == list(range(len(original))) and
             _positions(training["original_eligible_positions"]) == [original[p] for p in eligible],
             "native training position binding")


def _complete_support(support):
    """Check the shared native complete-vector selection and qualification schema."""
    _fields(support, "input_id variables features units basis source clock policy time_compression requested_positions eligible_positions excluded_positions eligible_instants observations use_status support_id".split())
    features, units = support["features"], support["units"]
    _require(isinstance(features,list) and features and all(_text(v) for v in features) and
             len(set(features)) == len(features) and isinstance(units,dict) and list(units) == features and
             all(_text(v) for v in units.values()) and isinstance(support["variables"],dict) and
             list(support["variables"]) == features, "native training feature schema")
    _require(support["basis"] in ("raw","daily_residual") and support["policy"] == "complete_case" and
             support["time_compression"] is False and
             _text(support["input_id"]) and _text(support["support_id"]) and
             isinstance(support["source"],dict) and _text(support["source"].get("id")) and
             isinstance(support["clock"],dict), "native training declarations")
    full = _positions(support["variables"][features[0]]["full_segment"]["positions"])
    requested = _positions(support["requested_positions"])
    eligible = _positions(support["eligible_positions"])
    excluded = _positions(support["excluded_positions"])
    _require(full == list(range(len(full))) and set(requested) <= set(full) and set(eligible) <= set(requested) and
             excluded == [p for p in requested if p not in set(eligible)], "native training position binding")
    source, clock = support["source"], support["clock"]
    _require(all(_text(source.get(k)) for k in (("segment",) if "segment" in source else ("site","station","instrument"))) and
             ("trusted_imputation" not in source or type(source["trusted_imputation"]) is bool), "native training source schema")
    _clock(clock, len(full))
    instants = support["eligible_instants"]
    _require(isinstance(instants,list) and len(instants) == len(eligible), "native training instant dimensions")
    times = [_time(t,strings=True) for t in instants]
    _require(all(a < b for a,b in zip(times,times[1:])), "native training instant order")
    _on_phase(times, clock, "native complete-case instant clock phase")
    observations = support["observations"]
    _observation_policy(observations)
    _require(support["use_status"] in _USE, "native training qualification")
    qualifications = []
    for name, variable in support["variables"].items():
        _fields(variable, "variable input_id basis source clock policy estimator positions selected_instants excluded_positions compute_status reason selected full_segment".split())
        _require(variable["variable"] == name and variable["input_id"] == identity(dict(
                     input=support["input_id"],variable=name,basis=support["basis"])) and
                 all(encode(variable[k]) == encode(support[k]) for k in ("basis","source","clock")) and
                 variable["policy"] == "complete_case" and variable["estimator"] == "point" and
                 _positions(variable["positions"]) == eligible and _positions(variable["excluded_positions"]) == [p for p in full if p not in set(eligible)] and
                 variable["selected_instants"] == instants, "native training variable binding")
        ok = len(eligible) >= observations["min_points"]
        _require(variable["compute_status"] == ("ok" if ok else "insufficient_support") and
                 (variable["reason"] is None if ok else _text(variable["reason"])), "native training selection availability")
        for key, positions in (("selected",eligible),("full_segment",full)):
            burden = variable[key]
            _fields(burden, ("positions","denominator","variables","support_id","weighting","duration_convention"))
            _require(_positions(burden["positions"]) == positions and type(burden["denominator"]) is int and
                     burden["denominator"] >= len(positions) and isinstance(burden["variables"],dict) and
                     list(burden["variables"]) == [name] and _text(burden["support_id"]) and
                     burden["weighting"] == "equal_cells" and
                     burden["duration_convention"] == support["clock"]["duration_convention"], "native training burden binding")
            detail = burden["variables"][name]
            _require(detail["denominator"] == burden["denominator"], "native training burden counts")
            _training_burden(detail,clock,len(positions))
        qualifications.append(_saved_signal_use(variable,observations))
    _require(support["use_status"] == min(qualifications,key=_USE.index), "native training burden/qualification binding")


def _observation_policy(observations):
    _fields(observations, ("values","min_points","max_imputed_fraction","max_reconstructed_duration"))
    _require(observations["values"] in ("provided","observed_only","trusted_imputation") and
             type(observations["min_points"]) is int and observations["min_points"] > 0 and
             (observations["max_imputed_fraction"] is None or _number(observations["max_imputed_fraction"]) and
              0 <= observations["max_imputed_fraction"] <= 1), "native training observation policy")
    if observations["max_reconstructed_duration"] is not None:
        _require(not isinstance(observations["max_reconstructed_duration"],(bool,int,float)) and
                 pd.Timedelta(observations["max_reconstructed_duration"]) > pd.Timedelta(0), "native training duration policy")


def _saved_signal_use(variable, observations, *, metric_status="primary_with_support"):
    """Compare native eligibility with saved counts, never source measurements."""
    name = variable["variable"]
    positions = _positions(variable["positions"])
    selected = variable["selected"]
    detail = selected["variables"][name]
    _require(selected["positions"] == positions and type(selected["denominator"]) is int and
             selected["denominator"] == detail["denominator"] == len(positions) and detail["unavailable_count"] == 0,
             "native training selected denominator")
    _training_burden(detail,variable["clock"],len(positions))
    if observations["values"] != "provided":
        _require(detail["unknown_touched_count"] == 0 and
                 (observations["values"] != "observed_only" or detail["touched_count"] == 0), "native training selected observation policy")
    if not positions:
        use = "insufficient_support"
    elif detail["unknown_touched_count"]:
        use = "unknown_provenance"
    elif metric_status != "primary_with_support":
        use = "sensitivity_only"
    elif observations["max_imputed_fraction"] is not None and detail["primary_gate"]["fraction"] > observations["max_imputed_fraction"]:
        use = "sensitivity_only"
    elif observations["max_reconstructed_duration"] is not None:
        seconds = detail["longest_reconstructed_seconds"]
        use = ("unverified_gap_geometry" if seconds is None else "sensitivity_only" if
               seconds > pd.Timedelta(observations["max_reconstructed_duration"]).value/1e9 else "primary_eligible")
    else:
        use = "primary_eligible"
    if "reference_id" in variable:
        _fields(variable,("reference_id","fit_support","fit_use_status","fit_observations","fit_scope"))
        fitted = variable["fit_support"]
        _require(_text(variable["reference_id"]) and variable["fit_scope"] in ("training_only","self") and
                 fitted.get("reference_id") is None and fitted["variable"] == name, "native residual training fit schema")
        _observation_policy(variable["fit_observations"])
        fit_use = _saved_signal_use(fitted,variable["fit_observations"])
        _require(variable["fit_use_status"] == fit_use, "native residual fit qualification binding")
        if fit_use == "unknown_provenance":
            use = "unknown_provenance"
        elif use == "primary_eligible" and (fit_use != "primary_eligible" or fitted["selected"]["variables"][name]["touched_count"]):
            use = "sensitivity_only"
    return use


def _training_burden(detail, clock, selected_count):
    """Validate the native saved influence/coverage and primary-use records."""
    _counts(detail,("denominator","touched_count","unknown_touched_count","unavailable_count"))
    _require(all(detail[k] <= detail["denominator"] for k in ("touched_count","unknown_touched_count","unavailable_count")),
             "native training burden counts")
    _numbers(detail,("numerator","unknown_numerator"))
    _require(0 <= detail["numerator"] <= detail["touched_count"] and
             0 <= detail["unknown_numerator"] <= detail["unknown_touched_count"], "native training burden numerator")
    for numerator,fraction in (("numerator","fraction"),("unknown_numerator","unknown_fraction")):
        _require(detail[fraction] is None if not detail["denominator"] else
                 _number(detail[fraction]) and detail[fraction] == detail[numerator]/detail["denominator"],
                 "native training burden fraction")
    _fields(detail, "fraction_kind operation_counts linear_influence contributor_coverage touched_cell_coverage primary_gate longest_reconstructed_cells longest_reconstructed_seconds".split())
    denominator = detail["denominator"]
    for key in ("linear_influence","contributor_coverage","touched_cell_coverage","primary_gate"):
        fraction = detail[key]
        _fields(fraction,("numerator","denominator","fraction"))
        _numbers(fraction,("numerator",))
        _counts(fraction,("denominator",))
        _require(0 <= fraction["numerator"] <= fraction["denominator"] <= denominator and
                 (fraction["fraction"] is None if not fraction["denominator"] else
                  _number(fraction["fraction"]) and fraction["fraction"] == fraction["numerator"]/fraction["denominator"]),
                 "native training burden fraction schema")
    linear, coverage, touched, gate = (detail[k] for k in
        ("linear_influence","contributor_coverage","touched_cell_coverage","primary_gate"))
    _require(linear["denominator"]+coverage["denominator"] == denominator and
             linear["numerator"]+coverage["numerator"] == detail["numerator"] and
             touched["denominator"] == gate["denominator"] == denominator and touched["numerator"] == detail["touched_count"] and
             linear["numerator"] <= gate["numerator"] <= min(linear["numerator"]+coverage["denominator"],detail["touched_count"]) and
             gate.get("fraction_kind") == "linear_influence_with_nonlinear_touched_cells", "native training burden component binding")
    kind = ("contributor_coverage" if coverage["denominator"] == denominator and denominator else
            "mixed_influence_and_coverage" if coverage["denominator"] else "linear_influence")
    _require(detail["fraction_kind"] == kind and
             all(detail[k] <= selected_count for k in ("touched_count","unknown_touched_count")), "native training burden interpretation")
    operations = detail["operation_counts"]
    _require(isinstance(operations,dict) and all(_text(k) and type(v) is int and v >= 0 for k,v in operations.items()) and
             sum(operations.values()) == selected_count, "native training operation counts")
    cells, seconds = detail["longest_reconstructed_cells"], detail["longest_reconstructed_seconds"]
    if clock["mode"] == "regular":
        _require(type(cells) is int and 0 <= cells <= detail["touched_count"] and _number(seconds) and
                 bool(cells) == bool(detail["touched_count"]) and
                 seconds == cells*clock["cadence"].value/1e9, "native training reconstruction duration")
    else:
        _require(cells is None and seconds is None, "native training irregular reconstruction duration")


def _evaluation_support(support):
    selected = _positions(support["positions"])
    evaluated = _positions(support.get("evaluation_positions",selected))
    _require(set(evaluated) <= set(selected), "evaluation support subset")
    if "outside" in support:
        outside = support["outside"]
        excluded = _positions(outside["positions"])
        excluded_set = set(excluded)
        _counts(outside, ("below","above"))
        _require(excluded_set <= set(selected) and len(excluded) == outside["below"]+outside["above"] and
                 evaluated == [p for p in selected if p not in excluded_set], "evaluation exclusion binding")
    if "evaluation_burden" in support:
        _require(support["evaluation_burden"]["positions"] == evaluated, "evaluation burden binding")
    return evaluated


def _same(first, second):
    # Native tables may expose a missing numeric cell as NaN while their
    # description keeps None. This applies only to duplicated scalar fields.
    return ((_missing(first) and _missing(second)) or
            (_number(first) and _number(second) and first == second) or encode(first) == encode(second))


def _time(value, *, strings=False, nullable=False):
    if nullable and value is None:
        return None
    _require(type(value) is str if strings else isinstance(value, pd.Timestamp), "native time type")
    parsed = pd.Timestamp(value)
    _require(not pd.isna(parsed) and parsed.tz is not None, "aware nonmissing time")
    return parsed


def _geometry(row, *, strings=False, nullable=False):
    _fields(row, ("start", "end", "label"))
    start, end, label = [_time(row[k], strings=strings, nullable=nullable) for k in ("start", "end", "label")]
    _require(start is None or end is None or start < end, "ordered window geometry")
    _require(label == start or label == end, "window label geometry")
    return start, end


def _timeline(timeline, rows, result_id):
    _fields(timeline, ("source_result_id", "rows", "episodes", "transitions", "max_gap", "noise_policy", "warning"))
    _require(timeline["source_result_id"] == result_id, "timeline result binding")
    _require(isinstance(timeline["max_gap"], pd.Timedelta) and timeline["max_gap"] > pd.Timedelta(0), "timeline gap")
    _require(timeline["noise_policy"] in {"include", "barrier"} and _text(timeline["warning"]), "timeline declarations")
    _require(all(isinstance(timeline[k], pd.DataFrame) for k in ("rows", "episodes", "transitions")), "timeline tables")
    required = {"row_id", "start", "end", "native_label", "display_label", "status", "reason"}
    _require(set(timeline["rows"].columns) == required, "timeline row columns")
    saved = timeline["rows"].to_dict("records")
    _require(len(saved) == len(rows) and all(set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
             for a,b in zip(saved,rows)), "timeline row correspondence")
    by_id = {r["row_id"]:r for r in rows}
    positions = {r["row_id"]:i for i,r in enumerate(rows)}
    _require(all(a["start"] < b["start"] for a,b in zip(rows,rows[1:])), "timeline ordered support")
    included = [r["row_id"] for r in rows if r["status"] == "assigned" or r["status"] == "noise" and timeline["noise_policy"] == "include"]
    episode_ids = []
    episodes = timeline["episodes"].to_dict("records")
    for episode in episodes:
        _fields(episode, ("row_ids", "native_label", "status", "start", "end", "covered_seconds", "elapsed_span_seconds"))
        ids = episode["row_ids"]
        _require(isinstance(ids, list) and ids and all(_text(i) and i in by_id for i in ids), "timeline episode row IDs")
        members = [by_id[i] for i in ids]
        _require([positions[i] for i in ids] == list(range(positions[ids[0]],positions[ids[-1]]+1)), "timeline episode barrier")
        coverage_end = members[0]["end"]
        covered_ns = coverage_end.value-members[0]["start"].value
        for member in members[1:]:
            _require(member["start"] - coverage_end <= timeline["max_gap"], "timeline episode gap")
            covered_ns += max(0,member["end"].value-max(member["start"],coverage_end).value)
            coverage_end = max(coverage_end,member["end"])
        _require(all(r["status"] == episode["status"] and _same(r["native_label"],episode["native_label"]) for r in members), "timeline episode labels")
        start, end = _time(episode["start"]), _time(episode["end"])
        _require(start == members[0]["start"] and end == max(r["end"] for r in members), "timeline episode geometry")
        _numbers(episode, ("covered_seconds", "elapsed_span_seconds"))
        _require(0 <= episode["covered_seconds"] <= episode["elapsed_span_seconds"] and
                 episode["covered_seconds"] == covered_ns/1e9 and
                 episode["elapsed_span_seconds"] == (end.value-start.value)/1e9, "timeline episode duration")
        episode_ids.extend(ids)
    _require(episode_ids == included, "timeline episode completeness")
    transitions = timeline["transitions"].to_dict("records")
    _unique(transitions, ("from_row_id", "to_row_id"))
    expected_links = []
    for first, second in zip(episodes,episodes[1:]):
        a, b = first["row_ids"][-1], second["row_ids"][0]
        continuous = positions[b] == positions[a]+1 and second["start"]-first["end"] <= timeline["max_gap"]
        if continuous:
            _require(first["status"] != second["status"] or not _same(first["native_label"],second["native_label"]),
                     "timeline continuous episode split")
            expected_links.append((a,b))
    _require([(r["from_row_id"],r["to_row_id"]) for r in transitions] == expected_links, "timeline transition barriers/gaps/completeness")
    for transition in transitions:
        _fields(transition, ("from_row_id", "to_row_id", "from_label", "to_label", "lower", "upper", "meaning"))
        a, b = transition["from_row_id"], transition["to_row_id"]
        _require(a in included and b in included and included.index(a) < included.index(b), "timeline transition row IDs")
        first, second = by_id[a], by_id[b]
        _require(_same(transition["from_label"],first["native_label"]) and _same(transition["to_label"],second["native_label"]), "timeline transition labels")
        _require(_time(transition["lower"]) == first["start"] and
                 _time(transition["upper"]) == max(first["end"],second["end"]) and _text(transition["meaning"]), "timeline transition geometry")


def _array(value, *, ndim=1, integer=False, finite=False):
    _require(isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in ("iu" if integer else "iuf"), "numeric array type/dimensions")
    if finite:
        _require(np.isfinite(value).all(), "finite numeric array required")


def _masses(value):
    _array(value, finite=True)
    _require(len(value) > 0 and np.all(value >= 0) and np.isclose(value.sum(),1,rtol=0,atol=1e-12), "normalized nonnegative masses")


def _bin_edges(value):
    _array(value, finite=True)
    with np.errstate(over="ignore", invalid="ignore"):
        widths = np.diff(value)
    _require(len(value) >= 2 and np.isfinite(widths).all() and np.all(widths > 0), "finite increasing bin edges")


def _numeric_frame(frame, columns=None):
    _require(isinstance(frame, pd.DataFrame), "numeric table")
    columns = list(frame) if columns is None else columns
    _require(set(columns) <= set(frame), "numeric table columns")
    for column in columns:
        _require(all(_number(v) for v in frame[column]), "numeric table values")


def _spectrum(value, *, welch=False):
    _fields(value, ("frequency_hz", "psd", "physical_frequency_hz", "physical_psd") if welch else ("frequency_hz", "psd"))
    for prefix in ("", "physical_"):
        if prefix and not any(prefix+k in value for k in ("frequency_hz", "psd")):
            continue
        frequency, power = value[prefix+"frequency_hz"], value[prefix+"psd"]
        _array(frequency)
        _array(power)
        _require(frequency.shape == power.shape, "spectrum array dimensions")
        _require(np.isfinite(frequency).all() and np.all(frequency >= 0) and np.all(np.diff(frequency) > 0), "spectrum frequency geometry")
        _require(not np.any(power < 0), "spectrum nonnegative PSD")
        if welch and not prefix:
            _require(np.isfinite(power).all(), "Welch successful spectrum must be finite")


def _welch_geometry(support, value):
    """Saved Welch grids; `value` is the spectrum itself, never a summary value."""
    _fields(support, ("requested","resolved","raw_segments"))
    _counts(support, ("raw_segments",))
    resolved = support["resolved"]
    if resolved is None:
        _require(value is None and type(support["raw_segments"]) is int and support["raw_segments"] == 0, "Welch unavailable geometry")
        return
    _counts(resolved, ("nperseg","nfft","overlap","hop"))
    _numbers(resolved, ("sampling_hz","physical_resolution_hz","display_spacing_hz","theoretical_nyquist_hz"))
    length, fft, sampling = resolved["nperseg"], resolved["nfft"], resolved["sampling_hz"]
    _require(all(type(resolved[k]) is int for k in ("nperseg","nfft","overlap","hop")) and length >= 2 and fft >= length and
             0 <= resolved["overlap"] < length and resolved["hop"] == length-resolved["overlap"] and sampling > 0, "Welch FFT geometry")
    # The producer's sampling frequency is exactly 1e9 / declared cadence in ns.
    _require(sampling == 1e9/support["clock"]["cadence"].value, "Welch sampling/cadence binding")
    requested = support["requested"]
    for key, default in (("nperseg",length),("nfft",length),("overlap",length//2)):
        _require(resolved[key] == (default if requested[key] is None else requested[key]), "Welch resolved geometry")
    # Bound generated geometry by the arrays already present in the payload,
    # not by an untrusted FFT size. Unavailable results need only endpoints.
    positive = resolved["physical_positive_frequencies_hz"]
    _array(positive,finite=True)
    _require(len(positive) == length//2, "Welch physical grid dimensions")
    if value is not None:
        _require(len(value["frequency_hz"]) == fft//2+1 and
                 len(value["physical_frequency_hz"]) == length//2+1, "Welch full frequency grid dimensions")
    physical = np.arange(len(positive)+1)*(1/(length*(1/sampling)))
    display_spacing = 1/(fft*(1/sampling))
    display_end = (fft//2)*display_spacing
    for key, expected in (("physical_resolution_hz",sampling/length),("display_spacing_hz",sampling/fft),
                          ("theoretical_nyquist_hz",sampling/2)):
        _require(np.isclose(resolved[key],expected,rtol=1e-12,atol=0), "Welch frequency spacing")
    _require(encode(resolved["physical_positive_frequencies_hz"]) == encode(physical[1:]) and
             resolved["physical_positive_range_hz"] == [physical[1],physical[-1]] and
             resolved["display_positive_range_hz"] == [display_spacing,display_end], "Welch declared frequency grid")
    lower, upper = physical[1], min(physical[-1],display_end)
    interval = [lower,upper] if support["raw_segments"] and lower <= upper else None
    _require(_same(resolved["supported_positive_range_hz"],interval), "Welch supported frequency interval")
    if value is not None:
        _require(support["raw_segments"] > 0, "Welch available segment count")
        display = np.arange(len(value["frequency_hz"]))*display_spacing
        for key, expected in (("frequency_hz",display),("physical_frequency_hz",physical)):
            _require(encode(value[key]) == encode(expected), "Welch full frequency grid binding")
        if fft == length:
            _require(encode(value["psd"]) == encode(value["physical_psd"]), "Welch equal-grid PSD binding")


def _irregular_frequencies(support, value):
    """Per-ordinate support; returns the number of available pooled ordinates."""
    _fields(support, ("requested", "resolved", "frequency_support"))
    frequency = support["requested"]["frequency_hz"]
    _array(frequency, finite=True)
    _require(len(frequency) > 0 and np.all(frequency > 0) and np.all(np.diff(frequency) > 0), "irregular frequency geometry")
    _require(encode(frequency) == encode(support["resolved"]["frequency_hz"]), "irregular frequency settings binding")
    rows = support["frequency_support"]
    _require(isinstance(rows,list) and len(rows) == len(frequency), "irregular frequency support dimensions")
    if value is not None:
        _require(encode(value["frequency_hz"]) == encode(frequency), "irregular output frequency binding")
    available = 0
    for i, row in enumerate(rows):
        _fields(row, ("frequency_hz", "compute_status", "reason", "psd", "raw_segments"))
        _counts(row, ("raw_segments",))
        ok = row["raw_segments"] >= 2
        _require(row["frequency_hz"] == frequency[i] and
                 row["compute_status"] == ("ok" if ok else "insufficient_segments") and
                 (row["reason"] is None if ok else _text(row["reason"])), "irregular ordinate status binding")
        _require(_number(row["psd"]) if ok else row["psd"] is None, "irregular ordinate availability")
        if value is not None:
            _require(_same(value["psd"][i],row["psd"]), "irregular output/support ordinate binding")
        available += ok
    return available


_FIT_POSITIONS = ("positions", "outside_interval", "nonpositive_frequency", "nonpositive_or_nonfinite_power")


def _fit_selection(selection):
    """Saved log-fit ordinate selection schema (value copy or summary support)."""
    _fields(selection, (*_FIT_POSITIONS, "inherited_range_hz", "frequency_exclusions", "resolved_interval_hz"))
    for key in _FIT_POSITIONS:
        positions = selection[key]
        _require(isinstance(positions,list) and all(type(p) is int and p >= 0 for p in positions) and
                 positions == sorted(set(positions)), "fit selection positions")
    for key in ("inherited_range_hz", "resolved_interval_hz"):
        bounds = selection[key]
        if bounds is not None:
            _require(isinstance(bounds,(list,np.ndarray)) and len(bounds) == 2 and
                     all(_number(v) and v > 0 for v in bounds) and bounds[0] <= bounds[1], "fit selection bounds")
    _require(isinstance(selection["frequency_exclusions"],list), "fit frequency exclusions")
    for row in selection["frequency_exclusions"]:
        _fields(row, ("position", "reason"))
        _counts(row, ("position",))
        _require(row["reason"] == "outside_frequency_support", "fit frequency exclusion reason")


def _fit(value):
    _fields(value, ("compute_status", "reason", "single", "broken", "candidates", "selection", "preferred", "delta_aicc"))
    ok = value["compute_status"] == "ok"
    _require(value["compute_status"] in {"ok", "insufficient_support"} and
             (value["reason"] is None if ok else _text(value["reason"])), "fit status/reason")
    _require(isinstance(value["candidates"],list) and
             (isinstance(value["single"],dict) if ok else value["single"] is None and
              value["broken"] is None and not value["candidates"]), "fit availability")
    selection = value["selection"]
    _fit_selection(selection)
    _require(value["preferred"] is None or value["preferred"] in {"single","broken"}, "fit preferred model")
    _numbers(value, ("delta_aicc",), nullable=True)
    if ok:
        _fields(value, ("resolved_candidates_hz", "comparison_status", "semantics"))
        candidates = value["resolved_candidates_hz"]
        _array(candidates)
        _require(np.isfinite(candidates).all() and np.all(candidates > 0) and
                 np.all(np.diff(candidates) > 0) and len(candidates) == len(value["candidates"]), "fit candidate geometry")
        _require(_text(value["semantics"]) and len(selection["positions"]) == value["single"]["n"], "fit selection count")
    else:
        _require(value["preferred"] is None and value["delta_aicc"] is None, "unavailable fit comparison")
    for fit in [value["single"], value["broken"], *value["candidates"]]:
        if fit is None:
            continue
        if fit.get("comparison_status") == "insufficient_arm_points":
            _numbers(fit, ("break_hz", "low_n", "high_n"))
            _counts(fit, ("low_n", "high_n"))
            _require(fit["break_hz"] > 0 and fit["low_n"]+fit["high_n"] == value["single"]["n"], "fit arm domain/count")
            continue
        _numbers(fit, ("n", "k", "rss", "intercept", "beta_low", "beta_high"))
        _numbers(fit, ("r2", "break_hz", "break_period_seconds"), nullable=True)
        _numbers(fit, ("aicc",), nullable=True)
        _fields(fit, ("comparison_status", "r2_reason"))
        _counts(fit, ("n", "k"))
        _require(fit["rss"] >= 0 and fit["n"] >= 3 and fit["k"] in {3,5} and
                 (fit["r2"] is None or fit["r2"] <= 1+1e-12) and
                 (fit["break_hz"] is None and fit["break_period_seconds"] is None or
                  fit["break_hz"] is not None and fit["break_hz"] > 0 and
                  fit["break_period_seconds"] is not None and fit["break_period_seconds"] > 0), "fit numerical domain")
        _require(fit["comparison_status"] in {"ok", "exact_fit", "insufficient_aicc_points"} and
                 (fit["aicc"] is None if fit["comparison_status"] != "ok" else _number(fit["aicc"])), "fit AICc availability")
        _require(_text(fit["r2_reason"]) if fit["r2"] is None else fit["r2_reason"] is None, "fit R2 reason")
    if ok:
        _require(all(row["break_hz"] == frequency for row,frequency in zip(value["candidates"],candidates)), "fit candidate binding")
        full = [row for row in value["candidates"] if row["comparison_status"] != "insufficient_arm_points"]
        _require((value["broken"] is not None) == bool(full) and (not full or
                 any(encode(value["broken"]) == encode(row) for row in full)), "fit retained broken candidate")
        available = value["broken"] is not None and value["single"]["aicc"] is not None and value["broken"]["aicc"] is not None
        _require(value["comparison_status"] == ("descriptive_working_aicc" if available else "unavailable") and
                 (_number(value["delta_aicc"]) and value["preferred"] is not None if available else
                  value["delta_aicc"] is None and value["preferred"] is None), "fit comparison availability")


def _distribution(value):
    _numbers(value, ("fi1", "entropy_nats", "occupied_bins", "eligible_count", "pseudocount"))
    _require(value["fi1"] >= 0, "nonnegative FI1")
    # The native mass primitive permits normalization error up to 1e-12.
    _require(value["entropy_nats"] >= -1e-12, "FI1 entropy domain")
    for key in ("probabilities", "counts", "edges"):
        _array(value[key], integer=key == "counts", finite=True)
    _require(len(value["probabilities"]) == len(value["counts"]) == len(value["edges"])-1, "distribution dimensions")
    _masses(value["probabilities"])
    _bin_edges(value["edges"])
    _counts(value, ("occupied_bins", "eligible_count"))
    _require(np.all(value["counts"] >= 0) and value["counts"].sum() == value["eligible_count"] and
             np.count_nonzero(value["counts"]) == value["occupied_bins"] and value["pseudocount"] >= 0, "distribution counts")


def _coherence_status(data, statuses):
    _fields(data, ("compute_status", "reason"))
    status = data["compute_status"]
    _require(_text(status) and status in statuses, "coherence compute status")
    ok = status == "ok"
    _require(data["reason"] is None if ok else _text(data["reason"]), "coherence status/reason")
    return ok


def _coherence_summary(data):
    ok = _coherence_status(data, {"ok", "insufficient_support", "insufficient_band_support", "undefined_spectrum"})
    _fields(data, ("band_mean",))
    _require(_number(data["band_mean"]) if ok else data["band_mean"] is None, "coherence band mean availability")
    return ok


def _coherence_values(data, frequency=None, *, required=(), allow_padding=False):
    _fields(data, required)
    if "frequency_hz" in data:
        frequency = data["frequency_hz"]
        _array(frequency, finite=True)
        _require(np.all(frequency >= 0) and np.all(np.diff(frequency) > 0), "coherence frequency geometry")
        if "physical_spacing_hz" in data:
            _numbers(data, ("physical_spacing_hz",))
            spacing = data["physical_spacing_hz"]
            _require(spacing > 0, "coherence physical grid spacing")
            if "run_id" in data:
                _counts(data, ("nperseg", "nfft"))
                _numbers(data, ("sampling_hz",))
                length, nfft, sampling = data["nperseg"], data["nfft"], data["sampling_hz"]
                _require(length >= 4 and length % 2 == 0 and nfft >= length and nfft % length == 0 and sampling > 0 and
                         (allow_padding or nfft == length) and len(frequency) == nfft//2+1,
                         "coherence FFT grid geometry")
                _require(np.isclose(spacing,sampling/length,rtol=1e-12,atol=spacing*1e-12), "coherence physical resolution")
                spacing = sampling/nfft
            _require(np.allclose(np.diff(frequency),spacing,rtol=1e-12,atol=spacing*1e-12), "coherence display grid spacing")
        if "run_id" in data and len(frequency):
            _require(len(frequency) >= 3 and frequency[0] == 0, "coherence run grid")
    for key in ("sxx", "syy", "sxy_real", "sxy_imag", "raw_coherence", "phase_rad", "corrected", "untruncated_corrected"):
        if key in data:
            if key in {"corrected", "untruncated_corrected"} and data[key] is None:
                continue
            _array(data[key])
            if key in {"sxx", "syy"}:
                _require(not np.any(data[key] < 0), "coherence nonnegative auto spectra")
            if key in {"raw_coherence", "corrected"}:
                ordinates = data[key]
                _require(not np.isinf(ordinates).any() and np.all(~np.isfinite(ordinates) | ((ordinates >= 0) & (ordinates <= 1))), "coherence ordinate domain")
            if key == "phase_rad":
                phase = data[key]
                _require(not np.isinf(phase).any() and np.all(np.isnan(phase) | (np.abs(phase) <= np.pi)), "coherence phase domain")
            # The report estimator retains a proposed short grid without
            # estimating its ordinates when fewer than three bins are available.
            unavailable = (key == "raw_coherence" and len(data[key]) == 0 and
                data.get("method_id") == "coherence.report_equal_run_grid_v1" and
                data.get("compute_status") == "insufficient_band_support" and
                frequency is not None and len(frequency) < 3)
            _require(frequency is not None and (data[key].shape == frequency.shape or unavailable), "coherence ordinate dimensions")
    if {"raw_coherence", "sxx", "syy"} <= data.keys():
        available = np.isfinite(data["raw_coherence"])
        _require(all(np.all(np.isfinite(data[k][available]) & (data[k][available] > 0))
                     for k in ("sxx", "syy")), "coherence raw/auto-spectrum availability")
    for key in ("raw_threshold", "corrected_threshold", "band_mean", "significant_fraction"):
        if key in data:
            _numbers(data, (key,), nullable=True)
            _require(_missing(data[key]) or 0 <= data[key] <= 1, "coherence scalar domain")
    if data.get("significant") is not None:
        flags = data["significant"]
        _require(isinstance(flags, (list, np.ndarray)) and
                 (not isinstance(flags, np.ndarray) or flags.ndim == 1) and
                 all(_missing(v) or isinstance(v, (bool, np.bool_)) for v in flags), "coherence decision flags")
        _require(frequency is not None and len(flags) == len(frequency), "coherence decision dimensions")
    if "inference_status" in data:
        _fields(data, _COHERENCE_INFERENCE)
        _numbers(data, ("alpha",))
        _require(data["inference_status"] in {"unavailable", "approximate_white_noise"}, "coherence inference status")
        _require(0 < data["alpha"] < 1 and data["null"] in {"gaussian_white", "colored", "unspecified"} and
                 data["decision_scale"] == "raw_strict_greater" and data["multiplicity"] == "none_pointwise_only", "coherence inference declarations")
        _require((data["corrected"] is None) == (data["untruncated_corrected"] is None), "coherence correction availability")
        if data["inference_status"] == "unavailable":
            _require(_text(data["reason"]) and all(data[k] is None for k in
                     ("raw_threshold", "corrected_threshold", "significant")), "unavailable coherence inference")
        else:
            _numbers(data, ("raw_threshold", "corrected_threshold"))
            _require(data["reason"] is None and data["null"] == "gaussian_white" and
                     all(data[k] is not None for k in ("corrected", "untruncated_corrected", "raw_threshold", "corrected_threshold", "significant")),
                     "available coherence inference")


def _coherence_pool_support(pool, support):
    """Check retained geometry and availability, without pooling or inference."""
    _fields(pool, ("settings", "band_positions", "inference_positions", "physical_band_ordinates"))
    settings, effective, inference = pool["settings"], pool["effective"], pool["inference"]
    _require(encode(settings) == encode(support["requested"]) == encode(support["resolved"]), "coherence settings binding")
    frequency, raw = pool["spectra"]["frequency_hz"], pool["spectra"]["raw_coherence"]
    accepted = sorted((r for r in support["runs"] if r["compute_status"] == "ok"), key=lambda r:r["run_id"])
    _require(accepted and all(r["nfft"] == settings["nfft"] and r["nperseg"] == settings["nperseg"] and
             encode(r["frequency_hz"]) == encode(frequency) for r in accepted), "coherence pool/run grid binding")
    short, long = settings["period_band"]
    _require(isinstance(short,pd.Timedelta) and isinstance(long,pd.Timedelta) and
             pd.Timedelta(0) < short < long, "coherence period band")
    band = [i for i,f in enumerate(frequency) if 0 < i < len(frequency)-1 and 1e9/long.value <= f <= 1e9/short.value]
    factor = settings["nfft"] // settings["nperseg"]
    _require(encode(pool["band_positions"]) == encode(band) and
             encode(pool["inference_positions"]) == encode([i for i in band if i % factor == 0]), "coherence band positions")
    _counts(pool, ("physical_band_ordinates",))
    # Count the declared physical grid, not the padded display ordinates.
    physical = np.arange(1,settings["nperseg"]//2) * (1/(settings["nperseg"]*(1/accepted[0]["sampling_hz"])))
    _require(pool["physical_band_ordinates"] == np.count_nonzero((physical >= 1e9/long.value) & (physical <= 1e9/short.value)),
             "coherence physical band count")
    expected = ("insufficient_band_support" if pool["physical_band_ordinates"] < 3 else
                "undefined_spectrum" if not np.isfinite(raw[band]).all() else "ok")
    _require(pool["compute_status"] == expected and
             (expected != "ok" or len(band) >= 3), "coherence band availability")
    _numbers(effective, ("n_eff", "approximate_dof"))
    _counts(effective, ("raw_segments",))
    _fields(effective, ("normalized_segment_weights",))
    weights = effective["normalized_segment_weights"]
    _require(isinstance(weights,list) and len(weights) == len(accepted) and
             pool["run_ids"] == [r["run_id"] for r in accepted], "coherence weighted run support")
    for run, weight in zip(accepted,weights):
        _counts(run, ("raw_segments", "points", "hop"))
        _array(weight,finite=True)
        _require(1 <= run["hop"] <= run["nperseg"] and run["points"] >= run["nperseg"] and
                 run["raw_segments"] == 1+(run["points"]-run["nperseg"])//run["hop"] and
                 isinstance(run["segment_starts"],list) and len(run["segment_starts"]) == len(weight) == run["raw_segments"] and
                 all(type(start) is int and start == i*run["hop"] for i,start in enumerate(run["segment_starts"])) and
                 np.all(weight > 0), "coherence segment support")
    _require(effective["raw_segments"] == sum(len(w) for w in weights) and
             np.isclose(sum(w.sum() for w in weights),1,rtol=0,atol=1e-12), "coherence effective segment count/weights")
    neff = effective["n_eff"]
    _require(neff > 0 and effective["raw_segments"] >= 1 and effective["approximate_dof"] == 2*neff,
             "coherence effective averages domain")
    _require(neff <= effective["raw_segments"] or np.isclose(neff,effective["raw_segments"],rtol=1e-12,atol=0),
             "coherence effective averages exceed segment support")
    corrected = neff > 1
    _require((inference["corrected"] is not None) == corrected and
             (inference["inference_status"] == "approximate_white_noise") == (corrected and settings["null"] == "gaussian_white") and
             inference["null"] == settings["null"] and inference["alpha"] == settings["alpha"],
             "coherence effective inference availability")
    physical_bins = np.array([0 < i < len(frequency)-1 and i % factor == 0 for i in range(len(frequency))])
    available = physical_bins & np.isfinite(raw)
    if corrected:
        for key in ("corrected", "untruncated_corrected"):
            values = inference[key]
            _require(np.isfinite(values[available]).all() and np.isnan(values[~available]).all(), "coherence physical-bin correction availability")
    if inference["significant"] is not None:
        _require(all(isinstance(flag,(bool,np.bool_)) if ok else _missing(flag)
                     for flag,ok in zip(inference["significant"],available)), "coherence physical-bin decision availability")


def _native_value(method, value):
    # This finite dispatch validates existing scientific payloads. Unknown
    # caller-created records still use the documented closed typed vocabulary.
    if method.startswith("individual."):
        _require(_number(value), "individual scalar value")
        if method in {"individual.sample_sd_v1", "individual.iqr_v1", "individual.population_variance_v1"}:
            _require(value >= 0, "individual nonnegative dispersion")
        elif method == "individual.exact_lagged_pearson_v1":
            _require(-1 <= value <= 1, "individual correlation domain")
    elif method in {"spectral.welch_mean_v1", "spectral.ls_projection_density_wosa_v1"}:
        _spectrum(value, welch=method == "spectral.welch_mean_v1")
    elif method in ("spectral.log_models_v1", "spectral.irregular_log_models_v1"):
        _fit(value)
    elif method == "spectral.peaks_v1":
        _numeric_frame(value["table"], ["rank", "position", "frequency_hz", "period_seconds", "power", "prominence", "cycles"])
        rows = value["table"].to_dict("records")
        _require([row["rank"] for row in rows] == list(range(1,len(rows)+1)), "peak consecutive ranks")
        _fields(value, ("rejected", "eligible_positions", "excluded_positions", "empty_reason"))
        _require(isinstance(value["rejected"],list) and
                 (value["empty_reason"] is None if rows else _text(value["empty_reason"])), "peak availability")
        for key in ("eligible_positions", "excluded_positions"):
            positions = value[key]
            _require(isinstance(positions,list) and all(type(p) is int and p >= 0 for p in positions) and
                     positions == sorted(set(positions)), "peak support positions")
        _require(not set(value["eligible_positions"]) & set(value["excluded_positions"]), "peak support partition")
        for row in rows+value["rejected"]:
            _numbers(row, ("position", "frequency_hz", "period_seconds", "power", "prominence", "cycles"))
            _counts(row, ("position",))
            _require(row["position"] in value["eligible_positions"] and row["frequency_hz"] > 0 and
                     row["period_seconds"] > 0 and np.isclose(row["period_seconds"]*row["frequency_hz"],1,rtol=1e-12,atol=0) and
                     all(row[k] >= 0 for k in ("power","prominence","cycles")), "peak numerical domain/geometry")
        _unique(rows+value["rejected"], ("position",))
        _require(all(row.get("reason") in {"outside_period_range","insufficient_cycles","max_peaks"} for row in value["rejected"]), "peak rejection reason")
        _require([(row["prominence"],row["frequency_hz"]) for row in rows] ==
                 sorted(((row["prominence"],row["frequency_hz"]) for row in rows),key=lambda pair:(-pair[0],pair[1])), "peak ordering")
    elif method == "spectral.band_power_v1":
        _numbers(value, ("power", "positive_frequency_power"))
        _numbers(value, ("fraction",), nullable=True)
        _fields(value, ("fraction_status",))
        power, total, fraction = value["power"], value["positive_frequency_power"], value["fraction"]
        _require(0 <= power <= total*(1+1e-12), "band nonnegative power domain")
        _require((fraction is None and value["fraction_status"] == "undefined_zero_denominator") if total == 0 else
                 (_number(fraction) and 0 <= fraction <= 1+1e-12 and value["fraction_status"] == "ok"), "band fraction availability/domain")
    elif method.startswith("harmonic."):
        _numbers(value, ("g",) if "g" in value else ("T",))
        _numbers(value, ("p_value", "critical", "alpha"))
        _require(isinstance(value["rejected"], (bool, np.bool_)), "harmonic decision")
        statistic = "g" if method == "harmonic.fisher_g_exact_v1" else "T"
        _fields(value, (statistic,"critical_scale"))
        _require(0 <= value[statistic] <= 1+(0 if statistic == "g" else 1e-12) and
                 0 <= value["p_value"] <= 1 and 0 <= value["critical"] <= 1+1e-12 and
                 0 < value["alpha"] < 1 and value["critical_scale"] == statistic, "harmonic probability domain")
        if statistic == "T":
            _numbers(value, ("lambda_",))
            _require(0 < value["lambda_"] < 1, "harmonic lambda domain")
    elif method.startswith("pca."):
        for key in ("eigenvalues", "fractions"):
            _array(value[key], finite=True)
        for key in ("covariance", "eigenvectors", "scaled_loadings"):
            _array(value[key], ndim=2, finite=True)
        _masses(value["fractions"])
        _require(np.all(value["eigenvalues"] >= 0) and np.all(np.diff(value["eigenvalues"]) <= 0), "PCA eigenvalue domain/order")
        _numeric_frame(value["scores"])
        _numbers(value, ("leading_value", "leading_fraction", "pc90", "participation_ratio", "rank", "rank_tolerance", "roundoff_negative_eigenvalues"))
        _counts(value, ("pc90", "rank", "roundoff_negative_eigenvalues"))
        n = len(value["eigenvalues"])
        _require(value["leading_value"] == value["eigenvalues"][0] > 0 and
                 value["leading_fraction"] == value["fractions"][0] and 0 < value["leading_fraction"] <= 1 and
                 1 <= value["pc90"] <= n and 1 <= value["rank"] <= n and value["rank_tolerance"] > 0 and
                 value["roundoff_negative_eigenvalues"] <= n and 1-1e-12 <= value["participation_ratio"] <= n*(1+1e-12),
                 "PCA summary domain/binding")
        _require(isinstance(value["rank_deficient"], (bool, np.bool_)), "PCA rank flag")
    elif method.startswith("fi1."):
        _distribution(value)
    elif method.startswith("density_fisher."):
        _numbers(value, ("J_coordinate", "J_raw", "center", "scale", "evaluation_count"))
        _counts(value, ("evaluation_count",))
        _require(value["J_coordinate"] >= 0 and value["J_raw"] >= 0 and value["scale"] > 0,
                 "density nonnegative information/positive scale")
        for key in ("widths", "centers", "heights", "counts", "probabilities", "edges", "points", "density", "derivative", "fisher_integrand"):
            if key in value:
                _array(value[key], integer=key == "counts", finite=True)
        _numbers(value, ("tail_mass",), nullable=True)
        _require(value["tail_mass"] is None or 0 <= value["tail_mass"] <= 1+1e-12, "density tail mass domain")
        if "counts" in value:
            n = len(value["counts"])
            _require(all(len(value[k]) == n for k in ("widths", "centers", "heights", "probabilities"))
                     and len(value["edges"]) == n+1, "histogram dimensions")
            _masses(value["probabilities"])
            _bin_edges(value["edges"])
            _require(np.all(value["counts"] >= 0) and np.all(value["widths"] > 0) and
                     np.all(np.diff(value["centers"]) > 0) and np.all(value["heights"] >= 0), "histogram geometry domain")
        else:
            _require(all(len(value[k]) == len(value["points"]) for k in ("density", "derivative", "fisher_integrand")), "KDE dimensions")
            _require(np.all(np.diff(value["points"]) > 0) and np.all(value["density"] >= 0) and
                     np.all(value["fisher_integrand"] >= 0), "KDE geometry domain")
    elif method.startswith("graph."):
        _numbers(value, ("energy", "connected_components", "isolated_mass", "occupied_count", "empty_count"))
        _counts(value, ("connected_components", "occupied_count", "empty_count"))
        _require(value["energy"] >= 0 and 0 <= value["isolated_mass"] <= 1+1e-12, "graph energy/mass domain")
        for key in ("probabilities", "weights", "node_signal", "per_edge", "cell_volumes", "counts", "observed_masses", "smoothed_masses"):
            _array(value[key], integer=key == "counts", finite=True)
        for key in ("edges", "nodes"):
            _array(value[key], ndim=2, integer=True)
        if value["volumes"] is not None:
            _array(value["volumes"], finite=True)
        for edges in value["partition_edges"]:
            _bin_edges(edges)
        n, e = len(value["nodes"]), len(value["edges"])
        _require(1 <= value["connected_components"] <= n and value["occupied_count"]+value["empty_count"] == n,
                 "graph node count domain")
        _require(value["edges"].shape == (e,2) and
                 all(len(value[k]) == n for k in ("probabilities", "node_signal", "cell_volumes", "counts", "observed_masses", "smoothed_masses")) and
                 all(len(value[k]) == e for k in ("weights", "per_edge")), "graph geometry dimensions")
        for key in ("probabilities", "observed_masses", "smoothed_masses"):
            _masses(value[key])
        _require(all(np.all(value[k] >= 0) for k in ("weights", "node_signal", "per_edge", "counts")) and
                 np.all(value["cell_volumes"] > 0), "graph numeric domain")
        if value["volumes"] is not None:
            _require(len(value["volumes"]) == n and np.all(value["volumes"] > 0), "graph volume geometry")
        edges = value["edges"]
        _require(np.all(edges >= 0) and np.all(edges < n), "graph edge range")
        pairs = [tuple(sorted(pair)) for pair in edges.tolist()]
        _require(all(a != b for a,b in pairs) and len(set(pairs)) == len(pairs), "graph unique non-self edges")
        nodes, partitions = value["nodes"], value["partition_edges"]
        _require(nodes.shape[1] == len(partitions) and len(set(map(tuple,nodes.tolist()))) == n and
                 all(np.all(nodes[:,j] >= 0) and np.all(nodes[:,j] < len(partition)-1)
                     for j,partition in enumerate(partitions)), "graph node geometry")
    elif method.startswith("network."):
        _numbers(value, ("mean_absolute_connectivity", "strong_count", "density"))
        _counts(value, ("strong_count",))
        _require(all(0 <= value[k] <= 1 for k in ("mean_absolute_connectivity","density")), "network summary domain")
    elif method.startswith("coherence."):
        fields = (_COHERENCE_SPECTRA + ["phase_rad", "band_mean", "significant_fraction"] + _COHERENCE_INFERENCE
                  if method in {"coherence.pooled_spectra_v1", "coherence.historical_modern_unavailable"} else _COHERENCE_REPORT)
        _coherence_values(value, required=fields)
        _numbers(value, ("band_mean",))
        if "significant_fraction" in value:
            _require(_number(value["significant_fraction"]) if value["inference_status"] == "approximate_white_noise"
                     else value["significant_fraction"] is None, "coherence significant fraction availability")


def _missing(value):
    return value is None or value is pd.NA or isinstance(value, (float, np.floating)) and np.isnan(value)


def _scalar_status(record, key):
    _status(record)
    _fields(record, (key,))
    _require(_number(record[key]) if record["compute_status"] == "ok" else _missing(record[key]), key + " status/value mismatch")


def _unique(rows, keys):
    seen = set()
    for row in rows:
        _fields(row, keys)
        key = dumps(encode(tuple(row[k] for k in keys)))
        _require(key not in seen, "duplicate primary key")
        seen.add(key)


def validate_keys(tables):
    for rows in tables.values():
        _unique(rows, ("result_id", "row_key"))
        for row in rows:
            _require(_text(row["metric_id"]), "metric identity must be text")
            for key in ("compute_status", "metric_status", "use_status", "reason", "representation", "config_id", "reference_id"):
                _require(row[key] is None or _text(row[key]), key + " must be text or null")
            units = row["units"]
            _require(units is None or _text(units) or isinstance(units, dict) and all(_text(v) for v in units.values()), "unit symbol or mapping")


def validate_item(item, sources, references):
    kind = item["kind"]
    _require(set(item) == {"kind", "sources", "labels", "record", *_EXTRA[kind]}, "adapter fields")
    names = item["sources"]
    _require(isinstance(names, list) and names and all(_text(s) for s in names), "source names")
    if len(set(names)) != len(names) or any(s not in sources for s in names):
        raise ValueError("result/source binding mismatch")
    _require(isinstance(item["labels"], dict) and isinstance(item["record"], dict), "labels/record mappings")
    for key in _EXTRA[kind]:
        if key not in {"timeline", "reference"}:
            _require(isinstance(item[key], list) and all(isinstance(r, dict) for r in item[key]), key + " rows")
    declared = [sources[n] for n in names]
    inputs = {s["identity"] for s in declared}
    signals = {v for s in declared for v in s["signal_identities"].values()}

    def bound(value, allowed=inputs):
        if not _text(value) or value not in allowed:
            raise ValueError("scientific input/source binding mismatch")

    def source_fields(input_id, record, fields=("source",)):
        bound(input_id)
        _fields(record, fields)
        matches = [s for s in declared if s["identity"] == input_id]
        _require(all(encode(record[k]) == encode(s[k]) for s in matches for k in fields),
                 "retained source declaration binding mismatch")
        return matches[0]

    def signal(support, basis=None):
        """Bind a signal declaration; return its matched source snapshot or None."""
        _fields(support, ("input_id",))
        allowed = inputs | signals
        if "variable" in support:
            allowed = {s["signal_identities"].get(support["variable"]) for s in declared}
        bound(support["input_id"], allowed)
        matches = [s for s in declared if support["input_id"] in s["signal_identities"].values()]
        if matches:
            _fields(support, ("variable", "basis", "source", "clock"))
            for source in matches:
                representation = source.get("representation", {})
                expected_basis = representation.get("basis", source["metadata"][support["variable"]]["representation"])
                _require(support["basis"] == expected_basis and (basis is None or basis == expected_basis),
                         "signal representation/source binding mismatch")
                _require(all(encode(support[k]) == encode(source[k]) for k in ("source","clock")),
                         "signal source/clock binding mismatch")
                _require(support.get("reference_id") == representation.get("reference_id"), "signal representation/reference binding mismatch")
                frozen = references.get(support.get("reference_id"))
                if frozen is not None and frozen.get("method_id") == "daily.median_distinct_dates_v1":
                    variable = frozen["variables"][support["variable"]]
                    _require(encode(support["fit_support"]) == encode(variable["support"]) and
                             support["fit_use_status"] == variable["use_status"] and
                             encode(support["fit_observations"]) == encode(frozen["observations"]) and
                             support["fit_scope"] == frozen["fit_scope"], "signal daily reference binding")
        return matches[0] if matches else None

    def variables(support, basis=None):
        _require(isinstance(support, dict), "variable support mapping")
        for value in support.values():
            signal(value, basis)

    def collective_support(support):
        _fields(support, ("input_id", "variables"))
        source = source_fields(support["input_id"], support, ("source","clock"))
        _require(list(support["variables"]) == [v for v in source["metadata"] if v in support["variables"]] and
                 all(support["units"][name] == source["metadata"][name]["unit"] for name in support["variables"]),
                 "collective feature/unit source binding")
        variables(support["variables"], support["basis"])

    def graph_sources(support):
        selections = [support]
        if "observed_only_counterpart" in support:
            selections.append(support["observed_only_counterpart"]["support"])
        for selection in selections:
            collective_support(selection["candidate"])
            collective_support(selection["evaluation"])

    def collective_result(record):
        """Source-bound PCA/graph support, then its derived declarations (and counterpart)."""
        support = record["support"]
        if record["method_id"] == "pca.frozen_projection_v1":
            collective_support(support)
            _pca_graph_declarations(record, support["training_support"])
            return
        graph_sources(support)
        _pca_graph_declarations(record, support["training"])
        if "observed_only_counterpart" in support:
            _pca_graph_declarations(support["observed_only_counterpart"], support["training"])

    def coherence(r, counterpart=None):
        """Pair records and their observed-only counterpart; no spectra are recomputed."""
        support, value, method = r["support"], r["value"], r["method_id"]
        modern = method in {"coherence.pooled_spectra_v1", "coherence.historical_modern_unavailable"}
        _fields(support, ("first", "second", "runs", "units", "variables", "observations", "audit", "requested", "resolved"))
        observations, audit = support["observations"], support["audit"]
        if counterpart is not None:
            # The counterpart repeats the same pair and settings under the native observed-only policy.
            outer = counterpart["support"]
            _require(encode(observations) == encode(dict(values="observed_only", min_points=outer["observations"]["min_points"],
                     max_imputed_fraction=None, max_reconstructed_duration=None)) and
                     all(encode(support[k]) == encode(outer[k]) for k in ("variables", "units", "requested", "resolved")) and
                     all(support[k]["input_id"] == outer[k]["input_id"] for k in ("first", "second")) and
                     r["representation"] == counterpart["representation"], "coherence counterpart binding")
        _observation_policy(observations)
        a, b = [signal(support[k], r["representation"])["metadata"][support[k]["variable"]]["unit"] for k in ("first", "second")]
        _require(support["units"] == dict(first=a, second=b) and
                 support["variables"] == [support["first"]["variable"], support["second"]["variable"]], "coherence pair unit/variable binding")
        cross = f"({a})*({b})/Hz"
        _declared(r, "exploratory_coherence", dict(frequency_hz="Hz", sxx=f"({a})^2/Hz", syy=f"({b})^2/Hz", sxy_real=cross,
            sxy_imag=cross, phase_rad="rad", raw_coherence="1", corrected="1", untruncated_corrected="1", raw_threshold="1",
            corrected_threshold="1", band_mean="1", significant_fraction="1") if modern else
            dict(frequency_hz="Hz", raw_coherence="1", band_mean="1"))
        uses, counts = [], []
        for key in ("first", "second"):
            positions, _ = _signal_selection(support[key], observations, estimator="regular")
            _require(support[key]["policy"] == "all_runs", "coherence contributing selection policy")
            uses.append(_saved_signal_use(support[key], observations, metric_status="exploratory_coherence"))
            counts.append(len(positions))
        use = min(uses, key=_USE.index)
        short = min(counts) < observations["min_points"]
        if short and use != "unknown_provenance":
            use = "insufficient_support"
        _require(isinstance(audit, dict) and isinstance(audit.get("reasons"), list), "coherence pair audit")
        if "unknown_lineage" in audit["reasons"] or use == "unknown_provenance":
            use = "unknown_provenance"
        _use(r, use)
        historical = {"coherence.historical_modern_unavailable": "unavailable_backend",
                      "coherence.report_exact_unresolved": "unresolved_compatibility"}.get(method)
        _require(r["compute_status"] == historical if historical else
                 (not short or r["compute_status"] == "insufficient_support") and
                 (use != "unknown_provenance" or r["compute_status"] != "ok"), "coherence pair qualification/availability")
        for run in support["runs"]:
            accepted = _coherence_status(run, {"ok", "insufficient_support"})
            _coherence_values(run, required=_COHERENCE_SPECTRA + ["physical_spacing_hz"] if accepted else (), allow_padding=modern)
        _fields(support, ("pooling",) if modern else ("aggregation",))
        if modern:
            pool = support["pooling"]
            if counterpart is not None and pool.get("compute_status") == "unresolved_weights":
                # Observed-only support changed the accepted runs; caller weights no longer name them.
                _require(support["requested"]["weighting"] == "run_weights" and set(pool) == {"compute_status", "reason", "pool_id"} and
                         pool["pool_id"] is None and _text(pool["reason"]) and value is None, "coherence unresolved counterpart weights")
            else:
                _fields(pool, ("spectra", "inference", "effective", "band_mean", "significant_fraction", "compute_status", "reason"))
                ok = _coherence_summary(pool)
                spectra, inference = pool["spectra"], pool["inference"]
                if spectra is not None:
                    _coherence_values(spectra, required=_COHERENCE_SPECTRA + ["phase_rad"])
                    _coherence_values(inference, spectra["frequency_hz"], required=_COHERENCE_INFERENCE)
                    _coherence_pool_support(pool,support)
                else:
                    _require(inference is None and pool["effective"] is None and pool["compute_status"] != "ok", "unavailable coherence pool")
                _coherence_values(pool, required=("band_mean", "significant_fraction"))
                has_decisions = ok and inference is not None and inference["inference_status"] == "approximate_white_noise"
                _require(_number(pool["significant_fraction"]) if has_decisions else pool["significant_fraction"] is None,
                         "coherence significant fraction availability")
                if value is not None:
                    _require(ok, "coherence outer/support availability")
                    expected = dict(**spectra, **inference, band_mean=pool["band_mean"], significant_fraction=pool["significant_fraction"])
                    _require(encode(value) == encode(expected), "coherence value/support disagreement")
        else:
            aggregate = support["aggregation"]
            _coherence_values(aggregate, required=_COHERENCE_REPORT + ["compute_status", "reason"])
            ok = _coherence_summary(aggregate)
            if ok:
                _require(np.isfinite(aggregate["raw_coherence"]).all(), "successful report aggregate must be finite")
            if value is not None:
                _require(ok, "coherence outer/support availability")
                _require(encode(value) == encode({k:aggregate[k] for k in _COHERENCE_REPORT}), "coherence value/support disagreement")
        if counterpart is None:
            _require(("observed_only_counterpart" in support) == (audit.get("observed_only_required") is True),
                     "coherence observed-only counterpart availability")
            if "observed_only_counterpart" in support:
                nested = support["observed_only_counterpart"]
                _require(isinstance(nested, dict) and set(nested) == _NATIVE and nested["method_id"] == method and
                         isinstance(nested["support"], dict) and "observed_only_counterpart" not in nested["support"],
                         "coherence counterpart schema")
                _native(nested, references)
                coherence(nested, r)

    r = item["record"]
    if kind in _COMPOSITE:
        _require(set(r) == set(_COMPOSITE[kind].split()), "composite record fields")
        for key in ("method_id", "config_id", "basis", "identity", "grid_id"):
            if key in r:
                _require(_text(r[key]), key + " must be text")
    if kind in {"metric", "peaks", "spectrum", "pca", "coherence", "density"}:
        _require(set(r) == _NATIVE, "native record fields")
        method = r["method_id"]
        _require(_text(method) and (method in _NATIVE_METHODS or not method.startswith(_RESERVED)), "unknown native method identity")
        if kind in _METHODS:
            _require(method in _METHODS[kind], "adapter method identity")
        _native(r, references)
        support, value = r["support"], r["value"]
        # Dispatch on the declared method, never on which fields remain.
        if method.startswith("coherence."):
            coherence(r)
        elif method == "pca.frozen_projection_v1" or method.startswith("graph."):
            collective_result(r)
        elif method == _REFERENCE_SUMMARY:
            _reference_summary(r)
        elif method in _SIGNAL_METHODS:
            refused = method in (_FISHER, _SIEGEL) and support.get("input_route") == "supplied_spectrum_refused"
            source = None if refused and "input_id" not in support else signal(support, r["representation"])
            _signal_method(r, None if source is None else source["metadata"][support["variable"]]["unit"])
        elif "input_id" in support:
            signal(support, r["representation"])
        if value is not None and kind == "peaks":
            _fields(value, ("table",))
            _require(isinstance(value["table"], pd.DataFrame), "peak table")
        if value is not None and kind == "spectrum":
            _spectrum(value)
    elif kind == "fi1":
        _status(r)
        _fields(r, ("support", "summary", "window_settings", "method_id", "config_id"))
        unit = signal(r["support"], r["basis"])["metadata"][r["support"]["variable"]]["unit"]
        _fields(r["support"], ("observations", "windows", "positions", "selected"))
        observations = r["support"]["observations"]
        _observation_policy(observations)
        _require(isinstance(r["summary"], dict) and isinstance(r["window_settings"], dict), "FI1 summary/settings")
        _require(r["method_id"] == "fi1.ordered_mass_roughness_v1" and r["metric_status"] == "historical_sensitivity_index" and
                 r["units"] == "1" and r["measurement_unit"] == unit and
                 r["fit_scope"] in ("frozen_reference", "global_self", "window_self") and
                 (r["reference_id"] is None) == (r["fit_scope"] == "window_self"), "FI1 trajectory method/unit/scope declaration")
        _counts(r["summary"], ("eligible_windows", "scheduled_windows", "flagged_count", "available_flags", "delta_count"))
        _numbers(r["summary"], ("median_fi1", "iqr_fi1", "median_entropy_nats", "delta_median", "delta_mad", "threshold"), nullable=True)
        _unique(item["rows"], ("start", "end"))
        schedule = r["support"]["windows"]
        _require(isinstance(schedule,list) and len(item["rows"]) == len(schedule) == r["summary"]["scheduled_windows"], "FI1 schedule completeness")
        _require(encode(r["window_settings"]) == encode(r["support"]["window_settings"]), "FI1 window settings binding")
        _require(all(isinstance(r["window_settings"][k],pd.Timedelta) and r["window_settings"][k] > pd.Timedelta(0)
                     for k in ("window","step")), "FI1 window durations")
        for i, (row, window) in enumerate(zip(item["rows"], schedule)):
            _fields(row, "start end label fi1 entropy_nats eligible_count evaluation_count compute_status metric_status use_status reason basis edge_id reference_id method_id config_id units distribution support abs_delta abrupt_flag".split())
            start, end = _geometry(row, strings=True)
            _require(end-start == r["window_settings"]["window"] and
                     row["label"] == row[r["window_settings"]["label"]], "FI1 window settings geometry")
            _require(isinstance(window, dict) and all(row[k] == window[k] for k in ("start", "end", "label")) and
                     isinstance(row["support"], dict) and all(encode(row["support"].get(k)) == encode(v) for k, v in window.items()),
                     "FI1 schedule geometry binding")
            _counts(row, ("eligible_count", "evaluation_count"))
            _require(row["eligible_count"] == window["eligible_cells"], "FI1 eligible count binding")
            _numbers(row, ("abs_delta",), nullable=True)
            _require(_missing(row["abs_delta"]) or row["abs_delta"] >= 0, "FI1 absolute delta domain")
            _scalar_status(row, "fi1")
            _scalar_status(row, "entropy_nats")
            _fields(row, ("support", "abrupt_flag"))
            _require(row["basis"] == r["basis"], "FI1 row representation binding mismatch")
            signal(row["support"], row["basis"])
            _require(row["method_id"] == r["method_id"] and row["edge_id"] == row["support"].get("edge_id") and
                     encode(row["support"].get("observations")) == encode(observations), "FI1 row method/settings binding")
            _fi1_evaluation(row, row["support"], row["distribution"], unit, "burden")
            evaluated = _evaluation_support(row["support"])
            _require(row["evaluation_count"] == len(evaluated), "FI1 evaluation count binding")
            _require(row["reference_id"] == identity(row["support"]["fit"]), "FI1 row reference binding mismatch")
            if r["reference_id"] is not None:
                _require(row["reference_id"] == r["reference_id"], "FI1 trajectory reference binding mismatch")
            _require((row["distribution"] is not None) == (row["compute_status"] == "ok"), "FI1 distribution availability")
            if row["distribution"] is not None:
                _distribution(row["distribution"])
                _require(row["distribution"]["eligible_count"] == len(evaluated), "FI1 distribution evaluation binding")
                _require(all(_same(row[k],row["distribution"][k]) for k in ("fi1", "entropy_nats")), "FI1 distribution summary binding")
            _require(_missing(row["abrupt_flag"]) or isinstance(row["abrupt_flag"], (bool, np.bool_)), "FI1 flag")
            _require(_missing(row["abrupt_flag"]) == _missing(row["abs_delta"]), "FI1 change availability")
            previous = item["rows"][i-1] if i else None
            adjacent = previous is not None and previous["compute_status"] == row["compute_status"] == "ok" and (
                start-pd.Timestamp(previous["start"]) == r["window_settings"]["step"])
            _require(not _missing(row["abs_delta"]) if adjacent else _missing(row["abs_delta"]), "FI1 adjacent change availability")
        trajectory = _saved_signal_use(r["support"], observations, metric_status="historical_sensitivity_index")
        _use(r, "unknown_provenance" if any(row["use_status"] == "unknown_provenance" for row in item["rows"]) else trajectory)
        summary = r["summary"]
        _require(r["compute_status"] == ("ok" if summary["eligible_windows"] else "insufficient_support") and
                 (r["reason"] is None if summary["eligible_windows"] else _text(r["reason"])), "FI1 overall availability")
        _require(summary["eligible_windows"] == sum(row["compute_status"] == "ok" for row in item["rows"]) and
                 summary["available_flags"] == summary["delta_count"] == sum(not _missing(row["abs_delta"]) for row in item["rows"]) and
                 summary["flagged_count"] == sum(bool(row["abrupt_flag"]) for row in item["rows"] if not _missing(row["abrupt_flag"])), "FI1 summary counts")
        for fields, available in (( ("median_fi1","iqr_fi1","median_entropy_nats"), summary["eligible_windows"] > 0),
                                  (("delta_median","delta_mad","threshold"), summary["delta_count"] > 0)):
            _require(all(_number(summary[k]) if available else summary[k] is None for k in fields), "FI1 summary availability")
            _require(all(summary[k] is None or summary[k] >= (-1e-12 if k == "median_entropy_nats" else 0)
                         for k in fields), "FI1 summary domain")
    elif kind == "network":
        _status(r)
        _fields(r, ("input_id", "method_id", "config_id", "features"))
        settings = r["requested"]
        arguments = "features basis missingness min_points threshold window step complete label include_partial historical valid_pairs".split()
        _fields(settings, arguments)
        _require(set(settings) == set(arguments)|{"closure","threshold_roundoff_tolerance"} and
                 settings["basis"] in {"raw","daily_residual"} and settings["missingness"] in {"pairwise","listwise"} and
                 type(settings["min_points"]) is int and settings["min_points"] >= 2 and
                 _number(settings["threshold"]) and 0 <= settings["threshold"] <= 1 and
                 settings["closure"] == "left" and settings["threshold_roundoff_tolerance"] == 1e-12,
                 "network settings")
        _require(all(type(settings[k]) is bool for k in ("complete","include_partial","historical","valid_pairs")) and
                 settings["label"] in {"start","end"}, "network window flags")
        if settings["window"] is None:
            _require(settings["step"] is None and settings["complete"] and not settings["include_partial"] and
                     settings["label"] == "start", "network nonrolling settings")
        else:
            _require(all(isinstance(settings[k],pd.Timedelta) and settings[k] > pd.Timedelta(0) for k in ("window","step")), "network window durations")
        observations = r["observations"]
        _fields(observations, ("values", "min_points", "max_imputed_fraction", "max_reconstructed_duration"))
        _require(observations["values"] in {"provided","observed_only","trusted_imputation"} and
                 type(observations["min_points"]) is int and observations["min_points"] >= 1, "network observation policy")
        fraction = observations["max_imputed_fraction"]
        _require(fraction is None or _number(fraction) and 0 <= fraction <= 1, "network imputation fraction")
        if observations["max_reconstructed_duration"] is not None:
            _require(pd.Timedelta(observations["max_reconstructed_duration"]) > pd.Timedelta(0), "network reconstructed duration")
        _require(encode(settings) == encode(r["resolved"]), "network requested/resolved settings")
        _require(r["config_id"] == identity(dict(spec=settings, observations=r["observations"], start=r["start"], end=r["end"])), "network configuration binding")
        _require(r["features"] == settings["features"] and r["basis"] == settings["basis"] and
                 r["method_id"] == "network.signed_pearson_v1", "network declaration binding")
        network_source = source_fields(r["input_id"], r, ("source","clock"))
        _require(r["metric_status"] == "exploratory_association" and isinstance(r["features"], list) and
                 encode(r["units"]) == encode({f: network_source["metadata"][f]["unit"] for f in r["features"]}),
                 "network qualification/unit declaration")
        _unique(item["rows"], ("window",))
        _unique(item["edges"], ("window", "a", "b"))
        _counts(r, ("window_count",))
        _require(len(item["rows"]) == r["window_count"], "network window completeness")
        expected_status = ("ok" if all(row["compute_status"] == "ok" for row in item["rows"]) else "undefined_pairs") if item["rows"] else "insufficient_support"
        _require(r["compute_status"] == expected_status and
                 (r["reason"] is None if expected_status == "ok" else _text(r["reason"])), "network overall status")
        features = r["features"]
        _require(isinstance(features,list) and len(features) >= 2 and all(_text(v) for v in features) and len(set(features)) == len(features), "network features")
        pair_count = len(features)*(len(features)-1)//2
        _require(len(item["edges"]) == len(item["rows"])*pair_count, "network edge completeness")
        pairs = [(a,b) for i,a in enumerate(features) for b in features[i+1:]] if item["rows"] else []
        _require([(e["window"],e["a"],e["b"]) for e in item["edges"]] ==
                 [(i,a,b) for i in range(len(item["rows"])) for a,b in pairs], "network edge completeness")
        for number, row in enumerate(item["rows"]):
            start, end = _geometry(row, nullable=r["requested"]["window"] is None)
            _require(row["label"] == row[r["requested"]["label"]], "network label geometry")
            if r["requested"]["window"] is not None:
                _require(end-start == r["requested"]["window"] and (not number or
                         start-item["rows"][number-1]["start"] == r["requested"]["step"]), "network scheduled geometry")
            _counts(row, ("window", "p", "pair_denominator", "valid_pair_count"))
            _require(row["window"] == number and row["p"] == len(features) and row["pair_denominator"] == len(pairs), "network window denominators")
            _native(row, references)
            for key in ("input_id", "source", "features", "units", "basis", "clock", "reference_ids", "requested", "resolved", "observations", "backend", "backend_version"):
                _require(encode(row["support"][key]) == encode(r[key]), "network window settings/support binding")
            _require(row["config_id"] == r["config_id"] and row["method_id"] == r["method_id"], "network window configuration binding")
            _require(row["metric_status"] == "exploratory_association" and row["units"] == "1", "network window qualification/unit declaration")
            if row["value"] is not None:
                _native_value("network.signed_pearson_v1", row["value"])
            for field in ("mean_absolute_connectivity", "strong_count", "density") + (
                    ("historical_diagonal_diluted_mean", "historical_double_counted_links") if r["requested"]["historical"] else ()):
                _scalar_status(row, field)
                _require(_same(row[field], None if row["value"] is None else row["value"][field]), "network duplicated summary binding")
            edges = [e for e in item["edges"] if e["window"] == number]
            _require(encode(row["support"]["pairs"]) == encode([dict(a=e["a"],b=e["b"],support=e["support"]) for e in edges]), "network pair support binding")
            _require(row["valid_pair_count"] == sum(e["eligible"] is True or isinstance(e["eligible"],np.bool_) and bool(e["eligible"]) for e in edges), "network valid pair count")
            strong_count = sum(bool(e["strong"]) for e in edges if e["eligible"])
            _require((row["compute_status"] == "ok") == (row["valid_pair_count"] == row["pair_denominator"]), "network full-pair availability")
            if row["value"] is not None:
                _require(row["strong_count"] == strong_count <= row["pair_denominator"] and
                         row["density"] == row["strong_count"]/row["pair_denominator"], "network count/density binding")
                if settings["historical"]:
                    _counts(row["value"], ("historical_double_counted_links",))
                    _require(row["value"]["historical_double_counted_links"] == 2*strong_count and
                             0 <= row["value"]["historical_diagonal_diluted_mean"] <= 1, "network historical domain/count")
            if r["requested"]["valid_pairs"]:
                subset = row["valid_pairs_summary"]
                _counts(subset, ("denominator",))
                available = subset["denominator"] > 0
                _require(subset["denominator"] == row["valid_pair_count"] and subset["compute_status"] == ("ok" if available else "insufficient_support"), "network subset status/count")
                _fields(subset, ("mean_absolute_connectivity", "strong_count", "density", "method_id"))
                _require(all(_number(subset[k]) if available else subset[k] is None for k in ("mean_absolute_connectivity", "strong_count", "density")), "network subset availability")
                if available:
                    _native_value("network.eligible_pairs_only_v1",subset)
                    _require(subset["strong_count"] == strong_count <= subset["denominator"] and
                             subset["density"] == strong_count/subset["denominator"], "network subset count/density binding")
            bound(row["support"]["input_id"])
            for pair in row["support"]["pairs"]:
                variables(pair["support"], r["basis"])
        for row in item["edges"]:
            _scalar_status(row, "correlation")
            _require(_missing(row["correlation"]) or -1 <= row["correlation"] <= 1, "network correlation domain")
            for key in ("input_id", "basis", "config_id", "method_id", "backend", "backend_version", "requested", "resolved"):
                _require(encode(row[key]) == encode(r[key]), "network edge settings binding")
            _require(encode(row["reference_ids"]) == encode({k:r["reference_ids"][k] for k in (row["a"],row["b"])}), "network edge reference binding")
            bound(row["input_id"])
            variables(row["support"], r["basis"])
            _require(row["metric_status"] == "exploratory_association" and row["units"] == "1" and
                     isinstance(row["support"], dict) and list(row["support"]) == [row["a"], row["b"]], "network edge qualification/unit declaration")
            uses = []
            for pair_support in row["support"].values():
                _signal_selection(pair_support, r["observations"], estimator="point")
                _require(pair_support["policy"] == "complete_case", "network pair selection policy")
                uses.append(_saved_signal_use(pair_support, r["observations"], metric_status="exploratory_association"))
            _use(row, min(uses, key=_USE.index))
            _counts(row, ("count", "window"))
            _fields(row, ("positions", "instants"))
            _require(isinstance(row["positions"],list) and isinstance(row["instants"],list) and
                     len(row["positions"]) == len(row["instants"]) == row["count"], "network edge support count")
            positions = _positions(row["positions"])
            instants = [_time(instant,strings=True) for instant in row["instants"]]
            _require(all(a < b for a,b in zip(instants,instants[1:])), "network instant order")
            window = item["rows"][row["window"]]
            _require(set(positions) <= set(window["support"]["positions"]), "network edge/window support")
            _require(all((window["start"] is None or window["start"] <= instant) and
                         (window["end"] is None or instant < window["end"]) for instant in instants), "network edge/window instants")
            for pair_support in row["support"].values():
                _require(pair_support["positions"] == positions and pair_support["selected_instants"] == row["instants"],
                         "network retained pair support binding")
            _require(isinstance(row["eligible"], (bool, np.bool_)) and
                     (_missing(row["strong"]) or isinstance(row["strong"], (bool, np.bool_))), "network edge flags")
            _require(row["eligible"] == (row["compute_status"] == "ok") and _missing(row["strong"]) == (row["compute_status"] != "ok"), "network edge availability")
        for number, window in enumerate(item["rows"]):
            _use(window, min((e["use_status"] for e in item["edges"] if e["window"] == number), key=_USE.index))
        _use(r, min((w["use_status"] for w in item["rows"]), key=_USE.index) if item["rows"] else "insufficient_support")
    elif kind == "matrix":
        _status(r)
        _fields(r, ("source", "support", "settings", "method_id"))
        members = r["source"]["members"]
        _require(isinstance(members, dict), "collective members")
        features = r["support"]["features"]
        _require(r["method_id"] == "collective.explicit_cell_aggregation_v1" and r["metric_status"] == "audited_preparation" and
                 isinstance(features, dict) and features and list(features) == list(r["variables"]) and
                 all(isinstance(f, dict) and f.get("use_status") in _USE and isinstance(f.get("eligible_positions"), list) and
                     f.get("compute_status") == ("ok" if f["eligible_positions"] else "insufficient_support") for f in features.values()),
                 "collective feature qualification")
        _use(r, min((f["use_status"] for f in features.values()), key=_USE.index))
        for member in members.values():
            bound(member["source_id"])
        _unique(item["rows"], ("feature", "start", "end"))
        settings = r["settings"]
        start, end = _time(settings["start"]), _time(settings["end"])
        cadence = settings["cadence"]
        _require(isinstance(cadence,pd.Timedelta) and cadence > pd.Timedelta(0) and end > start and
                 (end-start).value % cadence.value == 0, "matrix declared geometry")
        _require(len(item["rows"]) == ((end-start).value // cadence.value) * len(r["variables"]), "matrix cell completeness")
        _require(r["compute_status"] == ("ok" if all(row["compute_status"] == "ok" for row in item["rows"]) else "incomplete_matrix") and
                 all(features[v]["eligible_positions"] == [(row["start"]-start).value // cadence.value for row in item["rows"]
                     if row["feature"] == v and row["compute_status"] == "ok"] for v in features), "collective cell/feature availability")
        for row in item["rows"]:
            _fields(row, ("feature", "start", "end", "label", "compute_status", "reason", "stations", "contributing_stations", "contributor_count", "lineage_id"))
            left, right = _geometry(row)
            _require(row["feature"] in r["variables"] and start <= left < end and right-left == cadence and
                     (left-start).value % cadence.value == 0 and row["label"] == row[settings["label"]], "matrix cell geometry binding")
            _require(row["compute_status"] in {"ok", "unavailable_cell"} and
                     (row["reason"] is None and _text(row["lineage_id"]) if row["compute_status"] == "ok"
                      else _text(row["reason"]) and row["lineage_id"] is None), "matrix cell status")
            _counts(row, ("contributor_count",))
            _unique(row["stations"], ("station",))
            _require(isinstance(row["contributing_stations"],list) and all(_text(s) for s in row["contributing_stations"]), "collective station identities")
            contributing = [s for s in row["stations"] if s["reason"] is None]
            _require(row["contributing_stations"] == [s["station"] for s in contributing] and
                     row["contributor_count"] == sum(s["count"] for s in contributing) and
                     (row["compute_status"] == "ok") == bool(contributing), "collective contributor membership/count binding")
            for station in row["stations"]:
                _fields(station, ("station", "expected_count", "count", "coverage", "contributors", "rejected", "observed_only_contributors", "observed_only_coverage", "reason", "observed_only_reason"))
                _counts(station, ("expected_count", "count"))
                _require(_text(station["station"]) and all(station[k] is None or _text(station[k]) for k in ("reason","observed_only_reason")), "collective station availability")
                _numbers(station, ("count", "coverage", "observed_only_coverage"))
                _require(station["expected_count"] > 0 and station["count"] == len(station["contributors"]) <= station["expected_count"] and
                         station["coverage"] == station["count"] / station["expected_count"] and
                         station["observed_only_coverage"] == len(station["observed_only_contributors"]) / station["expected_count"], "matrix station denominator")
                for category in ("contributors", "observed_only_contributors", "rejected"):
                    for contributor in station[category]:
                        member = members.get(contributor["member"])
                        if member is None or contributor.get("source_id", member["source_id"]) != member["source_id"]:
                            raise ValueError("collective contributor/source binding mismatch")
                        _require(member["station"] == station["station"], "collective contributor station binding")
                        bound(member["source_id"])
                        if category != "rejected" or contributor.get("reason") != "missing_timestamp":
                            _fields(contributor, ("source_id","source"))
                            source_fields(member["source_id"], contributor)
                        else:
                            _require(set(contributor) == {"member","tick","reason"}, "missing timestamp rejection fields")
    elif kind == "graph":
        _fields(r, ("rows", "configurations", "grid_id"))
        _counts(r, ("configurations",))
        _require(len(r["rows"]) == len(item["rows"]) == r["configurations"], "graph row count")
        available = any(row["compute_status"] == "ok" for row in item["rows"])
        _require(all(_number(r[k]) if available else r[k] is None for k in ("median","minimum","maximum")), "graph range availability")
        _unique(item["rows"], ("pcs", "bins"))
        for row, saved in zip(item["rows"], r["rows"]):
            _fields(row, "pcs bins requested_states rank realized_bins reference_id config_id compute_status reason metric_status use_status energy connected_components occupied_count empty_count result".split())
            _counts(row, ("pcs", "bins", "requested_states"))
            # graph_state_grid has a fixed 2..5 x 3..6 request grid. Validate
            # those producer domains before exponentiating an untrusted count.
            _require(2 <= row["pcs"] <= 5 and 3 <= row["bins"] <= 6, "graph grid configuration domain")
            if not _missing(row["rank"]):
                _counts(row, ("rank",))
            _status(row)
            native = row["result"]
            _native(native, references)
            if native["value"] is not None:
                _native_value("graph.state_mass_v1", native["value"])
            _require(set(row) == set(saved) and all(_same(row[k],saved[k]) for k in row), "graph retained row disagreement")
            for key in ("compute_status", "metric_status", "use_status", "reason", "config_id", "reference_id"):
                _require(row[key] == native[key], "graph row status/reference disagreement")
            settings = native["support"]["requested"]
            _require(row["pcs"] == settings["pca_components"] and row["bins"] == settings["bins"] and
                     row["requested_states"] == row["bins"] ** row["pcs"] and
                     (native["support"]["requested_states"] is None or row["requested_states"] == native["support"]["requested_states"]) and
                     _same(row["rank"],native["support"]["rank"]), "graph configuration binding")
            for key, value_key in (("energy","energy"), ("connected_components","connected_components"),
                                   ("occupied_count","observed_occupied_count"), ("empty_count","observed_empty_count")):
                _scalar_status(row, key)
                _require(_same(row[key],None if native["value"] is None else native["value"][value_key]), "graph summary binding")
            if row["realized_bins"] is not None:
                _require(isinstance(row["realized_bins"],list) and all(_number(v) and v > 0 and int(v) == v for v in row["realized_bins"]), "graph realized bins")
                _require(row["realized_bins"] == [len(e)-1 for e in native["support"]["partition_edges"]], "graph partition binding")
            collective_result(native)
        if available:
            energies = [row["energy"] for row in item["rows"] if row["compute_status"] == "ok"]
            _require(r["minimum"] == min(energies) <= r["median"] <= r["maximum"] == max(energies), "graph range domain/binding")
    elif kind == "regimes":
        _status(r)
        _fields(r, ("source", "support", "reference_id", "method_id"))
        source_fields(r["source"]["source_id"], r["source"])
        variables(r["support"]["variables"])
        selection = r["support"]
        _fields(selection, ("row_ids", "positions", "variables", "observations", "use_status"))
        _require(r["method_id"] == "regimes.candidate_clusters_v1" and r["metric_status"] == "exploratory_candidate_regimes" and
                 isinstance(selection["positions"], list) and isinstance(r["training"], dict) and
                 selection["use_status"] in (("unknown_provenance", "sensitivity_only") if selection["positions"] else ("insufficient_support",)),
                 "regime method/selection qualification")
        _use(r, "unknown_provenance" if "unknown_provenance" in (selection["use_status"], r["training"].get("use_status"))
             else selection["use_status"])
        _require(identity(item["reference"]) == r["reference_id"], "regime numerical reference binding")
        _require(encode(r["training"]) == encode(item["reference"]), "regime training reference binding")
        if r["membership"] is not None:
            _array(r["membership"], ndim=2)
        _counts(r, ("cluster_count", "denominator", "assigned", "noise", "excluded", "failed"))
        _unique(item["rows"], ("row_id",))
        _require([x["row_id"] for x in item["rows"]] == [x["row_id"] for x in r["source"]["rows"]], "regime row geometry binding")
        for row, geometry in zip(item["rows"],r["source"]["rows"]):
            _fields(row, ("row_id", "start", "end", "native_label", "display_label", "status", "reason"))
            _require(_text(row["row_id"]), "regime row identity")
            _require(all(isinstance(row[k], pd.Timestamp) and row[k].tz is not None for k in ("start", "end"))
                     and row["end"] >= row["start"], "regime row times")
            _require(row["start"] == geometry["start"] and row["end"] == geometry["end"], "regime stored geometry binding")
            _require(row["status"] in {"assigned", "noise", "excluded_missing", "excluded_quality", "failed"}, "regime row status")
            if row["status"] in {"assigned", "noise"}:
                _require(type(row["native_label"]) is str or _number(row["native_label"]), "native label")
                _require(row["reason"] is None, "regime assigned reason")
            else:
                _require(_missing(row["native_label"]) and _text(row["reason"]), "unassigned label/reason")
            display = row["display_label"]
            _require(_missing(display) or _number(display) and display >= 0 and int(display) == display, "regime display label")
            _require(_same(display,r["display_mapping"].get(row["native_label"])), "regime display binding")
        counts = dict(assigned=sum(row["status"] == "assigned" for row in item["rows"]),
            noise=sum(row["status"] == "noise" for row in item["rows"]),
            excluded=sum(row["status"].startswith("excluded") for row in item["rows"]),
            failed=sum(row["status"] == "failed" for row in item["rows"]))
        _require(r["denominator"] == len(item["rows"]) and all(r[k] == v for k,v in counts.items()), "regime row counts")
        cluster_sizes = {}
        for row in item["rows"]:
            if row["status"] == "assigned":
                label = row["native_label"]
                cluster_sizes[label] = cluster_sizes.get(label,0)+1
        _require(isinstance(r["cluster_sizes"],dict) and isinstance(r["display_mapping"],dict) and
                 set(r["cluster_sizes"]) == set(cluster_sizes) == set(r["display_mapping"]), "regime cluster mappings")
        _counts(r["cluster_sizes"], list(cluster_sizes))
        _counts(r["display_mapping"], list(cluster_sizes))
        _require(r["cluster_count"] == len(cluster_sizes) and r["cluster_sizes"] == cluster_sizes and
                 sorted(r["display_mapping"].values()) == list(range(len(cluster_sizes))), "regime cluster count/display binding")
        _require(isinstance(r["fractions"],dict) and set(r["fractions"]) == set(counts), "regime fraction mapping")
        for key, count in counts.items():
            value = r["fractions"][key]
            _require(_number(value) and value == count/r["denominator"] if r["denominator"] else value is None, "regime fraction denominator binding")
        _require(r["compute_status"] in {"ok", "insufficient_support", "fit_failed", "not_converged"} and
                 (r["reason"] is None and counts["assigned"]+counts["noise"] > 0 if r["compute_status"] == "ok"
                  else _text(r["reason"]) and counts["assigned"]+counts["noise"] == 0), "regime overall availability")
        if r["membership"] is not None:
            _require(r["compute_status"] == "ok" and len(r["membership"]) == counts["assigned"]+counts["noise"] and
                     np.isfinite(r["membership"]).all(), "regime membership availability/dimensions")
        if item["timeline"] is not None:
            _timeline(item["timeline"], item["rows"], r["identity"])
    elif kind == "sensitivity":
        _fields(r, ("method", "identity", "variable_order", "geometry", "completion_provenance"))
        _require(_text(r["method"]), "sensitivity method")
        _require(isinstance(r["references"],list) and len(r["references"]) == 2 and
                 r["design"] in {"fixed","allowed_training_refit"}, "sensitivity preprocessing declarations")
        _unique(item["routes"], ("route_id",))
        routes = {row["route_id"]: row for row in item["routes"]}
        for row in item["routes"]:
            _scalar_status(row, "estimate")
            _fields(row, ("source_id", "route", "draw_id", "variable", "spectrum", "fit"))
            _require(row["route"] in {"W0", "I0", "Im", "Wc"} and row["variable"] in r["variable_order"], "sensitivity route declaration")
            if row["route"] == "Wc":
                _require(_text(row["draw_id"]) and row["draw_id"] in r["completion_provenance"], "sensitivity completion draw binding")
            else:
                _require(row["draw_id"] is None, "sensitivity baseline draw")
            if row["source_id"] is not None:
                bound(row["source_id"])
                if row["route"] == "Wc":
                    provenance = r["completion_provenance"][row["draw_id"]]
                    _require(isinstance(provenance,dict) and provenance.get("draw_id") == row["draw_id"] and
                             any(s["identity"] == row["source_id"] and encode(s["source"].get("imputation")) == encode(provenance)
                                 for s in declared), "sensitivity completion provenance binding")
            else:
                _require(row["compute_status"] in {"unavailable_reference", "unstarted"}, "missing sensitivity source")
            _require((row["spectrum"] is not None) == (row["source_id"] is not None), "sensitivity spectrum availability")
            if row["compute_status"] == "ok":
                _require(isinstance(row["fit"],dict) and row["reason"] is None, "sensitivity fit availability")
            _require(row["metric_status"] == "descriptive_sensitivity", "sensitivity route qualification declaration")
            _use(row, "sensitivity_only" if row["spectrum"] is None else row["spectrum"]["use_status"])
            if row["spectrum"] is not None:
                _native(row["spectrum"], references)
                nested = row["spectrum"]
                source = source_fields(row["source_id"], nested["support"])
                reference = r["references"][0 if row["route"] in {"W0","I0"} else 1]
                reference_id = None if reference is None else identity(reference)
                basis = source["metadata"][row["variable"]]["representation"] if reference is None else "daily_residual"
                _require(nested["representation"] == row["signature"]["basis"] == basis and
                         nested["reference_id"] == row["reference_id"] == row["signature"]["reference_id"] == reference_id and
                         nested["support"]["variable"] == row["variable"] and
                         row["signature"]["unit"] == source["metadata"][row["variable"]]["unit"],
                         "sensitivity spectrum/signature binding mismatch")
                expected_method = "spectral.ls_projection_density_wosa_v1" if row["route"] in {"I0","Im"} else "spectral.welch_mean_v1"
                _require(row["spectrum"]["method_id"] == expected_method, "sensitivity route method binding")
                _signal_method(nested, source["metadata"][row["variable"]]["unit"])
                if row["spectrum"]["value"] is not None:
                    _spectrum(row["spectrum"]["value"])
            if row["fit"] is not None:
                _fit(row["fit"])
                _require(row["spectrum"] is not None and row["spectrum"]["compute_status"] == "ok" and
                         row["compute_status"] == row["fit"]["compute_status"] and row["reason"] == row["fit"]["reason"], "sensitivity fit status binding")
                if row["compute_status"] == "ok":
                    _require(_same(row["estimate"],row["fit"]["single"]["beta_low"]), "sensitivity fitted estimate binding")
            _fields(row["signature"], ("interval", "basis", "unit", "variable", "model", "frequency_hz", "min_cycles", "reference_id"))
            _require(row["signature"]["variable"] == row["variable"] and
                     encode(row["signature"]["interval"]) == encode(r["interval"]) and
                     encode(row["signature"]["frequency_hz"]) == encode(r["frequency_hz"]), "sensitivity signature binding")
            _require(row["reference_id"] == row["signature"]["reference_id"], "sensitivity reference binding mismatch")
        _unique(item["contrasts"], ("contrast", "baseline_id", "candidate_id"))
        for row in item["contrasts"]:
            _scalar_status(row, "signed_delta")
            _scalar_status(row, "absolute_delta")
            for key in ("baseline", "candidate"):
                _require(row[key+"_id"] in routes, "contrast route binding")
                _fields(row[key], ("source_id", "route_id", "draw_id", "variable", "compute_status"))
                saved = routes[row[key+"_id"]]
                _require(all(row[key][k] == saved[k] for k in ("source_id", "route_id", "draw_id", "variable", "compute_status")), "contrast source/route binding")
                _require(set(row[key]) == set(saved) and all(_same(row[key][k],saved[k]) for k in saved), "contrast retained route binding")
            _fields(row, ("differences", "interpretation"))
            first, second = row["baseline"], row["candidate"]
            differences = [key for key in sorted(set(first["signature"])|set(second["signature"]))
                           if identity(first["signature"].get(key)) != identity(second["signature"].get(key))]
            available = first["compute_status"] == second["compute_status"] == "ok"
            status = "incomparable" if differences else "ok" if available else "unavailable"
            reason = "scientific_signature_mismatch" if differences else None if available else "unavailable_route"
            _require(row["differences"] == differences and row["compute_status"] == status and
                     row["reason"] == reason, "contrast endpoint availability/signature binding")
            _require(row["metric_status"] == "descriptive_sensitivity" and _text(row["interpretation"]), "contrast qualification")
            _use(row, "unknown_provenance" if "unknown_provenance" in (first["use_status"], second["use_status"]) else "sensitivity_only")
            if status == "ok":
                _require(row["absolute_delta"] == abs(row["signed_delta"]), "contrast absolute delta binding")
        _require([(row["variable"],row["route"],row["draw_id"]) for row in item["routes"]] ==
                 [(v,route,draw) for v in r["variable_order"] for route,draw in
                  [("W0",None),("I0",None),("Im",None)]+[("Wc",draw) for draw in r["completion_provenance"]]], "sensitivity route completeness")
        by_route = {(row["variable"],row["route"],row["draw_id"]):row["route_id"] for row in item["routes"]}
        expected = []
        for variable in r["variable_order"]:
            pairs = [("baseline_estimator",("W0",None),("I0",None)), ("deletion",("I0",None),("Im",None))]
            pairs += [(label,(route,None),("Wc",draw)) for draw in r["completion_provenance"] for label,route in (("completion","W0"),("mixed","Im"))]
            expected.extend((label,by_route[(variable,*a)],by_route[(variable,*b)]) for label,a,b in pairs)
        _require([(row["contrast"],row["baseline_id"],row["candidate_id"]) for row in item["contrasts"]] == expected, "sensitivity contrast completeness")

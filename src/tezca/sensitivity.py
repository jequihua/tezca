"""Owned matched-support sensitivity comparisons; never constructs completions."""

import numpy as np
import pandas as pd

from ._owned import identity, owned, text_field
from .data import AnalysisSeries, ResultRecord
from .individual import _subset, apply_daily_reference, fit_daily_reference
from .irregular import IrregularSpec, irregular_spectrum
from .spectral import WelchSpec, power_spectrum
from .spectral_summary import ModelSpec, _positive, _real_array, fit_log_models
from .support import ObservationPolicy, _aware_timestamp, positions_array


class SensitivityReport:
    def __init__(self, routes, contrasts, description):
        self._routes, self._contrasts, self._description = owned(routes), owned(contrasts), owned(description)
        self._identity = identity(dict(routes=routes, contrasts=contrasts, description=description))

    @property
    def identity(self):
        return self._identity

    @property
    def routes(self):
        return owned(pd.DataFrame(self._routes))

    @property
    def contrasts(self):
        return owned(pd.DataFrame(self._contrasts))

    def describe(self):
        return dict(owned(self._description), identity=self.identity)


def _aligned(first, second):
    if not isinstance(first, AnalysisSeries) or not isinstance(second, AnalysisSeries):
        raise ValueError("comparison inputs must be AnalysisSeries")
    if (not first.values.index.equals(second.values.index) or list(first.values) != list(second.values)
            or identity(first.metadata) != identity(second.metadata) or identity(first.time.record) != identity(second.time.record)):
        raise ValueError("input time/variable order/units/basis/clock mismatch")


def validate_completions(masked, completions, *, complete=None):
    """Validate supplied draws against their exact masked source and original masks."""
    if not isinstance(masked, AnalysisSeries) or not isinstance(completions, dict):
        raise ValueError("masked AnalysisSeries and explicit draw mapping required")
    available = np.isfinite(masked.values.to_numpy(dtype=float, na_value=np.nan))
    if complete is not None:
        _aligned(masked, complete)
        truth = complete.values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(truth).all():
            raise ValueError("complete empirical reference must be finite; use None for natural gaps")
        if not np.array_equal(masked.values.to_numpy(dtype=float, na_value=np.nan)[available],truth[available]):
            raise ValueError("masked input changed retained reference values")
    for draw, data in completions.items():
        text_field(draw, "draw ID")
        if data is None:  # A declared unstarted request is still a report row.
            continue
        _aligned(masked, data)
        x = data.values.to_numpy(dtype=float, na_value=np.nan)
        if not np.array_equal(x[available],masked.values.to_numpy(dtype=float, na_value=np.nan)[available]):
            raise ValueError("completion changed observed/supplied retained cells")
        finite_replacements = ~available & np.isfinite(x)
        if np.any(data.contribution[finite_replacements] != "imputed"):
            raise ValueError("finite replacements must be explicitly imputed")
        if not np.array_equal(data.contribution[available],masked.contribution[available]):
            raise ValueError("completion changed retained contribution provenance")
        if any(not np.array_equal(data.qualification[k],v) for k,v in masked.qualification.items()):
            raise ValueError("completion must preserve original qualification masks")
        original_lineage, completed_lineage = masked.lineage,data.lineage
        for j,variable in enumerate(masked.values):
            original = original_lineage.get(variable, [None] * len(masked.values))
            completed = completed_lineage.get(variable, [None] * len(data.values))
            for p in np.flatnonzero(available[:,j]):
                old,new = original[p],completed[p]
                if identity(None if old is None else old.record) != identity(None if new is None else new.record):
                    raise ValueError("completion changed retained contributor lineage")
        provenance = data.source.get("imputation", {})
        if (provenance.get("input_id") != masked.identity or provenance.get("draw_id") != draw
                or not provenance.get("method_id") or not provenance.get("plan_id")):
            raise ValueError("completion needs matching masked input_id, draw_id, method_id and plan_id")


def gap_geometry(masked):
    """Missing-cell geometry, with exact regular durations and no invented irregular coverage."""
    values = masked.values.to_numpy(dtype=float, na_value=np.nan)
    missing = ~np.isfinite(values)
    cadence = masked.time.cadence
    seconds = None if cadence is None else cadence.value/1e9
    rows = []
    for j, variable in enumerate(masked.values):
        positions = np.flatnonzero(missing[:,j])
        groups = []
        for p in positions:
            contiguous = (groups and p == groups[-1][-1]+1 and seconds is not None
                and int(masked.time.index.asi8[p])-int(masked.time.index.asi8[groups[-1][-1]]) == cadence.value)
            if contiguous:
                groups[-1].append(int(p))
            else:
                groups.append([int(p)])
        longest = max(map(len,groups),default=0)
        rows.append(dict(variable=variable, missing_cells=len(positions), positions=positions.tolist(),
            missing_seconds=None if seconds is None else len(positions)*seconds,
            longest_block_cells=longest if seconds is not None else None,
            longest_block_seconds=None if seconds is None else longest*seconds,
            blocks=groups, denominator=len(values)))
    return dict(variables=rows, shared_missing_positions=np.flatnonzero(missing.all(axis=1)).tolist(),
        partial_missing_positions=np.flatnonzero(missing.any(axis=1)&~missing.all(axis=1)).tolist(),
        missing_cells=int(missing.sum()), duration_convention="grid_cells" if seconds is not None else "unknown_irregular_coverage")


def compare_estimates(baseline, candidate, *, contrast):
    """Compare owned estimate records with explicit scientific signatures.

    This small record operation also permits independent literal controls. The
    public orchestration below derives records from actual methods and support.
    """
    for row in (baseline,candidate):
        if not isinstance(row,dict) or not isinstance(row.get("signature"),dict):
            raise ValueError("estimates require explicit signatures")
    a,b = owned(baseline),owned(candidate)
    differences = [key for key in sorted(set(a["signature"])|set(b["signature"]))
                   if identity(a["signature"].get(key)) != identity(b["signature"].get(key))]
    available = a.get("compute_status") == b.get("compute_status") == "ok" and a.get("estimate") is not None and b.get("estimate") is not None
    reason = "scientific_signature_mismatch" if differences else None if available else "unavailable_route"
    delta = None if reason else np.asarray(b["estimate"])-np.asarray(a["estimate"])
    if delta is not None and not np.isfinite(delta).all():
        raise ValueError("available estimates must be finite")
    use = "unknown_provenance" if "unknown_provenance" in (a.get("use_status"),b.get("use_status")) else "sensitivity_only"
    return dict(contrast=contrast, baseline_id=a.get("route_id"), candidate_id=b.get("route_id"),
        signed_delta=None if delta is None else delta.tolist(), absolute_delta=None if delta is None else abs(delta).tolist(),
        compute_status="incomparable" if differences else "unavailable" if reason else "ok", reason=reason,
        differences=differences, metric_status="descriptive_sensitivity", use_status=use,
        baseline=a, candidate=b, interpretation="descriptive contrast, not additive causal attribution")


def _interval_data(data, interval):
    if data is None:
        return None
    start,end = interval
    index = data.time.index
    return _subset(data,np.flatnonzero((index>=start)&(index<end)))


def _training_pair(training, training_positions, masked_training, interval, design):
    if design not in {"fixed", "allowed_training_refit"}:
        raise ValueError("design must be fixed or allowed_training_refit")
    if training is None or training_positions is None:
        raise ValueError("fitted preprocessing requires explicit training and training_positions")
    pos = positions_array(training_positions,len(training.values))
    selected = _subset(training,pos)
    if any(interval[0] <= t < interval[1] for t in selected.time.index):
        raise ValueError("training support overlaps evaluation interval")
    if design == "fixed":
        if masked_training is not None:
            raise ValueError("masked_training requires allowed_training_refit")
        return selected,selected
    if masked_training is None:
        raise ValueError("allowed refit requires explicit masked training data")
    _aligned(training,masked_training)
    finite = np.isfinite(masked_training.values.to_numpy(dtype=float,na_value=np.nan))
    if not np.array_equal(masked_training.values.to_numpy(dtype=float,na_value=np.nan)[finite],
                          training.values.to_numpy(dtype=float,na_value=np.nan)[finite]):
        raise ValueError("refit training may mask but cannot alter retained training values")
    return selected,_subset(masked_training,pos)


def _spectral_fit(spectrum, frequency, model, min_cycles):
    record = spectrum.summary()
    if record["compute_status"] != "ok":
        return None, record["compute_status"], record["reason"]
    value, support = record["value"],record["support"]
    irregular = record["method_id"].startswith("spectral.ls_projection")
    f = value["frequency_hz"] if irregular else value["physical_frequency_hz"]
    p = value["psd"] if irregular else value["physical_psd"]
    selected = []
    for requested in frequency:
        hits = np.flatnonzero(np.isclose(f,requested,rtol=1e-10,atol=0))
        if len(hits) != 1 or not np.isfinite(p[hits[0]]) or p[hits[0]] <= 0:
            return None,"unsupported_common_frequency","requested common frequency lacks a positive supported ordinate"
        j = int(hits[0])
        duration = support["frequency_support"][j]["min_occupied_span_seconds"] if irregular else support["longest_run_seconds"]
        if requested*duration < min_cycles:
            return None,"insufficient_cycles","common frequency fails declared cycles"
        selected.append(j)
    resolution = support["resolved"]["physical_resolution_hz"]
    if len(frequency)<3 or np.any(np.diff(frequency) < resolution*(1-1e-10)):
        return None,"insufficient_frequency_resolution","common fit requires three actual frequencies separated by physical resolution"
    fit = fit_log_models(frequency,p[selected],spec=model)
    fit["source_frequency_positions"] = selected
    return fit,fit["compute_status"],fit["reason"]


def spectral_sensitivity(complete, masked, completions, *, variables, welch, irregular, model,
                         fit_frequency_hz, min_cycles, daily_spec=None, training=None,
                         training_positions=None, design="fixed", masked_training=None):
    """Run W0/I0/Im and each Wc on a declared common fit domain; keep abstentions."""
    validate_completions(masked,completions,complete=complete)
    if not isinstance(welch,WelchSpec) or not isinstance(irregular,IrregularSpec) or not isinstance(model,ModelSpec):
        raise ValueError("explicit WelchSpec, IrregularSpec and ModelSpec required")
    variables = list(variables)
    if not variables or len(set(variables)) != len(variables) or any(v not in masked.metadata for v in variables):
        raise ValueError("explicit distinct known variables required")
    frequency = _real_array(fit_frequency_hz)
    if frequency.ndim != 1 or not len(frequency) or not np.isfinite(frequency).all() or np.any(frequency<=0) or np.any(np.diff(frequency)<=0):
        raise ValueError("common frequencies must be increasing positive finite")
    min_cycles = _positive(min_cycles,"min_cycles",zero=True)
    interval = irregular.record["interval"]
    if masked.time.mode != "regular":
        raise ValueError("four-way Welch comparison requires a declared regular slot grid")
    expected = pd.date_range(interval[0],interval[1],freq=masked.time.cadence,inclusive="left")
    if not _interval_data(masked,interval).time.index.equals(expected):
        raise ValueError("comparison interval must have an explicit complete slot index, retaining missing values")
    if (interval[0].value-irregular.record["anchor"].value)%irregular.record["hop"].value:
        raise ValueError("irregular anchor must align the first window with the comparison interval")
    if model.record["interval_hz"] is None or not np.allclose(model.record["interval_hz"],[frequency[0],frequency[-1]],rtol=1e-10,atol=0):
        raise ValueError("model interval must match the explicit common frequency endpoints")
    w,i = welch.record,irregular.record
    if (masked.time.mode != "regular" or w["recipe"] != "research_explicit" or w["scaling"] != "density"
            or w["taper"] != i["taper"] or not w["periodic"]
            or ("none" if w["detrend"] is False else w["detrend"]) != i["detrend"]
            or w["nperseg"]*masked.time.cadence != i["segment"]
            or (w["nperseg"]-(w["overlap"] if w["overlap"] is not None else w["nperseg"]//2))*masked.time.cadence != i["hop"]):
        raise ValueError("matched spectral comparison requires equal duration/hop/taper/detrend and regular supplied grid")
    refs = [None,None]
    if daily_spec is not None:
        original, altered = _training_pair(training,training_positions,masked_training,interval,design)
        refs[0] = fit_daily_reference(original,spec=daily_spec,training_positions=range(len(original.values)))
        refs[1] = refs[0] if design=="fixed" else fit_daily_reference(altered,spec=daily_spec,training_positions=range(len(altered.values)))
    elif training is not None or training_positions is not None or masked_training is not None or design != "fixed":
        raise ValueError("training/refit arguments require daily_spec")
    routes, contrasts = [],[]
    for variable in variables:
        group = {}
        inputs = [("W0",complete,None,False,refs[0]),("I0",complete,None,True,refs[0]),("Im",masked,None,True,refs[1])]
        inputs += [("Wc",data,draw,False,refs[1]) for draw,data in completions.items()]
        for route,data,draw,is_irregular,reference in inputs:
            signature = dict(interval=interval,basis="daily_residual" if reference else masked.metadata[variable]["representation"],
                unit=masked.metadata[variable]["unit"],variable=variable,model=model.record,frequency_hz=frequency,
                min_cycles=min_cycles,reference_id=reference.identity if reference else None)
            if model.record["basis"] is not None and model.record["basis"] != signature["basis"]:
                raise ValueError("fit model basis disagrees with preprocessing")
            row = dict(route=route,draw_id=draw,variable=variable,signature=signature,estimate=None,
                compute_status="unstarted" if route=="Wc" else "unavailable_reference",reason="no supplied draw" if route=="Wc" else "no complete empirical reference",
                use_status="sensitivity_only",metric_status="descriptive_sensitivity",source_id=None,
                reference_id=signature["reference_id"],spectrum=None,fit=None)
            if data is not None:
                selected = _interval_data(data,interval)
                observations = ObservationPolicy("observed_only" if route in {"W0","I0","Im"} else "provided")
                prepared = selected if reference is None else apply_daily_reference(selected,reference=reference,observations=observations)
                spectrum = (irregular_spectrum(prepared,variable=variable,spec=irregular,observations=observations)
                    if is_irregular else power_spectrum(prepared,variable=variable,spec=welch,observations=observations))
                fit,status,reason = _spectral_fit(spectrum,frequency,model,min_cycles)
                row.update(source_id=data.identity,spectrum=spectrum.summary(),fit=fit,compute_status=status,reason=reason,
                    estimate=fit["single"]["beta_low"] if fit is not None and status=="ok" else None,
                    use_status=spectrum.summary()["use_status"])
            row["route_id"] = identity(row)
            routes.append(row)
            group[(route,draw)] = row
        for label,a,b in (("baseline_estimator",("W0",None),("I0",None)),("deletion",("I0",None),("Im",None))):
            contrasts.append(compare_estimates(group[a],group[b],contrast=label))
        for draw in completions:
            for label,a in (("completion",("W0",None)),("mixed",("Im",None))):
                contrasts.append(compare_estimates(group[a],group[("Wc",draw)],contrast=label))
    return SensitivityReport(routes,contrasts,dict(method="sensitivity.four_route_spectral_v1",design=design,
        interval=interval,variable_order=variables,frequency_hz=frequency,geometry=gap_geometry(_interval_data(masked,interval)),
        references=[None if r is None else r.describe() for r in refs],
        completion_provenance={key:None if value is None else value.source["imputation"] for key,value in completions.items()},
        natural_gaps=complete is None,qualification="synthetic/empirical sensitivity, not reconstruction causality or ecological confidence"))


def summarize_draws(report):
    """Mean of separately computed draw metrics, with explicit exclusions."""
    rows = []
    for variable,group in report.routes.query("route == 'Wc'").groupby("variable",sort=False):
        available = group[(group.compute_status=="ok") & group.estimate.notna()]
        values = available.estimate.to_numpy(dtype=float)
        rows.append(dict(variable=variable,requested=len(group),denominator=len(values),
            excluded_draws=group.loc[~group.index.isin(available.index),"draw_id"].tolist(),
            mean=float(np.mean(values)) if len(values) else None,
            spread=float(np.ptp(values)) if len(values) else None,
            meaning="between-draw sensitivity, not a confidence interval"))
    return owned(pd.DataFrame(rows))


def compare_metric_results(baseline, candidate, *, baseline_interval, candidate_interval,
                           design="fixed"):
    """Compare existing FI1/PCA/graph/coherence records without vector-sign artifacts."""
    if not isinstance(baseline,ResultRecord) or not isinstance(candidate,ResultRecord):
        raise ValueError("comparison requires public ResultRecord inputs")
    if design not in {"fixed","retrospective_self","allowed_training_refit"}:
        raise ValueError("unknown comparison design")
    a,b = baseline.summary(),candidate.summary()
    if design != "fixed":
        scopes = [r["support"].get("fit_scope",r["support"].get("fit",{}).get("fit_scope")) for r in (a,b)]
        allowed = {"training_only"} if design=="allowed_training_refit" else {"self","global_self"}
        if any(scope not in allowed for scope in scopes):
            raise ValueError("reference-change design must match the actual recorded fit scopes")
    fields = {"fi1.ordered_mass_roughness_v1":["fi1","entropy_nats"],
        "pca.frozen_projection_v1":["eigenvalues","fractions","leading_value","leading_fraction","pc90","participation_ratio"],
        "graph.state_mass_v1":["energy"],"graph.state_density_v1":["energy"],
        "coherence.report_equal_run_v1":["band_mean"],"coherence.pooled_spectra_v1":["band_mean"]}
    if a["method_id"] not in fields or b["method_id"] not in fields:
        raise ValueError("unsupported metric family")
    def evaluation_instants(record):
        support = record["support"]
        method = record["method_id"]
        if method == "fi1.ordered_mass_roughness_v1":
            by_position = dict(zip(support["positions"],support["selected_instants"]))
            return [by_position[p] for p in support.get("evaluation_positions",support["positions"])]
        if method == "pca.frozen_projection_v1":
            return support["eligible_instants"]
        if method.startswith("graph."):
            return support["evaluation"]["eligible_instants"]
        return support["first"]["selected_instants"]+support["second"]["selected_instants"]
    def signature(record,interval):
        bounds = [_aware_timestamp(t,"interval") for t in interval]
        if len(bounds)!=2 or bounds[1]<=bounds[0]:
            raise ValueError("increasing comparison interval required")
        if any(not bounds[0] <= _aware_timestamp(t,"evaluation instant") < bounds[1]
               for t in evaluation_instants(record)):
            raise ValueError("actual evaluation support lies outside declared interval")
        support = record["support"]
        return dict(interval=bounds,method=record["method_id"],basis=record["representation"],
            units=record["units"],settings=support["requested"],
            run_weights=support.get("requested_run_weights"),
            reference_id=record["reference_id"] if design=="fixed" else "explicit_representation_change",
            feature_order=support.get("features",support.get("variables",support.get("variable"))),
            design=design)
    rows = []
    def result_id(record):
        value = record["value"]
        if value is not None and isinstance(value.get("scores"),pd.DataFrame):
            scores = value["scores"]
            value = dict(value,scores=dict(index=scores.index,columns=list(scores),values=scores.to_numpy()))
        return identity(dict(record,value=value))
    for field in fields[a["method_id"]]:
        records = []
        for rec,interval in ((a,baseline_interval),(b,candidate_interval)):
            records.append(dict(route_id=result_id(rec),compute_status=rec["compute_status"],use_status=rec["use_status"],
                estimate=None if rec["value"] is None else rec["value"].get(field),signature=signature(rec,interval),result=rec))
        row = compare_estimates(*records,contrast=field)
        row["representation_changed"] = a["reference_id"] != b["reference_id"]
        row["reference_ids"] = [a["reference_id"],b["reference_id"]]
        rows.append(row)
    return owned(pd.DataFrame(rows))


def compare_networks(baseline,candidate):
    """Align signed edges by actual window endpoints and node IDs, never by zeros."""
    a,b = baseline.describe(),candidate.describe()
    differences = [key for key in ("features","units","basis","requested","reference_ids","start","end") if identity(a[key])!=identity(b[key])]
    def edges(result):
        windows = {r["window"]:(r["start"],r["end"]) for r in result.summary.to_dict("records")}
        return {(windows[r["window"]],r["a"],r["b"]):r for r in result.edges.to_dict("records")}
    first,second = edges(baseline),edges(candidate)
    rows = []
    for key in sorted(set(first)|set(second),key=str):
        left,right = first.get(key),second.get(key)
        good = not differences and left is not None and right is not None and left["compute_status"]==right["compute_status"]=="ok"
        rows.append(dict(window=key[0],a=key[1],b=key[2],baseline=left,candidate=right,
            signed_delta=right["correlation"]-left["correlation"] if good else None,
            absolute_delta=abs(right["correlation"]-left["correlation"]) if good else None,
            support_changed=left is None or right is None or left["instants"]!=right["instants"],
            edge_inclusion_changed=(left["strong"]!=right["strong"]) if good else None,
            compute_status="incomparable" if differences else "ok" if good else "unavailable",
            differences=differences,metric_status="descriptive_sensitivity",
            use_status="unknown_provenance" if any(r and r["use_status"]=="unknown_provenance" for r in (left,right)) else "sensitivity_only"))
    summaries = []
    first_summary = {(r["start"],r["end"]):r for r in baseline.summary.to_dict("records")}
    second_summary = {(r["start"],r["end"]):r for r in candidate.summary.to_dict("records")}
    for window in sorted(set(first_summary)|set(second_summary),key=str):
        left,right = first_summary.get(window),second_summary.get(window)
        good = not differences and left is not None and right is not None and left["compute_status"]==right["compute_status"]=="ok"
        for field in ("mean_absolute_connectivity","strong_count","density"):
            delta = right[field]-left[field] if good else None
            summaries.append(dict(window=window,metric=field,signed_delta=delta,
                absolute_delta=None if delta is None else abs(delta),baseline=left,candidate=right,
                compute_status="incomparable" if differences else "ok" if good else "unavailable"))
    return owned(dict(edges=pd.DataFrame(rows),summaries=pd.DataFrame(summaries),differences=differences))

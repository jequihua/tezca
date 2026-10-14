"""Analyze synthetic inputs, save all result families, then reload in a new process.

Installed use: python -I -B delivery.py analyze NEW_DIRECTORY [--completion supplied]
Then: python -I -B delivery.py reload DIRECTORY --expected-identity DIRECTORY.identity.json
The companion result_exports.py supplies the small, explicit A-D baseline.
"""

import argparse
import json
from pathlib import Path
import runpy


def build_results(completion="observed", hidden_canary=987654321.0):
    """Return public export arguments; completions are supplied, never fitted here."""
    import numpy as np
    import pandas as pd
    from tezca.data import AnalysisSeries
    from tezca.exports import ResultItem
    from tezca.individual import DailySpec, fit_daily_reference, apply_daily_reference, describe_series
    from tezca.spectral import WelchSpec, power_spectrum
    from tezca.spectral_summary import BandSpec, band_power, ModelSpec, spectral_models, PeakSpec, spectral_peaks
    from tezca.irregular import IrregularSpec
    from tezca.sensitivity import spectral_sensitivity, validate_completions
    from tezca.regimes import RegimeFeatures, CustomClusterer, cluster_regimes, assign_regimes, summarize_regime_timeline
    from tezca.support import ObservationPolicy

    if completion not in {"observed", "supplied"}:
        raise ValueError("choose observed or supplied completion")
    arguments = runpy.run_path(str(Path(__file__).with_name("result_exports.py")))["build_results"]()
    sources, results = arguments["sources"], arguments["results"]
    start = pd.Timestamp("2000-02-01", tz="UTC")

    def series(values, name, *, time=None, **provenance):
        values = np.asarray(values, dtype=float)
        return AnalysisSeries(values, variables=["x"],
            time=pd.date_range(start, periods=len(values), freq="h") if time is None else time,
            metadata={"x": dict(unit="m", representation="raw")},
            source=dict(id=name, segment="synthetic", **provenance.pop("source", {})),
            clock=dict(mode="regular", cadence="1h"),
            **(provenance or dict(observed=np.isfinite(values))))

    def add(name, kind, result, selected, **extra):
        results[name] = ResultItem(kind, result, sources=selected, **extra)

    # The first two days are the only reference-fit inputs. Day 3/4 offsets
    # (100/102) discriminate accidental fitting on the evaluation population.
    training = series((np.arange(24) + np.array([0, 2])[:, None]).ravel(), "daily-training",
        time=pd.date_range(start-pd.Timedelta("48h"), periods=48, freq="h"))
    values = (np.arange(24) + np.array([100, 102, 104, 106])[:, None]).ravel().astype(float)
    values += np.tile([0.,20.,0.,-20.],24)
    hidden = np.array([4, 5, 28, 29])
    # This canary represents withheld validation truth. Strip it before the
    # public data boundary; neither reference fitting nor any export sees it.
    values[hidden] = hidden_canary
    values[hidden] = np.nan
    artificial = np.zeros(96, dtype=bool)
    artificial[hidden] = True
    masked = series(values, "daily-masked", contribution=np.where(np.isfinite(values), "observed", "unavailable"),
        qualification={"artificial": artificial})
    supplied = {}
    if completion == "supplied":
        for name, replacements in (("low", [11., 13., 17., 19.]), ("high", [12., 14., 18., 20.])):
            filled = masked.values.to_numpy().ravel()
            filled[hidden] = replacements  # explicitly supplied independent numbers
            supplied[name] = series(filled, "completion-"+name,
                contribution=np.where(artificial, "imputed", "observed"), qualification=masked.qualification,
                source=dict(imputation=dict(input_id=masked.identity, draw_id=name,
                    method_id="external_literal_completion", plan_id="delivery-synthetic-v1"), trusted_imputation=True))
        validate_completions(masked, supplied)
    selected = masked if completion == "observed" else supplied["low"]
    sources.update(training=training, daily_masked=masked, evaluation=selected)
    sources.update({"draw_"+k: v for k, v in supplied.items()})
    policy = ObservationPolicy("observed_only" if completion == "observed" else "provided")
    daily = fit_daily_reference(training, spec=DailySpec(civil_timezone="UTC"), training_positions=range(48))
    residual = apply_daily_reference(selected, reference=daily, observations=policy)
    retrospective = apply_daily_reference(selected, spec=DailySpec(civil_timezone="UTC"), observations=policy)
    sources.update(residual=residual, retrospective=retrospective)
    arguments["references"].append(daily)
    for label, source in (("evaluation", selected), ("residual", residual), ("retrospective", retrospective)):
        for index, metric in enumerate(describe_series(source, observations=policy)):
            add(label+"_"+str(index), "metric", metric, [label])
        add(label+"_welch", "spectrum", power_spectrum(source, variable="x",
            spec=WelchSpec(16, overlap=8, taper="boxcar", detrend=False), observations=policy), [label])
    add("too_many_cycles", "metric", band_power(
        power_spectrum(sources["signal"], variable="x", spec=WelchSpec(32, overlap=16, taper="boxcar", detrend=False)),
        spec=BandSpec((2, 8), min_cycles=100)), ["signal"])

    frequency = np.arange(2, 7)/(16*3600)
    settings = dict(variables=["x"], welch=WelchSpec(16, overlap=8, taper="boxcar", detrend=False),
        irregular=IrregularSpec(interval=[start, start+pd.Timedelta("96h")], segment="16h", hop="8h", anchor=start,
            frequency_hz=frequency, requested_band_hz=[frequency[0], frequency[-1]], f_ref_hz=1/3600, min_span="8h"),
        model=ModelSpec(interval_hz=[frequency[0], frequency[-1]], candidates_hz=[]), fit_frequency_hz=frequency,
        min_cycles=1, daily_spec=DailySpec(civil_timezone="UTC"), training=training, training_positions=range(48))
    training_values = training.values.to_numpy().ravel()
    training_values[0] = np.nan
    altered_training = series(training_values, "masked-training", time=training.time.index)
    for design in ("fixed", "allowed_training_refit"):
        extra = dict(masked_training=altered_training) if design != "fixed" else {}
        report = spectral_sensitivity(None, masked, supplied, design=design, **settings, **extra)
        add("daily_"+design, "sensitivity", report, ["daily_masked", *["draw_"+k for k in supplied]])

    # A tiny custom estimator demonstrates genuine frozen assignment with core
    # dependencies only. Nominal strings and noise -1 keep their native types.
    SignLabels = runpy.run_path(str(Path(__file__).with_name("delivery_labels.py")),
                               run_name="delivery_labels")["SignLabels"]

    def features(data, prefix):
        return RegimeFeatures(data, features=["x"], row_ids=[prefix+str(i) for i in range(len(data.values))],
            definitions={"x": dict(method_id="synthetic_coordinate_v1", reference_id=None, window=dict(kind="point"))})

    train_regimes = series([-2., -1., 1., 2.], "regime-training")
    future = series([-2., -2., np.nan, 0., 2., 2.], "regime-future",
        time=pd.DatetimeIndex([start+pd.Timedelta(hours=h) for h in (24, 25, 26, 34, 35, 36)]))
    fitted = cluster_regimes(features(train_regimes, "train"), clusterer=CustomClusterer(SignLabels,
        config=dict(rule="sign_relative_to_training_mean"), version="1", assignment=True, noise_labels=[-1]),
        preprocess=None, fit_scope="training_only", training_ids=["train"+str(i) for i in range(4)])
    assigned = assign_regimes(features(future, "future"), reference=fitted.reference)
    sources.update(regime_training=train_regimes, regime_future=future)
    arguments["references"].append(fitted.reference)
    for name, result, source in (("regime_fit", fitted, "regime_training"), ("regime_assignment", assigned, "regime_future")):
        add(name, "regimes", result, [source], timeline=summarize_regime_timeline(result, max_gap="1h", noise_policy="barrier"))

    # Carry the chosen observed/provided input into every stage, not just the
    # sensitivity comparison. Alignment exposes gaps without completing them.
    from tezca.fi1 import FI1Spec, fi1_windows
    from tezca.density_fisher import HistogramSpec, density_fisher
    from tezca.collective import CollectionMember, FeatureSpec, prepare_collective
    from tezca.pca import PCASpec, fit_pca_reference, project_pca
    from tezca.network import NetworkSpec, correlation_network
    from tezca.graph_state import graph_state_grid
    from tezca.coherence_modern import ModernCoherenceSpec, modern_coherence
    from tezca.pair_audit import PairLineage, signal_identity
    selected_spectrum = power_spectrum(selected, variable="x", observations=policy,
        spec=WelchSpec(16, overlap=8, taper="boxcar", detrend=False))
    add("evaluation_fit", "metric", spectral_models(selected_spectrum, spec=ModelSpec(candidates_hz=[])), ["evaluation"])
    add("evaluation_peaks", "peaks", spectral_peaks(selected_spectrum, spec=PeakSpec()), ["evaluation"])
    add("evaluation_fi1", "fi1", fi1_windows(selected, variable="x", anchor=start,
        spec=FI1Spec([-10,50,150]), window="24h", step="12h", observations=policy), ["evaluation"])
    add("evaluation_density", "density", density_fisher(selected, variable="x",
        spec=HistogramSpec([-10,50,150], measurement="rounded_continuous"), observations=policy), ["evaluation"])
    companion = series(np.cos(np.arange(96)*.37), "companion")
    sources["companion"] = companion
    members = {name:CollectionMember(data, variable="x", station="synthetic", family="synthetic",
        feature=name, semantics="state") for name,data in (("x",selected),("y",companion))}
    prepared = prepare_collective(members, features=[FeatureSpec(v,(v,)) for v in ("x","y")],
        start=start, end=start+pd.Timedelta("96h"), cadence="1h", observations=policy)
    add("evaluation_matrix", "matrix", prepared, ["evaluation","companion"])
    sources["prepared"] = prepared.series
    projection = fit_pca_reference(prepared.series, spec=PCASpec(features=["x","y"],basis="raw",scaling="none",ddof=1),
        training_positions=range(48), observations=policy)
    arguments["references"].append(projection)
    add("evaluation_pca", "pca", project_pca(prepared.series, reference=projection, observations=policy), ["prepared"])
    add("evaluation_network", "network", correlation_network(prepared.series,
        spec=NetworkSpec(features=["x","y"],basis="raw"), observations=policy), ["prepared"])
    add("evaluation_graph", "graph", graph_state_grid(prepared.series, features=["x","y"],basis="raw",
        max_states=100, observations=policy), ["prepared"])
    lineage = PairLineage({
        "a":dict(kind="reconstruction" if supplied else "measurement",signal_id=signal_identity(selected,"x"),
                 parents=["masked"] if supplied else [],donors=["external_literals"] if supplied else []),
        "masked":dict(kind="measurement",signal_id=signal_identity(masked,"x"),parents=[],donors=[]),
        "b":dict(kind="measurement",signal_id=signal_identity(companion,"x"),parents=[],donors=[])})
    add("evaluation_coherence", "coherence", modern_coherence(selected,companion,variable_a="x",variable_b="x",
        spec=ModernCoherenceSpec(basis="raw",period_band=("4h","16h"),nperseg=16,hop=8,taper="boxcar",detrend="none"),
        observations=policy,lineage=lineage,nodes=["a","b"]), ["evaluation","companion"])
    selected_regimes = assign_regimes(features(selected,"evaluation"),reference=fitted.reference,observations=policy)
    add("evaluation_regimes", "regimes", selected_regimes, ["evaluation"],
        timeline=summarize_regime_timeline(selected_regimes,max_gap="1h",noise_policy="barrier"))
    arguments["requested_policy"] = dict(example="delivery_v1", completion=completion)
    arguments["resolved_policy"] = dict(methods="explicit_per_result_specs", daily_training="separate_prior_48_hours",
        completion="external_literals" if supplied else "none", scientific_claim="synthetic_local_candidate")
    return arguments


def identity_record(manifest):
    """Keep this outside the bundle as the caller's expected selection."""
    return dict(expected_bundle_id=manifest["bundle_id"], expected_sources=manifest["source_identities"],
                expected_references=manifest["reference_identities"])


def summarize(loaded):
    results = loaded.details["results"]
    return dict(bundle_id=loaded.manifest["bundle_id"], families=len(loaded.manifest["families"])+1,
        eigenvalues=results["pca"]["record"]["value"]["eigenvalues"].tolist(),
        integrated_power=float(sum(results["welch"]["record"]["value"]["psd"])/(32*3600)),
        withheld_band=results["synoptic"]["record"]["compute_status"],
        withheld_reason=results["synoptic"]["record"]["reason"],
        daily_reference=results["residual_0"]["record"]["reference_id"],
        regime_labels=[row["native_label"] for row in results["regime_assignment"]["rows"]])


def main():
    from tezca.exports import export_results, load_results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "reload"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--completion", choices=("observed", "supplied"), default="observed")
    parser.add_argument("--expected-identity", type=Path)
    args = parser.parse_args()
    if args.action == "analyze":
        if args.expected_identity:
            parser.error("expected identity is a reload option")
        expected_path = args.directory.with_name(args.directory.name+".identity.json")
        if expected_path.exists():
            raise FileExistsError("expected identity destination already exists")
        manifest = export_results(args.directory, **build_results(args.completion))
        with expected_path.open("x", encoding="utf-8") as stream:
            json.dump(identity_record(manifest), stream, sort_keys=True)
        print(json.dumps(dict(bundle_id=manifest["bundle_id"], families=len(manifest["families"])+1)))
    else:
        expected = {} if args.expected_identity is None else json.loads(args.expected_identity.read_text(encoding="utf-8"))
        print(json.dumps(summarize(load_results(args.directory, **expected))))


if __name__ == "__main__":
    main()

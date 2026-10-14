"""Small explicit synthetic result producer and independent reload command.

Run after installation: python -I -B result_exports.py build NEW_DIRECTORY
Then, in a new process: python -I -B result_exports.py reload DIRECTORY
"""

import argparse
import json


def build_results():
    import numpy as np
    import pandas as pd
    from tezca.data import AnalysisSeries
    from tezca.exports import ResultItem
    from tezca.individual import describe_series, estimate_trend, autocorrelation
    from tezca.spectral import power_spectrum, WelchSpec
    from tezca.spectral_summary import spectral_models, spectral_peaks, band_power, ModelSpec, PeakSpec, BandSpec
    from tezca.fi1 import FI1Spec, fi1_windows
    from tezca.density_fisher import HistogramSpec, density_fisher
    from tezca.harmonic import HarmonicSpec, harmonic_diagnostics
    from tezca.collective import CollectionMember, FeatureSpec, prepare_collective
    from tezca.pca import PCASpec, fit_pca_reference, project_pca
    from tezca.network import NetworkSpec, correlation_network
    from tezca.coherence import ReportCoherenceSpec, report_coherence
    from tezca.coherence_modern import ModernCoherenceSpec, modern_coherence
    from tezca.graph_state import graph_state_grid
    from tezca.irregular import IrregularSpec, irregular_spectrum
    from tezca.sensitivity import spectral_sensitivity
    from tezca.regimes import RegimeFeatures, CustomClusterer, cluster_regimes, summarize_regime_timeline
    from tezca.support import ObservationPolicy

    start = pd.Timestamp("2000-01-01", tz="UTC")
    def data(values, name, names=("x",), contribution=None):
        values = np.asarray(values, dtype=float).reshape(-1, len(names))
        return AnalysisSeries(values, variables=list(names), time=pd.date_range(start, periods=len(values), freq="h"),
            metadata={v: dict(unit="m", representation="raw") for v in names}, source=dict(id=name, segment="synthetic"),
            clock=dict(mode="regular", cadence="1h"), contribution=contribution,
            observed=np.isfinite(values) if contribution is None else None)
    t = np.arange(128)
    signal = data(np.cos(2*np.pi*t/8), "export-signal")
    missing = data([np.nan]*8, "all-null")
    unknown = data([1, 2, 3, 4], "unknown", contribution=["unknown"]*4)
    matrix = data([[np.sqrt(6), 0], [-np.sqrt(6), 0], [0, np.sqrt(1.5)], [0, -np.sqrt(1.5)]], "matrix", ("x", "y"))
    sources = dict(signal=signal, missing=missing, unknown=unknown, matrix=matrix)
    results = {}
    def add(name, kind, result, source="signal", **kwargs):
        results[name] = ResultItem(kind, result, sources=[source] if isinstance(source, str) else source,
                                  labels=dict(site="synthetic", variable="x"), **kwargs)
    for name, src in (("signal", signal), ("missing", missing), ("unknown", unknown)):
        for i, result in enumerate(describe_series(src, observations=ObservationPolicy("provided"))):
            add(name+str(i), "metric", result, name)
    add("trend", "metric", estimate_trend(signal, variable="x"))
    add("acf", "metric", autocorrelation(signal, variable="x", lag="one_hour"))
    welch = WelchSpec(32, overlap=16, taper="boxcar", detrend=False)
    spectrum = power_spectrum(signal, variable="x", spec=welch)
    add("welch", "spectrum", spectrum)
    add("fit", "metric", spectral_models(spectrum, spec=ModelSpec(candidates_hz=[])))
    add("peaks", "peaks", spectral_peaks(spectrum, spec=PeakSpec()))
    add("synoptic", "metric", band_power(spectrum, spec=BandSpec((48, 240))))
    for i, result in enumerate(harmonic_diagnostics(signal, variable="x", spec=HarmonicSpec()).values()):
        add("harmonic"+str(i), "metric", result)
    gap_values = np.tile([0., 1., 2., 3.], 4)
    gap_values[5] = np.nan
    gaps = data(gap_values, "gaps")
    sources["gaps"] = gaps
    add("fi1", "fi1", fi1_windows(gaps, variable="x", anchor=start, spec=FI1Spec([-1, 1, 4]), window="4h", step="2h"), "gaps")
    add("density", "density", density_fisher(signal, variable="x", spec=HistogramSpec([-2, 0, 2], measurement="rounded_continuous")))
    members = {v: CollectionMember(matrix, variable=v, station="station", family="family", feature=v, semantics="state") for v in ("x", "y")}
    prepared = prepare_collective(members, features=[FeatureSpec(v, (v,)) for v in ("x", "y")], start=start, end=start+pd.Timedelta("4h"), cadence="1h")
    add("matrix", "matrix", prepared, "matrix")
    ref = fit_pca_reference(matrix, spec=PCASpec(features=["x", "y"], basis="raw", scaling="none", ddof=1), training_positions=[0, 1, 2, 3])
    add("pca", "pca", project_pca(matrix, reference=ref), "matrix")
    add("network", "network", correlation_network(matrix, spec=NetworkSpec(features=["x", "y"], basis="raw")), "matrix")
    add("coherence", "coherence", modern_coherence(signal, signal, variable_a="x", variable_b="x", spec=ModernCoherenceSpec(basis="raw", period_band=("4h", "32h"), nperseg=16, hop=8, taper="boxcar", detrend="none")))
    add("report", "coherence", report_coherence(signal, signal, variable_a="x", variable_b="x", spec=ReportCoherenceSpec(basis="raw", period_band=("4h", "32h"), taper="hann", detrend="constant")))
    add("grid", "graph", graph_state_grid(matrix, features=["x", "y"], basis="raw", max_states=100), "matrix")
    frequency = np.arange(2, 15)/(32*3600)
    irregular = IrregularSpec(interval=[start, start+pd.Timedelta("128h")], segment="32h", hop="16h", anchor=start,
        frequency_hz=frequency, requested_band_hz=[frequency[0], frequency[-1]], f_ref_hz=1/3600, min_span="16h")
    add("irregular", "spectrum", irregular_spectrum(signal, variable="x", spec=irregular))
    masked_values = signal.values
    masked_values.iloc[20] = np.nan
    masked = data(masked_values.to_numpy(), "masked")
    sources["masked"] = masked
    add("sensitivity", "sensitivity", spectral_sensitivity(signal, masked, {}, variables=["x"], welch=welch, irregular=irregular,
        model=ModelSpec(interval_hz=[frequency[0], frequency[-1]], candidates_hz=[]), fit_frequency_hz=frequency, min_cycles=1),
        ["signal", "masked"])

    class Labels:
        def fit_predict(self, frame):
            return np.array([9007199254740993, -1, 9007199254740995, "NA"], dtype=object)[:len(frame)]
    features = RegimeFeatures(matrix, features=["x", "y"], row_ids=["a", "9007199254740997", "c", "d"],
        definitions={v: dict(method_id="synthetic_coordinate_v1", reference_id=None, window=dict(kind="point")) for v in ("x", "y")})
    regimes = cluster_regimes(features, clusterer=CustomClusterer(Labels, config=dict(fixture="literal_labels"), version="1", noise_labels=[-1]), preprocess=None)
    add("regimes", "regimes", regimes, "matrix", timeline=summarize_regime_timeline(regimes, max_gap="1h", noise_policy="barrier"))
    return dict(results=results, sources=sources, references=[ref], requested_policy=dict(example="explicit_synthetic_v1"),
                resolved_policy=dict(observation="provided_only_where_explicit", methods="per_result_record"), compatibility=True)


def main():
    from tezca.exports import export_results, load_results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "reload"))
    parser.add_argument("directory")
    args = parser.parse_args()
    if args.action == "build":
        manifest = export_results(args.directory, **build_results())
        print(json.dumps(dict(bundle_id=manifest["bundle_id"], families=len(manifest["families"])+1)))
    else:
        loaded = load_results(args.directory)
        print(json.dumps(dict(bundle_id=loaded.manifest["bundle_id"], eigenvalues=loaded.details["results"]["pca"]["record"]["value"]["eigenvalues"].tolist())))


if __name__ == "__main__":
    main()

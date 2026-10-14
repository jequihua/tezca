"""Bounded synthetic example-local build/reload, not a general export schema."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build():
    from itertools import product
    import numpy as np
    import pandas as pd
    import tezca
    from tezca.collective import CollectionMember, FeatureSpec, prepare_collective
    from tezca.coherence import ReportCoherenceSpec, report_coherence
    from tezca.coherence_modern import ModernCoherenceSpec, modern_coherence
    from tezca.data import AnalysisSeries
    from tezca.graph_state import GraphStateSpec, fit_state_reference, graph_state_roughness, graph_state_grid
    from tezca.network import NetworkSpec, correlation_network
    from tezca.pair_audit import PairLineage, signal_identity
    from tezca.pca import PCASpec, fit_pca_reference, project_pca
    from tezca.support import ObservationPolicy

    def data(values, names, name, *, cadence="1h", contribution=None):
        values = np.asarray(values, dtype=float).reshape(-1, len(names))
        return AnalysisSeries(values, variables=names,
            time=pd.date_range("2000-01-01", periods=len(values), freq=cadence, tz="UTC"),
            metadata={v: dict(unit="m", representation="raw") for v in names},
            source=dict(id="synthetic-"+name, segment="collective-example"),
            clock=dict(mode="regular", cadence=cadence), contribution=contribution,
            observed=np.ones(values.shape, dtype=bool) if contribution is None else None)

    rows = [[np.sqrt(6), 0], [-np.sqrt(6), 0], [0, np.sqrt(1.5)], [0, -np.sqrt(1.5)]]
    source = data(rows, ["x", "y"], "matrix")
    collection = {v: CollectionMember(source, variable=v, station="synthetic-station", family="synthetic-family",
                                      feature=v, semantics="state") for v in ("x", "y")}
    prepared = prepare_collective(collection, features=[FeatureSpec(v, (v,)) for v in ("x", "y")],
        start=source.values.index[0], end=source.values.index[-1]+pd.Timedelta(hours=1), cadence="1h")
    matrix = prepared.series
    pca = fit_pca_reference(matrix, spec=PCASpec(features=["x", "y"], basis="raw", scaling="none", ddof=1), training_positions=list(range(4)))
    projection = project_pca(matrix, reference=pca)
    network_data = data(np.column_stack([np.arange(4), np.arange(4), -np.arange(4)]), list("xyz"), "network")
    network = correlation_network(network_data, spec=NetworkSpec(features=list("xyz"), basis="raw")).summary.iloc[0]
    t = np.arange(256)
    report_pair = [data(v, ["x"], name) for name, v in (("report-a", np.sin(.17*t)+.3*np.cos(.43*t)), ("report-b", np.sin(.17*t)+.2*np.cos(.31*t)))]
    def lineage(pair, copy=False):
        return PairLineage({key: dict(signal_id=signal_identity(value, "x"), kind="copy" if copy and key == "b" else "measurement",
                                      parents=["a"] if copy and key == "b" else [], donors=[])
                            for key, value in zip(("a", "b"), pair)})
    report_spec = ReportCoherenceSpec(basis="raw", period_band=("4h", "64h"), taper="hann", detrend="constant")
    report = report_coherence(*report_pair, variable_a="x", variable_b="x", spec=report_spec, lineage=lineage(report_pair), nodes=["a", "b"])
    t = np.arange(1024)
    phase_pair = [data(v, ["x"], name, cadence="1s") for name, v in (("phase-a", np.cos(2*np.pi*8*t/128)), ("phase-b", np.cos(2*np.pi*8*(t-3)/128)))]
    modern_spec = ModernCoherenceSpec(basis="raw", period_band=("4s", "64s"), nperseg=128, hop=128, taper="boxcar", detrend="none")
    modern = modern_coherence(*phase_pair, variable_a="x", variable_b="x", spec=modern_spec, lineage=lineage(phase_pair), nodes=["a", "b"])
    short_pair = [data([0, 1], ["x"], name) for name in ("short-a", "short-b")]
    insufficient = report_coherence(*short_pair, variable_a="x", variable_b="x", spec=report_spec, lineage=lineage(short_pair), nodes=["a", "b"])
    excluded = report_coherence(*report_pair, variable_a="x", variable_b="x", spec=report_spec, lineage=lineage(report_pair, copy=True), nodes=["a", "b"])
    unknown_data = data([.5, 2.5], ["x"], "unknown", contribution=["unknown", "observed"])
    state_spec = GraphStateSpec(features=["x"], basis="raw", partition="custom", edges=[[0, 1, 2, 3]], outside="exclude")
    unknown = graph_state_roughness(unknown_data, spec=state_spec, observations=ObservationPolicy("provided"))
    states = data([.5, 1.5, 2.5], ["x"], "state-training")
    state_ref = fit_state_reference(states, spec=state_spec, training_positions=[0, 1, 2])
    state = graph_state_roughness(data([.5, 2.5], ["x"], "state-evaluation"), reference=state_ref)
    grid = graph_state_grid(data(list(product((-1., 1.), repeat=3)), list("xyz"), "grid"), features=list("xyz"), basis="raw", max_states=100)
    # Defensive ownership checks are part of the user path, before serialization.
    view = prepared.matrix
    view.iloc[0, 0] = 999
    saved = state.summary()
    saved["value"]["counts"][:] = 999
    pca_view = pca.describe()
    pca_view["center"][:] = 999
    owned_outputs = bool(prepared.matrix.iloc[0, 0] != 999 and np.all(state.value["counts"] != 999) and np.all(pca.describe()["center"] != 999))
    grid_rows = [dict(pcs=row["pcs"], bins=row["bins"], status=row["compute_status"], energy=row["energy"],
                      reference_id=row["reference_id"], reason=row["reason"]) for row in grid.describe()["rows"]]
    package_path = Path(tezca.__file__).resolve()
    checkout = Path(__file__).resolve().parents[2]
    return dict(schema="collective_example_v1", synthetic_only=True,
        installed_outside_checkout=not package_path.is_relative_to(checkout), ecofill_absent=importlib.util.find_spec("ecofill") is None,
        matrix=prepared.matrix.to_numpy().tolist(), eigenvalues=projection.value["eigenvalues"].tolist(),
        participation_ratio=projection.value["participation_ratio"], pc90=projection.value["pc90"],
        network=network.value, report=dict(status=report.summary()["compute_status"], nperseg=report.summary()["support"]["runs"][0]["nperseg"],
                                         raw_segments=report.summary()["support"]["runs"][0]["raw_segments"]),
        modern=dict(phase_bin8=float(modern.summary()["support"]["pooling"]["spectra"]["phase_rad"][8]),
                    coherence_bin8=float(modern.summary()["support"]["pooling"]["spectra"]["raw_coherence"][8])),
        graph=dict(energy=state.value["energy"], masses=state.value["observed_masses"].tolist(), nodes=state.value["nodes"].tolist(),
                   components=state.value["connected_components"], grid=grid_rows),
        unavailable=dict(insufficient=dict(status=insufficient.summary()["compute_status"], value=insufficient.value),
                         unknown=dict(status=unknown.summary()["compute_status"], use_status=unknown.summary()["use_status"], value=unknown.value)),
        lineage_excluded=not excluded.summary()["support"]["audit"]["ranking_eligible"],
        references=dict(pca=pca.identity, projected=projection.summary()["reference_id"], state=state_ref.identity, graph=state.summary()["reference_id"]),
        support=dict(matrix_id=prepared.identity, pca_training_positions=pca.describe()["support"]["original_eligible_positions"],
                     graph_training_positions=state.summary()["support"]["training"]["original_eligible_positions"],
                     graph_evaluation_positions=state.summary()["support"]["contributing_positions"]),
        owned_outputs=owned_outputs, limits=["unresolved_Q001", "unresolved_Q002", "unresolved_Q003", "exploratory_not_ecological_validation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "reload"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        payload = build()
        args.path.write_text(json.dumps(dict(payload=payload, sha256=_digest(payload)), sort_keys=True, allow_nan=False), encoding="utf-8")
        print(json.dumps(dict(written=True, schema=payload["schema"])))
    else:
        envelope = json.loads(args.path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if payload.get("schema") != "collective_example_v1" or _digest(payload) != envelope["sha256"]:
            raise ValueError("example summary version or digest mismatch")
        print(json.dumps(dict(payload=payload, analysis_imported=any(k == "tezca" or k.startswith("tezca.") for k in sys.modules)), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

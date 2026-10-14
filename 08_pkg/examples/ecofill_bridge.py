"""Installed, synthetic producer -> export -> fresh-process Tezca reload.

python -I -B ecofill_bridge.py produce <new-directory>  # EcoFill 0.0.1, Python 3.12
python -I -B ecofill_bridge.py reload <directory>       # EcoFill is not required

Production creates the original EcoFill export and its sibling <directory>-results
Tezca bundle. Reload checks both, including complete cell provenance and masks;
--expected-bundle-id adds comparison against a previously recorded bundle identity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tezca.adapters.ecofill import from_ecofill, load_ecofill_export
from tezca.exports import ResultItem, export_results, load_results
from tezca.support import ObservationPolicy, select_support


OPTIONS = {"source": {"id": "synthetic-ecofill-example-v1", "segment": "eight-slots"},
           "representation": "raw", "clock": {"civil_timezone": "UTC", "clock_resolution": "resolved"}}


def snapshot(series):
    """JSON semantic comparison; only the documented input route is excluded."""
    frame = series.values
    values = frame.astype(object).where(frame.notna(), None).to_numpy().tolist()
    source = series.source
    del source["ecofill"]["route"]
    details = series.source_details
    details["cells"] = {field: table.to_numpy().tolist() for field, table in details["cells"].items()}
    return {"values": values, "timestamps": [t.isoformat() for t in frame.index],
            "variables": list(frame.columns), "metadata": series.metadata, "source": source,
            "source_details": details,
            "qualification": {name: mask.tolist() for name, mask in series.qualification.items()},
            "contribution": series.contribution.tolist(),
            "observed_only_positions": select_support(series, observations=ObservationPolicy("observed_only")).positions.tolist()}


def produce(directory):
    from ecofill import SeriesData, TimeSeriesSchema, VariableSpec, make_plan, ImputationResult
    index = pd.date_range("2000-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame({"x": [10, 999999, 30, np.nan, np.nan, np.nan, 70, 80]}, index=index)
    schema = TimeSeriesSchema(pd.Timedelta("1h"), (VariableSpec("x", "K"),))
    def mask(positions):
        return pd.DataFrame({"x": [i in positions for i in range(8)]}, index=index)
    data = SeriesData(frame, schema=schema, observed=mask([0, 1, 6, 7]), artificial=mask([1]),
                      qc_rejected=mask([2]), structural=mask([3]), inserted=mask([4]))
    plan = make_plan(data, method="persistence", mode="offline", max_gap=pd.Timedelta("4h"))
    completed = pd.DataFrame({"x": [10, 20, np.nan, np.nan, 50, 60, 70, 80]}, index=index)
    states = pd.DataFrame({"x": ["observed", "imputed", "qc_rejected", "structural", "imputed", "imputed", "observed", "observed"]}, index=index)
    methods = pd.DataFrame({"x": [None, "NA", None, None, "synthetic-second", "NA", None, None]}, index=index)
    reasons = pd.DataFrame({"x": [None, None, "NA", "structural_absence", None, None, None, None]}, index=index)
    quality = pd.DataFrame({"x": [np.nan, 1, np.nan, np.nan, 2, 3, np.nan, np.nan]}, index=index)
    result = ImputationResult(data, plan, completed, states, methods, reasons, mask([1, 4, 5]),
                              backend_identity={"name": "synthetic-supplied", "configuration": {"tags": ["owned"]}}, quality=quality)
    before_available = data.available
    before_completed = result.completed
    before_masks = {name: result.mask(name) for name in (*data.masks, "imputed")}
    converted = from_ecofill(result, **OPTIONS)
    pd.testing.assert_frame_equal(data.available, before_available)
    pd.testing.assert_frame_equal(result.completed, before_completed)
    for name, before in before_masks.items():
        pd.testing.assert_frame_equal(result.mask(name), before)
    result.export(directory)  # Public producer refuses to overwrite its destination.
    from tezca.individual import describe_series
    results = {}
    provenance = dict(source_details=converted.source_details,
                      qualification=converted.qualification, contribution=converted.contribution)
    for selection in ("observed_only", "provided"):
        for metric in describe_series(converted, observations=ObservationPolicy(selection, max_imputed_fraction=0.15)):
            name = selection + ":" + metric.summary()["method_id"]
            results[name] = ResultItem("metric", metric, sources=["ecofill"], labels=provenance)
    path = Path(directory)
    export_results(path.with_name(path.name + "-results"), results=results,
                   sources={"ecofill": converted}, requested_policy={"example": "ecofill_bridge"})
    return saved_snapshot(converted, directory)


def saved_snapshot(series, directory, *, expected_bundle_id=None):
    """Reload numerical snapshots; no estimator, reconstruction or model fitting."""
    path = Path(directory)
    saved = load_results(path.with_name(path.name + "-results"), expected_bundle_id=expected_bundle_id)
    details = saved.details
    for item in details["results"].values():
        labels = item["labels"]
        assert labels["source_details"]["provenance"] == series.source_details["provenance"]
        for name, table in series.source_details["cells"].items():
            pd.testing.assert_frame_equal(labels["source_details"]["cells"][name], table,
                                          check_dtype=False, check_freq=False)
        for name, mask in series.qualification.items():
            np.testing.assert_array_equal(labels["qualification"][name], mask)
        np.testing.assert_array_equal(labels["contribution"], series.contribution)
    result = snapshot(series)
    result["result_bundle"] = dict(bundle_id=saved.manifest["bundle_id"],
        source_identities=saved.manifest["source_identities"],
        metrics={name: dict(value=item["record"]["value"],
                           use_status=item["record"]["use_status"],
                           positions=item["record"]["support"]["positions"])
                 for name, item in details["results"].items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("produce", "reload"))
    parser.add_argument("directory")
    parser.add_argument("--expected-bundle-id")
    args = parser.parse_args()
    data = produce(args.directory) if args.operation == "produce" else saved_snapshot(
        load_ecofill_export(args.directory, **OPTIONS), args.directory, expected_bundle_id=args.expected_bundle_id)
    print(json.dumps(data, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

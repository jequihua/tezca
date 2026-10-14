"""Synthetic F01-shaped input only; no real source ingestion or domain admission."""

import json

import numpy as np
import pandas as pd

from tezca.collective import CollectionMember, FeatureSpec, prepare_collective, report_site_gate
from tezca.data import AnalysisSeries


def synthetic_f01():
    start, end = pd.Timestamp("2025-12-09", tz="UTC"), pd.Timestamp("2026-02-11", tz="UTC")
    collection = {}
    for j in range(6):
        cadence = "15min" if j < 3 else "30min"
        time = pd.date_range(start, end, freq=cadence, inclusive="left")
        values = (j + np.arange(len(time)) % 24).astype(float)
        data = AnalysisSeries(values, variables=["x"], time=time,
                              metadata={"x": dict(unit="1", representation="raw")},
                              source=dict(id=f"synthetic-f01-shaped-{j}", segment="synthetic"),
                              observed=np.ones((len(time), 1), dtype=bool),
                              clock=dict(mode="regular", cadence=cadence, phase=start))
        collection[f"s{j}"] = CollectionMember(data, variable="x", station=f"synthetic-{j}",
                                               family="quarter" if j < 3 else "half",
                                               feature=f"synthetic-variable-{j}", semantics="state")
    return prepare_collective(collection, features=[FeatureSpec(f"v{j}", (f"s{j}",)) for j in range(6)],
                              start=start, end=end, cadence="1h")


if __name__ == "__main__":
    result = synthetic_f01()
    gate = report_site_gate(result)
    print(json.dumps(dict(shape=list(result.matrix.shape), site_status=gate["compute_status"],
                          common_days=gate["common_days"], source="synthetic_only",
                          real_F01_admission="unresolved_Q002_Q003")))

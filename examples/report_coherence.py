"""Synthetic independently specified report coherence; exact parity unresolved."""
import json

import numpy as np
import pandas as pd

from tezca.coherence import ReportCoherenceSpec, report_coherence
from tezca.data import AnalysisSeries
from tezca.pair_audit import PairLineage, signal_identity


def main():
    t = np.arange(256)
    inputs = []
    for name, values in (("first", np.sin(.17*t) + .3*np.cos(.43*t)),
                         ("second", np.sin(.17*t) + .2*np.cos(.31*t))):
        inputs.append(AnalysisSeries(values, variables=["x"],
            time=pd.date_range("2025-01-01", periods=256, freq="h", tz="UTC"),
            metadata={"x": dict(unit="m", representation="raw")},
            source=dict(id="synthetic-" + name, segment="example"), observed=np.ones((256, 1), dtype=bool),
            clock=dict(mode="regular", cadence="1h")))
    lineage = PairLineage({name: dict(signal_id=signal_identity(data, "x"), kind="measurement", parents=[], donors=[])
                           for name, data in zip(("a", "b"), inputs)})
    result = report_coherence(*inputs, variable_a="x", variable_b="x", lineage=lineage, nodes=["a", "b"],
                             spec=ReportCoherenceSpec(basis="raw", period_band=("4h", "64h"), taper="hann", detrend="constant"))
    record = result.summary()
    run = record["support"]["runs"][0]
    return dict(label="synthetic", nperseg=run["nperseg"], segments=run["raw_segments"],
                compute_status=record["compute_status"], band_mean=result.value["band_mean"],
                exact_parity=record["support"]["exact_report_parity"])


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))

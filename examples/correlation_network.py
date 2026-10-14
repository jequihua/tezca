"""Synthetic 240-hour client network recipe; thresholds are descriptive."""
import json

import numpy as np
import pandas as pd

from tezca.data import AnalysisSeries
from tezca.network import correlation_network, report_network_spec


def main():
    x = np.arange(240)
    data = AnalysisSeries(np.column_stack([x, x, -x]), variables=["x", "y", "z"],
                          time=pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC"),
                          metadata={v: dict(unit="m", representation="raw") for v in "xyz"},
                          source=dict(id="synthetic-network", segment="example"),
                          clock=dict(mode="regular", cadence="1h"), observed=np.ones((240, 3), dtype=bool))
    result = correlation_network(data, spec=report_network_spec(features=list("xyz"), basis="raw"))
    return dict(label="synthetic", windows=len(result.summary), edges=len(result.edges),
                strong_counts=[v["strong_count"] for v in result.summary.value],
                use_status=result.describe()["use_status"], threshold_is_significance=False)


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))

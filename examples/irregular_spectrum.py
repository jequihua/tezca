"""Independent synthetic actual-time spectrum; no files or historical claims."""
import json

import numpy as np
import pandas as pd

from tezca.data import AnalysisSeries
from tezca.irregular import IrregularSpec, irregular_spectrum, irregular_spectral_models
from tezca.spectral_summary import ModelSpec


def main():
    n = np.arange(512)
    t = (n+.15*np.sin(2*np.pi*n/17))[n%11 != 0]
    start = pd.Timestamp("2000-01-01", tz="UTC")
    data = AnalysisSeries(np.sin(2*np.pi*t/8), time=start+pd.to_timedelta(t, unit="s"),
        variables=["x"], metadata={"x":{"unit":"m","representation":"raw"}},
        source={"id":"synthetic-irregular-example-v1","segment":"synthetic"},
        observed=np.ones((len(t),1),dtype=bool), clock={"mode":"irregular"})
    f = np.arange(2,31)/64
    spec = IrregularSpec(interval=[start,start+pd.Timedelta("512s")],segment="64s",hop="32s",
        anchor=start,frequency_hz=f,requested_band_hz=[f[0],f[-1]],f_ref_hz=1)
    result = irregular_spectrum(data,variable="x",spec=spec)
    fit = irregular_spectral_models(result,spec=ModelSpec(candidates_hz=[]),min_cycles=1)
    assert result.summary()["compute_status"] == "ok"
    assert fit.summary()["compute_status"] == "ok"
    peak = float(f[np.argmax(result.value["psd"])])
    assert abs(peak-.125) <= 1/64
    print(json.dumps(dict(peak_hz=peak,method=result.summary()["method_id"],
        beta=fit.value["single"]["beta_low"],use_status=fit.summary()["use_status"],
        parity=result.summary()["support"]["historical_parity"])))


if __name__ == "__main__":
    main()

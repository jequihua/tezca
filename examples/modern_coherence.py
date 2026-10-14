"""Bounded synthetic pooled coherence; no historical or ecological claim."""
import json
import numpy as np
import pandas as pd
from tezca.data import AnalysisSeries
from tezca.coherence_modern import ModernCoherenceSpec, modern_coherence
from tezca.pair_audit import PairLineage, signal_identity


def main():
    t = np.arange(1024)
    time = pd.date_range("2000-01-01", periods=len(t), freq="1s", tz="UTC")
    pair = [AnalysisSeries(values, time=time, variables=["x"], metadata={"x": dict(unit="m", representation="raw")},
                           source=dict(id=key, segment="synthetic-example"), clock=dict(mode="regular", cadence="1s"), observed=np.ones((len(t), 1), dtype=bool))
            for key, values in (("first", np.sin(.17*t)+.4*np.cos(.53*t)), ("second", np.sin(.17*(t-3))+.3*np.cos(.37*t)))]
    lineage = PairLineage({key: dict(signal_id=signal_identity(data, "x"), kind="measurement", donors=[])
                           for key, data in zip(("a", "b"), pair)})
    result = modern_coherence(*pair, variable_a="x", variable_b="x", lineage=lineage, nodes=["a", "b"],
        spec=ModernCoherenceSpec(basis="raw", period_band=("4s", "64s"), nperseg=128, hop=128, taper="boxcar", detrend="none"))
    record = result.summary()
    return dict(label="synthetic", compute_status=record["compute_status"],
                raw_segments=record["support"]["pooling"]["effective"]["raw_segments"],
                n_eff=record["support"]["pooling"]["effective"]["n_eff"],
                band_mean=result.value["band_mean"], inference_status=result.value["inference_status"],
                qualification="bounded independent method; no ecological validation or historical parity")


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))

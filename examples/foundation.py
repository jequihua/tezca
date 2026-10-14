"""Synthetic installed-wheel foundation path. Prints JSON; writes no files."""

import json

import numpy as np
import pandas as pd

from tezca.data import AnalysisSeries
from tezca.policy import AnalysisPolicy, RECIPES
from tezca.reference import ReferenceSpec, apply_reference, fit_reference
from tezca.support import ObservationPolicy, select_support


def main():
    frame = pd.DataFrame({"level": [0., 2., 4., 3., 3., 3.]},
                         index=pd.date_range("2000-01-01", periods=6, freq="h", tz="UTC"))
    original = frame.copy(deep=True)
    metadata = {"level": {"unit": "m", "representation": "raw", "tags": ["synthetic"]}}
    source = {"id": "foundation-example-v1", "segment": "synthetic"}
    clock = {"mode": "regular", "cadence": "1h", "civil_timezone": "UTC", "clock_resolution": "resolved"}
    data = AnalysisSeries(frame, metadata=metadata, source=source, observed=np.ones((6, 1), dtype=bool), clock=clock)
    policy = AnalysisPolicy(RECIPES[0])
    support = select_support(data, **policy.support_options())
    reference = fit_reference(data, spec=ReferenceSpec([0, 2, 4], scale=True),
                              observations=policy.observations(), training_positions=[0, 1, 2])
    evaluation = AnalysisSeries(frame.iloc[3:], metadata=metadata, source=source, observed=[[True]] * 3, clock=clock)
    result = apply_reference(evaluation, reference=reference, observations=policy.observations())
    null = AnalysisSeries(frame.iloc[:2] * np.nan, metadata=metadata, source=source, clock=clock)
    insufficient = apply_reference(null, reference=reference, observations=policy.observations())
    unknown_data = AnalysisSeries(frame.iloc[3:], metadata=metadata, source=source, clock=clock)
    unknown = apply_reference(unknown_data, reference=reference, observations=ObservationPolicy("provided"))
    unknown_observed = select_support(unknown_data, observations=ObservationPolicy("observed_only"))
    pd.testing.assert_frame_equal(frame, original)
    identity_before = data.identity
    frame.iloc[0, 0] = 999
    metadata["level"]["tags"].append("caller mutation")
    returned = data.values
    returned.iloc[0, 0] = 500
    result.value["counts"]["level"][0] = 100
    result.summary()["support"]["positions"].clear()
    assert data.values.iloc[0, 0] == 0
    assert data.metadata["level"]["tags"] == ["synthetic"]
    assert data.identity == identity_before
    assert result.value["counts"]["level"].tolist() == [0, 3]
    assert insufficient.value is None
    assert unknown.summary()["use_status"] == "unknown_provenance"
    assert unknown_observed.positions.tolist() == []
    print(json.dumps({"counts": result.value["counts"]["level"].tolist(),
                      "probabilities": result.value["probabilities"]["level"].tolist(),
                      "compute_status": result.summary()["compute_status"],
                      "metric_status": result.summary()["metric_status"],
                      "use_status": result.summary()["use_status"],
                      "selected_positions": support.positions.tolist(),
                      "fit_positions": reference.describe()["eligible_positions"],
                      "training_mean": reference.describe()["scaling"]["mean"].tolist(),
                      "training_std": reference.describe()["scaling"]["population_std"].tolist(),
                      "policy_id": policy.identity, "reference_id": reference.identity,
                      "insufficient_status": insufficient.summary()["compute_status"],
                      "insufficient_value": insufficient.value,
                      "unknown_status": unknown.summary()["use_status"],
                      "unknown_observed_status": unknown_observed.record["compute_status"],
                      "owned_input": True, "owned_result": True}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Installed, synthetic full-spectrum PCA and training projection example."""
import json

import numpy as np
import pandas as pd

from tezca.data import AnalysisSeries
from tezca.pca import PCASpec, fit_pca_reference, pca_summary, project_pca


def main():
    rows = np.array([[np.sqrt(6), 0], [-np.sqrt(6), 0], [0, np.sqrt(1.5)], [0, -np.sqrt(1.5)],
                     [10, 3], [11, 5], [12, 7], [13, 9]])
    data = AnalysisSeries(rows, variables=["x", "y"],
                          time=pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
                          metadata={v: dict(unit="m", representation="raw") for v in ("x", "y")},
                          source=dict(id="synthetic-pca", segment="example"),
                          clock=dict(mode="regular", cadence="1h"), observed=np.ones((8, 2), dtype=bool))
    spec = PCASpec(features=["x", "y"], basis="raw", scaling="none", ddof=1)
    reference = fit_pca_reference(data, spec=spec, training_positions=[0, 1, 2, 3])
    projected = project_pca(data, reference=reference)
    retrospective = pca_summary(data, spec=spec)
    return dict(label="synthetic", training_eigenvalues=projected.value["eigenvalues"].tolist(),
                training_pc90=projected.value["pc90"], training_pr=projected.value["participation_ratio"],
                projection_reference=projected.summary()["reference_id"],
                retrospective_reference=retrospective.summary()["reference_id"],
                retrospective_eigenvalues=retrospective.value["eigenvalues"].tolist())


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))

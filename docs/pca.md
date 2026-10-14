# Full-spectrum PCA

```python
from tezca.pca import PCASpec, fit_pca_reference, project_pca, pca_summary

spec = PCASpec(features=["temperature", "level"], basis="raw", scaling="zscore", ddof=1)
reference = fit_pca_reference(data, spec=spec, training_positions=range(100))
projection = project_pca(evaluation, reference=reference)
retrospective = pca_summary(evaluation, spec=spec)
```

Inputs are `AnalysisSeries` or `DailyResidual`, including `PreparedCollective.series`.
Features must be explicit, distinct and in input column order. Projection refuses
changed order, units or raw/daily-residual basis. Unselected columns may remain.
Unscaled covariance requires identical units; z-scaling admits heterogeneous units.
Centering is required. Both scaling variance and covariance use the declared ddof
(0 or 1). The default observation policy is `observed_only`; supplied completion
requires an explicit policy. No five-variable site gate applies to generic PCA.

Only complete eligible feature vectors enter covariance. Support records retain
requested, eligible and excluded positions and exact aware timestamps, clock holes,
per-variable support/burden and source/transform identities. Training support uses
positions within its owned subset and an `original_positions` mapping; it also
records `original_eligible_positions`. Held-out values do not enter fitting or
reference identity. Input/reference accessors and result tables return owned copies.

`fit_pca_reference` requires explicit training positions. Its `PCAReference` freezes
center/scales, full eigensystem, training identity/support, settings, NumPy version
and method identity. `project_pca` only computes evaluation scores in that reference;
the returned eigensystem describes **training covariance**, not evaluation covariance.
`pca_summary` is the separately identified retrospective self-fit route. Daily
reference qualifications remain visible. Stricter training use status persists,
and reconstructed training fits cannot promote observed evaluation to primary use.

Successful `ResultRecord.value` includes all p descending eigenvalues/fractions,
leading value/fraction, PC90 (smallest cumulative fraction >=.9, with 1e-12 rounding
tolerance), participation ratio `(sum(lambda))**2 / sum(lambda**2)`, covariance,
rank/tolerance, constant features and rank-deficiency diagnostics. `eigenvectors`
are unit columns; `scaled_loadings` are those columns times sqrt(eigenvalues).
Scores are `(evaluation - training_center) / training_scales @ eigenvectors`.
Result `units` (and reference `output_units`) name each numerical quantity:
unscaled scores/loadings use the input unit, covariance/eigenvalues/leading value
and rank tolerance use its square, and fractions/PR/unit eigenvectors/counts are
dimensionless (`1`). With z-scaling all these outputs are dimensionless. These
unit declarations are retained for unavailable results as well. Per-feature
physical input units remain in support, including heterogeneous scaled features.
Signs and bases within tied spaces are arbitrary; compare reconstruction/projectors.
Rank tolerance is machine epsilon times p times the largest absolute eigenvalue.
Negative values within that roundoff tolerance are explicitly counted and set to
zero; material negatives are unavailable, never silently repaired. Zero eigenvalues
remain in the full spectrum. Constant columns are retained for unscaled PCA;
z-scaling a constant is unavailable. No pairwise covariance or feature dropping occurs.

Compute statuses: `ok`, `insufficient_support` (fewer than required complete vectors),
`zero_scale`, `invalid_scale` (unrepresentable training scale, rejected before
division can erase a feature), `degenerate_input` (zero total covariance), `invalid_covariance`
(nonfinite arithmetic or material negative eigenvalue), and `invalid_projection`
(nonfinite scores). Unavailable values are `None` with reasons. Invalid specifications
raise `ValueError`; numerical/programming exceptions are not hidden. Metric status is
`primary_with_support`. Independent use status is `primary_eligible`, `sensitivity_only`,
`unknown_provenance`, `unverified_gap_geometry`, or `insufficient_support` following
support and inherited training qualifications. Numeric primary eligibility grants
neither ecological validity nor historical parity.

`examples/collective_pca.py` is an installed synthetic two-variable witness.
Named policy PCA recipes remain unresolved in `AnalysisPolicy.method_spec()`.
This separate API requires explicit physical features, basis, scaling, ddof and
training support; supplying them does not resolve the named recipe. Historical numerical parity, field validation and reuse rights
remain unresolved.

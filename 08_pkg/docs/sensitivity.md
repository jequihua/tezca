# Reconstruction and estimator sensitivity

`tezca.sensitivity.spectral_sensitivity` executes four separately identified
routes: complete empirical-reference Welch (`W0`), complete irregular (`I0`),
masked observed irregular (`Im`), and each supplied completed Welch (`Wc`).
It accepts explicit variables, `WelchSpec`, `IrregularSpec`, `ModelSpec`, actual
`fit_frequency_hz` and a cycle gate. See the runnable fixed synthetic example
`08_pkg/examples/reconstruction_sensitivity.py` for complete construction.

```python
report = spectral_sensitivity(complete, masked, {"draw_a": completed_a},
    variables=["x"], welch=welch_spec, irregular=irregular_spec,
    model=model_spec, fit_frequency_hz=common_frequencies, min_cycles=1)
routes, contrasts = report.routes, report.contrasts
description = report.describe()
```

The analysis interval comes from the irregular spec, with explicit endpoints.
All inputs share ordered timestamps, variable order, metadata, basis, units and
clock. The interval has a complete regular slot index retaining missing values;
the irregular anchor aligns its first window with that interval. Durations,
hops, taper and detrending match. Welch uses unpadded physical ordinates even
when its display is padded. Common frequencies must occur on both actual grids
(relative rounding tolerance1e-10), have positive supported densities, satisfy
cycles and be separated by at least physical1/T. No interpolation, extrapolation
or dense-grid inflation is used. At least three common ordinates are required.
The model's explicit interval equals the common-grid endpoints. Empty break
candidates request a single fit. Raw and daily-residual routes remain distinct.

Each route retains the full owned spectrum, fit diagnostics, source/method/config/
reference identities, selected rows, segment/effective-average definitions,
burden, statuses and reason. The report additionally retains missing-cell counts,
positions, partial/shared gaps and exact regular durations/longest blocks per
variable. Irregular gap durations remain unknown in the standalone geometry
helper; point spacing is not invented coverage. Complete reference values are
an empirical scoring reference, never population truth.

Signed deltas are candidate minus baseline; absolute deltas are their magnitudes:

| Contrast | Difference | Question |
| --- | --- | --- |
| `baseline_estimator` | I0−W0 | Estimator disagreement on complete support |
| `deletion` | Im−I0 | Effect of deletion on the irregular route |
| `completion` | Wc−W0 | Supplied completion contrast on Welch |
| `mixed` | Wc−Im | Requested mixed-estimator sensitivity |

There is no percentage division, causal decomposition, automatic ranking or
confidence interval. `compare_estimates` is the small record arithmetic helper;
its explicit signatures must agree. It permits literal oracle tests and cannot
certify the caller's declarations. The public spectral orchestration derives
signatures and estimates from actual API calls and validates inputs.

Completion mapping keys are unique draw IDs. A `None` value is an explicit
unstarted request. Every supplied draw preserves retained values, contribution
categories, original qualification masks and contributor lineage. Finite missing
cell replacements are explicitly imputed. `source.imputation` supplies matching
`input_id=masked.identity`, `draw_id`, nonempty `method_id` and `plan_id`.
Misalignment/mislabeling raises rather than producing plausible deltas. The
library never constructs or fits an imputer. Every draw is analyzed separately.
`summarize_draws(report)` reports mean(beta(draw)), range, requested count,
included denominator and excluded draw IDs. It does not analyze an averaged
completion. Between-draw spread is sensitivity, not calibrated uncertainty.

For daily residual preprocessing, supply `daily_spec`, independently declared
`training` and `training_positions`. Training instants must be outside the
evaluation interval. `design="fixed"` fits one reference on this support and
reuses it for all routes. `design="allowed_training_refit"` additionally requires
explicit `masked_training`, with the same alignment and unchanged retained
values. W0/I0 use the original training reference; Im/Wc use the refitted masked
training reference. Both identities and fit supports are retained. Comparisons
across different references return `incomparable`, exposing that representation
change rather than silently treating it as a fixed-analysis delta. Comparisons
within each shared reference remain available. The fixed qualification bank
uses raw data; residual/reference tests are separate controls.

The route's observation policy also applies during daily transformation: Wc
retains provided imputed values, while W0/I0/Im retain observed-only semantics.
An all-zero training template therefore preserves each draw's numerical spectrum
and its reconstructed support exactly.

`complete=None` permits natural-gap sensitivity and Wc−Im, while retaining missing
W0/I0 rows. No natural-gap error/RMSE field exists. Unknown input or training
qualification is never promoted. Route statuses include inherited estimator
limitations, `unavailable_reference`, `unstarted`, `unsupported_common_frequency`,
`insufficient_cycles` and `insufficient_frequency_resolution`. Contrast statuses
are `ok`, `unavailable` and `incomparable`, with the exact signature differences.
Metric status is `descriptive_sensitivity`; contrast use is `sensitivity_only`
or `unknown_provenance`. Nested accessors are defensive copies and operative
settings/results bind `report.identity`. No source_details payload is embedded.

For other existing methods, `compare_metric_results` accepts two public
`ResultRecord`s and explicit `baseline_interval`/`candidate_interval`. These
intervals declare the intended comparison domain. Every recorded contributing
instant must lie in its declared half-open interval; an incorrect declaration
raises `ValueError`. Gap-related differences within that domain remain allowed.
Shared signal support records expose owned `selected_instants` alongside row
positions, and FI1 comparisons use the actual in-range evaluation subset.
It compares FI1/entropy, full PCA
eigenvalues/fractions/leading values/PC90/participation ratio, graph energy and
coherence band means. Settings, basis, units, method and feature order must
agree, including modern coherence's explicit requested run-weight mapping.
Different pooling weights return `incomparable` with `run_weights` in the
differences. Default `design="fixed"` additionally requires the same reference.
Explicit `retrospective_self` or `allowed_training_refit` requires matching
recorded fit scopes and reports reference changes separately. Training-only
PCA projection eigenvalues describe training covariance, not evaluation
covariance; self PCA summaries have distinct retrospective semantics. Neither
comparison subtracts arbitrary eigenvector signs. Original results preserve
scores/loadings and full support for inspection.

Report-style equal-run coherence and modern pooled-spectra coherence remain
different methods; comparing their scalar fields is `incomparable`. Their pair
audits and ranking exclusions remain in the retained results; a descriptive
delta grants no pair eligibility or ecological validity. `compare_networks`
returns owned `edges` and `summaries` tables plus differences. Edges align by
actual window endpoints and node IDs, retaining signed correlation changes,
support changes and threshold-inclusion changes. Missing/undefined edges are
unavailable, never zero-filled; full scalar summaries retain their original
pair denominators, including an undefined zero-denominator case.

The fixed seed74002 bank accounts for nine mask geometries, three variables,
two supplied constant fixture draws and all routes:135 route rows and162
contrasts. All were computed in this bank; separate deterministic controls
exercise unavailable paths. The observed mean completed x beta was0.269566
without a mask,0.257643 for scattered24,0.670962 for block24,0.846427 for early72
and0.840775 for late72 (two draws each). These values describe deliberately crude
supplied fixtures, not reconstruction quality. See
[qualification and evidence limits](../../04_delivery/qualification.md) and the retained portable synthetic
record in `03_experiments/sensitivity/M004_fixed_bank.md`. Q001-Q003 remain open.

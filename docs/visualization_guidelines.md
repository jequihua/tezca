# Visualizing Tezca results outside Tezca

Owner decision D026 excludes all plotting from Tezca. This guide is for future
documentation, notebooks and application code that consume Tezca's saved
results. It is not a plotting API specification or a current implementation
task. Rendering libraries belong to the separate notebook/application
environment; Tezca has no plotting extra, renderer, figure objects, dashboard,
themes or automatic plotting on import, analysis, export or reload.

## Start from a reproducible result

Use the documented public result/export interfaces and a validated manifest.
Record the result format, Tezca/backend versions, source and mask fingerprints,
requested and resolved settings, formula and reference identities, and selected
output hashes. A hash establishes identity/integrity, not scientific validity
or publisher authenticity. Do not treat a matching hash as permission to use data.

Future notebook examples should have an explicit analysis/export stage and a
separate saved-result visualization stage. Restart the kernel and run all cells
in order to demonstrate that figures do not depend on hidden state. Do not
invent a future Tezca API in examples before that API exists and is documented.
Use admitted synthetic inputs by default. Never retain private observations,
machine paths or credentials in notebook cells, output or embedded metadata.

Read stored estimates, fits, thresholds, flags and support. Rendering must not
refit a spectrum, daily reference, PCA basis, clustering model or significance
threshold. A new scientific transformation belongs in an explicit analysis
step with a new identity. Document even display-only selections, reordering,
unit conversions and downsampling. If downsampling is necessary, show its rule
and retain the full exported data; do not hide narrow gaps or extrema.

## Make missingness and qualification visible

- Keep `compute_status`, `metric_status` and `use_status` distinct. A computed
  value can still be sensitivity-only or otherwise unqualified. Use the exact
  exported vocabulary and its reason, not an invented universal good/bad flag.
- Missing, unsupported, excluded, failed, unknown-origin and numeric zero are
  different states. Do not replace nulls with zero or silently drop all-null
  variables, failed configurations, skipped windows or absent frequency bins.
- Break lines at unsupported intervals. Show unavailable spans or a companion
  status panel; do not interpolate through a gap to improve appearance.
- Show the support relevant to that estimate: duration, actual observations,
  contributor/pair count, segments, effective averages and reconstruction burden
  as appropriate. A whole-series burden cannot describe every window or pair.
- Distinguish reconstructed linear influence from contributor coverage and
  touched nonlinear cells. In particular, median contributor fractions do not
  measure influence. Display the saved primary-gate basis without relabeling it.
- Keep supplied completion realizations separately identifiable. Unknown
  provenance must not acquire the visual styling of confirmed observations.

## Time, units and comparable quantities

Label the time zone, clock convention, cadence or irregular sampling, interval
closure and window label (start/center/end). Preserve UTC instants and document
any civil-time display conversion, including repeated or missing DST hours.
Do not compress elapsed time by plotting surviving samples as consecutive rows.
Window width and step differ; overlapping windows are dependent observations.

Every numerical axis needs a quantity and unit. State raw versus residual basis,
reference versus self-fitting, training period, feature selection and method
identity in the figure or its caption. Comparable-looking curves can represent
different estimands. Separate incompatible support, formula, reference or unit
groups rather than hiding them in a single average. A visual overlay alone does
not establish comparability, causality, significance or ecological health.

## Family-specific guidance

| Results | Recommended presentation and necessary qualifications |
| --- | --- |
| Individual dynamics and ACF | Use elapsed time, explicit realized lags and lag-specific pair support. Distinguish per-day trend from per-sample slope. Do not call an iid interval dependence-robust. Raw cycles and daily-reference residuals need distinct labels. |
| PSD, peaks and bands | Show the saved frequency range, physical resolution, sample/window support and peak-selection policy. A dense or padded grid is not better physical resolution. Frequency-to-period conversion reverses ordering and excludes zero; converting a density's frequency unit requires its Jacobian. Do not relabel frequency density as period density. Zero/negative/unsupported values on log axes need a visible exclusion policy, not an arbitrary positive epsilon. |
| Spectral fits and irregular spectra | Draw only the stored fitted model on its supported fit interval; show break/band limits and insufficient-cycle or resolution statuses. The irregular method's declared `f_ref` is a density scale, not an inferred Nyquist frequency. Do not extrapolate beyond supported frequencies or call an independent implementation historical parity. |
| FI1 and density Fisher | Show FI1 trajectories with partition, pseudocount and stored adjacent-window flags; do not draw changes across skipped strides. Distinguish discrete FI1, histogram-interior approximation, continuous density Fisher and harmonic Fisher-g. Show density coordinates, reference and boundary/tail limits. These quantities do not share a universal unit or interpretation. |
| PCA spectrum and modes | Show the full spectrum used for participation ratio/PC90, with scaling, ddof and reference identity. Distinguish eigenvectors from whichever loading definition was exported. Training eigenvalues differ from evaluation projection covariance. Eigenvector signs and tied eigenspaces require invariant interpretation. A truncated display must not imply a truncated effective-dimension calculation. |
| Networks and connectivity | Preserve signed unique edges, thresholds, actual pair support and eligible denominators. Constants/undefined pairs are not absent zero-weight edges. Use consistent node order or a documented display-only layout across comparisons; layout distance has no scientific meaning unless explicitly defined. Correlation is not directionality or causality. |
| Coherence and phase | Keep report-style run averages distinct from pooled modern estimates. Label raw/corrected scale, null assumptions, effective averages, significance and multiplicity policy. Show unsupported bins and pair-lineage exclusions. Rank only eligible comparable pairs. Specify ordered pairs and phase sign; do not imply a delay from unsupported or ambiguous phase. |
| Graph-state sensitivity | Display every requested feasible/infeasible PC/bin configuration, geometry, reference and mass/density definition. Variation across configurations is sensitivity, not a confidence interval. Do not select only the most favorable result or silently clip infeasible dimensions. |
| Reconstruction sensitivity | Distinguish complete-Welch, complete-irregular, masked-irregular and each supplied completion-Welch route. State which signed/absolute contrast is shown and whether support is matched. Mixed-estimator disagreement is not a causal reconstruction effect. Between-completion spread is not automatically calibrated uncertainty; natural gaps have no observed hidden-truth RMSE. |
| Candidate regimes and timelines | Treat cluster IDs as nominal labels with a stable, explicitly recorded color mapping. Preserve large integer/string label identities. Noise, excluded and failed rows need different representations. Respect interval overlap and max-gap barriers; do not connect or smooth across unsupported time, invent transitions or double-count duration. Clustering does not establish ecological regimes or change points. |

## Accessible, honest figures

Use redundant encodings: color plus line style, marker or text. Choose a
colorblind-friendly palette with readable contrast; use sequential scales for
ordered nonnegative values, diverging scales with an explicit meaningful center
for signed values, and categorical colors for nominal regimes. Keep equivalent
scales and labels consistent across panels. Do not use color alone for status.

Show legends, units, definitions and qualifying captions at a legible final
size. Avoid misleading axis truncation, unsupported smoothing, 3D decoration
and dual axes that suggest an arbitrary relationship. Disclose logarithmic or
nonlinear scales. Include text/alt descriptions and the accessible source table
when publishing. Choose output formats and resolution in the external rendering
environment; these are not Tezca responsibilities.

## Future example collection

The former eight client figure families can be demonstrated later in external
notebooks: dynamics/PSD, dominant periods, full PCA scree/effective dimension,
mode loadings, eligible coherence rankings, dynamic connectivity, FI1
trajectories/flags and reconstruction sensitivity. Density, graph and regime
examples can follow the same principles. This list is guidance, not a promise
of package functions or a requirement to create those notebooks in M005.

Each future example should include at least one unavailable or qualified result,
not only an ideal curve. Use a small fixed synthetic case, disclose its meaning,
validate saved inputs, and exercise the documented public interface. Keep the
rendering dependency/environment record with the notebook, outside Tezca's
dependency metadata. No automatic downloads or hidden upstream data access.

## Review before sharing an external visualization

1. Can another reader identify the exact validated saved result and reproduce
   the selection/display transformations in a clean notebook environment?
2. Are units, basis, time, support, reference and formula identities clear?
3. Are missingness, unknown provenance, qualifications and gap barriers visible?
4. Are every fit, threshold and scientific statistic saved analysis outputs?
5. Are uncertainty, sensitivity and ecological interpretation claims justified?
6. Are labels/colors accessible, values faithful, and source data rights honored?

Numerical verification remains necessary. Visual inspection is an additional
check of the external presentation, not a replacement for scientific evidence.

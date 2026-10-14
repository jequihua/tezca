# Choose a method

[Start here](../README.md) · [API cheat sheet](../../08_pkg/docs/api_cheatsheet.md)

Begin with the question and the available observations. Choose the units, clock,
eligible contributions and training interval explicitly. A named policy records
study choices; it does not turn every method declaration into an executable
specification. `AnalysisPolicy.method_spec()` still refuses unresolved scientific
methods. Use the method-specific constructors below, even after selecting a
recipe. Their details and signatures are in the linked pages and cheat sheet.

| Your question | Public entry point and choices | Read next |
| --- | --- | --- |
| What are the level, spread, trend and exact-lag association? | `tezca.individual.describe_series`, `estimate_trend`, `autocorrelation`; choose the variable, observations and actual lag. Trends use measurement units/day; ACF requires a regular clock. | [Individual dynamics](../../08_pkg/docs/individual.md) |
| What remains after the daily pattern? | `fit_daily_reference` with `DailySpec`, then `apply_daily_reference`; declare civil timezone and training positions. | [Daily reference](../../08_pkg/docs/individual.md) |
| Which periods occur on a regular clock? | `tezca.spectral.power_spectrum` with `WelchSpec`; declare segment length, taper and detrending. Contiguous runs remain separate. | [Welch spectra](../../08_pkg/docs/spectral.md) |
| What slope, peaks or band power does a spectrum describe? | `tezca.spectral_summary.spectral_models`, `spectral_peaks`, `band_power` with `ModelSpec`, `PeakSpec`, `BandSpec`; select actual frequency/period limits and check cycle/resolution gates. | [Spectral summaries](../../08_pkg/docs/spectral_summary.md) |
| Are raw periodic components unusual under a white-noise model? | `tezca.harmonic.harmonic_diagnostics` with `HarmonicSpec`; only the supported complete raw periodogram route has these null assumptions. | [Harmonic diagnostics](../../08_pkg/docs/harmonic.md) |
| How rough are ordered probability masses over time? | `tezca.fi1.fi1_discrete`, `fi1_windows` with `FI1Spec` or a fitted reference; fix the edges, pseudocount and window schedule. | [Discrete FI1](../../08_pkg/docs/fi1.md) |
| How does a distribution's density change with its coordinate? | `tezca.density_fisher.density_fisher` with `HistogramSpec` or `KDESpec`; choose a distinct interior approximation or continuous KDE functional, scale and boundary treatment. | [Density Fisher](../../08_pkg/docs/density_fisher.md) |
| How do I align several declared series without filling gaps? | `tezca.collective.prepare_collective` with `CollectionMember` and `FeatureSpec`; specify membership, cadence, reducers and coverage. | [Collective preparation](../../08_pkg/docs/collective.md) |
| How many variance directions and which loadings describe a matrix? | `tezca.pca.fit_pca_reference` with `PCASpec`, then `project_pca`; choose feature order, raw/residual basis and scaling on training data. | [PCA](../../08_pkg/docs/pca.md) |
| Which pairs correlate in each supported window? | `tezca.network.correlation_network` with `NetworkSpec`; set features, threshold and pair support. | [Signed networks](../../08_pkg/docs/network.md) |
| Which frequencies are shared between two series? | `tezca.coherence.report_coherence` with `ReportCoherenceSpec`, or `tezca.coherence_modern.modern_coherence` with `ModernCoherenceSpec`; report-style run averaging and pooled spectra have distinct formulas. Inspect pair lineage first. | [Report coherence](../../08_pkg/docs/coherence.md) / [pooled coherence](../../08_pkg/docs/coherence_modern.md) |
| How rough is occupancy on a declared graph of states? | `tezca.graph_state.graph_state_roughness` / `graph_state_grid`; choose feature basis, projection/partition geometry and mass or density definition. | [Graph-state roughness](../../08_pkg/docs/graph_state.md) |
| How do I analyze actual irregular instants? | `tezca.irregular.irregular_spectrum` with `IrregularSpec`; declare interval, segments and actual frequencies. This is independently defined projection-density WOSA, not a resampled Welch spectrum. | [Irregular spectra](../../08_pkg/docs/irregular.md) |
| How sensitive are results to deletion or supplied completions? | `tezca.sensitivity.spectral_sensitivity`, `compare_metric_results` and `compare_networks`; declare matching support and configurations. A difference between estimators is not an imputation effect by itself. | [Sensitivity](../../08_pkg/docs/sensitivity.md) |
| Which exploratory clusters and gap-safe episodes describe features? | `tezca.regimes.cluster_regimes`, `assign_regimes`, `summarize_regime_timeline`; declare `RegimeFeatures`, preprocessing and a `ClusterSpec` or `CustomClusterer`. | [Candidate regimes](../../08_pkg/docs/regimes.md) |
| How do I save exact results for another process? | `tezca.exports.ResultItem`, `export_results`, `load_results`; keep expected bundle/source/reference identities outside the bundle. | [Result exports](../../08_pkg/docs/exports.md) |

## Make the four choices explicit

**Raw or residual:** raw periodicity and residual memory answer different
questions. Fit daily templates on training rows, apply the frozen reference to
evaluation rows, and preserve both branches. Fitting a template on evaluation
data is allowed only as explicitly retrospective self-fitting; it is not an
out-of-sample result.

**Regular or irregular:** a regular clock may contain holes. Keep those holes;
do not drop missing timestamps and relabel the remaining values as consecutive.
Welch uses valid continuous runs. The irregular route uses actual elapsed time
and retains frequency-specific support.

**Observed or completed:** `ObservationPolicy("observed_only")` selects eligible
observations. `"provided"` admits supplied finite contributions while retaining
unknown and imputed provenance. `"trusted_imputation"` additionally needs the
caller to declare the imputation trust evidence. A finite value is not proof of
observation or scientific quality.

**Self-fitted or frozen:** use a `fit_*_reference` function with explicit training
positions/IDs when you need a stable comparison, then pass that reference to
evaluation. A `spec` route that fits on evaluation data is labelled self or
retrospective. The loaded export stores numerical descriptions; it does not
recreate an executable predictor.

The numerical routes above need NumPy, pandas and SciPy. Policy YAML needs
PyYAML; native KMeans/DBSCAN and StandardScaler need the models extra. Custom
regime factories work in core. EcoFill object conversion has its own admitted
version/environment; loading an EcoFill result export needs no EcoFill import.
See [optional integration troubleshooting](troubleshooting.md).

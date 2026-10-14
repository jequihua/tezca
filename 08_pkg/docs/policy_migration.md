# Migration of retained 006

The read-only source is `02_analysis/references/006_analysis_reference_policy.yaml`.
Lists are terminal leaves: the exact source path set contains 45 leaves.
`load_policy_yaml(text, target=...)` requires a named recipe target for
`manglaria_v1_10_compat`. Sensor/report targets have different scientific identities.
The loader never writes the source. YAML tests compare its exact bytes before
and after migration, as well as the independently enumerated exact path set.

The following table is the complete sensor/report migration. A mapped value is
retained as a method declaration, subject to strict validation. Changed
rows state the interpretation change even when the numeric value stays equal.
Unresolved rows identify a route but cannot supply an executable estimator.
Every runtime row carries source path, input value, disposition, destination,
resolved value and reason; `.describe()` preserves requested and resolved IDs.
The independent research recipe uses raw/residual branches, window support,
minimum raw-segment/resolution semantics and pooled spectra. All recipes record
unverified parity and the missing section 6.2 choices described in `foundation.md`.
Independent implemented methods take explicit specs through their own module
APIs; using them does not make `AnalysisPolicy.method_spec()` resolve a recipe.

| Source terminal path | Input value | Disposition | Destination and sensor/report resolved value or reason |
| --- | --- | --- | --- |
| `format_version` | 1 | changed | `schema_version=1`, separately versioned resolved identity |
| `profile` | manglaria_v1_10_compat | changed | `recipe`, explicit manglaria_sensor_20260911_compat / manglaria_report_v1_10_compat |
| `individual.min_analysis_points` | 256 | mapped | Same path/value; compatibility screen, additional estimator gates unresolved |
| `individual.anomaly` | median_minute_of_day | mapped | Same path/value; fitted reference/timezone/support requirements remain visible |
| `individual.trend_unit` | per_day | mapped | Same path/value; descriptive measurement-unit/day, inference unresolved |
| `individual.acf_lags` | [one_step, one_hour, twenty_four_hours] | mapped | Same path/value; exact cadence resolution, unsupported lags retained |
| `individual.psd.method` | welch | mapped | Same path/value; regular-run declaration, no irregular substitution |
| `individual.psd.max_nperseg` | 4096 | mapped | Same path/value |
| `individual.psd.min_nperseg` | 64 | mapped | Same path/value; must not exceed maximum |
| `individual.psd.target_segments` | 8 | changed | Same count, `segment_rule=floor_n_over_target_length`; not eight actual averages |
| `individual.psd.beta_sign` | beta_equals_negative_loglog_slope | mapped | Same path/value |
| `individual.psd.peak_prominence_log10` | 0.08 | mapped | Same path/value; tie/period selection requirements unresolved |
| `individual.psd.max_peaks` | 3 | mapped | Same path/value |
| `individual.psd.bands_hours.circadian` | [20, 28] | mapped | Same path/value; explicit spectral-summary API enforces resolution/cycle gates |
| `individual.psd.bands_hours.semidiurnal` | [10, 14] | mapped | Same path/value |
| `individual.psd.bands_hours.synoptic` | [48, 240] | mapped | Same path/value |
| `individual.fi1.label` | discrete_first_order_fisher_index_historical | mapped | Same path/value; not density Fisher or harmonic Fisher g |
| `individual.fi1.window_hours` | 72 | mapped | Same path/value |
| `individual.fi1.step_hours` | 6 | mapped | Same path/value |
| `individual.fi1.abrupt_k` | 3 | mapped | Same path/value; explicit adjacency/MAD rule still required |
| `collective.cadence` | 1h | mapped | Same path/value; reducers and source selection unresolved |
| `collective.min_family_overlap_days` | 30 | mapped | Same path/value; site criterion, not a generic reference gate |
| `collective.min_variable_coverage` | 0.65 | mapped | Same path/value; observed/completed coverage definition unresolved |
| `collective.min_variables` | 5 | mapped | Same path/value; site criterion, not a universal mathematical restriction |
| `collective.anomaly` | median_minute_of_day | changed | `preprocessing`: sensor PCA raw, network/coherence daily_residual; report PCA raw, network daily_residual, coherence unresolved |
| `collective.standardize` | true | mapped | Same path/value; actual fit scope and parameters must be supplied to later methods |
| `collective.pca.full_spectrum` | true | mapped | Same path/value |
| `collective.pca.pc90` | true | mapped | Same path/value |
| `collective.pca.effective_dimension` | participation_ratio | mapped | Same path/value; entire-spectrum declaration |
| `collective.pca.export_modes` | 3 | mapped | Same path/value; available modes must be reported, not fabricated |
| `collective.graph.window_hours` | 168 | mapped | Same path/value |
| `collective.graph.step_hours` | 24 | mapped | Same path/value |
| `collective.graph.correlation_threshold_abs` | 0.3 | mapped | Same path/value; association threshold, not significance |
| `coherence.corrected` | true | unresolved | `coherence.estimator=pinned_sensor_helper` / `report_run_average_corrected`; backend/correction/threshold definitions unverified |
| `coherence.min_raw_welch_segments_per_run` | 4 | mapped | Same path/value; overlapping segments are not independent replicates |
| `coherence.overlap_fraction` | 0.5 | mapped | Same path/value; integer overlap/taper remain unresolved |
| `coherence.alpha` | 0.05 | mapped | Same path/value; pointwise/null/multiplicity definitions unresolved |
| `coherence.exclude_direct_target_transforms` | true | mapped | Same path/value; later lineage filtering requires pair metadata |
| `interpretation.primary_use_max_imputed_fraction` | 0.15 | changed | Same numeric value, `burden_convention=relevant_support_v1`, with actual support/block checks |
| `interpretation.metric_status.psd_beta` | primary_with_support | mapped | Same path/value; qualified descriptor, no ecological health claim |
| `interpretation.metric_status.fi1_discrete` | historical_sensitivity_index | mapped | Same path/value |
| `interpretation.metric_status.pca_full_spectrum` | primary_with_support | mapped | Same path/value; provenance/comparison assumptions still apply |
| `interpretation.metric_status.functional_correlation` | exploratory_association | mapped | Same path/value |
| `interpretation.metric_status.coherence` | exploratory_frequency_coupling | mapped | Same path/value |
| `interpretation.metric_status.graph_state_fisher` | exploratory_resolution_sensitive | mapped | Same path/value |

The source does not fully specify station/reducer/priority rules, two-family
screening, endpoint/maximum reconstruction conventions, observed/completed
coverage, daily reference/replicates/fallback, inference, taper/overlap/detrending/
scaling/frequency/model criteria, harmonic null/backend, FI bins/pseudocount/
entropy/adjacency, graph geometry, coherence phase/effective averages/correction/
threshold/null/multiplicity, LSFT-WOSA, density/regime settings, or export/figure
versions. `.describe()['unresolved']` records these requirements, and
`method_spec()` rejects attempts to execute them. Compatibility means declared
intent until pinned implementations and fixtures are independently admitted.

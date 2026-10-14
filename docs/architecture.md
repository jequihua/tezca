# Tezca architecture and scientific boundary

For installation and user tasks, begin with [Start here](index.md)
and the [API cheat sheet](api_cheatsheet.md). This document describes implemented
boundaries and explicit scientific definitions.
The current foundation implements owned
data/provenance, time/support, strict policies and frozen scaling/fixed-edge
references; see `foundation.md`. Individual descriptions, trends, exact ACF and
civil daily references are available in `individual`, and regular Welch spectra
in `spectral`, with descriptive models, peaks and bands in `spectral_summary`,
discrete FI1 partitions and complete trajectories in `fi1`, and distinct
histogram-interior/KDE methods in `density_fisher`, and independent raw-periodogram
Fisher-g/Siegel diagnostics in `harmonic`, and explicit collective matrix
preparation with nested provenance and separate site screening in `collective`,
full-spectrum covariance PCA with frozen training projection in `pca`, and
exploratory signed rolling correlation networks in `network`, and source-bound
pair audits/report-style contiguous-run coherence in `pair_audit`/`coherence`, and
pooled spectra, phase and qualified white-noise inference in `coherence_modern`,
and mass/density graph energy, frozen state geometry and resolution grid in `graph_state`,
and independent actual-time projection-density LSFT-WOSA in `irregular`;
and matched four-route spectral/non-spectral comparisons in `sensitivity`.
[Candidate regimes](regimes.md) add capability-aware optional models, custom factories,
frozen preprocessing/assignment and support timelines. [Validated numerical exports and reload](exports.md)
and the installed `examples/delivery.py` route cover the complete local workflow.
This remains an unreleased local candidate: historical parity, real-data/domain
qualification and distribution authorization are distinct unresolved gates. See
the [qualification record](qualification.md), [sensitivity](sensitivity.md),
[irregular spectra](irregular.md), [individual dynamics](individual.md),
[spectra](spectral.md), [spectral summaries](spectral_summary.md) and the qualification limits.

```text
observations + metadata/masks ─┐
external completions ──────────┼─> owned analysis data -> audit/explicit preparation
optional EcoFill adapter ──────┘                         |
                         individual / collective / comparisons
                                         |
                               result tables + manifest
                                         |
                                 explicit exports / validated reload
```

Use ordinary arrays/pandas at the boundary, small method-specific specifications
and result records, and functions rather than a plugin server. Current modules
separate data/support/policy, method-specific analysis, references, optional
adapters and passive exports. The [API inventory](api_cheatsheet.md) records the
actual callable surface; the package does not supply a general pipeline engine.

Numerical core uses NumPy/pandas/SciPy. YAML configuration uses a
`yaml` extra and clustering scikit-learn a `models` extra. Tezca has no plotting
API, renderer or dependency. PCA uses native NumPy/SciPy linear algebra.
Source-specific MAT/HDF5 ingestion is caller-side preparation, not an implemented
Tezca adapter. EcoFill object support is optional. The package declares the
numerical core and those two extras. Core qualification does not establish
optional-feature compatibility. Exact current evidence is in the qualification
record. No automatic network/plugin installation occurs.

`AnalysisSeries` owns values, aligned time and metadata. Collections preserve
different cadences until explicit alignment. Distinguish original qualification
(including artificial/QC/structural/absent/inserted) from present contribution
(eligible observation/imputed/derived/unavailable/unknown) and lineage. Finite
does not imply observed. Keep raw versus residual branches, reference fitting
and retrospective self-fitting separate. Regular estimators require regular
time; irregular estimators consume actual elapsed times. Never drop missing
timestamps then pretend the resulting array retained its cadence.

For current EcoFill, use `SeriesData.available` and `mask('donor')`, or
`ImputationResult.completed`, public masks/methods/schema/provenance. `observed`
can include artificially hidden truth; `raw` must never feed analysis. The
current adapter admits the hash-bound EcoFill 0.0.1 public contract and result
export format 1; see `ecofill.md`. Exports use the numerical core without EcoFill,
while actual object compatibility has a separate installed Python 3.12 job.

Each metric carries `compute_status`, `metric_status`, and `use_status`, support,
units, representation, method/configuration/reference IDs, and origin burden.
Expected insufficient support yields a reasoned missing value; invalid config
raises; programming failures must not be disguised as data limitations. Owned
storage/defensive copies enforce immutability of nested frames and metadata.

Scientific identities remain distinct: FI1 discrete roughness; density Fisher
functional versus histogram interior approximation; graph-state energy; Fisher
g/Siegel harmonic tests. Preserve full PCA spectrum. Modern coherence pools
auto/cross spectra under common settings; report compatibility averages run
coherences under its own identity. Corrected values and significance thresholds
use compatible scales. Graph resolution ranges are sensitivity, not confidence
intervals. Regime labels are exploratory clusters, not ecological truth.

Recipes: `manglaria_sensor_20260911_compat`,
`manglaria_report_v1_10_compat`, and `manglaria_research_v1`. Old
`manglaria_v1_10_compat` requires a selected migration target. No implicit
completion, fallback estimator, clipped PC count, or universal ecological gate.
The 0.15 burden threshold and client site criteria are versioned study choices.
`AnalysisPolicy.method_spec()` retains explicit unresolved declarations and
refuses to return executable scientific specs. Calling an independent method
with its own explicit spec does not resolve a named policy or historical parity.

Canonical CSV/JSON results and compatibility exports are separate. Export must
refuse accidental overwrite, retain unsupported rows, preserve formula identity,
and record requested/resolved settings plus source/mask/reference/output hashes.
External applications consume saved values; Tezca provides no rendering. The
separate [visualization guidelines](visualization_guidelines.md) describe
future downstream work. Scientific identity does
not include nondeterministic wall-clock timestamps. Candidate examples run after
installation from a wheel, outside this checkout, across fresh processes.

The source checkout contains companion docs, examples and portable tests.
A wheel contains the import package and distribution metadata. No environments,
dependency stores, measurements or executable fitted models are bundled.

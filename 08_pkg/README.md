# Tezca: environmental time-series analysis

Tezca describes individual signals, collective variation, frequency coupling,
and sensitivity to missing or supplied replacement values. It accepts ordinary
arrays and pandas inputs with explicit time, units and provenance, and saves
numerical results that another process can validate without refitting.

**New here?** Use [Start here](../04_delivery/README.md), then the
[source/wheel quickstart](../04_delivery/guides/quickstart.md). The
[complete API cheat sheet](docs/api_cheatsheet.md) lists public module functions,
constructors, methods and accessors, including those outside root imports.

This is the unreleased `0.1.0.dev0` candidate for local testing. Historical parity,
real-data/domain evidence and reuse/distribution terms remain unresolved. Local
engineering checks do not settle those questions or authorize sharing and
publication. Read the [qualification record](../04_delivery/qualification.md)
for actual evidence, exact environments, limits and owners of the remaining work.

## Start with your question

| Task | Methods and guidance |
| --- | --- |
| Describe one signal | [Statistics, trends, exact ACF and daily residuals](docs/individual.md) |
| Describe periodicity and spectral shape | [Welch](docs/spectral.md), [models, peaks and bands](docs/spectral_summary.md), [harmonic diagnostics](docs/harmonic.md), [actual-time irregular spectra](docs/irregular.md) |
| Describe distributions | [Discrete FI1](docs/fi1.md), [histogram-interior and KDE density Fisher](docs/density_fisher.md) |
| Combine declared sources | [Collective preparation](docs/collective.md), [full-spectrum PCA](docs/pca.md), [signed networks](docs/network.md) |
| Compare coupled signals and states | [Report-style coherence](docs/coherence.md), [pooled coherence](docs/coherence_modern.md), [graph-state roughness](docs/graph_state.md) |
| Compare reconstructions or exploratory regimes | [Matched sensitivity](docs/sensitivity.md), [candidate regimes](docs/regimes.md) |
| Save and validate numerical results | [CSV/JSON exports and reload](docs/exports.md) |

The [method-selection guide](../04_delivery/guides/choose_methods.md) explains
raw versus residual, regular versus irregular, observed versus supplied and
self-fitted versus frozen choices. Use explicit method specs: named policies
retain unresolved scientific declarations, and `AnalysisPolicy.method_spec()`
does not automatically wire these implemented methods into an executable recipe.

## Installation and the complete example

This source checkout keeps `pyproject.toml`, `08_pkg/README.md` and `08_pkg/src`
together for builds. Follow the [source/wheel quickstart](../04_delivery/guides/quickstart.md)
to install directly from this checkout or build and install a local wheel using
approved offline dependencies. No retained wheel is bundled and no unpublished
PyPI installation by name is implied. This checkout is distinct from the earlier
wheel-only development handoff.

Run `08_pkg/examples/delivery.py analyze NEW_DIRECTORY` with the installed
interpreter, then exit and run `reload DIRECTORY --expected-identity DIRECTORY.identity.json`
in a fresh process. The supplied route adds `--completion supplied`. All inputs
are deterministic synthetic generators. Sibling helpers `result_exports.py` and
`delivery_labels.py` are included. `08_pkg/examples/read_bundle_stdlib.py DIRECTORY`
checks saved bytes using only the standard library. The quickstart gives copyable
commands, useful values, all 18 export families and the withheld-band reason.
The examples are source files, not installed modules.

## Inputs, dependencies and interpretation

The package declares Python >=3.11, NumPy >=2.0,<3, pandas >=2.2,<3 and
SciPy >=1.14,<2. Declared ranges are not proof of every version/platform.
Qualification lists the exact exercised Windows environments. Optional YAML
loading needs PyYAML (`yaml` extra); native KMeans/DBSCAN and StandardScaler need
scikit-learn (`models` extra). Core and custom-regime routes need neither extra.

The [EcoFill adapter](docs/ecofill.md) accepts the admitted EcoFill 0.0.1 public
objects in its separately qualified Python 3.12 environment. It consumes available
or completed values and explicit masks/provenance, excluding hidden truth.
Its format-1 export loader works in core without importing EcoFill or reading
`raw.csv`. No arbitrary EcoFill version is implied.

[Foundation guidance](docs/foundation.md) explains owned inputs/results, clocks,
masks, support and references. A finite value is not automatically observed.
Analysis never reconstructs missing values implicitly or compresses gaps into
continuous support. [Understand results](../04_delivery/guides/understand_results.md)
before interpreting computation, metric and use statuses. A primary-eligible
number is not an ecological, causal or historical-parity conclusion.

Tezca has no plotting or rendering API, dependency or notebook implementation.
Saved values can support downstream applications following the separate
[visualization guidelines](../docs/visualization_guidelines.md). No legacy code
or private measurements are included; no license or redistribution permission
is inferred from local installation.

For difficulties, use [troubleshooting](../04_delivery/guides/troubleshooting.md),
[migration and limitations](../04_delivery/guides/migration_and_limits.md), and
the [safe colleague-testing guide](../04_delivery/guides/colleague_testing.md).

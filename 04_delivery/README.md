# Start here: Tezca

Tezca helps you describe environmental time series, compare several variables,
and see how missing or supplied replacement values affect an analysis. It keeps
the timestamps, units, provenance (where each contribution came from), settings
and reasons for unavailable results alongside the numbers.

This is an **unreleased candidate for local testing**. It includes synthetic
examples and buildable package source; build a local wheel or install the checkout. Local checks
do not establish historical equivalence, validity on your field data, or
permission to redistribute the package or its reference material. Sharing and
publication require separate authorization. See the [qualification record](qualification.md)
for evidence, limits and the people who can resolve them.

## Your first session

Start with [the Windows quickstart](guides/quickstart.md). It walks through an
isolated installation, a synthetic analysis, saved results, and a second Python
process that validates and reads those results. You should see a cosine's
integrated power near **0.5**, two PCA eigenvalues near **4 and 1**, and a
long-period band deliberately withheld because its requested frequencies are
unsupported. A missing value with a reason is part of a useful result.

This source checkout contains root `pyproject.toml`, `08_pkg/src`, package docs
and examples, these guides and visible `.front_repo_sync.json` provenance.
It contains neither the earlier wheel-only candidate nor its binary/index/checksum
files. Dependencies are supplied separately by your operator. Run the quickstart
from the checkout root and keep generated outputs in unused sibling paths.

## Find your next step

| I want to… | Start with |
| --- | --- |
| Install, analyze, save and reload | [Quickstart](guides/quickstart.md) |
| Decide which method matches my question | [Choose a method](guides/choose_methods.md) |
| Understand a number, status, missing result or reference | [Understand results](guides/understand_results.md) |
| Fix an installation, clock, export or identity refusal | [Troubleshooting](guides/troubleshooting.md) |
| Move an older workflow and understand present limits | [Migration and limitations](guides/migration_and_limits.md) |
| Exercise the candidate and report a reproducible problem | [Colleague testing](guides/colleague_testing.md) and [bug template](guides/bug_report.md) |
| Find an exact import, constructor, function, method or accessor | [Complete API cheat sheet](../08_pkg/docs/api_cheatsheet.md) |
| Read table formats and reload guarantees | [Saved result format](../08_pkg/docs/exports.md) |

Tezca does not fill gaps automatically. You may explicitly supply completions,
but they remain distinguishable from observations. The core needs NumPy, pandas
and SciPy; YAML, native clustering models and EcoFill objects have separate
optional paths. Tezca writes numerical results and provides no plotting or
rendering capability. Future downstream work can use the [external visualization
guidelines](../docs/visualization_guidelines.md).

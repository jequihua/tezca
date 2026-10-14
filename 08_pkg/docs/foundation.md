# Installed foundation API

This development snapshot implements owned data, clock/support accounting,
strict policies and a useful frozen scaling/fixed-edge histogram reference.
Individual descriptions, elapsed-day trends, exact ACF and explicit civil daily
references are implemented; see [individual dynamics](individual.md).
Regular Welch spectra are implemented; see [spectral contracts](spectral.md).
Descriptive fits, peaks and density bands use [spectral summaries](spectral_summary.md).
Discrete FI1 and complete trajectories are available in [fi1](fi1.md).
[Density Fisher](density_fisher.md) implements histogram interior and Gaussian KDE routes.
[Harmonic diagnostics](harmonic.md) implement independently identified Fisher-g/Siegel
tests under an explicit raw-periodogram and iid Gaussian white-noise null.
[Collective preparation](collective.md) aligns and aggregates declared membership
with nested lineage and separate report-site gates. [PCA](pca.md) adds complete-vector
full-spectrum summaries and frozen training projection. [Networks](network.md)
provide signed edges and explicit rolling support. [Report-style coherence](coherence.md)
adds lineage audits and qualified run spectra. [Modern coherence](coherence_modern.md)
adds independent pooled spectra, phase and qualified white-noise thresholds.
[Graph-state roughness](graph_state.md) supplies explicit mass/density energy,
frozen geometry and retained PC/bin sensitivity configurations. This
snapshot also implements [independent LSFT-WOSA](irregular.md) and
[matched reconstruction sensitivity](sensitivity.md), plus explicit
[candidate regime APIs](regimes.md) with optional native models. Unresolved declarations in a policy
remain inspectable requirements; `method_spec()` raises rather than executing
an incomplete method or substituting a backend. Pinned numerical parity is
unverified. The five accepted toolchain witnesses remain regression checks.
Explicit specs are passed directly to the implemented module APIs; supplying
one does not change `AnalysisPolicy.method_spec()` into a recipe resolver.
For a first session use the [local-wheel quickstart](../../04_delivery/guides/quickstart.md)
and [complete API cheat sheet](api_cheatsheet.md).

Public imports are explicit and do not load YAML:

```python
from tezca.data import AnalysisSeries, ContributorLineage, ResultRecord
from tezca.policy import AnalysisPolicy, RECIPES
from tezca.support import ObservationPolicy, TimeAxis, select_support, pair_support
from tezca.reference import ReferenceSpec, fit_reference, apply_reference
```

For optional public EcoFill objects and format-1 result exports, use
`tezca.adapters.ecofill.from_ecofill` / `load_ecofill_export`; see
[the admitted adapter contract](ecofill.md). Neither import loads EcoFill;
export loading remains part of the standalone core.

## Complete example

With the wheel installed, run `python -I -B <checkout>/08_pkg/examples/foundation.py`
from an empty directory outside the checkout. The full verifier builds the
wheel offline, installs it into a fresh environment, and runs this exact
example in a separate isolated process. The example writes no files and needs
neither EcoFill nor YAML or models extras. Numerical result persistence is
documented in [exports.md](exports.md); no plotting capability is provided.

```python
import numpy as np
import pandas as pd
from tezca.data import AnalysisSeries
from tezca.policy import AnalysisPolicy
from tezca.reference import ReferenceSpec, fit_reference, apply_reference
from tezca.support import select_support

frame = pd.DataFrame({"level": [0., 2., 4., 3., 3., 3.]},
                     index=pd.date_range("2000-01-01", periods=6, freq="h", tz="UTC"))
metadata = {"level": {"unit": "m", "representation": "raw"}}
source = {"id": "synthetic-example-v1", "segment": "synthetic"}
clock = {"mode": "regular", "cadence": "1h", "civil_timezone": "UTC",
         "clock_resolution": "resolved"}
data = AnalysisSeries(frame, metadata=metadata, source=source,
                      observed=np.ones((6, 1), dtype=bool), clock=clock)
policy = AnalysisPolicy("manglaria_sensor_20260911_compat")
support = select_support(data, **policy.support_options())
reference = fit_reference(data, spec=ReferenceSpec([0, 2, 4]),
                          observations=policy.observations(), training_positions=[0, 1, 2])
evaluation = AnalysisSeries(frame.iloc[3:], metadata=metadata, source=source,
                            observed=[[True]] * 3, clock=clock)
result = apply_reference(evaluation, reference=reference, observations=policy.observations())
assert result.value["counts"]["level"].tolist() == [0, 3]
assert result.summary()["compute_status"] == "ok"
assert result.summary()["metric_status"] == "exploratory_reference_summary"
assert result.summary()["use_status"] == "sensitivity_only"
```

The retained example also evaluates all-null inputs (`value=None`,
`compute_status=insufficient_support`) and undeclared finite inputs
(`use_status=unknown_provenance`). The latter can be analyzed under `provided`
but select no points under `observed_only` or `trusted_imputation`. It checks
that caller mutations and mutations of result accessors leave owned data,
metadata, counts and support intact. The example JSON includes the policy and
reference identities, training mean 2, population SD sqrt(8/3), and statuses.

## Data and qualification

`AnalysisSeries` accepts a numeric pandas frame, or one-/two-dimensional arrays
with `time` and `variables`. Every column requires metadata with an explicit
unit symbol and representation. A source requires `id` and either `segment`
or all of `site`, `station`, `instrument`. Unit symbols are caller declarations;
the foundation neither infers physical dimensions nor converts units. Empty,
whitespace-containing units, nonnumeric/infinite values, incomplete identities
and unaligned masks fail. Nullable numeric dtypes remain intact; numeric object
columns and all-null object columns become nullable Float64. All-null variables
and zero-row aware frames remain represented.

Choose either `observed` (an aligned boolean mask) or `contribution` (aligned
`observed`, `imputed`, `derived`, `unavailable`, `unknown` strings). Undeclared
finite inputs are unknown. Nonfinite cells are unavailable. Qualification flags
are independent: `present`, `quality_accepted`, `qc_rejected`, `absent`,
`inserted`, `structural`, `artificial`. Undeclared original qualification of
imputed/derived/unknown finite inputs is represented by unset original flags;
completion cannot prove the original observation existed. Observations require
present/accepted qualification and cannot be artificial, rejected, absent,
inserted or structural. Structural cells cannot contain finite contributions.
The returned `qualification` mapping always lists all seven flags in sorted
name order, independent of the supplied mapping's order and of the process hash
seed; membership and masks are unchanged.

A finite artificial cell is rejected unless its current contribution explicitly
declares an imputed replacement. An original accepted/artificial flag can thus
remain true while the current contribution is imputed. No benchmark truth
storage or accessor feeds analysis. Imputation requires `source.imputation`
with `method_id` and `plan_id`; the trusted-imputation gate additionally needs
an explicit `source.trusted_imputation=True` declaration. This declaration is
caller-supplied evidence, not a library certification of an imputer.

For a supplied aggregate, each derived cell needs a `ContributorLineage` in
the variable's row-aligned `lineage` list. Contributor IDs must be unique and
nonempty; origins are observed/imputed/unknown. Units must agree with the
output. Linear means record normalized nonnegative finite weights and influence
fraction. Medians conservatively retain touched flags and call the weighted
contributor fraction coverage, without claiming nonlinear influence. Unknown
ancestry cannot pass provenance gates even with zero declared coverage weight.
The foundation records supplied aggregation; `collective.prepare_collective`
builds explicit matrices without reconstruction. Lineage also supports sums,
duration-weighted rates, integration and circular means, owned parent records,
original contributor details and optional observed-only counterparts. Nonlinear
ancestry keeps coverage/touched semantics through subsequent reducers.
Derived lineage may declare per-cell `trusted_imputation`; where present this
overrides the source-wide default for that cell, while unknown ancestry always
withholds eligibility. Collective preparation derives this declaration from the
actual contributors rather than unrelated collection members.
Accessors copy frames (including nested object cells),
arrays, source/variable metadata, lineage and results.

Optional `source_details={"provenance": producer_json, "cells": tables}` supplies
a separately owned payload: finite JSON producer metadata and either no cell
tables or exactly aligned `state`, `method`, `reason` DataFrames with optional
`quality`. The constructor validates the fields/alignment, normalizes missing
cells and computes a versioned content digest once; `source.details` is reserved
when details are supplied. `.source_details` returns complete defensive copies,
or `None` for ordinary sources. `.source` contains the computed compact summary;
series identity binds it without repeatedly traversing full details. Ordinary
caller source records retain their existing behavior. See [the EcoFill migration
and digest specification](ecofill.md#unreleased-source-detail-accessor-migration).

## Clocks and support

Time must be an ordered unique timezone-aware `DatetimeIndex` without NaT.
Stored second, millisecond, microsecond and nanosecond indices are converted
exactly to an owned nanosecond clock for arithmetic, preserving instants and
timezone. Input indices are unchanged. Dates, phases, durations or regular cell
endpoints outside nanosecond representation raise a descriptive `ValueError`.
Phase differences and elapsed spans use Python integers to avoid signed int64
overflow; seconds in duration records retain submicrosecond contributions.
`clock_resolution="resolved"` declares resolved DST/folds; unresolved clocks
fail. `storage_timezone` must match supplied timestamps, while `civil_timezone`
can differ. Regular mode requires a positive fixed cadence; phase is explicit
or resolved to the first supplied timestamp and saved. Off-grid samples fail;
omitted grid timestamps remain holes and break runs. No filling or compression
occurs. Irregular mode retains actual elapsed time, rejects regular cadence/
phase declarations, and cannot resolve a duration to a sample lag.

`TimeAxis.resolve_lag()` accepts `one_step`, `one_hour`, `twenty_four_hours`,
or explicit durations. It returns requested/realized lag and sample count.
At 2 h cadence, 1 h has `unsupported_exact_lag` and no realized lag; 4 h gives
two samples and one step gives 2 h. Rounding is never used.

`select_support()` accepts `longest_run`, `all_runs`, `windows`, `irregular`,
and `complete_case` under a compatible estimator contract. Runs refer to
original positions, not compressed values; equal longest runs choose the
earliest. A regular estimator consumes each returned run/window on its own
timestamps; the union of positions is not permission to concatenate runs into
one regular signal. Complete-case selection preserves joint row identities.

Windows require positive duration/step, left closure `[start,end)`, explicit
anchor (default first timestamp) and start/end labels. Regular durations and
steps must be exact cadence multiples. Every requested window through the
record extent remains in the report, including empty, partial and skipped rows,
with endpoints, support identity, counts and a reason. `complete=True` requires
all expected regular cells; `complete=False` retains eligible points explicitly.
For irregular windows, completeness concerns supplied rows; it cannot certify
unobserved grid coverage. Tail windows remain insufficient if incomplete.
`window_settings` saves duration, step, resolved anchor, completeness, label and
closure, and participates in window and selection identities even when two
settings happen to select the same cells. Non-window selections reject supplied
window/step/anchor, `complete=False`, and `label="end"`; ordinary neutral defaults
remain valid. Only left closure is supported.

Regular duration uses cell coverage `[timestamp,timestamp+cadence)`: 72 hourly
cells cover 72 h although their timestamp span is 71 h. Full duration includes
omitted grid cells; supplied, selected, excluded duration and missing-grid count
are separate. Full-segment burden uses the expected grid-cell denominator;
selected and pair burden use their actual rows. Each window records both
`window_cells` coverage and `burden` on eligible metric rows. Counts/fractions
use equal-cell weighting and retain unknown/unavailable counts. Every burden
records numerator, denominator, support identity and longest reconstructed
block in cells/seconds. Pair results keep first/second variable identities and
union touched-cell coverage separately, including overlapping reconstruction.

Each variable also saves `operation_counts` and `fraction_kind`. The legacy
`numerator`/`fraction` is labelled `linear_influence`, `contributor_coverage`, or
`mixed_influence_and_coverage`; the mixed value is a descriptive sum, not an
influence estimate. Separate `linear_influence`, `contributor_coverage` and
`touched_cell_coverage` records each contain numerator, denominator and fraction.
Contributor coverage averages only nonlinear cells. The linear component uses
the remaining support cells (including unavailable/omitted cells in full-grid
accounting); touched coverage uses the entire relevant denominator. Empty
component denominators yield `fraction=None`. Per-variable pair records retain
these same semantics on joint eligible rows; their union is touched coverage
and does not infer overlapping contributor influence.

Irregular durations are unknown unless justified `coverage_seconds` and a
positive `max_coverage_seconds` are explicitly supplied. Coverage cannot exceed
the declared bound or overlap the next sample. It never automatically extends
through a long gap. Irregular reconstructed-block adjacency remains unknown.
Coverage weights describe durations; the current reference estimator uses
equal point weights, saved in its scaling record.

`ObservationPolicy` separates supplied/observed/trusted values, minimum points,
and optional maximum imputed fraction and reconstructed duration. `use_status()`
checks relevant support, unknown ancestry and metric qualification. The recipe
0.15 threshold is `relevant_support_v1`, a study convention; it does not certify
ecological validity. A computed exploratory quantity remains sensitivity-only.
The explicit `primary_gate` basis is
`linear_influence_with_nonlinear_touched_cells`: ordinary imputed cells count
one, linear means count their imputed influence fraction, and each median with
any imputed ancestor counts one regardless of contributor weight. This sum is
divided by the relevant support denominator. Thus 72 medians each with one
imputed parent out of ten retain contributor coverage 0.1, but gate/touched
coverage 1.0 and are sensitivity-only at 0.15. All-observed medians have zero
burden; a four-parent linear mean with one imputed has influence/gate fraction
0.25. This conservative median rule bounds exposure, not actual nonlinear
influence, and can withhold primary use even if imputation did not change a
median. Unknown ancestry still takes precedence, including zero-weight unknown
parents. Longest reconstructed blocks continue to use touched cells.

## Frozen references and results

`ReferenceSpec` requires finite increasing raw-coordinate edges and explicitly
records scaling, population ddof=0, minimum points, point-compatible support,
and outside `raise`/`exclude` policy. The same supplied edges apply per variable;
variable order, unit and representation must match when applying a reference.
Bins include the final edge, like NumPy histogram. No edges are clipped or refit.

`fit_reference(..., training_positions=...)` fits only eligible selected training
rows and records their positions, inclusive timestamp range, source, clock/
support identity, variable order/units, mean, SD, edges/spec identities and scope.
Missing support and zero/unrepresentable scaling return reasoned reference
statuses. Invalid specs or unsupported support combinations raise.
For series with separate details, the frozen source record contains only the
compact summary/digest and originating series identity. Fitting and describing
references do not read or copy the full producer or cell payload.

`apply_reference()` requires exactly one `reference` or `spec`. The `spec` route
fits on evaluation input and labels the operation `self`/`retrospective`.
The frozen route never refits and recomputes counts/probabilities from current
eligible values. `outside="raise"` produces `OutsideReferenceError` with owned
below/above diagnostics; `exclude` records excluded rows, drift counts and the
remaining evaluation denominator/burden. It never silently clips. Too few
in-range rows return `outside_reference` and a missing value.

`ResultRecord` owns values and support, exposes `.value`, `.summary()`,
`.describe()` and `.to_frame()`, and retains units/representation and
method/config/reference IDs. `compute_status`, `metric_status`, `use_status`
are independent declarations; numeric feasibility cannot promote a scientific
qualification. A missing result requires `value=None` and a reason. Configuration
errors raise; programming errors propagate. These objects promise defensive
ownership through the public API, not protection from deliberate private-field
modification in Python.

## Policies and persistence

`AnalysisPolicy(recipe, settings, operational=...)` merges validated ordinary
Python settings into exactly the three named recipes. `.describe()` returns
requested/resolved settings and IDs, migration rows, unverified parity and
unresolved requirements. `.to_dict()` / `.from_dict()` roundtrip the resolved
policy; save `.describe()` as well when original requests and migration evidence
are needed. Scientific IDs are canonical by key order and exclude `created_at`
and `run_label`. Changed scientific settings change identity. Returned mappings
are owned copies.

`support.complete_windows` and `support.window_label` operate only under
`selection="windows"`, resolving to `True` and `"start"` by default. They are
`None` in resolved non-window policies and omitted from `support_options()`.
Non-window `False`/`"end"` requests fail shared Python/YAML validation, as do
window/step hours. The neutral legacy values `True`/`"start"` canonicalize to
`None` outside windows, so their resolved scientific identity matches defaults;
requested identities still preserve the supplied request. Window policies
forward completeness and label explicitly and roundtrip their resolved values.

The sensor route requests the earliest longest completed run, raw descriptive
and residual PSD/FI branches, and its unverified pinned coherence helper. The
report route requests all valid runs and average-run corrected coherence; its
individual PSD/FI and coherence preprocessing remain unresolved where the source
does not identify a complete pinned transform. The research route requests
windows, both raw/residual branches, a segment-count/resolution rule and pooled
auto/cross spectra. Independent explicit estimators are available, but source-specific
fit, null, taper, normalization and backend choices stay visible in `unresolved`.

YAML is optional. `load_policy_yaml()` / `dump_policy_yaml()` import PyYAML only
on demand and report `install tezca[yaml]` when absent. Loading rejects unsafe
constructors, aliases, merge keys, duplicates at all depths, unknown fields and
nonmapping/malformed roots. YAML and Python use the same scalar/range/cross-field
validation. The old profile requires an explicit `target`; see the complete
[45-leaf migration table](policy_migration.md) and [qualification and evidence limits](../../04_delivery/qualification.md).

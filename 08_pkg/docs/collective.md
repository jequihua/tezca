# Explicit collective preparation

`tezca.collective` implements `CollectionMember`, `FeatureSpec`,
`prepare_collective` and the separate `report_site_gate`. Each member declares
one source variable, physical feature, station, family and measurement semantics.
Features select member IDs explicitly. Multiple stations require an explicit
`station_reducer="median"` and `spatial_group`; otherwise they stay separate.
Members of one feature must agree on physical feature, family, units, semantics
and raw/residual representation. No aliases or conversions are inferred.

```python
from tezca.collective import CollectionMember, FeatureSpec, prepare_collective

collection = {"local": CollectionMember(data, variable="level", station="A",
                                      family="hydro", feature="water_level",
                                      semantics="state")}
result = prepare_collective(
    collection, features=[FeatureSpec("level_A", ("local",), min_coverage=.75)],
    start="2000-01-01T00:00:00Z", end="2000-01-02T00:00:00Z",
    cadence="1h", label="start")
matrix, audit, observed_counterpart = result.matrix, result.audit, result.observed_only
```

Inputs require declared regular clocks. Source cadence divides the target
cadence exactly, and source phase aligns with the explicit target start. The
end-exclusive interval contains whole bins. Each source timestamp owns
`[timestamp,timestamp+source_cadence)`; no interpolation, boundary duplication,
gap compression or implicit clock repair occurs. `label="end"` changes the
matrix/audit display labels only: `result.series` always uses interval starts
so downstream duration/window calculations retain actual cell ownership.

Coverage is eligible source slots / expected source slots per station and bin,
including missing timestamps in the denominator. The threshold is inclusive;
`ObservationPolicy.min_points` also applies within bins. Default selection is
observed-only; callers explicitly select `provided` or `trusted_imputation`.
`priority=("local", "remote")` orders every declared feature member and selects
the first eligible source per station/slot. Simultaneous eligible alternatives
without priority raise. Alternates at one station must share cadence. Rejected
and lower-priority contributors remain in the audit, with original qualification
flags, source identities, exact positions, timestamps and reasons.

Semantics and reducers are deliberately small:

| Semantics | Reducer | Output |
| --- | --- | --- |
| `state` | `mean` | Equal-slot state mean, same units |
| `accumulation` | `sum` | Sum of interval accumulations, same units |
| `rate` | `rate_mean` | Explicit duration-weighted mean, same rate units |
| `rate` | `integral` | Sum of rate × duration / declared `rate_period`; explicit `output_unit` |
| `angle` | `circular_mean` | Circular mean in declared `deg` or `rad` |

Integration units are the caller's explicit dimensional declaration; no general
unit library validates them. A circular resultant magnitude <=1e-12 is missing,
with `zero_circular_resultant`. Station medians currently support state variables
only. Every bin and feature remains present when unavailable.

Lineage preserves the contributor tree, original qualifications and origins,
and IDs/counts at each level. Means/sums/rate reducers record normalized linear
contribution exposure; medians and circular means record contributor coverage.
Linear lineage stores local coefficients and their composed total on original
contributors. Mean coefficients normalize locally; sum coefficients remain
unnormalized, and rate-integral coefficients include duration / rate period.
Parent totals therefore compose before burden normalization. Grouping a sum into
bins with different contributor counts cannot change its reconstruction fraction.
Nonlinear lineage has no claimed linear coefficients or coefficient total.
An outer linear reducer over nonlinear ancestors stays conservative: any
imputed ancestor activates the nonlinear touched-cell gate. Unknown ancestry
cannot become observed. The observed-only counterpart independently reruns
priority, coverage and reducers on observed ancestors, including available
counterparts of nested aggregates; it never treats a touched median as observed.
This is an alternative calculation, not a claim that excluded contributors had
no influence. Full producer payloads remain with their source; contributor
records retain compact source summaries/digests.

`matrix`, `series`, `audit`, `observed_only` and `describe()` return owned copies.
Descriptions bind requested/resolved explicit settings, NumPy version, source
and feature order, units/basis, exact eligible positions, support/burden and
method `collective.explicit_cell_aggregation_v1`. `reference_id` is `None` since
preparation does not fit a reference. Per-cell compute statuses are `ok` and
`unavailable_cell`; overall statuses are `ok` and `incomplete_matrix`.
Reasons are `no_eligible_contributors`, `insufficient_within_bin_coverage` and
`zero_circular_resultant`; rejected source records also use `missing_timestamp`,
`unavailable_or_ineligible`, and `lower_source_priority`. Metric status is
`audited_preparation`. Use status follows the explicit support policy:
`primary_eligible`, `sensitivity_only`, `unknown_provenance`,
`unverified_gap_geometry`, or `insufficient_support`. Numeric feasibility is not
domain admission or ecological validity. Invalid specifications raise.

`support.features` records each feature's eligible positions, relevant burden
and compute/use statuses. A feature's unavailable rows cannot dilute its use gate;
`support.full_grid_burden` separately describes exposure over the requested grid.
Overall use status conservatively prioritizes unknown provenance, unavailable
features, sensitivity-only, unverified geometry, then primary eligibility.
Each derived cell stores the imputation trust of its actual contributing
ancestry. Ordinary observed and unused sources do not strip that trust, while
any untrusted reconstructed contributor prevents trusted-imputation eligibility
for its output. This qualification survives nested aggregation. Membership uses
frame column labels and is independent of metadata dictionary insertion order.

`report_site_gate(result, coverage_basis="observed_only")` is a separate numeric
screen. `provided` coverage is available explicitly. It requires two declared
families, at least 30 days of common covered target cells, variable coverage
>=.65 over the whole requested grid, and at least five retained variables.
A family covers a cell when at least one of its declared features is eligible;
common coverage intersects all declared families. It reports exact positions,
denominator, excluded variables and reasons. Status is `ok` or `site_gate_failed`;
failure reasons are `fewer_than_two_families`, `fewer_than_30_common_days`, and
`fewer_than_five_retained_variables`, with variable reason `coverage_below_0.65`.
It never restricts generic two-/three-variable preparation.

`08_pkg/examples/collective_preparation.py` constructs synthetic quarter-/half-hour
inputs over 64 end-exclusive days: 1,536 hourly rows × six columns with four/two
contributors per hour. Its units and membership are synthetic declarations.
The real F01 candidate remains `unresolved_Q002_Q003`; passing this numeric
screen grants no raw-data admission, historical parity or publication rights.
Policy recipes remain unresolved in `AnalysisPolicy.method_spec()`. Declare
membership, reducers, priority, coverage basis and source semantics through this
separate API; doing so does not resolve the recipe. The resulting series
can feed [full-spectrum PCA and training projection](pca.md), [signed networks](network.md),
and [report-style coherence](coherence.md) with explicit pair lineage.
[Modern coherence](coherence_modern.md) pools spectra and qualifies correction,
phase and white-noise inference. [Graph-state roughness](graph_state.md) separates
representation, occupancy and frozen geometry, retaining all16 resolution requests.

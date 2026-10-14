# Signed correlation networks

```python
from tezca.network import NetworkSpec, correlation_network, report_network_spec

spec = NetworkSpec(features=["level", "temperature", "salinity"], basis="raw",
                   missingness="pairwise", min_points=12, threshold=.3,
                   window="168h", step="24h", complete=False, valid_pairs=True)
result = correlation_network(data, spec=spec, start=aware_start, end=aware_end)
summary, edges = result.summary, result.edges
client = report_network_spec(features=["level", "temperature"], basis="raw")
```

`AnalysisSeries`, `DailyResidual`, and prepared collective `.series` inputs are
accepted. Features are explicit, distinct physical variables; no five-feature site
gate applies. Omitting window/step computes a single point-support network.
`observations` defaults to `ObservationPolicy("observed_only")`. Neither supplied
completion nor unknown lineage becomes an observation. No interpolation occurs.
Correlation is dimensionless; original feature units remain in support and may differ.

Each unique undirected pair has one signed Pearson correlation, count, exact common
positions/instants, per-variable support/burden, eligibility, statuses and reason.
Pairwise selection intersects the two variables; listwise selection intersects all
declared variables. Constants remain nodes. Every pair requires the larger of
`min_points` and observation-policy minimum points. Correlation positively rescales
each input before centering to avoid overflow; only roundoff within 1e-12 outside
[-1,1] is clipped. Absolute-threshold inclusion uses the same 1e-12 equality tolerance.
The threshold describes association strength, never significance or causality.

For p nodes, `mean_absolute_connectivity = 2*sum(i<j, abs(r))/(p*(p-1))`.
`strong_count` counts each `abs(r)>=threshold` pair once; `density` divides this count
by `p*(p-1)/2`. If any declared pair is undefined, all full-network metrics are missing;
the denominator and p stay unchanged. Optional `valid_pairs_summary` uses only defined
pairs, names its own method and reports its explicit denominator. A zero valid-pair
denominator produces missing values, never zeros. `historical=True` separately
exposes `historical_diagonal_diluted_mean` (full mean times `(p-1)/p`) and
`historical_double_counted_links` (twice the unique count), also missing when the full
network is unavailable. These formulas do not assert historical implementation parity.

Rolling windows require a regular declared cadence, exact cadence-multiple duration
and step, and phase-aligned aware start/end. End is exclusive. Defaults span the
first timestamp through last timestamp plus cadence. Labels may be start or end;
labels do not change membership. By default only windows fully contained in the
requested interval are requested; `include_partial=True` also retains trailing
windows, with support truncated at the requested end. `complete=True` requires each
pair to cover every expected cell of the full window, including clock holes;
`complete=False` allows incomplete pair support subject to minimum counts.
Empty input needs explicit rolling boundaries. A too-short interval has zero
requested windows and an `insufficient_support` container status with a reason.

`report_network_spec` is the explicit client recipe: 168-hour windows, 24-hour steps,
complete pair support and absolute threshold .3. For 240 hourly cells it requests
four windows starting at hours 0/24/48/72. A wholly unavailable requested window is
retained with all its undefined edges. Global named policy recipes retain an
explicit unresolved declaration in `AnalysisPolicy.method_spec()`. Use this
separate network API with physical features, basis, missingness, pair minimum
and boundaries; supplying those choices does not resolve the policy declaration.

`NetworkResult.summary` and `.edges` are independently owned tables; `.describe()`
owns metadata and `.identity` binds settings, source, support and reference records.
The summary has flat numeric columns for export plus the canonical `ResultRecord`
value/status/support fields. Edges retain settings, method/backend/version and
daily-transform reference identities. Relevant burden uses each pair's actual
common support; full-segment burden is separately labelled. Nonlinear coverage
keeps its conservative touched-cell gate. Unknown or stricter training qualification
survives observed evaluation; all known-provenance networks remain sensitivity-only.

Edge compute statuses are `ok`, `insufficient_support` (reasons
`gap_or_incomplete_window` or `too_few_common_points`), `constant_input`, and
`invalid_correlation`. Summary/container compute status is `ok` only when all its
pairs/windows succeed, otherwise `undefined_pairs`; a zero-window container is
`insufficient_support`. Missing table numeric cells may display as NaN; the
canonical unavailable summary value is `None` with a reason. Metric status is
`exploratory_association`. Use status is independently `sensitivity_only`,
`unknown_provenance` or `insufficient_support`. Invalid configurations raise.

`08_pkg/examples/correlation_network.py` exercises the installed synthetic client recipe.
Q001–Q003 still gate historical parity, real F01 admission and rights.

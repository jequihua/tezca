# Discrete FI1 and complete trajectories

`tezca.fi1` implements ordered probability-mass roughness
`4*sum(diff(sqrt(p))**2)` and entropy `-sum(p*log(p))` in nats, taking
`0*log(0)=0`. There are no boundary ghost bins. FI1 is dimensionless and
depends on the ordered partition; it is neither density Fisher nor FI2.
The array primitive `discrete_information(p)` requires normalized real masses.
The source definition is proposal 008 section 7.3/E11; this independent method
has identity `fi1.ordered_mass_roughness_v1`. Sensor-helper parity remains
unverified under Q001, as do real-data validity and redistribution rights.

```python
from tezca.fi1 import FI1Spec, fit_fi1_reference, fi1_discrete, fi1_windows

spec = FI1Spec([0, 1, 3], pseudocount=0, outside="raise")
reference = fit_fi1_reference(training, variable="x", spec=spec,
                              training_positions=list(range(48)))
result = fi1_discrete(evaluation, variable="x", reference=reference)
trajectory = fi1_windows(evaluation, variable="x", reference=reference,
                         anchor=evaluation.time.index[0])
rows = trajectory.table
summary = trajectory.describe()
```

Explicit finite increasing edges include the last edge. The separate
`FI1Spec(partition="local_fd_v1")` rule uses equal-width edges from training
minimum to maximum, width `2*IQR*N**(-1/3)`, and `ceil(range/width)` bins bounded
to 10–40; zero IQR uses 10. IQR uses linear quantiles. Its default count
pseudocount is 1e-12; explicit partitions default to zero. No upstream code
is copied or run. `partition="sensor_pinned"` yields `missing_backend`, never
a replacement masquerading as the absent helper. Policy FI1 declarations remain
unresolved in `AnalysisPolicy.method_spec()`. Call the FI1 API with explicit
settings and raw/residual basis; doing so does not resolve the policy declaration
or select or certify a pinned backend.

Each evaluation recomputes counts, smoothed probabilities, occupied bins before
smoothing, FI1 and entropy. Edges and their unit/identity, pseudocount and
normalization are saved. `outside="raise"` reports below/above positions;
`"exclude"` saves excluded count/mass and the conditional in-range denominator
and burden. It never clips. Constant training or evaluation samples yield
`degenerate_input`, including with explicit edges; all-null and too-short
inputs give `insufficient_support`. Too few in-range samples give
`outside_reference`. A degenerate sample is distinct from the valid arithmetic
of a supplied discrete mass vector with zero interior differences.

Supply exactly one spec or reference. Point evaluation with a spec fits a
retrospective `global_self` partition; references freeze training-only geometry.
Held-out values do not enter its fitting support or identity. Feature, units
and basis must agree at application. Inputs can be `AnalysisSeries` or an owned
daily residual, retaining the daily training reference and its qualification.
Fit/evaluation provenance and observation policy stay separate.
Evaluation supports, including each trajectory row and the trajectory's outer
support, save their own observation policy in `support["observations"]`.
Method status is always `historical_sensitivity_index`; feasible values are
sensitivity-only, and unknown evaluation or reference origin remains
`unknown_provenance`.

Windows require regular time and an explicit aware grid-aligned anchor at or
before the first timestamp. Defaults are complete 72 h intervals every 6 h,
left-closed/right-open, labeled by start; end labeling is explicit. The schedule
includes every requested interval whose end is at or before the record's final
cell endpoint: 168 hourly cells give 17 intervals. Short trailing intervals are
outside this full-interval schedule, not silently analyzed as complete windows.
Every scheduled interval remains a row when missing/excluded cells prevent a
calculation, with exact positions, expected/available/eligible counts, reason,
window and metric burden, independent statuses, source, basis and method IDs.
Outside exclusions additionally report the actual conditional evaluation count.
An irregular clock raises rather than inventing complete grid coverage.

The spec route uses one `global_self` partition by default. Explicit
`fit_scope="window_self"` fits each interval separately and records separate
reference/edge identities; it asks a different question. Frozen references
never refit. Accessors own nested tables, distributions and reference records.
`describe()` includes schedule/settings, identity and summary statistics.
`table` remains available even if every window is unsupported; the trajectory
then reports `insufficient_support`, missing medians and threshold, and each
row retains its more specific reason. Records shorter than a window have an
empty table and no eligible-window summary.

`adjacent_flags(times, values, step="6h")` computes absolute differences only
between consecutive available rows exactly one stride apart. Its threshold is
the median difference plus three times the unscaled MAD; comparison is strictly
greater than. Missing rows or strides never create bridge differences. Missing
delta/threshold yields a nullable decision, not False. Summaries retain eligible
window count, median/IQR FI1, median entropy, available/flagged counts and the
actual threshold. These are retrospective sensitivity flags, not calibrated
change-point or ecological significance tests.

The installed `08_pkg/examples/individual_dynamics.py` now demonstrates FI1=0 and
entropy log(2), 17 complete windows and a constant-input unavailable result,
alongside the earlier individual and spectral methods. Distinct
[density Fisher methods](density_fisher.md) and [harmonic diagnostics](harmonic.md)
are available under their own formula identities and assumptions.

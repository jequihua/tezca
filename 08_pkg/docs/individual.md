# Individual dynamics

`tezca.individual` provides `describe_series`, `estimate_trend`,
`autocorrelation`, `DailySpec`, `fit_daily_reference`, and
`apply_daily_reference`. Inputs are owned `AnalysisSeries`; residual results
also enter the same numerical functions. Descriptions return one `ResultRecord`
per variable/metric; concatenate each record's `to_frame()` for a table.
Trend/ACF take an explicit variable. Default observations are `observed_only`.
Point statistics use all eligible runs (actual irregular timestamps when
declared); a requested `longest_run` is the earliest tied longest run.
Only `support=None` chooses a default; invalid supplied selections raise.
For supplied `AnalysisSeries`, each variable's declared representation is
preserved in the result and support basis, including externally supplied residuals
or standardized inputs. The daily application route explicitly yields
`daily_residual` with its fitted-reference identity.

Mean/median/sample SD (`ddof=1`)/linear-interpolated IQR/min/max and population
variance each keep exact positions, source, units, basis, status and identity.
One point can describe a mean while sample SD and slope remain unavailable.
Constants retain a `constant_input` declaration in description records. Slope
is OLS against actual elapsed days with measurement units/day. Its default
uncertainty is unavailable because dependence-aware inference is unimplemented.
`uncertainty="iid_ols_compatibility"` exposes a named iid diagnostic with its
assumption, expressly untrusted for temporal inference.

ACF is Pearson correlation between exact lagged pairs inside continuous eligible
runs. It records original position pairs, pair count, endpoint-local burden,
requested/realized lag and refusal reasons. No endpoint pair can jump a missing
or excluded intervening cell. One step depends on cadence; durations never round.
Irregular clocks, constant margins and too few pairs give missing results.

Daily fitting requires `DailySpec(civil_timezone="Europe/Berlin")` or another
explicit IANA zone; it never consumes a default `TimeAxis.civil_timezone` as a
declaration. Timestamps must already be resolved. The only supported fold policy
`daily_median` combines readings for one civil date/minute first, then takes the
median across distinct dates. Default minimum is two dates; folded readings
count as one daily replicate. Spring minutes are never manufactured. Each of
1440 bins stores its distinct-date count and missing/available template value.

```python
from tezca.individual import DailySpec, fit_daily_reference, apply_daily_reference
reference = fit_daily_reference(data, spec=DailySpec(civil_timezone="UTC"),
                                training_positions=list(range(48)))
residual = apply_daily_reference(data, reference=reference)
```

The fit identity includes training values/qualification/support, explicit
positions/dates and all operative settings, excluding held-out values.
Exactly one reference/spec enters application. The spec route fits on the
evaluation record and is labeled retrospective self-fitting. Units, variables
and representation must match a frozen reference. Missing reference bins and
ineligible evaluation values stay missing; the raw input is preserved.
`residual.describe()` separates fitted support/burden from evaluation support.
Metric records include both. Unknown training provenance withholds primary use;
reconstructed training references remain sensitivity-only on clean evaluation.
This prevents a clean evaluation from concealing a qualified training transform.

Run `08_pkg/examples/individual_dynamics.py` with an installed wheel outside the
checkout (`python -I -B <example>`). It writes only JSON to stdout. Daily fitting
is a descriptive transform; residual memory and raw periodicity are separate
questions. Historical helper parity and real-data validity remain unverified.
Policy recipes still explicitly declare unresolved reference period/civil zone,
replicate and inference settings. They require these explicit API choices.
Welch is available through [explicit spectral specs](spectral.md).
Fits, ranked peaks and supported bands are available through [spectral summaries](spectral_summary.md).
Discrete FI1 and complete trajectories are available through [FI1 specs](fi1.md).
[Density Fisher](density_fisher.md) implements distinct histogram and KDE routes.
[Harmonic diagnostics](harmonic.md) provide independent Fisher-g/Siegel tests on
admitted raw periodograms. `08_pkg/examples/individual_milestone.py` connects these
families, fitted daily references and unavailable cases without optional extras.

# Understand a result

[Start here](../README.md) · [Spectral status vocabulary](../../08_pkg/docs/spectral_summary.md)

Read the number together with its units, actual contributing observations,
settings and three independent statuses. For a `ResultRecord`, `summary()` gives
the compact account, `describe()` includes support and identities, and `value`
returns an owned copy. Composite results also retain row-level and nested
statuses; an outer `ok` does not make every window, fit or threshold available.

| Field | What it tells you | Example |
| --- | --- | --- |
| `compute_status` | Could this requested calculation be made on this support? | `ok`, `insufficient_support`, `insufficient_frequency_resolution` |
| `metric_status` | What scientific interpretation does this method support? | `descriptive_spectral_summary`, `exploratory_candidate_regimes` |
| `use_status` | Is this result eligible under its declared support/provenance policy? | `primary_eligible`, `sensitivity_only`, `unknown_provenance` |

`primary_eligible` is a policy result, not certification of ecological meaning,
causality, data quality or historical equivalence. `ok` can coexist with
`sensitivity_only`. Invalid settings raise an exception; insufficient data usually
produces an explicit missing result with a reason.

## A useful value and a useful refusal

The [quickstart](quickstart.md) uses a cosine with unit amplitude. Its Welch
density integrates to approximately `0.5` in squared input units. The requested
48–240 hour band extends beyond the short segment's supported frequency range,
so its `value` is `None` and the status is `outside_frequency_support`. A regular
mean can still be available on that same source. Do not replace the missing band
with zero or infer that the long-period signal is absent. Collect longer suitable
support or ask a different, scientifically justified frequency question.

The PCA example yields eigenvalues approximately `[4.0, 1.0]`: two variance
directions, with the full spectrum retained. Component signs can be arbitrary;
comparing loadings requires the saved feature order, scaling and reference.
Regime labels are identifiers. Their numeric order is not a health ranking,
and a noise label differs from an excluded missing row.

## Read support, units and burden

Support means the exact rows, intervals, pairs or frequencies that contributed.
A window remains in a trajectory even when it is skipped. A pair correlation
may use fewer rows than either separate mean. Frequencies are in Hz; period
conversions use actual seconds. Trend is measurement units/day, Welch density
is unit²/Hz, integrated band power is unit², and ACF is dimensionless. FI1,
density Fisher and graph-state energy are different definitions and cannot be
interchanged because their names sound similar.

The reconstruction burden measures supplied imputed contributions on that
result's support. A linear mean can record a weighted imputed influence, whereas
a median records contributor coverage and whether any contributor was imputed.
The primary-use gate conservatively counts each touched nonlinear cell in full.
For example, one imputed parent out of ten gives a median contributor coverage
of `0.1` but a touched-cell gate of `1.0`. The `0.15` recipe cutoff is a study
choice, not a universal scientific threshold. Unknown ancestry remains unknown,
including an unknown parent with zero declared weight.

## Common unavailable outcomes

| Result or reason | Meaning and next action |
| --- | --- |
| `unknown_provenance` | Contribution origins were not established. Supply documented masks/lineage; do not relabel finite values as observed just to pass a gate. |
| `insufficient_support` | Too few eligible observations, pairs or windows. Inspect the saved counts and exclusions; obtain appropriate observations or report the limitation. |
| `insufficient_cycles` | The longest contributing run does not cover the required cycles. More separated short runs do not automatically supply a longer continuous run. |
| `insufficient_frequency_resolution` | Too few physical frequency ordinates, or no supported positive interval. Zero padding cannot create physical resolution. |
| `outside_frequency_support` | The requested band needs extrapolation. Preserve the refusal or justify a supported band before rerunning. |
| `outside_reference` | Evaluation values fall outside frozen geometry and too few remain. Inspect drift/exclusions; do not silently clip or refit on evaluation data. |
| `constant_input` / `degenerate_input` | A method needs variation or a defined denominator. Some descriptions remain meaningful; correlation or relative power may not. |
| Excluded pair / skipped window | Lineage, support or schedule rules excluded it. Retain its identity and reason; do not join across the gap. |

The [method pages](choose_methods.md) define their complete vocabularies, including
nested fit-comparison and peak-rejection reasons. Missing, zero, `False`, an empty
string and the string `"NA"` are distinct in canonical exports. CSV cells use
typed JSON to preserve those distinctions; use `load_results` or the supplied
standard-library consumer rather than guessing pandas dtype inference.

## References and saved identities

A training reference includes the training support, settings and identity.
Reusing it avoids changing the transformation to fit each evaluation set. Clean
evaluation data cannot erase unknown or reconstructed training provenance. Never
use deliberately hidden evaluation truth to fit preprocessing, choose a method
or replace a missing input.

Canonical numerical outputs and historical compatibility columns have separate
formula identities. A matching historical column name does not prove matching
calculations; unverified historical fields may be null with a reason. The
qualification appendix records what evidence is still needed.

Reload verifies saved structure, hashes and available source/reference bindings.
It does not refit, recompute omitted measurements, authenticate the author or
grant data rights. Expected identities from a separately trusted record detect
the wrong internally valid bundle. See [reload limits](migration_and_limits.md).

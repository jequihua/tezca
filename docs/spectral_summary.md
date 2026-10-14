# Descriptive spectral summaries

The separate `tezca.irregular.irregular_spectral_models` adapter applies the same
log-fitting definition to actual-time projection-density WOSA, with explicit
occupied-span cycles and resolved-frequency gates; see [irregular](irregular.md).
It preserves qualifications and does not convert that density into Welch.
The [four-route sensitivity API](sensitivity.md) explicitly selects shared
actual frequencies before invoking the same fit primitive on each route.

`tezca.spectral_summary` exposes `spectral_models`, `spectral_peaks` and
`band_power` on an owned Welch result. Each returns a ResultRecord carrying the
original basis, reference, variable/source, selected observations and burden,
the input spectrum digest and all summary settings. An optional requested basis
must match. Array primitives `fit_log_models`, `rank_peaks`, `integrate_band`
support independent ordinate controls; their outputs are owned, without invented
source provenance. The installed `individual_dynamics.py` example exercises all
three wrappers, including an hourly daily sine with 20–28 h band power 0.5.
Primitive geometry/ordinate inputs must be real numeric arrays; complex, boolean
and string arrays are rejected before conversion. Scalar settings reject booleans.
All three wrappers retain `descriptive_spectral_summary` metric status and
qualify primary-eligible inputs as `sensitivity_only`. Stricter input statuses,
including unknown provenance or a qualified training reference, remain unchanged.
Fit and evaluation support/burden and reference identity remain inspectable.

```python
from tezca.spectral_summary import ModelSpec, PeakSpec, BandSpec
from tezca.spectral_summary import spectral_models, spectral_peaks, band_power
models = spectral_models(spectrum, spec=ModelSpec())
peaks = spectral_peaks(spectrum, spec=PeakSpec(max_peaks=3))
bands = [band_power(spectrum, spec=BandSpec(hours))
         for hours in ([20, 28], [10, 14], [48, 240])]
```

ModelSpec declares a closed frequency interval, log base e or 10, sorted candidate
breaks (None resolves to the selected physical frequencies), and at least three
points per arm. Fits use unpadded Welch ordinates even when the displayed spectrum
is padded. The inherited Welch `supported_positive_range_hz` is intersected with
the requested interval. Zero, negative and nonfinite power and DC are counted and excluded;
there is no positive floor. The selected interval and realized candidate list
are retained. A continuous hinge fit reports beta as negative slope and both
break frequency in Hz and break period in seconds.
Frequency exclusions refer to original physical-grid positions and state
`outside_frequency_support`; the selected span is distinct from the available
support and requested bounds. Absent or degenerate inherited support is missing.
Candidates minimize RSS; an absolute RSS tie tolerance chooses the lower break.
The same declared tolerance applied to delta AICc chooses single on a model tie.
The working Gaussian log-residual AICc omits a common constant, counts k=3/5
including variance and estimated breakpoint, and requires n>k+1. RSS is retained
unchanged. RSS at or below the explicit numerical exact-fit tolerance (default
1e-24 log-coordinate squared units) has `exact_fit` status and no AICc or preferred
model; no epsilon is substituted. Constant log power has unavailable R2.
These are retrospective descriptive approximations, without process, long-memory
or ecological classification.

PeakSpec declares a closed period range in seconds, cycles on the longest
contributing run, maximum count and linear or log10 ordinate prominence.
Excluded ordinates split the search; only interior local maxima are candidates.
The inherited frequency interval is applied before prominence, period/cycle
filtering and truncation. Outside ordinates cannot contribute to prominence.
Peak tables retain original display-grid positions, and frequency exclusions,
eligible positions and their span retain the selection path without renumbering.
Period/cycle filtering precedes truncation. Sorting is descending prominence then
ascending frequency. Empty tables retain float/int columns and reasons. Peak
power has the input ordinate units; log10 prominence is in log ordinate units.

BandSpec accepts increasing hours or Hz bounds and records increasing frequency
bounds. Density scaling is required. Piecewise-linear trapezoid integration
interpolates boundaries within the available physical/display intersection;
extrapolation is unavailable. The fraction denominator spans that whole positive
range and excludes DC. The result has input-unit squared power; zero denominator
leaves fraction undefined. At least three unpadded ordinates and the explicit
minimum cycles (default two on the longest contributing run) must pass before an
energy, including zero energy, is reported. Padding cannot earn resolution.

Recipes remain explicit unresolved declarations where historical evidence is
absent. These functions do not establish historical helper parity or admit data.

## Status and rejection vocabulary

Read the outer result's three statuses independently. `compute_status="ok"`
means that this summary was computed; `metric_status` remains
`descriptive_spectral_summary`, and use remains `sensitivity_only` or the
stricter inherited input qualification. A valid outer fit may still contain an
unavailable AICc comparison. Peak rejection reasons describe individual candidate
rows and do not turn an otherwise valid empty peak table into a failed result.

| Status or reason | Where it appears and the condition | What to do |
| --- | --- | --- |
| `insufficient_support` | Log fitting has fewer than three eligible positive ordinates; wrappers also preserve an unavailable input spectrum's status/reason. | Inspect selected ordinates and upstream support; retain a missing result if evidence is inadequate. |
| `insufficient_frequency_resolution` | A wrapper has no supported positive frequency interval, or a band has fewer than `min_ordinates` physical, unpadded bins. | Choose scientifically justified segment/band settings or obtain longer usable runs; padding cannot satisfy the gate. |
| `insufficient_cycles` | A band fails `min_cycles` at its low frequency on the longest contributing run, or an individual peak fails its cycle requirement. | Inspect the actual contiguous duration; additional short disjoint runs cannot masquerade as a longer run. |
| `outside_frequency_support` | Band/denominator bounds require extrapolation; model/peak frequency-exclusion rows use the same reason for ordinates outside inherited support. | Retain the refusal/exclusion or justify a supported interval. |
| `outside_period_range` | A candidate peak lies outside `PeakSpec.period_seconds`; recorded in that rejected row's `reason`. | Inspect the requested range before changing the scientific question. |
| `max_peaks` | An otherwise eligible peak falls beyond the retained count after prominence/frequency ordering. | Inspect the saved rejected candidates and requested count. |
| `insufficient_aicc_points` | A fitted single or broken row has `n <= k + 1`, except an exact-fit row takes precedence; `comparison_status` carries the reason and AICc is null. | Keep the descriptive fit but withhold model selection; more independent support may be needed. |
| `insufficient_arm_points` | A proposed break has too few ordinates on one side for `ModelSpec.min_arm`; this is a candidate-row `comparison_status`. | Keep the rejected break in the record; do not silently reduce the arm requirement. |
| `exact_fit` | A fitted row's RSS is at or below the declared exact-fit tolerance. AICc and the preferred-model comparison stay unavailable. | Report the exact descriptive fit without inventing an epsilon or a winning model. |
| `unavailable` | Overall model `comparison_status` when there is no usable broken fit or either model's AICc is unavailable. | Inspect single/broken/candidate rows for the actual cause; `preferred` and `delta_aicc` are null. |
| `descriptive_working_aicc` | Both model AICc values support the declared descriptive comparison. | Interpret under the recorded working log-residual model, not as process classification. |
| `degenerate_input` | Band integration has no finite nonnegative positive-frequency density. | Inspect the input spectrum; zero is not a substitute for a missing estimate. |
| `undefined_zero_denominator` | Band energy is available but total positive-frequency power is zero; `fraction_status` carries this value and the fraction is null. | Report available absolute power separately from the undefined fraction. |

An empty peak table has `empty_reason="no eligible local peaks"`; it can result
from a smooth spectrum or from the retained exclusions/rejections. Constant log
ordinates have a null R² with `r2_reason="constant log ordinates"`. Neither case
licenses replacing missing values with zero. Invalid specifications and basis
mismatches raise rather than returning one of these scientific statuses.

# Actual-time projection-density WOSA

`tezca.irregular` provides `IrregularSpec`, `irregular_spectrum` and the explicit
`irregular_spectral_models` adapter. The method identity is
`spectral.ls_projection_density_wosa_v1`. It is an independent LSFT-WOSA
definition; historical helper parity remains `unavailable_Q001`.

```python
from tezca.irregular import IrregularSpec, irregular_spectrum, irregular_spectral_models
from tezca.spectral_summary import ModelSpec
spec = IrregularSpec(
    interval=[start, end], segment="96h", hop="48h", anchor=start,
    frequency_hz=frequencies, requested_band_hz=[frequencies[0], frequencies[-1]],
    f_ref_hz=1/3600, taper="boxcar", detrend="none", min_points=3,
    min_span="48h", min_cycles=1, rank_tolerance=1e-10)
spectrum = irregular_spectrum(data, variable="x", spec=spec)
fit = irregular_spectral_models(spectrum,
    spec=ModelSpec(interval_hz=[frequencies[0], frequencies[-1]], candidates_hz=[]),
    min_ordinates=3, min_cycles=1)
```

Inputs are owned `AnalysisSeries` or `DailyResidual`. Ordered aware actual
instants are used on irregular observations and regular records with holes;
there is no filling, cadence inference or temporal compression. The existing
data constructor rejects unresolved clocks. Elapsed differences use integer
nanoseconds before converting to seconds, including s/ms/us/ns stored indices.
Observations default to observed-only. Explicit provided/trusted policies keep
reconstruction burden and unknown origin; daily training qualification survives.

Declare a half-open analysis interval, anchor, segment duration T, hop, positive
frequency grid and containing closed requested band. Full anchor-aligned windows
within the interval are requested, including windows with no eligible points.
When the interval is shorter than T its partial window remains unavailable.
One weight per requested window, in chronological order, is required if weights
are supplied; positive finite weights are normalized independently at each
frequency over supported segments. Results retain every segment, selected and
excluded original positions/reasons, occupied timestamp span, burden, individual
ordinates, traces, ranks, actual weights and contributing IDs. Missing ordinates
never enter pooling as zeros. Outside-interval rows are recorded separately.
Aggregate contributing positions, burden and use qualification include only
rows supporting ordinates with at least two valid segments. Singleton spectra
and their qualification remain visible in the individual segment diagnostics.

For actual elapsed segment times t, form C=[cos(2πft),sin(2πft)] and an
orthonormal two-column basis Q by SVD. Reject rank below two at the declared
relative singular-value tolerance. W is boxcar or the continuous periodic Hann
`.5-.5*cos(2*pi*t/T)` at those actual positions. R removes none, an ordinary
constant, or an ordinary linear trend in actual time. Define

`A=R W Q Qᵀ W R; B=A/trace(A); P=2 xᵀ B x/f_ref`.

The implementation stores two-column factors L with B=LLᵀ internally, avoiding
dense global projections. Units are input-unit squared/Hz. Positive `f_ref_hz`
is a declared density scale, **not** an inferred irregular Nyquist. Physical
resolution is 1/T; increasing the display grid density does not improve it.
No general irregular Parseval or amplitude-response correction is claimed.
For complete regular boxcar/no-detrend interior Fourier bins and f_ref=fs this
equals the one-sided periodogram. Periodic Hann/no-detrend has that Fourier-grid
limit as well. Detrending/off-grid settings retain their explicit trace
normalization and do not claim universal equality with Welch.

For normalized segment weights b, embed B in original observation coordinates
and form G=ΣbB. Shared row identities, including holes, determine overlap.
Each ordinate reports raw K, `n_eff=1/(2 trace(G²))`, approximate DOF `2*n_eff`,
and predicted relative variance `2 trace(G²)`. These match the first two
quadratic-form moments for iid equal-variance Gaussian observations. They are
not exact generic chi-square inference or calibrated colored-noise intervals.
At fixed admissible geometry the noise expectation is `2*sigma²/f_ref`.
Spectral-window leakage, deletion and amplitude-response effects remain.

Aggregate `compute_status` is `ok` when at least one ordinate has two supported
segments, otherwise `insufficient_support` and value is missing. Unavailable
array ordinates are NaN with `insufficient_segments` rows, while individual
ordinate status is `ok` or `unavailable`. Reasons are `partial_analysis_window`,
`insufficient_points`, `insufficient_occupied_span`, `insufficient_cycles`,
`rank_deficient_frequency`, `zero_projection_trace`, `nonfinite_phase` or
`nonfinite_projection_density`. Constants retain `constant_input`; their
projection is still defined, but positive log-fitting may be unavailable.
Invalid specs/weights raise `ValueError`. Metric status is
`descriptive_irregular_projection_density`; use is sensitivity-only or retains
stricter provenance/support restrictions. These are descriptive spectra.

The explicit model adapter uses the existing single/broken log-ordinate fitting
definition. Empty `candidates_hz` requests only the single fit. It selects actual
positive supported frequencies within the requested interval, requires cycles on
the minimum occupied span of contributing segments, then greedily retains
ascending ordinates separated by at least 1/T. Every exclusion is recorded;
no interpolation or extrapolation invents bins. `min_ordinates>=3` and
`min_cycles>=0` are explicit adapter gates bound into identity. Too few qualified
points return `insufficient_frequency_resolution` or `insufficient_cycles`;
input failure statuses propagate. Metric status remains
`descriptive_spectral_summary`. Basis, training reference, support and unknown
qualification survive. The existing Welch-only peak/band/harmonic wrappers do
not implicitly accept this different estimand.

Run `examples/irregular_spectrum.py` with the installed package for a
synthetic periodic input and fitted-summary path. Required witnesses and fixed
control results are recorded in [qualification and evidence limits](qualification.md).
Policies retain unresolved historical LSFT declarations; this explicit API does
not silently resolve their missing source settings. Real-data qualification and
rights remain Field validation and reuse rights.

# Independent harmonic diagnostics

`tezca.harmonic` implements Fisher's g and Siegel's T as separate harmonic
diagnostics. These are dimensionless periodogram concentration statistics,
distinct from ordered-state FI1 and inverse-square-unit density Fisher.
Definitions follow [Antoch (1995), Critical values of Fisher's and Siegel's
test](https://dml.cz/dmlcz/124677), sections 1–2, equations 1.2–1.3 and 2.1,
as specified in proposal 008 section 7.2 and the M002 scientific contract.
Implementation is independent; no historical helper, source code or table is
copied. Q001 still gates exact pinned sensor-helper parity.

```python
from tezca.harmonic import (
    HarmonicSpec, calibrate_siegel, harmonic_diagnostics, compatibility_fields,
)

# A length-129 complete raw record has 64 admitted positive Fourier bins.
calibration = calibrate_siegel(m=64, alpha=0.05, lambda_=0.6,
                              replications=19999, seed=2026101101)
results = harmonic_diagnostics(data, variable="x", spec=HarmonicSpec(),
                               calibration=calibration)
fisher, siegel = results["fisher"], results["siegel"]
compatibility = compatibility_fields(results)
```

Inputs select one variable explicitly. The public API uses owned `AnalysisSeries`
and shared observation/support rules. Input must be raw, complete and regularly
spaced under the declared iid Gaussian white-noise null. The null is **assumed,
not proven by the data**. Provided reconstructed inputs require an explicit
observation policy, retain their burden and remain sensitivity-only; unknown
origin remains unknown. All results use `assumed_white_noise_diagnostic` metric
status and cannot become primary merely because a test rejects. No field-data,
ecological or colored-process calibration is claimed.

## Fourier geometry and Fisher g

`HarmonicSpec` records alpha (default 0.05), lambda (0.6), centering (True),
backend (`independent`), estimator (`rectangular_periodogram`), taper (`boxcar`),
nfft (None resolving to input length), normalization and null. Supported raw
execution computes one unaveraged ordinary DFT: `I_j=abs(DFT_j)**2/N` and
`q_j=I_j/sum(I)`, for j=1,...,floor((N-1)/2). DC and even-N Nyquist are omitted.
Measurement-squared powers and Hz coordinates are retained alongside normalized
q; frequency conversion uses exact integer nanosecond cadence, including
sub-microsecond intervals. Both g and T have units one. There is no
selected-frequency-band route.

Mean removal is applied only with `center=True`, and the actual removed mean is
recorded. Even with centering disabled, omitting DC makes the non-DC statistic
mathematically offset-invariant; that is not an implicit mean-removal operation.
Multiplying amplitudes by three leaves g/T unchanged. `fourier_proportions`
exposes the finite-array primitive, accepting either one row or a batch along
the last axis; it has no claim to supplied-data provenance or validated null.

Fisher uses `g=max(q)` and method ID `harmonic.fisher_g_exact_v1`.
For m included bins the upper tail is
`sum_k (-1)**(k-1)*choose(m,k)*(1-k*g)**(m-1)`, retaining positive bases only.
`fisher_tail(g,m)` handles the support endpoints explicitly. To avoid catastrophic
cancellation it represents the supplied binary float g as an exact rational,
sums integer powers exactly, checks the probability domain and converts only
the final ratio to float. It does not clip probabilities or introduce epsilon.
Very small final probabilities may underflow to zero at float precision.
`fisher_critical(m,alpha)` solves the upper tail with a bracketed Brent root
(absolute/relative tolerances 1e-14). Rejection uses p<=alpha. Exact arithmetic
has increasing cost with m; it is not a large-scale approximate tail algorithm.

## Siegel calibration and threshold scale

`siegel_statistic(q, alpha, lambda_)` computes
`T=sum(max(q-lambda*a_m(alpha),0))`, where a_m is Fisher's critical g.
Its decision threshold is on **T's scale**, not g's. Method identity is
`harmonic.siegel_dirichlet_mc_v1` and calibration is an explicit operation.
Without calibration, Fisher can compute while Siegel returns `not_computed`
with a missing value and decision.

`calibrate_siegel` draws B rows of m independent Exponential(1) variables with
NumPy Generator/PCG64 and normalizes each row. It stores a defensive sorted T
bank in a `SiegelCalibration`, recording m, alpha, lambda, B, seed, RNG, NumPy
version, null, simulation definition and conventions. This is a reusable null
record, not a fit to evaluation values. Applying it checks the method,
alpha/lambda/null/backend and included-bin count. No evaluation sample changes
the calibration; parameter mismatches raise. The record is owned, and results
carry its identity and compact metadata without copying the entire simulation
bank into every result.

Construction accepts only the validated simulation parameters (m, alpha,
lambda, replication count and seed), whether called through `calibrate_siegel`
or `SiegelCalibration`. The bank, order, domain, conventions, threshold and
resolution metadata are derived together by that constructor; callers cannot
supply or replace them. `describe()` returns inspectable owned output, not an
importable record. Arbitrary mappings, including modified counts or critical
values, are rejected at construction. Record deserialization is unsupported.

The conservative upper-tail p is `(1+count(T_sim>=T))/(B+1)`, with ties included.
Rejection is p<=alpha. Let R be the number of attainable positive integer ranks
r for which r/(B+1)<=alpha. For R>0 the reported critical T is sorted_T[B-R]
(zero-based): reject only for T strictly greater than that value. Equality does
not reject; including ties in the count makes this order-statistic rule consistent
with the p-value rule. Metadata exposes both zero-based index and one-based
order. With B=19999 and alpha=0.05 the index is 18999, order 19000 and minimum
p is 1/20000. A budget with minimum p>alpha yields
`insufficient_calibration_resolution`, no critical value and no decision.
The reported minimum p is resolution, not zero uncertainty or exact significance.

## Refusals and compatibility

Welch, nonrectangular taper and padded nfft requests return `unsupported_estimator`.
Supplying any spectrum result is refused without recomputing raw data; callers
must explicitly supply the raw series for that route. Missing/excluded values or
timestamp holes return `unsupported_support`; irregular clocks return
`unsupported_time_geometry`; fitted daily residuals or other transformed bases
return `unsupported_basis`; colored-noise declarations return `unsupported_null`.
Constants and exact even-length signals confined to DC/Nyquist give
`degenerate_input` with missing decisions. The latter are detected by constant
even and odd subsequences before FFT rounding can manufacture admitted power;
no epsilon or power floor is used. The array primitive rejects these degenerate
rows explicitly, while the public series API retains reasoned unavailable rows.
All-null/too-short inputs give
`insufficient_support`. Numerical failures are not concealed by broad catches.
`backend="sensor_pinned"` returns `missing_backend` and missing decisions.
Unknown option names or inconsistent configuration raise. None of these cases
becomes a false rejection decision, zero information or a silent fallback.

Each Fisher/Siegel `ResultRecord` retains source, input/config/method/reference
identity, variable, basis, exact eligible positions, clocks, burden, units,
requested/resolved settings, null and independent compute/metric/use statuses.
Returned frames, arrays, specs and calibration dictionaries are defensive copies.

`compatibility_fields(results)` returns exactly six historical field names:
`fisher_harmonic_rejected`, `fisher_g`, `fisher_g_critical`,
`siegel_harmonic_rejected`, `siegel_T`, `siegel_T_critical`.
It includes a mandatory sidecar with formula, independent method/backend,
null/support/status and reference metadata. Unavailable fields are None.
This is field-name compatibility, not pinned numerical parity or the later
general CSV/JSON exporter.

The fixed qualification uses the declared PCG64 calibration seed 2026101101
(B=19999,m=64), a separate 1024-by-129 normal bank with seed 2026101102, and 64
signal records with seed 2026101103. Tests use direct DFT and independent
probability arithmetic, preserve the predeclared null count range [25,85] and
signal minimum 60/64, and print compact settings/counts. No seed search or
threshold tuning is part of this method. `08_pkg/examples/individual_milestone.py`
connects all current individual families with unavailable outcomes, shared
support/reference checks and ownership checks. It writes JSON to stdout only
and needs no optional dependencies. Collective methods, exports and a separate
milestone-wide review remain outside this implemented slice.

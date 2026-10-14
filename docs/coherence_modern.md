# Pooled coherence and qualified phase inference

```python
from tezca.coherence_modern import ModernCoherenceSpec, modern_coherence

spec = ModernCoherenceSpec(basis="raw", period_band=("4s", "64s"),
    nperseg=128, hop=64, taper="hann", detrend="constant", alpha=.05,
    null="gaussian_white", weighting="equal_segments")
result = modern_coherence(first, second, variable_a="level", variable_b="temperature",
    spec=spec, lineage=lineage, nodes=["a", "b"], target="a")
```

Inputs are owned `AnalysisSeries` or `DailyResidual` with matching explicit basis,
cadence and phase. Supply the source-bound `PairLineage` described in
[pair audits](coherence.md). Exact common valid runs preserve clock holes; no
segment crosses a gap and no observation is interpolated or completed. Default
observations are observed-only. Duplicate/copy/target transforms, unknown ancestry
and shared donors keep the same ranking exclusions as report coherence. All
results remain exploratory and inherit contributing and training provenance.

The independently specified method is `coherence.pooled_spectra_v1`. Fixed even
segment length L >=4, integer hop 1..L, FFT length a positive integer multiple of L, boxcar/Hann taper,
periodic/symmetric convention and none/constant/linear detrending are explicit.
Runs shorter than L remain reasoned excluded records. Accepted runs retain
segment starts, raw counts, unused tails, exact positions and original spectra.
Shared report/modern support code evaluates constancy on actual contributors.

`pool_coherence_runs(runs, spec=spec, run_weights=None)` is also an owned numerical
primitive for declared run spectra, without provenance certification. Each run
contains `run_id`, `raw_segments`, `segment_starts` starting at zero with fixed
hop, `sampling_hz`, `nperseg`, `hop`, `nfft`, `taper` (the actual vector), `detrend`,
`scaling="density"`, `constant_input`, `frequency_hz`, `sxx`, `syy`, `sxy_real` and
`sxy_imag`. All runs must have identical settings and exact frequency grids.
Stable IDs canonically order runs and bind input spectra; reversing input order
preserves the result and pool identity. No adaptive-grid pooling or extrapolation
is supported. Raw run means come from identically normalized segment FFT products.

Equal-segment weighting assigns each periodogram the same weight. For
`weighting="run_weights"`, explicit positive run weights are split equally over
that run's segments. `modern_coherence` accepts weights keyed by every accepted
run's ISO UTC start timestamp; the numerical primitive uses stable run IDs.
Weights must name exactly the accepted runs. Other weighting settings raise.
Normalize segment weights by their sum, pool Sxx, Syy and complex Sxy first,
then compute C2=abs(Sxy/sqrt(Sxx)/sqrt(Syy))**2. This differs from averaging run
coherences, notably when phases cancel. PSD units are squared input units/Hz;
CSD units are the input-unit product/Hz. Coherence is dimensionless. CSD is
conj(X)*Y and phase is arg(Sxy) in radians: delay of the second signal is negative;
swapping signals conjugates Sxy and reverses phase. Zero cross power has undefined
phase. Constant contributing runs conservatively make the pooled band undefined;
zero/nonfinite auto power is undefined, never zero coherence or perfect certainty.

`effective_averages(run_segment_weights, window=..., hop=...)` implements

    n_eff = (sum a)^2 / (sum a_i^2 + 2 sum_{i<j, same run} a_i*a_j*rho_ij^2)

where rho is the taper dot product at the segment offset divided by taper energy.
No overlap term crosses runs. The support records raw K, n_eff, approximate DOF
2*n_eff, normalized segment weights and the weighted overlap term. This reuses
the PSD overlap calculation. The assumption is independent stationary Gaussian
white-noise channels and independent stationary runs; overlapping taper counts
are approximate and are not independent ecological replicates.

For n_eff>1, `coherence_inference` saves the untruncated value
(n_eff*C2-1)/(n_eff-1) and its named positive part max(0,value). For alpha in (0,1),
raw threshold is 1-alpha**(1/(n_eff-1)); corrected threshold applies the same
positive-part transform. Decisions always use raw C2 strictly greater than the
raw threshold. At high alpha the positive part is non-injective at zero, so a
comparison on the corrected scale need not give the same decision. Raw values
are clipped only for roundoff <=1e-10; material range failures raise.

Only `null="gaussian_white"` supplies `inference_status="approximate_white_noise"`.
`colored` and `unspecified` return `unavailable` with None thresholds and decisions
and a reason; descriptive raw coherence and the named algebraic correction may
remain available, without calibrated bias or false-alarm claims for those nulls.
n_eff<=1 also returns None correction/threshold/decision and refuses single-
periodogram certainty. Correction/inference uses only physical segment Fourier
bins; intermediate zero-padded display bins have missing correction and decisions.
DC and Nyquist are excluded from correction/inference and band summaries; their
saved raw spectra remain inspectable. Nonfinite bins have
None decisions. No positive decision is inferred from a missing value.

At least three physical interior frequencies in the requested short/long period
band are required; zero padding does not increase that count. `band_mean` averages
raw coherence on the declared FFT grid within the band. `significant_fraction`
is the unadjusted fraction of physical segment Fourier bins in that band exceeding the raw threshold, only when
the band and inference are available. `multiplicity="none_pointwise_only"` means
neither familywise nor FDR significance across bins, pairs or windows.

The owned `ResultRecord` keeps source/variables/units, exact eligible and actual
contributing support/burden, raw/residual and training references, requested and
resolved settings, backend/version, run spectra, pool identity and diagnostics.
Configuration identity binds the complete observation policy, including the
minimum-point gate even when it leaves the underlying numerical pool unchanged.
Its compute statuses are `ok`, `insufficient_support`, `insufficient_band_support`,
`undefined_spectrum`, `unknown_provenance`, `unresolved_weights` (a weighted
observed-only counterpart needs replacement weights), and `unavailable_backend`.
Unavailable values are None with reasons; diagnostic support remains. Metric
status is `exploratory_coherence`; use status is `sensitivity_only`,
`insufficient_support` or `unknown_provenance`. A failing support minimum cannot
erase unknown contributing or training provenance. Shared donors require an
owned observed-only counterpart, or its reasoned unavailability when explicit
weights no longer describe that support.

`route="historical_modern"` returns `unavailable_backend` with its distinct
`coherence.historical_modern_unavailable` identity; the missing historical helper is
not silently replaced. Global policy recipes retain unresolved definitions;
callers must supply the independent explicit specification. Field validity and
reuse rights remain unverified.

`examples/modern_coherence.py` exercises the installed synthetic API. The tests
use the predeclared seeds 73005/73006 (128 white-noise replicates each) and 73007
(16 coupled controls), print actual counts/rates, and retain their fixed bands.
This bounded synthetic calibration does not establish ecological validation.

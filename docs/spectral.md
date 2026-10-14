# Regular Welch spectra

For actual irregular instants or masked regular records, the explicit
[projection-density LSFT-WOSA API](irregular.md) supplies a separately named
estimator with frequency-specific support and overlap moments.
[Sensitivity comparisons](sensitivity.md) execute both on declared common fit
frequencies, preserving baseline disagreement, deletion and completion contrasts.

```python
from tezca.spectral import WelchSpec, power_spectrum
spectrum = power_spectrum(data, variable="x", spec=WelchSpec(512))
frequency, psd = spectrum.value["frequency_hz"], spectrum.value["psd"]
```

The input is `AnalysisSeries` or owned `DailyResidual`. Source declaration,
basis and fitted daily reference support/use gates survive. Result arrays,
settings and support are defensive copies; `to_frame()` gives one summary row.
Null, short and irregular input return missing values with explicit reasons.
Default observations are observed-only; supplied/reconstructed/unknown inputs
require explicit `ObservationPolicy` and retain their exact contributing burden.

Every SciPy option is set explicitly following the
[SciPy 1.17 Welch contract](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.welch.html).
The initial implementation supports real one-sided spectra, mean averaging,
boxcar/Hann windows with explicit periodic/symmetric convention, no/constant/
linear detrending, density (`unit^2/Hz`) or spectrum (`unit^2`) scaling, integer
length/overlap/FFT length and all-runs/earliest-longest-run selection. Other
choices reject. Defaults are periodic Hann, constant detrend, density, half
overlap, FFT equal to segment length, and all runs. Sampling rate comes from
the declared cadence; frequency units are Hz. All requested and resolved
settings, normalization/window sums and backend version appear in the record.

Research requires explicit `nperseg`. `WelchSpec(recipe="sensor_length")`
instead uses floor(selected eligible N/8), bounded by declared 64 and 4096.
Those bounds/divisor can be changed only on that route. It independently
implements the recorded length recipe and claims no pinned sensor parity.
Oversized requests abstain; SciPy never shortens the segment. `min_segments`
is a separate explicit raw count gate. Automatic research resolution selection
remains unresolved in the policy; no compatibility length fallback is used.

No FFT crosses omitted/excluded cells. Each regular run uses identical settings;
its mean PSD receives its actual segment count as weight. Raw segment positions,
candidate/selected/used/rejected runs and unused trailing cells remain inspectable.
Burden is on unique contributing sample positions. Coverage and longest supported
run durations exclude gaps; the cycle diagnostic is frequency times that longest
used duration. This diagnostic does not count independent environmental cycles.

Physical resolution is fs/nperseg; display spacing is fs/nfft. The saved unpadded
positive Fourier grid counts physical ordinates. Padding adds display samples
without resolving new frequencies. The theoretical Nyquist (fs/2), available
display grid range and unpadded physical grid range are recorded separately;
an odd FFT length has no returned Nyquist ordinate. The supported positive
interval begins at the physical fundamental and ends at the lesser of the
physical and displayed grid maxima. Padding cannot expand physical support.
An empty interval or no contributing segment records no supported interval.
Frequency fitting and peak/band summaries are implemented in
[spectral summaries](spectral_summary.md). [Harmonic diagnostics](harmonic.md)
are available through an explicit raw rectangular-periodogram route; they refuse
these Welch spectra rather than assigning classical independent-bin thresholds.

The approximate effective-average count is K²/[K+2Σc_ij²], where c_ij is the
normalized inner product of overlapping taper windows within the same run.
Cross-run and nonoverlapping segment pairs have zero penalty. It assumes equal
segment weights and Gaussian white noise; this is neither independent field
replication nor a colored-noise significance calibration. Actual K and the
versioned approximation remain separate. Constants retain a named flag; a
detrended zero spectrum remains zero with relative summaries still unavailable.

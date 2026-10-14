# Pair audit and report-style coherence

```python
from tezca.pair_audit import PairLineage, signal_identity, audit_pair
from tezca.coherence import ReportCoherenceSpec, report_coherence

lineage = PairLineage({
    "a": dict(signal_id=signal_identity(first, "level"), kind="measurement", parents=[], donors=[]),
    "b": dict(signal_id=signal_identity(second, "temperature"), kind="measurement", parents=[], donors=[]),
})
spec = ReportCoherenceSpec(basis="raw", period_band=("4h", "64h"),
                           taper="hann", detrend="constant", min_nperseg=64, max_nperseg=256)
result = report_coherence(first, second, variable_a="level", variable_b="temperature",
                          spec=spec, lineage=lineage, nodes=["a", "b"], target="a")
```

Inputs are `AnalysisSeries` or `DailyResidual`; raw/residual basis must match the
explicit spec. Clock cadence and phase must agree. Exact common aware timestamps
are intersected; missing timestamps and ineligible values split contiguous runs.
No resampling, gap compression or interpolation of observations occurs. The default
observation policy is observed-only; completion needs an explicit policy.

PairLineage is a caller-declared graph, bound to the actual selected source/value/
variable/basis/reference identities by `signal_identity`. It is not inferred from
numeric correlation. Node kinds are `measurement`, `copy`, `deterministic`,
`reconstruction` and `unknown`. Measurement roots have no parents, copies have
one, deterministic transforms have at least one; parents name transitive graph
nodes. `donors` explicitly lists reconstruction-source IDs (empty means none;
missing means unknown). Ancestor donors are retained. Missing ancestors and unknown
nodes produce `unknown_lineage`; cycles, malformed graphs and wrong input bindings
raise. Equal nonmissing signal identities and declared copy links are resolved as
equivalent aliases before transitive traversal, including intermediate aliases.
Every alias contributes its ancestry, donor and unknown declarations. Graph
descriptions are independently owned and identified.

`audit_pair` records ordered sources/representations, references, transitive ancestry,
deterministic ancestry, unknown ancestry and shared donors. Duplicate identities,
direct/indirect deterministic pair transforms and target descendants cannot enter
target/environment rankings. Audit reasons are `duplicate_signal`,
`copy_or_deterministic_transform`, `target_descendant`, `unknown_lineage` and
`shared_reconstruction_donors`. A legitimate tightly coupled measurement pair
remains eligible for descriptive comparison; that eligibility grants no causal or
primary ecological claim. Unknown contributing provenance also prevents ranking.
Shared donors force sensitivity qualification and an owned observed-only counterpart
with its own support/result or reasoned unavailability. Exact source details remain
in their originating owned inputs; summaries carry compact source/digest bindings.

The independent method is `coherence.report_equal_run_v1`. Declare boxcar/Hann
taper, periodic or symmetric convention, none/constant/linear detrending and a
short/long period band in duration units. Segments overlap by exactly half their
even length. Adaptive resolution chooses the largest even L in the declared bounds
with `K=1+floor((N-L)/(L/2)) >= min_segments`; min_segments must be >=4. Fixed
`nperseg` overrides adaptive bounds. N=256 and bounds 64..256 yield L=102,K=4;
fixed L=64 yields K=7. N=159 cannot supply four segments with minimum L=64.
Short runs remain excluded records. Counts are raw overlapping segments, not
independent ecological replicates or calibrated significance evidence.

Each accepted run saves its actual eligible and contributing positions, unused tail,
length, hop, segment starts/count, sampling rate, physical spacing, taper, detrend,
frequency grid, density-normalized Sxx/Syy, real/imaginary Sxy and raw magnitude-squared
coherence. CSD uses `conj(X)*Y`; one-sided density scaling doubles interior FFT bins.
PSD units are squared input units per Hz; CSD uses the product of input units per Hz;
coherence is dimensionless. Constancy is checked only on actual segment contributors;
unused tail values cannot affect this qualification. Constant or zero-auto-power bins are undefined,
not zero or perfect coherence. Numerical coherence overshoot <=1e-10 is treated as
roundoff; material overshoot raises. Burden counts actual segment-contributing cells,
including nonlinear touched-cell gates, rather than unused tails or whole input.

`aggregate_report_runs` sorts stable run IDs, intersects all interior positive
frequency supports with the requested band, and forms a zero-anchored grid at the
coarsest physical spacing. Linear interpolation of run coherences stays within
that intersection; input order never chooses the grid. DC and Nyquist are excluded
from the summary but remain in saved spectra. Equal-run weights average coherence
at each ordinate; `band_mean` is the equal-ordinate mean. At least three common
physical ordinates are required. Any undefined common-band ordinate makes the band
summary unavailable. Originals, common range/grid, run IDs and weights remain in
support. This is an average of run coherences, not pooled cross/auto spectra.

The `ResultRecord` has owned values/support/configuration/reference records and
independent statuses. Compute vocabulary: `ok`, `insufficient_support` (no accepted
run or policy minimum not met), `insufficient_band_support` (fewer than three common
interior ordinates), `undefined_spectrum` (constant/zero/nonfinite power),
`unknown_provenance`, and `unresolved_compatibility`. Unavailable values are `None`
with reasons; their audit/run/band records remain inspectable. Metric status is
`exploratory_coherence`; use status is `sensitivity_only`, `unknown_provenance` or
`insufficient_support`. Invalid configurations and clock/basis mismatches raise.
Daily-reference IDs and stricter training qualifications are preserved.
When policy support is insufficient and contributing or training provenance is
unknown, compute status stays `insufficient_support`, use status stays
`unknown_provenance`, and ranking remains excluded.

`route="exact_report_compatibility"` returns `unresolved_compatibility` under
`coherence.report_exact_unresolved`. Q001 must recover the inputs, helper versions
and historical interpolation behavior; passing synthetic witnesses establishes
only the independently specified method. Global policy coherence recipes likewise
retain unresolved settings in `AnalysisPolicy.method_spec()`. Call this separate
API with the explicit specification and lineage; supplying them does not resolve
the named policy or automatically substitute another backend.

`08_pkg/examples/report_coherence.py` is a bounded installed synthetic example.
[Modern pooling/correction/phase](coherence_modern.md) has its own explicit method.
[Graph-state roughness](graph_state.md) has separate frozen representation/geometry.
Q002/Q003 still gate real F01 admission and rights.

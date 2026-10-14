# Univariate density Fisher

`tezca.density_fisher` provides two distinct estimators of marginal distribution
sharpness. Derivatives are with respect to the measurement coordinate, not time.
Neither measures temporal predictability or ecological health. Source definitions
are proposal 003 section 4 and 008 section 7.3; the location-family interpretation
follows [Ly et al., A Tutorial on Fisher Information](https://arxiv.org/abs/1705.01064).
No external implementation is copied and exact historical parity remains unverified.

```python
from tezca.density_fisher import (
    HistogramSpec, KDESpec, fit_density_reference, density_fisher,
)

histogram = density_fisher(data, variable="x", spec=HistogramSpec([0, 1, 3]))
spec = KDESpec(bandwidth=0.3, bounds=[-16, 16])
reference = fit_density_reference(training, variable="x", spec=spec,
                                  training_positions=training_positions)
smooth = density_fisher(evaluation, variable="x", reference=reference)
```

The API selects one variable explicitly and never flattens a multivariate frame.
It accepts owned `AnalysisSeries` or daily residuals. Default observations are
observed-only; provided/completed values need an explicit observation policy.
Point weighting is equal across eligible samples, on actual regular or irregular
support; it does not invent time weights or fill gaps. Method status is
`density_fisher_approximation` and use is sensitivity-only; unknown evaluation or
training provenance stays unknown. Burden, source, basis, daily transform and
density-reference support remain independently inspectable. The evaluation
support saves its own observation policy in `support["observations"]`, distinct
from the fitted reference's training policy.

Supply exactly one spec or reference. A spec fits retrospectively with
`fit_scope="self"`; a reference freezes training geometry, population scaling and
bandwidth. References contain no training density to reuse at evaluation. Counts
or the Gaussian mixture are recomputed from current eligible evaluation samples.
Held-out values do not enter reference identity. Variable, units and basis must
match; getters return defensive copies, including arrays and nested support.

Coordinates default to raw. Explicit `coordinates="standardized"` uses
`z=(x-center)/scale`, with population SD (ddof=0) fitted only on the declared
training support. Edges, bandwidth and numerical bounds are specified in these
analysis coordinates. Results separately report `J_coordinate` and
`J_raw=J_coordinate/scale**2`, with coordinate and raw inverse-square units.
Self-standardized shape has a separate question label from frozen-reference
coordinates and raw-amplitude comparisons. There is no nonlinear conversion.

Both specs require continuous measurements. `measurement="rounded_continuous"`
explicitly interprets tied values as rounded measurements to which continuous
density smoothing is applied; otherwise ties raise. Discrete/mixed declarations
are rejected. Constant samples yield `degenerate_input`, and null/too-short
samples yield `insufficient_support`. These limitations remain distinct from a
valid zero interior energy. Numerical/configuration errors are not hidden by
broad exception handling.

## Histogram interior approximation

`HistogramSpec` has method identity `density_fisher.histogram_interior_v1`.
For masses p, widths w and centers c, heights are `d=p/w` and the result is
`4*sum(diff(sqrt(d))**2/diff(c))`. `histogram_interior(p, edges)` exposes this
independent array primitive. This is the derivative energy of a linear
interpolant of square-root heights on the interior center intervals. Boundary
contributions are omitted; the interpolant's normalization is not assumed.
An equal-height bounded histogram can therefore give zero interior energy
without establishing zero full location Fisher.

Choose explicit edges, `binning="equal_width", bins=N`, or
`binning="quantile", bins=N, ties="raise"|"merge"`. Quantiles use linear
interpolation. Explicit edges conflict with automatic counts/ties. Merging
records original/final edges and reduced bin count, reconnecting consecutive
remaining centers; zero-width cells are never used. The final edge is included.
The only boundary mode is `interior`; unsupported corrections reject.

Counts, smoothed masses, heights, centers, widths, occupied bins and integration
interval are returned. Pseudocount is dimensionless counts per bin, default zero.
`outside="raise"` reports below/above positions; `"exclude"` retains excluded
mass and the conditional in-range normalization, positions and burden. Too few
in-range points give `outside_reference`. No clipping or winsorization occurs.

`legacy_density_v16()` returns the independently specified historical choices:
self-standardization, 20 quantile bins, explicit tied-edge merging, zero
pseudocount, minimum 100 samples and interior energy. Its recipe identity says
independent and parity unverified; it requires positive widths and does not
reproduce the historical silent width filter. General policy declarations stay
unresolved in `AnalysisPolicy.method_spec()`. Call the density-Fisher API with
an explicit method spec; that separate call does not resolve the recipe.

## Gaussian KDE location functional

`KDESpec` has method identity `density_fisher.gaussian_kde_location_v1` and
targets `J=integral(p_prime**2/p) dx`. It uses an equal-weight Gaussian mixture
with an analytic derivative. `bandwidth` is a positive absolute analysis-coordinate
scale, or the explicit Scott rule `sample_SD(ddof=1)*N**(-1/5)` fitted on training
values. This is distinguished from SciPy's covariance-factor convention in its
[Gaussian KDE documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.gaussian_kde.html).
The physical kernel support is the real line with no boundary correction.
Bounded/reflected/periodic models are unsupported and rejected explicitly.

`gaussian_kde_geometry(samples, points, bandwidth=...)` returns actual density,
analytic derivative and Fisher integrand at each supplied point. The implementation
uses scaled Gaussian weights to calculate the score and `p*score**2`, avoiding
a density floor or division of two underflowed values. Tails may underflow at
machine precision; no epsilon, trimming renormalization or probability clipping
is introduced.

`bounds` are a finite numerical interval, distinct from physical kernel support.
Composite Simpson integration uses the requested odd `points` (default 1025)
and a doubled grid (2049). Returned arrays are on the finer grid. The difference
divided by 15 estimates quadrature error; it must meet `atol+rtol*abs(J)`.
The coarse grid spacing must also be at most bandwidth/4 to resolve kernels.
Otherwise computation is `insufficient_numerical_resolution`, with diagnostics.
Defaults are atol=1e-8 and rtol=1e-6, in coordinate Fisher units and relative
units respectively; this is numerical checking, not statistical uncertainty.

`integration="finite_partial"` reports only the integral on the numerical
interval. It retains the original normalized mixture, exact Gaussian-CDF mass
inside/outside, numerical quadrature mass and no full-domain claim. The fixed
N(0,4) quantile witness on [-2,2] has mass about 0.68, not one.

The default `integration="full_bounded_tails"` qualifies a full-domain estimate
by bounding the omitted Fisher contribution. For a Gaussian component with
standardized bounds a and b, its omitted derivative energy is
`(Phi(a)-a*phi(a)+Phi(-b)+b*phi(b))/h**2`. Cauchy–Schwarz bounds the mixture's
pointwise Fisher integrand by the average component integrands, so averaging
these omitted component energies is an upper bound for the omitted mixture
energy. The bound must be <= `tail_tolerance` (default 1e-8, coordinate Fisher
units). Otherwise a full-domain request returns `insufficient_integration_domain`
with no value. Nothing expands the frozen geometry or silently switches routes.
Successful results are labeled `full_domain_estimate_with_tail_bound`, retain
that bound and the separate quadrature-error estimate, and do not claim an exact
infinite-domain integral. Finite partial results retain the bound as a diagnostic
without imposing the full-domain threshold. Tolerance for that threshold is
inapplicable on the partial route.

The fixed software controls use 4096 deterministic Gaussian midpoint quantiles,
absolute h=0.3 and [-16,16]. They compare with J=1/4, an independently computed
finite-difference derivative, doubled quadrature, translation and affine laws.
They do not establish optimal bandwidth, population calibration or real-data
validity. The installed `individual_dynamics.py` example exercises both methods
and an unavailable constant case. Separate [harmonic diagnostics](harmonic.md)
are available on the admitted raw-periodogram/white-noise route.

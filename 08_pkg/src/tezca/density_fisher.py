"""Univariate histogram interior energy and Gaussian-KDE location Fisher."""

from numbers import Real

import numpy as np
import scipy
from scipy.integrate import simpson
from scipy.special import ndtr

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations, _training_signal
from .reference import FrozenReference, OutsideReferenceError, ReferenceSpec
from .support import positions_array, reconstruction_burden, select_support


HISTOGRAM = "density_fisher.histogram_interior_v1"
KDE = "density_fisher.gaussian_kde_location_v1"
METRIC = "density_fisher_approximation"


def _real_vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real one-dimensional array")
    return raw.astype(float)


def _common(coordinates, measurement, min_points):
    if coordinates not in {"raw", "standardized"}:
        raise ValueError("coordinates must be raw or standardized")
    if measurement not in {"continuous", "rounded_continuous"}:
        raise ValueError("density Fisher requires continuous or explicitly rounded_continuous measurements")
    if type(min_points) is not int or min_points < 2:
        raise ValueError("min_points must be an integer >=2")
    return dict(coordinates=coordinates, measurement=measurement, min_points=min_points,
                weighting="equal_points", scaling_ddof=0,
                smoothing_interpretation="continuous_density_of_rounded_measurements"
                                        if measurement == "rounded_continuous" else "continuous_density")


class HistogramSpec:
    def __init__(self, edges=None, *, binning="explicit", bins=None, ties=None,
                 pseudocount=0, outside="raise", coordinates="raw",
                 measurement="continuous", min_points=2, boundary="interior"):
        common = _common(coordinates, measurement, min_points)
        if binning not in {"explicit", "equal_width", "quantile"} or boundary != "interior":
            raise ValueError("histograms require explicit/equal_width/quantile bins and interior boundary")
        if binning == "explicit":
            if bins is not None or ties is not None:
                raise ValueError("explicit edges conflict with automatic bins/ties")
            edges = ReferenceSpec(edges, scale=False).record["edges"]
        else:
            if edges is not None or type(bins) is not int or bins < 2:
                raise ValueError("automatic partitions require bins >=2 and no explicit edges")
        if binning == "quantile":
            if ties is None:
                ties = "raise"
            if ties not in {"raise", "merge"}:
                raise ValueError("quantile ties must be raise or merge")
        elif ties is not None:
            raise ValueError("ties applies only to quantile binning")
        if outside not in {"raise", "exclude"}:
            raise ValueError("outside must be raise or exclude")
        if (isinstance(pseudocount, (bool, np.bool_)) or not isinstance(pseudocount, Real)
                or not np.isfinite(pseudocount) or pseudocount < 0):
            raise ValueError("pseudocount must be nonnegative finite counts per bin")
        self._record = dict(**common, method_id=HISTOGRAM, edges=edges, binning=binning,
            bins=bins, ties=ties, pseudocount=pseudocount, outside=outside, boundary=boundary,
            edge_coordinates="analysis", last_edge_included=True,
            derivative="adjacent_square_root_heights", integration="interior_center_intervals",
            boundary_contributions="omitted", estimator_status="approximation")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class KDESpec:
    def __init__(self, *, bandwidth, bounds, integration="full_bounded_tails", points=1025,
                 atol=1e-8, rtol=1e-6, tail_tolerance=1e-8, coordinates="raw",
                 measurement="continuous", min_points=2, boundary="none", physical_support="real_line"):
        common = _common(coordinates, measurement, min_points)
        if not (isinstance(bandwidth, str) and bandwidth == "scott"):
            if (isinstance(bandwidth, (bool, np.bool_)) or not isinstance(bandwidth, Real)
                    or not np.isfinite(bandwidth) or bandwidth <= 0):
                raise ValueError("bandwidth must be a positive absolute analysis scale or scott")
            bandwidth = float(bandwidth)
        bounds = _real_vector(bounds, "bounds")
        if len(bounds) != 2 or bounds[1] <= bounds[0]:
            raise ValueError("bounds must be an increasing finite pair in analysis coordinates")
        if integration not in {"full_bounded_tails", "finite_partial"}:
            raise ValueError("integration must be full_bounded_tails or finite_partial")
        if type(points) is not int or points < 33 or points % 2 != 1:
            raise ValueError("points must be an odd integer >=33 for composite Simpson quadrature")
        for name, value in (("atol", atol), ("rtol", rtol), ("tail_tolerance", tail_tolerance)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if integration == "finite_partial" and tail_tolerance != 1e-8:
            raise ValueError("tail_tolerance applies only to full_bounded_tails")
        if boundary != "none" or physical_support != "real_line":
            raise ValueError("Gaussian KDE supports only the real line with no boundary correction")
        self._record = dict(**common, method_id=KDE, bandwidth=bandwidth, bounds=bounds,
            integration=integration, points=points, atol=atol, rtol=rtol,
            tail_tolerance=tail_tolerance if integration == "full_bounded_tails" else None,
            boundary=boundary, physical_support=physical_support, kernel="gaussian",
            derivative="analytic_gaussian_mixture_score", quadrature="composite_Simpson_with_doubled_grid",
            bandwidth_coordinates="analysis_absolute", bounds_coordinates="analysis",
            estimator_status="smoothed_density_approximation", outside="all_real_samples_retained")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def legacy_density_v16():
    """Independently specified historical choices; exact backend parity unverified."""
    result = HistogramSpec(binning="quantile", bins=20, ties="merge", coordinates="standardized",
                           measurement="rounded_continuous", min_points=100)
    result._record.update(recipe="legacy_density_v16_independent", parity="unverified",
                          width_filter="none_positive_widths_required")
    return result


def histogram_interior(probabilities, edges):
    p = _real_vector(probabilities, "probabilities")
    edges = ReferenceSpec(edges, scale=False).record["edges"]
    if len(p) != len(edges)-1 or (p < 0).any() or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("probabilities must be normalized masses matching the edges")
    widths = np.diff(edges)
    centers = edges[:-1]+widths/2
    if not np.isfinite(widths).all() or not np.isfinite(centers).all():
        raise ValueError("histogram widths and centers must be representable and finite")
    heights = p/widths
    energy = float(4*np.sum(np.diff(np.sqrt(heights))**2/np.diff(centers)))
    if not np.isfinite(energy) or not np.isfinite(heights).all():
        raise ValueError("histogram geometry/energy is not representable")
    return dict(J_coordinate=energy, widths=widths, centers=centers, heights=heights,
                integration_bounds=[float(centers[0]), float(centers[-1])],
                boundary_contributions="omitted", full_domain_claim=False)


def gaussian_kde_geometry(samples, points, *, bandwidth):
    """Equal-weight Gaussian mixture density, analytic derivative and Fisher integrand."""
    samples, points = _real_vector(samples, "samples"), _real_vector(points, "points")
    if (not len(samples) or isinstance(bandwidth, (bool, np.bool_)) or not isinstance(bandwidth, Real)
            or not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("KDE requires samples and positive finite absolute bandwidth")
    density, derivative, integrand = [], [], []
    for start in range(0, len(points), 128):
        z = (points[start:start+128, None]-samples[None, :])/bandwidth
        log_kernel = -.5*z*z
        if not np.isfinite(log_kernel).all():
            raise ValueError("KDE distances cannot be represented")
        maximum = log_kernel.max(axis=1)
        weights = np.exp(log_kernel-maximum[:, None])
        total = weights.sum(axis=1)
        p = np.exp(maximum+np.log(total)-np.log(len(samples))-np.log(bandwidth)-.5*np.log(2*np.pi))
        score = -np.sum(weights*z, axis=1)/(total*bandwidth)
        density.extend(p)
        derivative.extend(p*score)
        integrand.extend(p*score**2)
    arrays = dict(points=points, density=np.asarray(density), derivative=np.asarray(derivative),
                  fisher_integrand=np.asarray(integrand))
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("KDE density or derivative cannot be represented")
    return arrays


def _kde_integral(samples, bandwidth, settings):
    lo, hi = settings["bounds"]
    n = 2*settings["points"]-1
    grid = np.linspace(lo, hi, n)
    geometry = gaussian_kde_geometry(samples, grid, bandwidth=bandwidth)
    integrand = geometry["fisher_integrand"]
    fine = float(simpson(integrand, x=grid))
    coarse = float(simpson(integrand[::2], x=grid[::2]))
    error = abs(fine-coarse)/15
    left, right = (lo-samples)/bandwidth, (hi-samples)/bandwidth
    phi_left = np.exp(-left**2/2)/np.sqrt(2*np.pi)
    phi_right = np.exp(-right**2/2)/np.sqrt(2*np.pi)
    left_mass, right_mass = float(ndtr(left).mean()), float(ndtr(-right).mean())
    tail_bound = float(np.mean(ndtr(left)-left*phi_left+ndtr(-right)+right*phi_right)/bandwidth**2)
    partial = settings["integration"] == "finite_partial"
    status, reason = "ok", None
    # Resolve each kernel before trusting a refinement difference; no narrow-peak aliasing.
    if (hi-lo)/(settings["points"]-1) > bandwidth/4 or error > settings["atol"]+settings["rtol"]*abs(fine):
        status, reason = "insufficient_numerical_resolution", "grid spacing or Simpson refinement fails requested tolerance"
    elif not partial and tail_bound > settings["tail_tolerance"]:
        status, reason = "insufficient_integration_domain", "omitted Fisher tail upper bound exceeds requested tolerance"
    record = dict(**geometry, bandwidth=bandwidth, quadrature_points=n,
        requested_points=settings["points"], J_coordinate=fine,
        integral_kind="finite_partial" if partial else "full_domain_estimate_with_tail_bound",
        integrated_mass=1-left_mass-right_mass, quadrature_mass=float(simpson(geometry["density"], x=grid)),
        tail_mass=left_mass+right_mass, left_tail_mass=left_mass, right_tail_mass=right_mass,
        omitted_fisher_upper_bound=tail_bound, quadrature_error_estimate=error,
        tail_bound_method="Gaussian_mixture_convexity_component_second_moment_bound",
        renormalized=False, integration_bounds=[float(lo), float(hi)],
        full_domain_claim=not partial and status == "ok", numerical_status=status, numerical_reason=reason)
    return record


class DensityReference(FrozenReference):
    pass


def _sample_status(x, settings, minimum):
    if len(x) < minimum:
        return "insufficient_support", "too few eligible samples"
    if np.ptp(x) == 0:
        return "degenerate_input", "constant samples cannot identify a continuous density"
    if settings["measurement"] == "continuous" and len(np.unique(x)) != len(x):
        raise ValueError("tied samples require explicit rounded_continuous smoothing interpretation")
    return "ok", None


def fit_density_reference(data, *, variable, spec, training_positions, observations=None, fit_scope="training_only"):
    if not isinstance(spec, (HistogramSpec, KDESpec)) or fit_scope not in {"training_only", "self"}:
        raise ValueError("density fitting requires a method-specific spec and training_only/self scope")
    original = _ColumnSignal(data, variable)
    positions = positions_array(training_positions, len(original.values))
    signal = _training_signal(data, variable, positions)
    observations = _observations(observations)
    selection = select_support(signal, observations=observations, estimator="point",
                               policy="all_runs" if signal.time.mode == "regular" else "irregular")
    used = selection.positions
    x = signal.values.iloc[used, 0].to_numpy(dtype=float, na_value=np.nan)
    settings = spec.record
    status, reason = _sample_status(x, settings, max(settings["min_points"], observations.min_points))
    center, scale = 0., 1.
    edges, bandwidth = None, None
    partition = None
    if status == "ok":
        if settings["coordinates"] == "standardized":
            center, scale = float(x.mean()), float(x.std(ddof=0))
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            status, reason = "degenerate_input", "unrepresentable fitted scaling"
        else:
            z = (x-center)/scale
            if not np.isfinite(z).all():
                status, reason = "degenerate_input", "unrepresentable analysis coordinates"
            elif settings["method_id"] == HISTOGRAM:
                if settings["binning"] == "explicit":
                    original_edges = settings["edges"]
                elif settings["binning"] == "equal_width":
                    original_edges = np.linspace(z.min(), z.max(), settings["bins"]+1)
                else:
                    original_edges = np.quantile(z, np.linspace(0, 1, settings["bins"]+1))
                edges = original_edges.copy()
                if settings["binning"] == "quantile" and np.any(np.diff(edges) == 0):
                    if settings["ties"] == "raise":
                        raise ValueError("quantile edges tie; explicitly request ties=merge")
                    edges = np.unique(edges)
                edges = ReferenceSpec(edges, scale=False).record["edges"]
                partition = dict(original_edges=original_edges, actual_bins=len(edges)-1,
                    merged_edges=len(original_edges)-len(edges), adjacency="consecutive_remaining_centers")
            else:
                bandwidth = (float(z.std(ddof=1)*len(z)**(-1/5)) if settings["bandwidth"] == "scott"
                             else float(settings["bandwidth"]))
                if not np.isfinite(bandwidth) or bandwidth <= 0:
                    status, reason = "degenerate_input", "unrepresentable fitted bandwidth"
    return DensityReference(dict(method_id=settings["method_id"], spec=settings, spec_id=spec.identity,
        variable=variable, unit=signal.metadata[variable]["unit"], basis=signal.basis,
        center=center, scale=scale, scaling_ddof=0, edges=edges, partition=partition,
        bandwidth=bandwidth, geometry_id=identity(dict(edges=edges, bandwidth=bandwidth,
            center=center, scale=scale, bounds=settings.get("bounds"))),
        training_positions=positions.tolist(), eligible_training_positions=positions[used].tolist(),
        support=signal.support_record(selection), observations=observations.__dict__,
        use_status=signal.use(used, observations, METRIC), fit_scope=fit_scope,
        semantics="retrospective" if fit_scope == "self" else "training_only",
        compute_status=status, reason=reason))


def density_fisher(data, *, variable, spec=None, reference=None, observations=None):
    if (spec is None) == (reference is None):
        raise ValueError("supply exactly one of spec or reference")
    signal = _ColumnSignal(data, variable)
    observations = _observations(observations)
    if spec is not None:
        reference = fit_density_reference(data, variable=variable, spec=spec, observations=observations,
            training_positions=np.arange(len(signal.values)), fit_scope="self")
    if not isinstance(reference, DensityReference):
        raise ValueError("reference must be DensityReference")
    fit = reference.describe()
    if (variable != fit["variable"] or signal.metadata[variable]["unit"] != fit["unit"] or signal.basis != fit["basis"]):
        raise ValueError("density reference variable, units and basis must match")
    settings = fit["spec"]
    selection = select_support(signal, observations=observations, estimator="point",
                               policy="all_runs" if signal.time.mode == "regular" else "irregular")
    positions = selection.positions
    x = signal.values.iloc[positions, 0].to_numpy(dtype=float, na_value=np.nan)
    status, reason = fit["compute_status"], fit["reason"]
    if status == "ok":
        status, reason = _sample_status(x, settings, max(settings["min_points"], observations.min_points))
    support = signal.support_record(selection)
    support.update(fit=fit, requested=settings, resolved=dict(settings, bandwidth=fit["bandwidth"], edges=fit["edges"]),
                   geometry_id=fit["geometry_id"], measurement_unit=fit["unit"],
                   backend=dict(numpy=np.__version__, scipy=scipy.__version__), observations=owned(observations.__dict__),
                   evaluation_positions=positions.tolist(), evaluation_burden=reconstruction_burden(signal, positions))
    value = None
    if status == "ok":
        z = (x-fit["center"])/fit["scale"]
        if not np.isfinite(z).all():
            status, reason = "degenerate_input", "unrepresentable evaluation coordinates"
        elif settings["method_id"] == HISTOGRAM:
            edges = fit["edges"]
            outside = (z < edges[0]) | (z > edges[-1])
            diagnostics = dict(below=int(np.sum(z < edges[0])), above=int(np.sum(z > edges[-1])),
                               positions=positions[outside].tolist())
            if outside.any() and settings["outside"] == "raise":
                raise OutsideReferenceError({variable: diagnostics})
            support.update(outside=diagnostics, excluded_mass=float(outside.mean()),
                normalization="conditional_on_in_range" if outside.any() else "all_eligible",
                evaluation_positions=positions[~outside].tolist(),
                evaluation_burden=reconstruction_burden(signal, positions[~outside]))
            positions, z = positions[~outside], z[~outside]
            status, reason = _sample_status(z, settings, max(settings["min_points"], observations.min_points))
            if status == "insufficient_support":
                status, reason = "outside_reference", "too few evaluation samples inside fitted edges"
            if status == "ok":
                counts = np.histogram(z, bins=edges)[0]
                smooth = counts.astype(float)+settings["pseudocount"]
                if not np.isfinite(smooth.sum()):
                    raise ValueError("pseudocount normalization is not representable")
                p = smooth/smooth.sum()
                value = dict(**histogram_interior(p, edges), counts=counts, probabilities=p,
                    edges=edges, occupied_bins=int(np.count_nonzero(counts)),
                    histogram_mass=1., integral_kind="histogram_interior", pseudocount=settings["pseudocount"],
                    tail_mass=None, tail_information="unidentified_omitted_boundaries")
        else:
            numerical = _kde_integral(z, fit["bandwidth"], settings)
            status, reason = numerical["numerical_status"], numerical["numerical_reason"]
            if status == "ok":
                value = numerical
            else:
                support["integration_diagnostics"] = numerical
    if value is not None:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            raw_J = float(np.divide(value["J_coordinate"], np.float64(fit["scale"])**2))
        value.update(J_raw=raw_J,
                     coordinate_units="1" if settings["coordinates"] == "standardized" else fit["unit"],
                     raw_units=f"1/{fit['unit']}^2", coordinate_J_units="1" if settings["coordinates"] == "standardized" else f"1/{fit['unit']}^2",
                     center=fit["center"], scale=fit["scale"], evaluation_count=len(positions),
                     question="raw_amplitude" if settings["coordinates"] == "raw" else
                              "self_standardized_shape" if fit["fit_scope"] == "self" else "fixed_reference_coordinates")
        if not np.isfinite(raw_J):
            status, reason = "degenerate_input", "raw-coordinate Fisher cannot be represented with the fitted scale"
            support["coordinate_diagnostics"] = value
            value = None
    use = signal.use(positions, observations, METRIC)
    if fit["use_status"] == "unknown_provenance":
        use = "unknown_provenance"
    return ResultRecord(value, compute_status=status, reason=reason, metric_status=METRIC, use_status=use,
        support=support, units=f"1/{fit['unit']}^2", method_id=settings["method_id"],
        config_id=identity(dict(spec=fit["spec_id"], observations=observations.__dict__)),
        reference_id=reference.identity, representation=signal.basis)

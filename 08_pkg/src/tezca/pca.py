"""Complete-vector PCA and frozen training representation projection."""

import numpy as np
import pandas as pd

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations, _training_signal
from .reference import FrozenReference
from .support import positions_array, select_support


class PCASpec:
    def __init__(self, *, features, basis, scaling, ddof, center=True):
        if isinstance(features, str):
            raise ValueError("features must be a sequence of names")
        features = list(features)
        if not features or len(set(features)) != len(features) or any(
                not isinstance(v, str) or not v.strip() for v in features):
            raise ValueError("features must be distinct explicit variable names")
        if basis not in {"raw", "daily_residual"} or scaling not in {"none", "zscore"}:
            raise ValueError("declare raw/daily_residual basis and none/zscore scaling")
        if type(ddof) is not int or ddof not in {0, 1}:
            raise ValueError("ddof must be 0 or 1")
        if center is not True:
            raise ValueError("covariance PCA requires explicit centering")
        self._record = dict(features=features, basis=basis, scaling=scaling, ddof=ddof,
                            center=True, support="complete_case", weighting="equal_vectors")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class PCAReference(FrozenReference):
    """Owned center, scales, full eigensystem and training support."""


def _stricter(statuses):
    order = ["unknown_provenance", "insufficient_support", "sensitivity_only",
             "unverified_gap_geometry", "primary_eligible"]
    return min(statuses, key=order.index)


def _support(data, settings, observations, within=None):
    features = settings["features"]
    signals = [_ColumnSignal(data, v) for v in features]
    if any(s.basis != settings["basis"] for s in signals):
        raise ValueError("selected feature basis must match the declared PCA basis")
    frame = data.values
    if [v for v in frame.columns if v in features] != features:
        raise ValueError("selected features must follow input column order")
    units = {s.variable: s.metadata[s.variable]["unit"] for s in signals}
    if settings["scaling"] == "none" and len(set(units.values())) > 1:
        raise ValueError("unscaled PCA requires compatible identical units")
    requested = np.arange(len(frame)) if within is None else positions_array(within, len(frame))
    eligible = np.logical_and.reduce([s.eligible(observations)[:, 0] for s in signals])
    positions = requested[eligible[requested]]
    records, statuses = {}, []
    for signal in signals:
        selected = select_support(signal, observations=observations, policy="complete_case",
                                  estimator="point", within=positions)
        records[signal.variable] = signal.support_record(selected)
        statuses.append(signal.use(positions, observations))
    record = dict(policy="complete_case", time_compression=False, features=features,
                  units=units, basis=settings["basis"], input_id=data.identity,
                  source=signals[0].source, clock=signals[0].time.record,
                  requested_positions=requested.tolist(), eligible_positions=positions.tolist(),
                  excluded_positions=np.setdiff1d(requested, positions).tolist(),
                  eligible_instants=[t.isoformat() for t in frame.index[positions]],
                  variables=records, observations=observations.__dict__,
                  use_status=_stricter(statuses))
    record["support_id"] = identity(record)
    return frame.loc[:, features].iloc[positions].to_numpy(dtype=float), positions, record


def fit_pca_reference(data, *, spec, training_positions=None, fit_scope="training_only", observations=None):
    if not isinstance(spec, PCASpec):
        raise ValueError("spec must be PCASpec")
    if fit_scope not in {"training_only", "self"}:
        raise ValueError("fit_scope must be training_only or self")
    if (fit_scope == "training_only") != (training_positions is not None):
        raise ValueError("training_only requires positions; self uses the evaluation input")
    observations = _observations(observations)
    settings = spec.record
    requested = (np.arange(len(data.values)) if training_positions is None else
                 positions_array(training_positions, len(data.values)))
    training = _training_signal(data, settings["features"][0], requested)._data
    values, positions, support = _support(training, settings, observations)
    support.update(position_scope="training_subset", original_positions=requested.tolist(),
                   original_eligible_positions=requested[positions].tolist())
    p, ddof = len(settings["features"]), settings["ddof"]
    status, reason, center, scales, result = "ok", None, None, None, None
    if len(values) < max(2, ddof + 1, observations.min_points):
        status, reason = "insufficient_support", "fewer than the required complete eligible vectors"
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            center = values.mean(axis=0)
            centered = values - center
            sd = centered.std(axis=0, ddof=ddof)
            scales = sd if settings["scaling"] == "zscore" else np.ones(p)
            covariance = None
            if np.isfinite(center).all() and np.isfinite(scales).all() and np.all(scales > 0):
                transformed = centered / scales
                covariance = transformed.T @ transformed / (len(values) - ddof)
        if not np.isfinite(scales).all():
            status, reason = "invalid_scale", "training scale exceeds finite precision"
        elif settings["scaling"] == "zscore" and np.any(sd == 0):
            status, reason = "zero_scale", "z-scaling has a constant selected feature"
        elif covariance is None or not np.isfinite(covariance).all():
            status, reason = "invalid_covariance", "covariance exceeds finite precision"
        else:
            eigenvalues, vectors = np.linalg.eigh(covariance)
            eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
            tolerance = np.finfo(float).eps * p * max(float(np.max(np.abs(eigenvalues))), 1e-300)
            if eigenvalues[-1] < -tolerance:
                status, reason = "invalid_covariance", "material negative covariance eigenvalue"
            elif eigenvalues[0] <= tolerance:
                status, reason = "degenerate_input", "zero total covariance"
            else:
                negative_count = int(np.sum(eigenvalues < 0))
                eigenvalues = np.maximum(eigenvalues, 0)
                relative = eigenvalues / eigenvalues[0]
                fractions = relative / relative.sum()
                # Ratio formulation avoids overflow in the equivalent squared-sum formula.
                result = dict(covariance=covariance, eigenvalues=eigenvalues, fractions=fractions,
                              leading_value=float(eigenvalues[0]), leading_fraction=float(fractions[0]),
                              pc90=int(np.searchsorted(np.cumsum(fractions) + 1e-12, .9) + 1),
                              participation_ratio=float(1 / np.sum(fractions ** 2)),
                              rank=int(np.sum(eigenvalues > tolerance)), rank_tolerance=tolerance,
                              rank_deficient=bool(np.sum(eigenvalues > tolerance) < p),
                              roundoff_negative_eigenvalues=negative_count,
                              constant_features=[v for v, s in zip(settings["features"], sd) if s == 0],
                              eigenvectors=vectors, scaled_loadings=vectors * np.sqrt(eigenvalues),
                              eigenvector_definition="unit columns, descending eigenvalue order",
                              scaled_loading_definition="eigenvectors * sqrt(eigenvalues)")
    score_unit = "1" if settings["scaling"] == "zscore" else next(iter(support["units"].values()))
    variance_unit = "1" if score_unit == "1" else f"({score_unit})^2"
    output_units = {name: variance_unit for name in ("covariance", "eigenvalues", "leading_value", "rank_tolerance")}
    output_units.update({name: score_unit for name in ("scores", "scaled_loadings")})
    output_units.update({name: "1" for name in ("fractions", "leading_fraction", "participation_ratio",
                                               "eigenvectors", "pc90", "rank", "roundoff_negative_eigenvalues")})
    record = dict(compute_status=status, reason=reason, spec=settings, spec_id=spec.identity,
                  requested=settings, resolved=settings, fit_scope=fit_scope,
                  semantics="retrospective" if fit_scope == "self" else "training_only",
                  support=support, center=center, scales=scales, eigensystem=result,
                  training_id=training.identity,
                  output_units=output_units,
                  method_id="pca.complete_covariance_v1", backend="numpy.linalg.eigh",
                  backend_version=np.__version__)
    return PCAReference(record)


def project_pca(data, *, reference, observations=None):
    """Project complete evaluation vectors without fitting evaluation parameters."""
    if not isinstance(reference, PCAReference):
        raise ValueError("reference must be PCAReference")
    frozen = reference.describe()
    settings = frozen["spec"]
    # Selected columns must occur in the same relative order; unrelated columns may remain.
    selected_order = [v for v in data.values.columns if v in settings["features"]]
    if selected_order != settings["features"]:
        raise ValueError("evaluation feature order must match frozen order")
    observations = _observations(observations)
    values, positions, support = _support(data, settings, observations)
    if support["units"] != frozen["support"]["units"]:
        raise ValueError("evaluation units must match frozen units")
    support.update(fit_scope=frozen["fit_scope"], training_id=frozen["training_id"],
                   training_support=frozen["support"], requested=settings, resolved=settings,
                   reference_id=reference.identity, backend=frozen["backend"],
                   backend_version=frozen["backend_version"], refitted=False)
    status, reason, value = frozen["compute_status"], frozen["reason"], None
    if status == "ok" and len(positions) < observations.min_points:
        status, reason = "insufficient_support", "too few complete evaluation vectors"
    if status == "ok":
        with np.errstate(over="ignore", invalid="ignore"):
            scores = ((values - frozen["center"]) / frozen["scales"]) @ frozen["eigensystem"]["eigenvectors"]
        if not np.isfinite(scores).all():
            status, reason = "invalid_projection", "scores exceed finite precision"
        else:
            value = owned(frozen["eigensystem"])
            value["scores"] = pd.DataFrame(scores, index=data.values.index[positions],
                                            columns=[f"PC{i + 1}" for i in range(len(settings["features"]))])
    use = _stricter([support["use_status"], frozen["support"]["use_status"]])
    if use == "primary_eligible" and frozen["fit_scope"] == "training_only" and any(
            v["selected"]["variables"][name]["touched_count"]
            for name, v in frozen["support"]["variables"].items()):
        use = "sensitivity_only"
    return ResultRecord(value, compute_status=status, reason=reason,
                        metric_status="primary_with_support", use_status=use, support=support,
                        units=frozen["output_units"],
                        method_id="pca.frozen_projection_v1", config_id=identity(dict(spec=settings, observations=observations.__dict__)),
                        reference_id=reference.identity, representation=settings["basis"])


def pca_summary(data, *, spec, observations=None):
    """Retrospective self-fit, distinctly identified from training projection."""
    reference = fit_pca_reference(data, spec=spec, observations=observations, fit_scope="self")
    return project_pca(data, reference=reference, observations=observations)

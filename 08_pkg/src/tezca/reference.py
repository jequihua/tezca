"""Frozen shared scaling and fixed-edge histogram references.

This useful representation primitive does not implement density Fisher, graph,
spectral or regime estimators. Histogram occupancy is recomputed at evaluation.
Source bindings snapshot AnalysisSeries.source; full producer details stay in
the originating series and are identified by its computed detail digest.
"""

import numpy as np
import pandas as pd

from ._owned import identity, owned
from .data import AnalysisSeries, ResultRecord
from .support import ObservationPolicy, positions_array, reconstruction_burden, select_support, use_status


class ReferenceSpec:
    def __init__(self, edges, *, scale=True, min_points=2, outside="raise", support="all_runs"):
        raw_edges = np.asarray(edges)
        if raw_edges.dtype.kind not in "iuf" or raw_edges.ndim != 1 or len(raw_edges) < 2:
            raise ValueError("edges must be a one-dimensional real numeric sequence")
        edges = raw_edges.astype(float)
        if not np.isfinite(edges).all() or np.any(edges[1:] <= edges[:-1]):
            raise ValueError("edges must be finite and strictly increasing")
        if type(scale) is not bool or type(min_points) is not int or min_points < 1:
            raise ValueError("scale must be boolean and min_points a positive integer")
        if outside not in {"raise", "exclude"}:
            raise ValueError("outside must explicitly be raise or exclude")
        if support not in {"longest_run", "all_runs", "complete_case", "irregular"}:
            raise ValueError("reference support must be a point-estimator-compatible selection")
        self._record = {"edges": edges.copy(), "scale": scale, "min_points": min_points,
                        "outside": outside, "support": support, "scaling_ddof": 0,
                        "edge_coordinates": "raw", "last_edge_included": True}

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class FrozenReference:
    def __init__(self, record):
        self._record = owned(record)
        self._id = identity(self._record)

    @property
    def identity(self):
        return self._id

    def describe(self):
        return owned(self._record)


class OutsideReferenceError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = owned(diagnostics)
        super().__init__("evaluation contains values outside frozen edges; request an explicit exclude policy")


def fit_reference(series, *, spec, observations=None, training_positions=None, fit_scope="training_only"):
    if not isinstance(series, AnalysisSeries) or not isinstance(spec, ReferenceSpec):
        raise ValueError("fit_reference requires AnalysisSeries and ReferenceSpec")
    if fit_scope not in {"training_only", "self"}:
        raise ValueError("fit_scope must be training_only or self")
    if fit_scope == "training_only" and training_positions is None:
        raise ValueError("training_only fitting requires explicitly selected training_positions")
    if fit_scope == "self" and training_positions is not None:
        raise ValueError("self fitting uses evaluation input and is explicitly retrospective")
    observations = ObservationPolicy("observed_only") if observations is None else observations
    settings = spec.record
    training = np.arange(len(series.values)) if training_positions is None else positions_array(training_positions, len(series.values))
    selection = select_support(series, observations=observations, policy=settings["support"],
                               estimator="point", within=training)
    positions = selection.positions
    frame = series.values
    values = frame.iloc[positions].to_numpy(dtype=float, na_value=np.nan)
    status, reason = "ok", None
    mean, std = None, None
    if len(positions) < max(settings["min_points"], observations.min_points):
        status, reason = "insufficient_support", "too few eligible training rows"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            status, reason = "degenerate_input", "training scale cannot be represented with finite precision"
        elif settings["scale"] and (std == 0).any():
            status, reason = "degenerate_input", "zero population scale on eligible training support"
    record = {"compute_status": status, "reason": reason, "fit_scope": fit_scope,
              "semantics": "training_only" if fit_scope == "training_only" else "retrospective",
              "source_id": series.identity, "source": series.source,
              "variables": list(frame.columns),
              "units": {v: series.metadata[v]["unit"] for v in frame.columns},
              "representations": {v: series.metadata[v]["representation"] for v in frame.columns},
              "training_positions": training.tolist(), "eligible_positions": positions.tolist(),
              "time_start": frame.index[positions[0]].isoformat() if len(positions) else None,
              "time_last": frame.index[positions[-1]].isoformat() if len(positions) else None,
              "time_range_closure": "both", "support": selection.record,
              "observations": observations.__dict__, "spec": settings, "spec_id": spec.identity,
              "scaling": {"mean": mean, "population_std": std, "ddof": 0,
                          "enabled": settings["scale"], "weighting": "equal_points"},
              "edges": settings["edges"].copy(), "edges_id": identity(settings["edges"])}
    return FrozenReference(record)


def apply_reference(series, *, reference=None, spec=None, observations=None):
    """Exactly one route: evaluate a frozen reference or retrospectively self-fit."""
    if (reference is None) == (spec is None):
        raise ValueError("supply exactly one of spec or reference")
    if not isinstance(series, AnalysisSeries):
        raise ValueError("evaluation requires an AnalysisSeries")
    observations = ObservationPolicy("observed_only") if observations is None else observations
    if spec is not None:
        reference = fit_reference(series, spec=spec, observations=observations, fit_scope="self")
    if not isinstance(reference, FrozenReference):
        raise ValueError("reference must be a FrozenReference")
    frozen = reference.describe()
    frame = series.values
    if list(frame.columns) != frozen["variables"]:
        raise ValueError("evaluation variable order must match the fitted reference")
    for variable in frame.columns:
        if series.metadata[variable]["unit"] != frozen["units"][variable]:
            raise ValueError("evaluation units must match the fitted reference")
        if series.metadata[variable]["representation"] != frozen["representations"][variable]:
            raise ValueError("evaluation representation must match the fitted reference")
    selection = select_support(series, observations=observations, policy=frozen["spec"]["support"], estimator="point")
    positions = selection.positions
    config_id = identity({"spec_id": frozen["spec_id"], "observations": observations.__dict__})
    support = selection.record
    support["fit_scope"] = frozen["fit_scope"]
    support["semantics"] = frozen["semantics"]
    status = frozen["compute_status"]
    reason = frozen["reason"]
    if status == "ok" and len(positions) < max(frozen["spec"]["min_points"], observations.min_points):
        status, reason = "insufficient_support", "too few eligible evaluation rows"
    common = dict(metric_status="exploratory_reference_summary", units="1",
                  method_id="fixed_edges_and_population_scaling_v1", config_id=config_id,
                  reference_id=reference.identity, representation="frozen_reference")
    use = use_status(series, positions, observations, metric_status=common["metric_status"])
    if status != "ok":
        return ResultRecord(None, compute_status=status, reason=reason, use_status=use, support=support, **common)
    values = frame.iloc[positions].to_numpy(dtype=float, na_value=np.nan)
    edges = frozen["edges"]
    outside = (values < edges[0]) | (values > edges[-1])
    diagnostics = {v: {"below": int(np.sum(values[:, j] < edges[0])),
                       "above": int(np.sum(values[:, j] > edges[-1])),
                       "positions": positions[outside[:, j]].tolist()}
                   for j, v in enumerate(frame.columns)}
    if outside.any() and frozen["spec"]["outside"] == "raise":
        raise OutsideReferenceError(diagnostics)
    inside_rows = ~outside.any(axis=1)
    used = positions[inside_rows]
    support.update(outside=diagnostics, outside_policy=frozen["spec"]["outside"],
                   evaluation_positions=used.tolist(), evaluation_denominator=len(used),
                   excluded_outside_positions=positions[~inside_rows].tolist())
    support["evaluation_burden"] = reconstruction_burden(series, used)
    use = use_status(series, used, observations, metric_status=common["metric_status"])
    if len(used) < max(frozen["spec"]["min_points"], observations.min_points):
        return ResultRecord(None, compute_status="outside_reference", reason="too few rows within fixed edges",
                            use_status=use, support=support, **common)
    accepted = values[inside_rows]
    scaling = frozen["scaling"]
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = (accepted - scaling["mean"]) / scaling["population_std"] if scaling["enabled"] else accepted.copy()
    if not np.isfinite(transformed).all():
        return ResultRecord(None, compute_status="degenerate_input", reason="evaluation scaling cannot be represented with finite precision",
                            use_status=use, support=support, **common)
    counts, probabilities = {}, {}
    for j, variable in enumerate(frame.columns):
        count = np.histogram(accepted[:, j], bins=edges)[0]
        counts[variable] = count
        probabilities[variable] = count / count.sum()
    value = {"scaled": pd.DataFrame(transformed, index=frame.index[used], columns=frame.columns),
             "counts": counts, "probabilities": probabilities, "edges": edges,
             "last_edge_included": True, "scaled_units": "1" if scaling["enabled"] else frozen["units"],
             "edge_units": frozen["units"]}
    return ResultRecord(value, compute_status="ok", use_status=use, support=support, **common)

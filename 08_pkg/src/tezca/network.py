"""Exploratory signed correlation networks on explicit elapsed-time windows."""

from itertools import combinations

import numpy as np
import pandas as pd

from ._owned import identity, owned
from .data import ResultRecord
from .individual import _ColumnSignal, _observations
from .support import _aware_timestamp, positive_duration, select_support


class NetworkSpec:
    def __init__(self, *, features, basis, missingness="pairwise", min_points=3,
                 threshold=.3, window=None, step=None, complete=True, label="start",
                 include_partial=False, historical=False, valid_pairs=False):
        if isinstance(features, str):
            raise ValueError("features must be a sequence of names")
        features = list(features)
        if len(features) < 2 or any(not isinstance(v, str) or not v.strip() for v in features) or len(set(features)) != len(features):
            raise ValueError("network requires at least two distinct feature names")
        if basis not in {"raw", "daily_residual"} or missingness not in {"pairwise", "listwise"}:
            raise ValueError("declare raw/daily_residual basis and pairwise/listwise missingness")
        if type(min_points) is not int or min_points < 2:
            raise ValueError("min_points must be an integer >=2")
        if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite within [0,1]")
        if any(type(v) is not bool for v in (complete, include_partial, historical, valid_pairs)) or label not in {"start", "end"}:
            raise ValueError("window/export flags must be boolean and label must be start or end")
        if (window is None) != (step is None):
            raise ValueError("window and step must be declared together")
        if window is not None:
            window, step = positive_duration(window, "window"), positive_duration(step, "step")
        elif include_partial or label != "start" or not complete:
            raise ValueError("window settings require a rolling window")
        self._record = dict(features=features, basis=basis, missingness=missingness,
                            min_points=min_points, threshold=threshold, window=window, step=step,
                            complete=complete, label=label, include_partial=include_partial,
                            closure="left", historical=historical, valid_pairs=valid_pairs,
                            threshold_roundoff_tolerance=1e-12)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def report_network_spec(*, features, basis, missingness="pairwise", min_points=3,
                        historical=False, valid_pairs=False):
    """Client recipe: complete 168 h windows, 24 h steps, abs(r)>=.3."""
    return NetworkSpec(features=features, basis=basis, missingness=missingness,
                       min_points=min_points, threshold=.3, window="168h", step="24h",
                       historical=historical, valid_pairs=valid_pairs)


class NetworkResult:
    def __init__(self, summary, edges, record):
        self._summary, self._edges, self._record = owned(summary), owned(edges), owned(record)
        self._identity = identity(dict(summary=summary.to_dict("records"), edges=edges.to_dict("records"), record=record))

    @property
    def summary(self):
        return owned(self._summary)

    @property
    def edges(self):
        return owned(self._edges)

    @property
    def identity(self):
        return self._identity

    def describe(self):
        return owned(self._record)


def _windows(axis, settings, start, end):
    index = axis.index
    if settings["window"] is None:
        left = _aware_timestamp(start, "start") if start is not None else (index[0] if len(index) else None)
        right = _aware_timestamp(end, "end") if end is not None else (
            index[-1] + axis.cadence if len(index) and axis.mode == "regular" else None)
        if left is not None and right is not None and right <= left:
            raise ValueError("end must be after start")
        positions = np.arange(len(index))
        if left is not None:
            positions = positions[index[positions] >= left]
        if right is not None:
            positions = positions[index[positions] < right]
        return [(left, right, positions, None)]
    if axis.mode != "regular":
        raise ValueError("rolling networks require an explicit regular clock")
    if any(settings[v].value % axis.cadence.value for v in ("window", "step")):
        raise ValueError("window and step must be exact cadence multiples")
    if not len(index) and (start is None or end is None):
        raise ValueError("empty rolling input requires explicit start and end")
    left = _aware_timestamp(start, "start") if start is not None else index[0]
    limit = _aware_timestamp(end, "end") if end is not None else index[-1] + axis.cadence
    if limit <= left:
        raise ValueError("end must be after start")
    phase = axis.record["phase"]
    if phase is not None and any((t.value - phase.value) % axis.cadence.value for t in (left, limit)):
        raise ValueError("window boundaries must match the declared phase")
    windows = []
    while left < limit:
        right = left + settings["window"]
        if right > limit and not settings["include_partial"]:
            break
        positions = np.flatnonzero((index >= left) & (index < min(right, limit)))
        expected = int(settings["window"].value // axis.cadence.value)
        windows.append((left, right, positions, expected))
        left += settings["step"]
    return windows


def _correlation(x, y):
    # Positive rescaling before centering avoids overflow in means and squares.
    a, b = x / max(float(np.max(np.abs(x))), 1e-300), y / max(float(np.max(np.abs(y))), 1e-300)
    a, b = a - a.mean(), b - b.mean()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None, "constant_input", "at least one feature is constant on pair support"
    r = float(np.dot(a / norm_a, b / norm_b))
    if not np.isfinite(r) or abs(r) > 1 + 1e-12:
        return None, "invalid_correlation", "correlation exceeds numerical range"
    return float(np.clip(r, -1, 1)), "ok", None


def _use(statuses):
    return min(statuses, key=["unknown_provenance", "insufficient_support", "sensitivity_only",
                              "unverified_gap_geometry", "primary_eligible"].index)


def correlation_network(data, *, spec, observations=None, start=None, end=None):
    if not isinstance(spec, NetworkSpec):
        raise ValueError("spec must be NetworkSpec")
    observations = _observations(observations)
    settings, frame = spec.record, data.values
    features = settings["features"]
    signals = {v: _ColumnSignal(data, v) for v in features}
    if any(s.basis != settings["basis"] for s in signals.values()):
        raise ValueError("feature basis must match declared network basis")
    axis = next(iter(signals.values())).time
    windows = _windows(axis, settings, start, end)
    masks = {v: s.eligible(observations)[:, 0] for v, s in signals.items()}
    complete_mask = np.logical_and.reduce(list(masks.values()))
    reference_ids = {v: s.reference_id for v, s in signals.items()}
    config_id = identity(dict(spec=settings, observations=observations.__dict__, start=start, end=end))
    p, pair_count = len(features), len(features) * (len(features) - 1) // 2
    summaries, edges = [], []
    for number, (left, right, window_positions, expected) in enumerate(windows):
        current, pair_records = [], []
        for a, b in combinations(features, 2):
            mask = complete_mask if settings["missingness"] == "listwise" else masks[a] & masks[b]
            positions = window_positions[mask[window_positions]]
            support, statuses = {}, []
            for v in (a, b):
                selection = select_support(signals[v], observations=observations, policy="complete_case",
                                           estimator="point", within=positions)
                support[v] = signals[v].support_record(selection)
                statuses.append(signals[v].use(positions, observations, "exploratory_association"))
            count = len(positions)
            r, status, reason = None, "ok", None
            if expected is not None and settings["complete"] and count != expected:
                status, reason = "insufficient_support", "gap_or_incomplete_window"
            elif count < max(settings["min_points"], observations.min_points):
                status, reason = "insufficient_support", "too_few_common_points"
            else:
                r, status, reason = _correlation(frame[a].iloc[positions].to_numpy(dtype=float),
                                                  frame[b].iloc[positions].to_numpy(dtype=float))
            strong = None if r is None else abs(r) + settings["threshold_roundoff_tolerance"] >= settings["threshold"]
            edge = dict(window=number, a=a, b=b, correlation=r, strong=strong, eligible=status == "ok",
                        count=count, positions=positions.tolist(), instants=[t.isoformat() for t in axis.index[positions]],
                        compute_status=status, reason=reason, metric_status="exploratory_association",
                        use_status=_use(statuses), units="1", support=support,
                        input_id=data.identity, basis=settings["basis"], config_id=config_id,
                        method_id="network.signed_pearson_v1", backend="numpy", backend_version=np.__version__,
                        requested=settings, resolved=settings,
                        reference_ids={v: reference_ids[v] for v in (a, b)})
            edge["support_id"] = identity(dict(input=data.identity, positions=positions, features=[a, b], observations=observations.__dict__))
            current.append(edge)
            pair_records.append(dict(a=a, b=b, support=support))
        valid = [e for e in current if e["eligible"]]
        all_valid = len(valid) == pair_count
        full_value = None
        if all_valid:
            full_value = dict(mean_absolute_connectivity=sum(abs(e["correlation"]) for e in valid) / pair_count,
                              strong_count=sum(e["strong"] for e in valid),
                              density=sum(e["strong"] for e in valid) / pair_count)
            if settings["historical"]:
                full_value.update(historical_diagonal_diluted_mean=full_value["mean_absolute_connectivity"] * (p - 1) / p,
                                  historical_double_counted_links=full_value["strong_count"] * 2)
        support = dict(input_id=data.identity, source=next(iter(signals.values())).source,
                       features=features, units={v: s.metadata[v]["unit"] for v, s in signals.items()},
                       basis=settings["basis"], clock=axis.record, positions=window_positions.tolist(),
                       expected_cells=expected, pairs=pair_records, reference_ids=reference_ids,
                       requested=settings, resolved=settings, observations=observations.__dict__,
                       backend="numpy", backend_version=np.__version__, threshold_is_significance=False)
        result = ResultRecord(full_value, compute_status="ok" if all_valid else "undefined_pairs",
                              reason=None if all_valid else "at least one declared pair is undefined",
                              metric_status="exploratory_association", use_status=_use([e["use_status"] for e in current]),
                              support=support, units="1", representation=settings["basis"],
                              method_id="network.signed_pearson_v1", config_id=config_id,
                              reference_id=identity(reference_ids) if any(reference_ids.values()) else None).summary()
        result.update(window=number, start=left, end=right, label=left if settings["label"] == "start" else right,
                      p=p, pair_denominator=pair_count, valid_pair_count=len(valid))
        for field in ("mean_absolute_connectivity", "strong_count", "density"):
            result[field] = full_value[field] if full_value is not None else None
        if settings["historical"]:
            for field in ("historical_diagonal_diluted_mean", "historical_double_counted_links"):
                result[field] = full_value[field] if full_value is not None else None
        if settings["valid_pairs"]:
            result["valid_pairs_summary"] = dict(denominator=len(valid), compute_status="ok" if valid else "insufficient_support",
                mean_absolute_connectivity=sum(abs(e["correlation"]) for e in valid) / len(valid) if valid else None,
                strong_count=sum(e["strong"] for e in valid) if valid else None,
                density=sum(e["strong"] for e in valid) / len(valid) if valid else None,
                method_id="network.eligible_pairs_only_v1")
        summaries.append(result)
        edges.extend(current)
    record = dict(method_id="network.signed_pearson_v1", input_id=data.identity,
                  source=next(iter(signals.values())).source, features=features,
                  units={v: s.metadata[v]["unit"] for v, s in signals.items()}, basis=settings["basis"],
                  clock=axis.record, requested=settings, resolved=settings,
                  observations=observations.__dict__, config_id=config_id, reference_ids=reference_ids,
                  start=start, end=end, backend="numpy", backend_version=np.__version__,
                  window_count=len(windows), compute_status=("ok" if all(r["compute_status"] == "ok" for r in summaries)
                      else "undefined_pairs") if windows else "insufficient_support",
                  reason=(None if all(r["compute_status"] == "ok" for r in summaries) else "at least one requested network contains undefined pairs")
                      if windows else "no requested window fits the declared interval",
                  metric_status="exploratory_association", use_status=_use([r["use_status"] for r in summaries]) if summaries else "insufficient_support")
    return NetworkResult(pd.DataFrame(summaries), pd.DataFrame(edges), record)

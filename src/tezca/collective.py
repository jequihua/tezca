"""Explicit, non-reconstructing collective preparation and report-site screening."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ._owned import identity, owned, text_field, unit_field
from .data import AnalysisSeries, ContributorLineage
from .individual import _ColumnSignal
from .support import (ObservationPolicy, _aware_timestamp, positive_duration,
                      reconstruction_burden, use_status)


class CollectionMember:
    """One declared physical variable from one station/source; clocks stay separate."""

    def __init__(self, series, *, variable, station, family, feature, semantics):
        if not isinstance(series, AnalysisSeries) or variable not in series.metadata:
            raise ValueError("member requires AnalysisSeries and an existing variable")
        for name, value in (("station", station), ("family", family), ("feature", feature)):
            text_field(value, name)
        if semantics not in {"state", "accumulation", "rate", "angle"}:
            raise ValueError("unknown measurement semantics")
        self._series = series
        self._record = dict(variable=variable, station=station, family=family,
                            feature=feature, semantics=semantics, source_id=series.identity)

    @property
    def record(self):
        return owned(self._record)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    members: tuple[str, ...]
    reducer: str = "mean"
    min_coverage: float = 1.0
    station_reducer: str | None = None
    spatial_group: str | None = None
    priority: tuple[str, ...] = ()
    output_unit: str | None = None
    rate_period: str | None = None

    def __post_init__(self):
        text_field(self.name, "feature name")
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "priority", tuple(self.priority))
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("members must be unique and nonempty")
        if self.reducer not in {"mean", "sum", "rate_mean", "integral", "circular_mean"}:
            raise ValueError("unknown reducer")
        if type(self.min_coverage) not in (int, float) or not np.isfinite(self.min_coverage) or not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0,1]")
        if self.station_reducer not in {None, "median"}:
            raise ValueError("only explicit median station aggregation is supported")
        if (self.station_reducer is None) != (self.spatial_group is None):
            raise ValueError("station median requires an explicit spatial_group")
        if self.spatial_group is not None:
            text_field(self.spatial_group, "spatial_group")
        if self.priority and (len(set(self.priority)) != len(self.priority) or set(self.priority) != set(self.members)):
            raise ValueError("priority must order every member exactly once")
        if self.output_unit is not None:
            unit_field(self.output_unit)
        if self.reducer == "integral":
            if self.output_unit is None or self.rate_period is None:
                raise ValueError("rate integration requires output_unit and rate_period")
            positive_duration(self.rate_period, "rate_period")
        elif self.rate_period is not None:
            raise ValueError("rate_period only applies to integration")


class PreparedCollective:
    """Owned start-cell series, separately labelled matrix, audit and observed counterpart."""

    def __init__(self, series, audit, observed_only, record, labels):
        self._series = series
        self._audit = owned(audit)
        self._observed_only = owned(observed_only)
        self._record = owned(record)
        self._labels = labels.copy()

    @property
    def series(self):
        return owned(self._series)

    @property
    def matrix(self):
        result = self._series.values
        result.index = self._labels.copy()
        return result

    @property
    def observed_only(self):
        return owned(self._observed_only)

    @property
    def audit(self):
        return owned(self._audit)

    @property
    def identity(self):
        return self._record["identity"]

    def describe(self):
        return owned(self._record)


def _reduce(values, durations, spec, unit):
    if not len(values):
        return None, "no_eligible_contributors"
    values = np.asarray(values, dtype=float)
    if spec.reducer == "circular_mean":
        radians = np.deg2rad(values) if unit == "deg" else values
        vector = np.mean(np.exp(1j * radians))
        if abs(vector) <= 1e-12:
            return None, "zero_circular_resultant"
        value = np.angle(vector) % (2 * np.pi)
        value = np.rad2deg(value) if unit == "deg" else value
        period = 360 if unit == "deg" else 2 * np.pi
        return (0.0 if abs(value - period) <= 1e-12 else float(value)), None
    if spec.reducer == "sum":
        return float(values.sum()), None
    if spec.reducer == "integral":
        return float(np.dot(values, durations) / positive_duration(spec.rate_period, "rate_period").value), None
    if spec.reducer == "rate_mean":
        return float(np.average(values, weights=durations)), None
    return float(values.mean()), None


def prepare_collective(collection, *, features, start, end, cadence, label="start", observations=None):
    """Aggregate left-closed cells inside an explicit end-exclusive interval; never fill.

    Regular source cells must tile target bins exactly. Priority resolves eligible
    sources per station/slot. Coverage denominators include missing source slots.
    The observed-only counterpart repeats reducers/coverage with observed ancestors.
    """
    if not isinstance(collection, dict) or not collection or any(
            not isinstance(m, CollectionMember) for m in collection.values()):
        raise ValueError("collection must map explicit IDs to CollectionMember")
    for key in collection:
        text_field(key, "member ID")
    features = tuple(features)
    if not features or any(not isinstance(f, FeatureSpec) for f in features) or len({f.name for f in features}) != len(features):
        raise ValueError("features require unique explicit FeatureSpec names")
    start, end = _aware_timestamp(start, "start"), _aware_timestamp(end, "end")
    cadence = positive_duration(cadence, "cadence")
    span = end.value - start.value
    if span <= 0 or span % cadence.value or label not in {"start", "end"}:
        raise ValueError("interval must contain whole target bins; label is start or end")
    observations = ObservationPolicy(values="observed_only") if observations is None else observations
    if not isinstance(observations, ObservationPolicy):
        raise ValueError("observations must be ObservationPolicy")
    index = pd.date_range(start, end, freq=cadence, inclusive="left")
    labels = index if label == "start" else index + cadence
    values = np.full((len(index), len(features)), np.nan)
    observed_values = values.copy()
    lineages, metadata, audit = {}, {}, []
    prepared = {}
    for key, member in collection.items():
        data, declaration = member._series, member.record
        axis = data.time
        if axis.mode != "regular" or cadence.value % axis.cadence.value:
            raise ValueError("source cadence must divide target cadence exactly")
        if any((int(tick) - start.value) % axis.cadence.value for tick in axis.index.asi8):
            raise ValueError("source phase must align with the requested target bins")
        j = list(data.values.columns).index(declaration["variable"])
        original_lineage = data.lineage.get(declaration["variable"])
        observed_values_member = data.values.to_numpy(dtype=float, na_value=np.nan)[:, j].copy()
        observed_mask = data.eligible(ObservationPolicy(values="observed_only"))[:, j]
        observed_values_member[~observed_mask] = np.nan
        if original_lineage:
            for p, parent in enumerate(original_lineage):
                if parent is not None and "observed_only_value" in parent.record:
                    observed_values_member[p] = parent.record["observed_only_value"]
        prepared[key] = dict(data=data, declaration=declaration, axis=axis, j=j,
                             values=data.values.to_numpy(dtype=float, na_value=np.nan)[:, j],
                             eligible=data.eligible(observations)[:, j],
                             trusted=data.eligible(ObservationPolicy(values="trusted_imputation"))[:, j],
                             observed=np.isfinite(observed_values_member), observed_values=observed_values_member,
                             origins=data.contribution[:, j], lineage=original_lineage,
                             qualification=data.qualification,
                             lookup={int(t): i for i, t in enumerate(axis.index.asi8)})
    for column, spec in enumerate(features):
        if set(spec.members) - set(collection):
            raise ValueError("feature references unknown collection members")
        members = [prepared[key] for key in spec.members]
        declarations = [m["declaration"] for m in members]
        units = {m["data"].metadata[m["declaration"]["variable"]]["unit"] for m in members}
        bases = {m["data"].metadata[m["declaration"]["variable"]]["representation"] for m in members}
        semantics = {d["semantics"] for d in declarations}
        if len(units) != 1 or len(bases) != 1 or len(semantics) != 1 or len({d["feature"] for d in declarations}) != 1 or len({d["family"] for d in declarations}) != 1:
            raise ValueError("feature members must share units, basis, semantics, physical feature and family")
        unit, basis, semantic = next(iter(units)), next(iter(bases)), next(iter(semantics))
        allowed = {"state": {"mean"}, "accumulation": {"sum"}, "rate": {"rate_mean", "integral"}, "angle": {"circular_mean"}}
        if spec.reducer not in allowed[semantic] or (semantic == "angle") != (unit in {"deg", "rad"}):
            raise ValueError("reducer, semantics and angle units are incompatible")
        if spec.station_reducer and semantic != "state":
            raise ValueError("station median is supported only for state measurements")
        output_unit = spec.output_unit or unit
        if spec.reducer != "integral" and output_unit != unit:
            raise ValueError("no implicit unit conversion")
        stations = {}
        for key in spec.members:
            stations.setdefault(prepared[key]["declaration"]["station"], []).append(key)
        if len(stations) > 1 and spec.station_reducer is None:
            raise ValueError("stations remain separate unless a spatial median is explicitly requested")
        for keys in stations.values():
            if len({prepared[k]["axis"].cadence.value for k in keys}) != 1:
                raise ValueError("alternate sources at one station must have identical cadence")
        metadata[spec.name] = dict(unit=output_unit, representation=basis,
                                   semantics="accumulation" if spec.reducer == "integral" else semantic,
                                   family=declarations[0]["family"], feature=declarations[0]["feature"],
                                   stations=list(stations), spatial_group=spec.spatial_group)
        lineages[spec.name] = []
        for row, timestamp in enumerate(index):
            station_values, station_observed, station_parents, station_ids, station_audits = [], [], [], [], []
            for station, keys in stations.items():
                step = prepared[keys[0]]["axis"].cadence.value
                expected = cadence.value // step
                chosen, observed_chosen, rejected = [], [], []
                for tick in range(timestamp.value, timestamp.value + cadence.value, step):
                    candidates, observed_candidates = [], []
                    for key in keys:
                        m = prepared[key]
                        position = m["lookup"].get(tick)
                        if position is None:
                            rejected.append(dict(member=key, tick=tick, reason="missing_timestamp"))
                            continue
                        detail = dict(member=key, position=position, tick=tick,
                                      source_id=m["data"].identity, source=m["data"].source,
                                      qualification={q: bool(mask[position, m["j"]]) for q, mask in m["qualification"].items()})
                        if m["eligible"][position]:
                            candidates.append((key, position, detail))
                        else:
                            rejected.append(dict(**detail, reason="unavailable_or_ineligible"))
                        if m["observed"][position]:
                            observed_candidates.append((key, position, detail))
                    for choices, selected in ((candidates, chosen), (observed_candidates, observed_chosen)):
                        if len(choices) > 1 and not spec.priority:
                            raise ValueError("ambiguous duplicate sources require explicit priority")
                        if choices:
                            if spec.priority:
                                choices.sort(key=lambda c: spec.priority.index(c[0]))
                            selected.append(choices[0])
                            if selected is chosen:
                                rejected.extend(dict(**item[2], reason="lower_source_priority") for item in choices[1:])
                coverage = len(chosen) / expected
                def reduce_selected(selected, *, observed=False):
                    if len(selected) / expected < spec.min_coverage or len(selected) < observations.min_points:
                        return None, "insufficient_within_bin_coverage"
                    return _reduce([prepared[k]["observed_values" if observed else "values"][p] for k, p, _ in selected],
                                   np.full(len(selected), step, dtype=float), spec, unit)
                value, reason = reduce_selected(chosen)
                observed_value, observed_reason = reduce_selected(observed_chosen, observed=True)
                record = dict(station=station, expected_count=int(expected), count=len(chosen), coverage=coverage,
                              contributors=[d for _, _, d in chosen], rejected=rejected,
                              observed_only_contributors=[d for _, _, d in observed_chosen],
                              observed_only_coverage=len(observed_chosen) / expected,
                              value=value, reason=reason, observed_only_value=observed_value,
                              observed_only_reason=observed_reason)
                station_audits.append(record)
                if observed_value is not None:
                    station_observed.append(observed_value)
                if value is None:
                    continue
                operation = {"mean": "linear_mean", "sum": "linear_sum", "rate_mean": "rate_mean",
                             "integral": "rate_integral", "circular_mean": "circular_mean"}[spec.reducer]
                parents = [prepared[k]["lineage"][p] if prepared[k]["lineage"] else None for k, p, _ in chosen]
                origins = [prepared[k]["origins"][p] if prepared[k]["origins"][p] != "derived" else "unknown" for k, p, _ in chosen]
                lineage = ContributorLineage([f"{k}:{p}" for k, p, _ in chosen], origins, unit=output_unit,
                                             weights=(np.full(len(chosen), step / positive_duration(spec.rate_period, "rate_period").value)
                                                      if spec.reducer == "integral" else None),
                                             operation=operation, parents=parents, details=[d for _, _, d in chosen],
                                             observed_only_value=observed_value,
                                             trusted_imputation=all(prepared[k]["trusted"][p] for k, p, _ in chosen))
                station_values.append(value)
                station_parents.append(lineage)
                station_ids.append(station)
            result = None
            if station_values:
                values[row, column] = float(np.median(station_values)) if spec.station_reducer else station_values[0]
                result = (ContributorLineage(station_ids, ["unknown"] * len(station_ids), unit=output_unit,
                                             operation="median", parents=station_parents,
                                             observed_only_value=float(np.median(station_observed)) if station_observed else None,
                                             trusted_imputation=all(p.record["trusted_imputation"] for p in station_parents))
                          if spec.station_reducer else station_parents[0])
            if station_observed:
                observed_values[row, column] = float(np.median(station_observed)) if spec.station_reducer else station_observed[0]
            lineages[spec.name].append(result)
            audit.append(dict(feature=spec.name, start=timestamp, end=timestamp + cadence, label=labels[row],
                              stations=station_audits, contributing_stations=station_ids,
                              contributor_count=sum(s["count"] for s in station_audits if s["value"] is not None),
                              compute_status="ok" if result else "unavailable_cell",
                              reason=None if result else ";".join(sorted({s["reason"] for s in station_audits})),
                              lineage_id=result.identity if result else None))
    settings = dict(features=[asdict(f) for f in features], start=start, end=end, cadence=cadence,
                    label=label, closure="left", observations=asdict(observations), circular_tolerance=1e-12)
    source = dict(id=identity(dict(members={k: m.record for k, m in collection.items()}, settings=settings)),
                  segment="explicit_collective", members={k: m.record for k, m in collection.items()})
    frame = pd.DataFrame(values, index=index, columns=[f.name for f in features])
    series = AnalysisSeries(frame, metadata=metadata, source=source,
                            clock=dict(mode="regular", cadence=cadence, phase=start),
                            contribution=np.where(np.isfinite(values), "derived", "unavailable"), lineage=lineages)
    feature_support = {}
    for j, variable in enumerate(frame.columns):
        signal = _ColumnSignal(series, variable)
        positions = np.flatnonzero(np.isfinite(values[:, j])).tolist()
        feature_support[variable] = dict(eligible_positions=positions,
                                         burden=reconstruction_burden(signal, positions),
                                         use_status=use_status(signal, positions, observations),
                                         compute_status="ok" if positions else "insufficient_support")
    statuses = {s["use_status"] for s in feature_support.values()}
    overall_use = next(s for s in ("unknown_provenance", "insufficient_support", "sensitivity_only",
                                   "unverified_gap_geometry", "primary_eligible") if s in statuses)
    record = dict(identity=identity(dict(series=series.identity, settings=settings)), source=source,
                  settings=settings, method_id="collective.explicit_cell_aggregation_v1",
                  backend=dict(name="numpy", version=np.__version__), reference_id=None,
                  variables=list(metadata), metadata=metadata, support=dict(start=start, end=end,
                  closure="left", eligible_positions={v: np.flatnonzero(np.isfinite(values[:, j])).tolist()
                                                       for j, v in enumerate(metadata)},
                  full_grid_burden=reconstruction_burden(series), features=feature_support),
                  compute_status="ok" if np.isfinite(values).all() else "incomplete_matrix",
                  metric_status="audited_preparation", use_status=overall_use,
                  reason=None if np.isfinite(values).all() else "unavailable cells retained; see audit",
                  real_F01_admission="unresolved_Q002_Q003")
    return PreparedCollective(series, pd.DataFrame(audit), pd.DataFrame(observed_values, index=labels,
                                                                      columns=frame.columns), record, labels)


def report_site_gate(prepared, *, coverage_basis="observed_only"):
    """Named report-site screen: >=2 families, >=30 common covered days, .65, >=5.

    Common-family duration counts target cells with at least one eligible feature
    in every declared family. Variable denominator is the entire requested grid.
    This numeric screen neither admits real data nor certifies QC/site semantics.
    """
    if not isinstance(prepared, PreparedCollective) or coverage_basis not in {"observed_only", "provided"}:
        raise ValueError("supply PreparedCollective and explicit supported coverage_basis")
    frame = prepared.observed_only if coverage_basis == "observed_only" else prepared.matrix
    metadata = prepared._series.metadata
    valid = np.isfinite(frame.to_numpy(dtype=float))
    families = sorted({m["family"] for m in metadata.values()})
    coverage = {v: float(valid[:, j].mean()) if len(frame) else 0 for j, v in enumerate(frame.columns)}
    retained = [v for v in frame if coverage[v] >= .65]
    common = np.ones(len(frame), dtype=bool)
    for family in families:
        columns = [j for j, v in enumerate(frame) if metadata[v]["family"] == family]
        common &= valid[:, columns].any(axis=1)
    days = int(common.sum()) * prepared._series.time.cadence.value / (86400 * 1e9)
    reasons = []
    if len(families) < 2:
        reasons.append("fewer_than_two_families")
    if days < 30:
        reasons.append("fewer_than_30_common_days")
    if len(retained) < 5:
        reasons.append("fewer_than_five_retained_variables")
    return dict(recipe="report_site_screen_v1", source_id=prepared.identity, coverage_basis=coverage_basis,
                denominator=len(frame), families=families, common_positions=np.flatnonzero(common).tolist(),
                common_days=days, coverage=coverage, retained_variables=retained,
                excluded={v: "coverage_below_0.65" for v in frame if v not in retained},
                compute_status="ok" if not reasons else "site_gate_failed", reasons=reasons,
                domain_admission="unresolved_Q002_Q003")

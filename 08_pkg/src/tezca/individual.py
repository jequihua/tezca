"""Descriptive dynamics and explicit training-only civil daily references."""

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ._owned import identity, owned
from .data import AnalysisSeries, ResultRecord
from .reference import FrozenReference
from .support import ObservationPolicy, positions_array, positive_duration, select_support, use_status


class DailySpec:
    def __init__(self, *, civil_timezone, fold_policy="daily_median", min_dates=2):
        try:
            ZoneInfo(civil_timezone)
        except (ZoneInfoNotFoundError, TypeError) as error:
            raise ValueError("daily fitting requires an explicit known civil_timezone") from error
        if fold_policy != "daily_median":
            raise ValueError("fold_policy must be daily_median: one median per date/minute")
        if type(min_dates) is not int or min_dates < 2:
            raise ValueError("min_dates must be an integer >=2")
        self._record = dict(civil_timezone=civil_timezone, fold_policy=fold_policy,
                            min_dates=min_dates, clock_bins="minute_of_day",
                            aggregation="median_of_per_date_medians")

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


class DailyReference(FrozenReference):
    pass


class DailyResidual:
    """Owned residual values; original qualifications and fit burden stay separate."""
    def __init__(self, series, values, reference, evaluation):
        self._series = series
        self._values = owned(values)
        self._reference = reference
        self._evaluation = owned(evaluation)
        self._identity = identity(dict(source=series.identity, reference=reference.identity,
                                       values=values.to_numpy(), evaluation=evaluation))

    @property
    def values(self):
        return owned(self._values)

    @property
    def identity(self):
        return self._identity

    @property
    def reference_id(self):
        return self._reference.identity

    def describe(self):
        return dict(basis="daily_residual", source_id=self._series.identity,
                    reference_id=self.reference_id, fit=self._reference.describe(),
                    evaluation=owned(self._evaluation))


class _ColumnSignal:
    """One metric variable, sharing the owned input without copying cell details."""
    def __init__(self, data, variable):
        if not isinstance(data, (AnalysisSeries, DailyResidual)):
            raise ValueError("input must be AnalysisSeries or DailyResidual")
        base = data._series if isinstance(data, DailyResidual) else data
        if variable not in base.metadata:
            raise ValueError("unknown variable")
        self._base, self._data, self.variable = base, data, variable
        self._j = list(base.values.columns).index(variable)
        self.basis = "daily_residual" if isinstance(data, DailyResidual) else base.metadata[variable]["representation"]
        self.reference_id = data.reference_id if isinstance(data, DailyResidual) else None

    @property
    def values(self):
        return self._data.values[[self.variable]]

    @property
    def time(self):
        return self._base.time

    @property
    def source(self):
        return self._base.source

    @property
    def metadata(self):
        return {self.variable: self._base.metadata[self.variable]}

    @property
    def contribution(self):
        result = self._base.contribution[:, [self._j]]
        result[~np.isfinite(self.values.to_numpy(dtype=float, na_value=np.nan))] = "unavailable"
        return result

    @property
    def origin(self):
        return {k: v[:, [self._j]] for k, v in self._base.origin.items()}

    @property
    def identity(self):
        return identity(dict(input=self._data.identity, variable=self.variable, basis=self.basis))

    def eligible(self, observations):
        return self._base.eligible(observations)[:, [self._j]] & np.isfinite(
            self.values.to_numpy(dtype=float, na_value=np.nan))

    def support_record(self, selection):
        record = selection.record
        record.update(variable=self.variable, source=self.source, input_id=self.identity,
                      basis=self.basis, clock=self.time.record,
                      selected_instants=[t.isoformat() for t in self.time.index[selection.positions]])
        if self.reference_id:
            fit = self._data._reference.describe()
            record.update(reference_id=self.reference_id,
                          fit_support=fit["variables"][self.variable]["support"],
                          fit_use_status=fit["variables"][self.variable]["use_status"],
                          fit_observations=fit["observations"], fit_scope=fit["fit_scope"])
        return record

    def use(self, positions, observations, metric_status="primary_with_support"):
        result = use_status(self, positions, observations, metric_status=metric_status)
        if self.reference_id:
            fit = self._data._reference.describe()["variables"][self.variable]
            # Evaluation cannot erase an unqualified or reconstructed training fit.
            if fit["use_status"] == "unknown_provenance":
                return "unknown_provenance"
            if result == "primary_eligible" and (fit["use_status"] != "primary_eligible"
                    or fit["support"]["selected"]["variables"][self.variable]["touched_count"]):
                return "sensitivity_only"
        return result


def _observations(observations):
    return ObservationPolicy("observed_only") if observations is None else observations


def _point_selection(signal, observations, support):
    return select_support(signal, observations=observations, policy=support, estimator="point")


def _result(signal, selection, observations, *, metric, value, config, units=None,
            status="ok", reason=None, metric_status="primary_with_support", extra=None):
    support = signal.support_record(selection)
    support.update(metric=metric, requested=owned(config), resolved=owned(config),
                   observations=owned(observations.__dict__))
    if extra:
        support.update(owned(extra))
    return ResultRecord(value, compute_status=status, reason=reason, metric_status=metric_status,
                        use_status=signal.use(selection.positions, observations, metric_status),
                        support=support, units=units or signal.metadata[signal.variable]["unit"],
                        method_id=metric, config_id=identity(dict(settings=config, observations=observations.__dict__)),
                        reference_id=signal.reference_id, representation=signal.basis)


def describe_series(data, *, observations=None, support=None):
    """Return per-variable/per-metric records, including missing sample SD."""
    observations = _observations(observations)
    base = data._series if isinstance(data, DailyResidual) else data
    if support is None:
        support = "all_runs" if base.time.mode == "regular" else "irregular"
    records = []
    for variable in base.values.columns:
        signal = _ColumnSignal(data, variable)
        selection = _point_selection(signal, observations, support)
        values = signal.values.iloc[selection.positions, 0].to_numpy(dtype=float, na_value=np.nan)
        for metric in ("mean", "median", "sample_sd", "iqr", "minimum", "maximum", "population_variance"):
            required = max(observations.min_points, 2 if metric == "sample_sd" else 1)
            value, status, reason = None, "insufficient_support", "too few eligible points"
            if len(values) >= required:
                calculators = {"mean": np.mean, "median": np.median, "sample_sd": lambda x: np.std(x, ddof=1),
                               "iqr": lambda x: np.diff(np.quantile(x, [0.25, 0.75], method="linear"))[0],
                               "minimum": np.min, "maximum": np.max, "population_variance": np.var}
                with np.errstate(over="ignore", invalid="ignore"):
                    value = float(calculators[metric](values))
                status, reason = "ok", None
                if not np.isfinite(value):
                    value, status, reason = None, "degenerate_input", "statistic exceeds finite precision"
            unit = signal.metadata[variable]["unit"]
            records.append(_result(signal, selection, observations, metric="individual." + metric + "_v1",
                                   value=value, status=status, reason=reason,
                                   units=f"{unit}^2" if metric == "population_variance" else unit,
                                   config=dict(support=support, ddof=1 if metric == "sample_sd" else 0,
                                               quantile_method="linear"),
                                   extra=dict(constant_input=bool(len(values) and np.ptp(values) == 0))))
    return records


def estimate_trend(data, *, variable, observations=None, support=None, uncertainty="unavailable"):
    if uncertainty not in {"unavailable", "iid_ols_compatibility"}:
        raise ValueError("uncertainty must be unavailable or iid_ols_compatibility")
    observations = _observations(observations)
    signal = _ColumnSignal(data, variable)
    if support is None:
        support = "all_runs" if signal.time.mode == "regular" else "irregular"
    selection = _point_selection(signal, observations, support)
    positions = selection.positions
    config = dict(support=support, time_unit="elapsed_days", uncertainty=uncertainty)
    extra = dict(uncertainty=dict(method=uncertainty, p_value=None,
                 reason="no dependence-aware temporal inference implemented", assumptions=None))
    value, status, reason = None, "insufficient_support", "trend requires at least two eligible times"
    if len(positions) >= max(2, observations.min_points):
        ticks = signal.time.index.asi8[positions]
        days = np.array([(int(tick) - int(ticks[0])) / 86400e9 for tick in ticks])
        values = signal.values.iloc[positions, 0].to_numpy(dtype=float, na_value=np.nan)
        fit = linregress(days, values)
        value, status, reason = float(fit.slope), "ok", None
        extra.update(intercept=float(fit.intercept), elapsed_days=days)
        if not np.isfinite(value):
            value, status, reason = None, "degenerate_input", "trend exceeds finite precision"
        if uncertainty == "iid_ols_compatibility":
            extra["uncertainty"] = dict(method=uncertainty,
                p_value=float(fit.pvalue) if np.isfinite(fit.pvalue) else None,
                reason="iid compatibility diagnostic; untrusted temporal inference",
                assumptions="independent identically distributed errors", trusted_temporal_inference=False)
    return _result(signal, selection, observations, metric="individual.elapsed_day_ols_v1", value=value,
                   units=signal.metadata[variable]["unit"] + "/day", status=status, reason=reason, config=config, extra=extra)


def autocorrelation(data, *, variable, lag="one_step", observations=None, support="all_runs"):
    observations = _observations(observations)
    signal = _ColumnSignal(data, variable)
    if support not in {"all_runs", "longest_run"}:
        raise ValueError("ACF requires all_runs or longest_run")
    if signal.time.mode != "regular":
        if lag not in {"one_step", "one_hour", "twenty_four_hours"}:
            positive_duration(lag, "lag")
        selection = _point_selection(signal, observations, "irregular")
        resolved = dict(requested=lag, samples=None, realized_seconds=None)
        status, reason = "unsupported_exact_lag", "exact sample ACF requires declared regular cadence"
    else:
        selection = select_support(signal, observations=observations, policy=support)
        resolved = signal.time.resolve_lag(lag)
        status, reason = resolved["compute_status"], resolved["reason"]
    pairs, value = [], None
    if status == "ok":
        allowed = set(selection.positions.tolist())
        count = resolved["samples"]
        for run in selection.record["runs"]:
            pairs.extend((i, i + count) for i in range(run["start_position"], run["stop_position"] - count)
                         if i in allowed and i + count in allowed)
        status, reason = "insufficient_support", "too few exact eligible pairs"
        if len(pairs) >= max(2, observations.min_points):
            values = signal.values.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
            a, b = values[np.array(pairs).T]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                status, reason = "degenerate_input", "constant lag-pair margin"
            else:
                value = float(np.corrcoef(a, b)[0, 1])
                status, reason = "ok", None
                if not np.isfinite(value):
                    value, status, reason = None, "degenerate_input", "pair correlation exceeds finite precision"
    used = np.unique(np.array(pairs, dtype=int)) if pairs else np.array([], dtype=int)
    # Re-select only pair endpoints so burden never uses unrelated rows.
    pair_selection = select_support(signal, observations=observations,
                                    policy="all_runs" if signal.time.mode == "regular" else "irregular",
                                    estimator="point", within=used)
    return _result(signal, pair_selection, observations, metric="individual.exact_lagged_pearson_v1",
                   value=value, units="1", status=status, reason=reason,
                   config=dict(support=support, lag=lag), extra=dict(lag=resolved, pairs=pairs, pair_count=len(pairs)))


def _subset(series, positions):
    """Training-only input identity; held-out values do not enter the fit record."""
    clock = series.time.record
    clock.pop("duration_convention")
    if clock["coverage_seconds"] is not None:
        clock["coverage_seconds"] = clock["coverage_seconds"][positions]
    return AnalysisSeries(series.values.iloc[positions], metadata=series.metadata, source=series.source,
                          clock=clock, contribution=series.contribution[positions],
                          qualification={k: v[positions] for k, v in series.qualification.items()},
                          lineage={k: [v[i] for i in positions] for k, v in series.lineage.items()})


def _training_signal(data, variable, positions):
    """Own just training values/qualification, retaining any fitted daily transform."""
    if isinstance(data, DailyResidual):
        training = DailyResidual(_subset(data._series, positions), data.values.iloc[positions],
                                 data._reference, {})
    else:
        training = _subset(data, positions)
    return _ColumnSignal(training, variable)


def fit_daily_reference(series, *, spec, training_positions=None, observations=None, fit_scope="training_only"):
    if not isinstance(series, AnalysisSeries) or not isinstance(spec, DailySpec):
        raise ValueError("daily fitting requires AnalysisSeries and DailySpec")
    if fit_scope not in {"training_only", "self"}:
        raise ValueError("fit_scope must be training_only or self")
    if (fit_scope == "training_only" and training_positions is None
            or fit_scope == "self" and training_positions is not None):
        raise ValueError("training requires positions; self fitting uses all evaluation input")
    observations = _observations(observations)
    positions = np.arange(len(series.values)) if fit_scope == "self" else positions_array(training_positions, len(series.values))
    training = _subset(series, positions)
    civil = training.time.index.tz_convert(spec.record["civil_timezone"])
    dates = np.array([x.date().isoformat() for x in civil])
    minutes = civil.hour * 60 + civil.minute
    variables = {}
    for variable in training.values.columns:
        signal = _ColumnSignal(training, variable)
        selection = _point_selection(signal, observations, "all_runs" if training.time.mode == "regular" else "irregular")
        eligible = selection.positions
        frame = pd.DataFrame(dict(date=dates[eligible], minute=minutes[eligible],
                             value=signal.values.iloc[eligible, 0].to_numpy(dtype=float, na_value=np.nan)))
        daily = frame.groupby(["date", "minute"])["value"].median()
        counts = daily.groupby("minute").count().reindex(range(1440), fill_value=0)
        template = daily.groupby("minute").median().reindex(range(1440))
        template[counts < spec.record["min_dates"]] = np.nan
        if selection.record["compute_status"] != "ok":
            template[:] = np.nan
        variables[variable] = dict(template=template.to_numpy(), date_counts=counts.to_numpy(),
            eligible_training_positions=positions[eligible].tolist(), support=signal.support_record(selection),
            use_status=signal.use(eligible, observations),
            unit=series.metadata[variable]["unit"], representation=series.metadata[variable]["representation"])
    return DailyReference(dict(method_id="daily.median_distinct_dates_v1", spec=spec.record,
        spec_id=spec.identity, training_id=training.identity, source=series.source,
        training_positions=positions.tolist(), training_dates=sorted(set(dates)), variables=variables,
        observations=observations.__dict__, fit_scope=fit_scope,
        semantics="retrospective" if fit_scope == "self" else "training_only"))


def apply_daily_reference(series, *, reference=None, spec=None, observations=None):
    if (reference is None) == (spec is None):
        raise ValueError("supply exactly one of spec or reference")
    if not isinstance(series, AnalysisSeries):
        raise ValueError("daily evaluation requires AnalysisSeries")
    observations = _observations(observations)
    if spec is not None:
        reference = fit_daily_reference(series, spec=spec, observations=observations, fit_scope="self")
    if not isinstance(reference, DailyReference):
        raise ValueError("reference must be DailyReference")
    fit = reference.describe()
    if list(series.values.columns) != list(fit["variables"]):
        raise ValueError("daily reference variable order must match")
    civil = series.time.index.tz_convert(fit["spec"]["civil_timezone"])
    minutes = civil.hour * 60 + civil.minute
    residual = series.values.astype(float)
    evaluations = {}
    for variable in residual.columns:
        fitted = fit["variables"][variable]
        if any(series.metadata[variable][key] != fitted[key] for key in ("unit", "representation")):
            raise ValueError("daily reference units and representation must match")
        signal = _ColumnSignal(series, variable)
        selection = _point_selection(signal, observations, "all_runs" if series.time.mode == "regular" else "irregular")
        valid = signal.eligible(observations).ravel()
        residual[variable] = residual[variable].to_numpy() - fitted["template"][minutes]
        residual.loc[~valid, variable] = np.nan
        evaluations[variable] = signal.support_record(selection)
        evaluations[variable]["supported_positions"] = np.flatnonzero(np.isfinite(residual[variable])).tolist()
        evaluations[variable]["unsupported_minute_positions"] = np.flatnonzero(valid & ~np.isfinite(residual[variable])).tolist()
    return DailyResidual(series, residual, reference, evaluations)

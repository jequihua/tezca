"""Public EcoFill 0.0.1 objects and strict result export format 1.

Object conversion imports EcoFill only when called. Export loading needs only
the numerical core and reads four fixed files, never hidden raw measurements.
"""

from collections import Counter
import csv
import importlib
from importlib.metadata import version
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from .._owned import identity, owned, text_field, unit_field
from ..data import AnalysisSeries
from ..support import TimeAxis, positive_duration


BASE_MASKS = ("raw_present", "observed", "donor", "absent", "qc_rejected",
              "structural", "artificial", "inserted")
RESULT_MASKS = BASE_MASKS + ("imputed",)
SUPPORTED_VERSION = "0.0.1"


def _mapping(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} requires exactly {sorted(fields)}")


def _json_record(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain finite JSON values") from error


def _schema(record):
    _mapping(record, {"timezone", "cadence_seconds", "time_policy", "variables"}, "schema")
    if record["time_policy"] != "strict":
        raise ValueError("EcoFill schema time_policy must be strict")
    text_field(record["timezone"], "schema.timezone")
    seconds = record["cadence_seconds"]
    if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("schema.cadence_seconds must be positive and finite")
    try:
        cadence = positive_duration(pd.Timedelta(seconds, unit="s"), "schema cadence")
    except (ValueError, OverflowError) as error:
        raise ValueError("schema cadence must be representable in nanoseconds") from error
    if cadence.value / 1e9 != seconds:
        raise ValueError("schema cadence must be exactly representable in nanoseconds")
    variables = record["variables"]
    if not isinstance(variables, list) or not variables:
        raise ValueError("schema.variables must be a nonempty list")
    names, parents = [], {}
    for item in variables:
        _mapping(item, {"name", "unit", "role", "derived_from"}, "schema variable")
        name = text_field(item["name"], "variable name")
        unit_field(item["unit"])
        if item["role"] not in ("target", "driver", "response", "external", "derived", "diagnostic"):
            raise ValueError("unsupported schema variable role")
        lineage = item["derived_from"]
        if (not isinstance(lineage, list) or any(not isinstance(p, str) for p in lineage)
                or len(set(lineage)) != len(lineage)):
            raise ValueError("schema lineage must list unique parent names")
        names.append(name)
        parents[name] = lineage
    if len(set(names)) != len(names):
        raise ValueError("schema variable names must be unique")
    for name, lineage in parents.items():
        if any(p not in names or p == name for p in lineage):
            raise ValueError("invalid schema lineage parent")
    # A topological removal checks cycles without guessing contributor weights.
    remaining = set(names)
    while remaining:
        ready = {name for name in remaining if not remaining.intersection(parents[name])}
        if not ready:
            raise ValueError("schema lineage cannot contain cycles")
        remaining -= ready
    return names, cadence


def _index(values, timezone):
    try:
        stamps = [pd.Timestamp(value) for value in values]
        if any(pd.isna(stamp) or stamp.tzinfo is None for stamp in stamps):
            raise ValueError("naive or missing timestamp")
        if any(stamp.utcoffset() != stamp.tz_convert(timezone).utcoffset() for stamp in stamps):
            raise ValueError("timestamp offsets disagree with schema timezone")
        return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_convert(timezone)
    except (TypeError, ValueError, OverflowError, KeyError) as error:
        raise ValueError("timestamps must be aware and match a valid schema timezone") from error


def _aligned(table, frame, name, *, boolean=False):
    if (not isinstance(table, pd.DataFrame) or not table.index.equals(frame.index)
            or not table.columns.equals(frame.columns)):
        raise ValueError(f"{name} must match timestamp/variable labels and order exactly")
    if boolean and (not all(pd.api.types.is_bool_dtype(t) for t in table.dtypes)
                    or table.isna().to_numpy().any()):
        raise ValueError(f"{name} must contain nonmissing booleans")
    return owned(table)


def _provenance(record):
    if not isinstance(record, dict) or type(record.get("format_version")) is not int or record["format_version"] != 1:
        raise ValueError("supported EcoFill export format_version is integer 1")
    required = {"data_fingerprint", "schema", "timestamps", "plan", "state_counts", "masks", "files"}
    if not required <= set(record):
        raise ValueError(f"provenance is missing required fields: {sorted(required - set(record))}")
    names, cadence = _schema(record["schema"])
    text_field(record["data_fingerprint"], "data_fingerprint")
    if not isinstance(record["timestamps"], list) or not record["timestamps"]:
        raise ValueError("provenance timestamps must be a nonempty list")
    if any(not isinstance(t, str) for t in record["timestamps"]):
        raise ValueError("provenance timestamps must be strings")
    inventory = record["masks"]
    if (not isinstance(inventory, list) or any(not isinstance(m, str) for m in inventory)
            or len(inventory) != len(RESULT_MASKS) or set(inventory) != set(RESULT_MASKS)):
        raise ValueError("unsupported provenance mask inventory")
    counts = record["state_counts"]
    if (not isinstance(counts, dict) or any(not isinstance(k, str) or type(v) is not int or v < 0
                                          for k, v in counts.items())):
        raise ValueError("state_counts must contain nonnegative integer counts")
    plan = record["plan"]
    _mapping(plan, {"data_fingerprint", "method", "mode", "max_gap_seconds", "period_seconds",
                    "search_periods", "state", "target_gaps", "support_interval", "gaps", "events"}, "plan")
    if plan["data_fingerprint"] != record["data_fingerprint"]:
        raise ValueError("plan and source fingerprint disagree")
    text_field(plan["method"], "plan.method")
    if plan["mode"] not in ("offline", "causal") or not isinstance(plan["gaps"], list) or not isinstance(plan["events"], list):
        raise ValueError("unsupported plan mode/gap/event records")
    if type(plan["max_gap_seconds"]) not in (int, float) or not np.isfinite(plan["max_gap_seconds"]) or plan["max_gap_seconds"] <= 0:
        raise ValueError("plan.max_gap_seconds must be positive and finite")
    if not isinstance(record["files"], list) or any(not isinstance(f, str) for f in record["files"]):
        raise ValueError("provenance.files must be a list of strings (never followed)")
    return names, cadence


def _text(value, name, *, missing=False):
    if value is None or pd.isna(value):
        if missing:
            return None
        raise ValueError(f"{name} cannot be missing")
    return text_field(value, name)


def _measurements(frame):
    # Numeric cells are immutable; preserve object dtype until checked conversion.
    # Mapping a copy function over them can make pandas infer lossy float storage.
    frame = frame.copy(deep=True)
    for name in frame.columns:
        if frame[name].dtype != object:
            continue
        present = frame[name].dropna()
        integers = [isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in present]
        if all(integers):
            dtype = "UInt64" if len(present) and min(present) >= 0 and max(present) > 2**63 - 1 else "Int64"
            try:
                frame[name] = pd.array(frame[name], dtype=dtype)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("integer measurements exceed the supported 64-bit range") from error
        elif any(integer and float(v) != int(v) for integer, v in zip(integers, present)):
            raise ValueError("mixed numeric measurements would round an integer")
    return frame


def _convert(frame, masks, schema, provenance, *, source, representation, clock,
             trusted_imputation, route, states=None, methods=None, reasons=None, quality=None):
    names, cadence = _schema(schema)
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != names or not len(frame):
        raise ValueError("values must have nonempty timestamps and exact schema variable order")
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != schema["timezone"]:
        raise ValueError("value timestamps must match schema timezone")
    if not isinstance(clock, dict) or not {"civil_timezone", "clock_resolution"} <= set(clock) or set(clock) - {"civil_timezone", "clock_resolution", "phase"}:
        raise ValueError("clock requires civil_timezone and clock_resolution; optional phase only")
    axis = TimeAxis(frame.index, mode="regular", cadence=cadence, **clock)
    ticks = axis.index.asi8
    if any(int(b) - int(a) != cadence.value for a, b in zip(ticks[:-1], ticks[1:])):
        raise ValueError("schema cadence requires consecutive timestamps; no clock repair is performed")
    expected = RESULT_MASKS if states is not None else BASE_MASKS
    if set(masks) != set(expected):
        raise ValueError("mask inventory does not match the supported object/result contract")
    masks = {name: _aligned(table, frame, name, boolean=True).to_numpy(dtype=bool)
             for name, table in masks.items()}
    present, observed = masks["raw_present"], masks["observed"]
    structural, inserted = masks["structural"], masks["inserted"]
    if ((observed | masks["qc_rejected"]) & ~present).any() or (observed & masks["qc_rejected"]).any():
        raise ValueError("original observed/QC masks contradict raw presence")
    if ((structural | inserted) & present).any() or (structural & inserted).any():
        raise ValueError("structural/inserted masks contradict original presence")
    if np.any(inserted.any(axis=1) != inserted.all(axis=1)):
        raise ValueError("inserted timestamps must mark every variable")
    if not np.array_equal(masks["absent"], ~present & ~structural & ~inserted):
        raise ValueError("absent mask contradicts original qualifications")
    if (masks["artificial"] & ~observed).any() or not np.array_equal(masks["donor"], observed & ~masks["artificial"]):
        raise ValueError("donor/artificial masks contradict original observations")
    for j, spec in enumerate(schema["variables"]):
        for parent in spec["derived_from"]:
            if (masks["artificial"][:, names.index(parent)] & observed[:, j] & ~masks["artificial"][:, j]).any():
                raise ValueError("hidden parent must hide its observed derived descendants")
    # AnalysisSeries validates real numeric values and independently owns these flags.
    frame = _measurements(frame)
    try:
        numeric = frame.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be real numeric measurements") from error
    finite = np.isfinite(numeric)
    if np.isinf(numeric).any():
        raise ValueError("infinite completion values are invalid")
    if (structural & finite).any():
        raise ValueError("structural cells cannot contain finite values")
    imputed = masks.get("imputed", np.zeros(frame.shape, dtype=bool))
    if (imputed & masks["donor"]).any() or (imputed & structural).any():
        raise ValueError("donor/imputed/structural masks overlap")
    if not np.array_equal(finite, masks["donor"] | imputed):
        raise ValueError("finite values must be exactly classified donor or imputed; hidden truth is forbidden")
    categories = np.full(frame.shape, "unavailable", dtype=object)
    categories[masks["donor"]] = "observed"
    categories[imputed] = "imputed"
    tables = {}
    if states is not None:
        _provenance(provenance)
        expected_index = _index(provenance["timestamps"], schema["timezone"])
        if not expected_index.equals(frame.index):
            raise ValueError("provenance timestamps disagree with values")
        tables = {"state": states, "method": methods, "reason": reasons}
        if quality is not None:
            tables["quality"] = quality
        if (quality is not None) != ("quality" in provenance):
            raise ValueError("quality table and provenance declaration disagree")
        tables = {name: _aligned(table, frame, name) for name, table in tables.items()}
        for i in range(len(frame)):
            for j in range(len(names)):
                state = _text(tables["state"].iat[i, j], "state")
                method = _text(tables["method"].iat[i, j], "method", missing=True)
                reason = _text(tables["reason"].iat[i, j], "reason", missing=True)
                if (state == "observed") != masks["donor"][i, j] or (state == "imputed") != imputed[i, j]:
                    raise ValueError("cell state disagrees with donor/imputed masks")
                if imputed[i, j] and method is None:
                    raise ValueError("imputed cells require a per-cell method")
                tables["state"].iat[i, j] = state
                tables["method"].iat[i, j] = method
                tables["reason"].iat[i, j] = reason
                if quality is not None:
                    value = tables["quality"].iat[i, j]
                    if pd.isna(value):
                        value = None
                    elif not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                        raise ValueError("cell quality must be finite numeric or missing")
                    else:
                        value = float(value)
                    tables["quality"].iat[i, j] = value
        if dict(Counter(tables["state"].to_numpy().ravel())) != provenance["state_counts"]:
            raise ValueError("provenance state_counts disagree with cells")
    if type(trusted_imputation) is not bool:
        raise ValueError("trusted_imputation must be an explicit boolean")
    if not isinstance(source, dict) or set(source) & {"ecofill", "imputation", "trusted_imputation", "details"}:
        raise ValueError("source must be a mapping without reserved adapter/trust fields")
    source = owned(source)
    source["trusted_imputation"] = trusted_imputation
    source["ecofill"] = {"distribution": "ecofill", "contract_version": SUPPORTED_VERSION,
                          "route": route, "data_fingerprint": provenance["data_fingerprint"]}
    if states is not None:
        source["imputation"] = {"method_id": "ecofill:per_cell_methods_v1", "plan_id": identity(provenance["plan"])}
    if isinstance(representation, str):
        representation = {name: representation for name in names}
    _mapping(representation, names, "representation")
    metadata = {spec["name"]: {"unit": spec["unit"], "role": spec["role"],
                              "derived_from": owned(spec["derived_from"]),
                              "representation": representation[spec["name"]]}
                for spec in schema["variables"]}
    qualification = {"present": present, "quality_accepted": observed,
                     **{name: masks[name] for name in ("qc_rejected", "absent", "inserted", "structural", "artificial")}}
    return AnalysisSeries(frame, metadata=metadata, source=source, contribution=categories,
                          source_details={"provenance": provenance, "cells": tables},
                          qualification=qualification, clock={"mode": "regular", "cadence": cadence, **clock})


def from_ecofill(value, *, source, representation, clock, trusted_imputation=False):
    """Convert an installed admitted public SeriesData or ImputationResult."""
    try:
        ecofill = importlib.import_module("ecofill")
    except ModuleNotFoundError as error:
        if error.name != "ecofill":
            raise
        raise ImportError("object conversion requires EcoFill 0.0.1 on Python 3.12; exports load without EcoFill") from error
    if version("ecofill") != SUPPORTED_VERSION or ecofill.__version__ != SUPPORTED_VERSION:
        raise ValueError("only the admitted EcoFill 0.0.1 snapshot is supported")
    if not isinstance(value, (ecofill.SeriesData, ecofill.ImputationResult)):
        raise ValueError("expected a public EcoFill SeriesData or ImputationResult")
    if not isinstance(value.schema, ecofill.TimeSeriesSchema):
        raise ValueError("expected a public EcoFill TimeSeriesSchema")
    if (not isinstance(value.schema.cadence, pd.Timedelta) or not isinstance(value.schema.variables, tuple)
            or any(not isinstance(v, ecofill.VariableSpec) or not isinstance(v.derived_from, tuple)
                   for v in value.schema.variables)):
        raise ValueError("malformed public EcoFill cadence/variable/lineage schema")
    try:
        frame = value.available if isinstance(value, ecofill.SeriesData) else value.completed
        schema = {"timezone": str(frame.index.tz), "cadence_seconds": value.schema.cadence.value / 1e9,
                  "time_policy": value.schema.time_policy,
                  "variables": [{"name": v.name, "unit": v.unit, "role": v.role,
                                 "derived_from": list(v.derived_from)} for v in value.schema.variables]}
    except (AttributeError, TypeError) as error:
        raise ValueError("malformed public EcoFill schema/value contract") from error
    if isinstance(value, ecofill.SeriesData):
        text_field(value.fingerprint, "data fingerprint")
        provenance = {"schema": schema, "data_fingerprint": value.fingerprint,
                      "timestamps": [stamp.isoformat() for stamp in frame.index]}
        return _convert(frame, {name: value.mask(name) for name in BASE_MASKS}, schema, provenance,
                        source=source, representation=representation, clock=clock,
                        trusted_imputation=trusted_imputation, route="series_object")
    provenance = _json_record(value.provenance())
    _provenance(provenance)
    if provenance["schema"] != schema or _json_record(value.plan.to_dict()) != provenance["plan"]:
        raise ValueError("public schema/plan disagree with result provenance")
    return _convert(frame, {name: value.mask(name) for name in RESULT_MASKS}, schema, provenance,
                    source=source, representation=representation, clock=clock,
                    trusted_imputation=trusted_imputation, route="result_object", states=value.states,
                    methods=value.methods, reasons=value.unresolved_reasons, quality=value.quality)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _csv(path, expected):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, csv.Error, UnicodeError) as error:
        raise ValueError(f"cannot read required export file {path.name}") from error
    if not rows or len(set(rows[0])) != len(rows[0]) or rows[0] != expected:
        raise ValueError(f"{path.name} headers must exactly match the declared schema")
    if any(len(row) != len(expected) for row in rows[1:]):
        raise ValueError(f"{path.name} has malformed row widths")
    return rows[1:]


def _number(value):
    if value == "":
        return np.nan
    if value.lstrip("+-").isdigit():
        return int(value)
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError("CSV measurements/quality must be numeric or empty") from error
    if not np.isfinite(number):
        raise ValueError("nonfinite CSV number; represent missing values with empty fields")
    return number


def load_ecofill_export(directory, *, source, representation, clock, trusted_imputation=False):
    """Load format 1 without EcoFill, following only four fixed file names."""
    directory = Path(directory)
    try:
        with (directory / "provenance.json").open(encoding="utf-8") as handle:
            provenance = json.load(handle, object_pairs_hook=_unique_json, parse_constant=_nonfinite)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("cannot read required provenance.json") from error
    provenance = _json_record(provenance)
    names, _ = _provenance(provenance)
    timezone = provenance["schema"]["timezone"]
    rows = _csv(directory / "completed.csv", ["timestamp", *names])
    if not rows:
        raise ValueError("completed.csv requires nonempty timestamp rows")
    index = _index([row[0] for row in rows], timezone)
    frame = pd.DataFrame([[_number(v) for v in row[1:]] for row in rows], index=index, columns=names, dtype=object)
    inventory = provenance["masks"]
    headers = [f"{mask}__{name}" for mask in inventory for name in names]
    mask_rows = _csv(directory / "masks.csv", ["timestamp", *headers])
    mask_index = _index([row[0] for row in mask_rows], timezone)
    if not mask_index.equals(index):
        raise ValueError("masks.csv timestamps must align with completed.csv exactly")
    if any(v not in ("True", "False") for row in mask_rows for v in row[1:]):
        raise ValueError("CSV masks require explicit True/False encodings")
    mask_values = np.array([[v == "True" for v in row[1:]] for row in mask_rows], dtype=bool)
    masks = {mask: pd.DataFrame(mask_values[:, i * len(names):(i + 1) * len(names)], index=index, columns=names)
             for i, mask in enumerate(inventory)}
    fields = ["timestamp", "variable", "state", "method", "reason"]
    if "quality" in provenance:
        fields.append("quality")
    cells = _csv(directory / "cells.csv", fields)
    tables = {field: pd.DataFrame(None, index=index, columns=names, dtype=object) for field in fields[2:]}
    seen = set()
    for row in cells:
        timestamp = _index([row[0]], timezone)[0]
        key = (timestamp, row[1])
        if key in seen or timestamp not in index or row[1] not in names:
            raise ValueError("cells.csv requires unique declared timestamp/variable keys")
        seen.add(key)
        for field, value in zip(fields[2:], row[2:]):
            tables[field].at[key] = (_number(value) if field == "quality" else None if value == "" else value)
    if len(seen) != len(index) * len(names):
        raise ValueError("cells.csv is missing timestamp/variable keys")
    return _convert(frame, masks, provenance["schema"], provenance, source=source,
                    representation=representation, clock=clock, trusted_imputation=trusted_imputation,
                    route="result_export", states=tables["state"], methods=tables["method"],
                    reasons=tables["reason"], quality=tables.get("quality"))

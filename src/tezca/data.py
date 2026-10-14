"""Owned numeric observations, independent provenance layers and result records."""

import json
from numbers import Real

import numpy as np
import pandas as pd

from ._owned import identity, owned, text_field, unit_field
from .support import TimeAxis


QUALIFICATIONS = {"present", "quality_accepted", "qc_rejected", "absent", "inserted",
                  "structural", "artificial"}
CONTRIBUTIONS = {"observed", "imputed", "derived", "unavailable", "unknown"}


def aligned_array(value, frame, name, *, boolean=False):
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(frame.index) or not value.columns.equals(frame.columns):
            raise ValueError(f"{name} labels must exactly align with values")
    result = np.asarray(value)
    if result.ndim == 1 and frame.shape[1] == 1:
        result = result[:, None]
    if result.shape != frame.shape:
        raise ValueError(f"{name} shape must equal {frame.shape}")
    if boolean and result.dtype != np.dtype(bool):
        raise ValueError(f"{name} must contain booleans")
    return result.copy()


class ContributorLineage:
    """Declared aggregation provenance; no aggregation or reconstruction is performed."""

    def __init__(self, contributor_ids, origins, *, unit, operation, weights=None,
                 parents=None, details=None, observed_only_value=None, trusted_imputation=None):
        ids = tuple(contributor_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("contributor IDs must be nonempty and unique")
        for item in ids:
            text_field(item, "contributor ID")
        origins = tuple(origins)
        if len(origins) != len(ids) or set(origins) - {"observed", "imputed", "unknown"}:
            raise ValueError("origins must align with contributors and retain known/unknown origin")
        if operation not in {"linear_mean", "linear_sum", "rate_mean", "rate_integral",
                             "median", "circular_mean"}:
            raise ValueError("unsupported contributor operation")
        raw_weights = np.ones(len(ids)) if weights is None else np.asarray(weights)
        if raw_weights.dtype.kind not in "iuf" or raw_weights.shape != (len(ids),):
            raise ValueError("weights must be aligned numeric values")
        weights = raw_weights.astype(float)
        with np.errstate(over="ignore", invalid="ignore"):
            total = weights.sum()
        if not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(total) or total <= 0:
            raise ValueError("weights must be finite, nonnegative, with positive total")
        fractions = weights / total
        parents = [None] * len(ids) if parents is None else list(parents)
        if len(parents) != len(ids) or any(p is not None and not isinstance(p, ContributorLineage)
                                          for p in parents):
            raise ValueError("parents must align and contain ContributorLineage or None")
        parent_records = [p.record if p is not None else None for p in parents]
        nonlinear = operation in {"median", "circular_mean"} or any(
            p is not None and p["fraction_kind"] == "coverage" for p in parent_records)
        # Means normalize local coefficients; sums and integrals do not. Compose
        # those coefficients before normalizing exposure over the leaf inputs.
        coefficients = weights if operation in {"linear_sum", "rate_integral"} else fractions
        coefficient_total = None
        if not nonlinear:
            parent_totals = np.array([p["coefficient_total"] if p else 1.0 for p in parent_records])
            with np.errstate(over="ignore", invalid="ignore"):
                effective = coefficients * parent_totals
                coefficient_total = float(effective.sum())
            if not np.isfinite(coefficient_total) or coefficient_total <= 0:
                raise ValueError("composed linear coefficients must have a finite positive total")
            fractions = effective / coefficient_total
        reconstructed = [p["reconstructed_fraction"] if p else float(o == "imputed")
                         for p, o in zip(parent_records, origins)]
        unknown = [p["unknown_fraction"] if p else float(o == "unknown")
                   for p, o in zip(parent_records, origins)]
        if details is not None and (not isinstance(details, (list, tuple)) or len(details) != len(ids)):
            raise ValueError("contributor details must align")
        self._record = {"contributor_ids": ids, "origins": origins, "unit": unit_field(unit),
                        "operation": operation, "weights": weights.copy(), "count": len(ids),
                        "coefficients": coefficients.copy() if not nonlinear else None,
                        "coefficient_total": coefficient_total,
                        "fraction_kind": "coverage" if nonlinear else "influence",
                        "reconstructed_fraction": float(np.dot(fractions, reconstructed)),
                        "unknown_fraction": float(np.dot(fractions, unknown)),
                        "unknown_origin": any(p["unknown_origin"] if p else o == "unknown"
                                              for p, o in zip(parent_records, origins)),
                        "touched_by_imputation": any(p["touched_by_imputation"] if p else o == "imputed"
                                                     for p, o in zip(parent_records, origins))}
        if any(p is not None for p in parents):
            self._record["parents"] = parent_records
        if details is not None:
            self._record["details"] = owned(details)
        if observed_only_value is not None:
            if isinstance(observed_only_value, (bool, np.bool_)) or not isinstance(observed_only_value, Real) or not np.isfinite(observed_only_value):
                raise ValueError("observed_only_value must be finite or None")
            self._record["observed_only_value"] = float(observed_only_value)
        if trusted_imputation is not None:
            if type(trusted_imputation) is not bool:
                raise ValueError("lineage trusted_imputation must be an explicit boolean")
            self._record["trusted_imputation"] = trusted_imputation
        self._identity = identity(self._record)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return self._identity


class _SourceDetails:
    """Owned producer JSON and aligned cell columns; digest established once."""

    def __init__(self, frame, details):
        if not isinstance(details, dict) or set(details) != {"provenance", "cells"}:
            raise ValueError("source_details requires exactly provenance and cells")
        if not isinstance(details["provenance"], dict):
            raise ValueError("detail provenance must be a finite JSON mapping")
        try:
            self._provenance = json.loads(json.dumps(details["provenance"], allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("detail provenance must be a finite JSON mapping") from error
        tables = details["cells"]
        required = {"state", "method", "reason"}
        if (not isinstance(tables, dict) or (tables and not required <= set(tables))
                or set(tables) - (required | {"quality"})):
            raise ValueError("detail cell fields must be empty or state/method/reason with optional quality")
        self._index = frame.index.copy(deep=True)
        self._columns = frame.columns.copy(deep=True)
        self._cells = {}
        for field in sorted(tables):
            table = tables[field]
            if (not isinstance(table, pd.DataFrame) or not table.index.equals(frame.index)
                    or not table.columns.equals(frame.columns)):
                raise ValueError("detail cell tables must align with values exactly")
            array = table.to_numpy(dtype=object, copy=True)
            for position in np.ndindex(array.shape):
                value = array[position]
                if value is None or pd.isna(value):
                    if field == "state":
                        raise ValueError("detail state cannot be missing")
                    array[position] = None
                elif field == "quality":
                    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                        raise ValueError("detail quality must be finite numeric or missing")
                    array[position] = float(value)
                else:
                    array[position] = text_field(value, f"detail {field}")
            self._cells[field] = array
        digest = identity({"representation_version": 1, "timestamps": self._index,
                           "variables": list(self._columns), "provenance": self._provenance,
                           "cells": self._cells})
        self._summary = {"representation_version": 1, "digest": digest,
                         "rows": len(frame), "variables": len(frame.columns),
                         "cell_fields": sorted(self._cells)}

    @property
    def summary(self):
        return owned(self._summary)

    @property
    def record(self):
        index, columns = self._index.copy(deep=True), self._columns.copy(deep=True)
        return {"provenance": owned(self._provenance),
                "cells": {field: pd.DataFrame(array.copy(), index=index, columns=columns)
                          for field, array in self._cells.items()}}


class AnalysisSeries:
    """Numeric frame with declared clock, metadata, qualifications and current origin.

    Arrays require explicit variable names. Undeclared finite values retain unknown
    origin. Artificial cells may contain only an explicitly declared imputed replacement.
    Optional source_details owns producer JSON and aligned state/method/reason/
    quality tables separately; source exposes their computed versioned digest.
    """

    def __init__(self, values, *, metadata, source, time=None, variables=None, clock=None,
                 observed=None, contribution=None, qualification=None, lineage=None,
                 source_details=None):
        if isinstance(values, pd.DataFrame):
            if time is not None or variables is not None:
                raise ValueError("frame already supplies time and variable labels")
            frame = owned(values)
        else:
            array = np.asarray(values)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or time is None or variables is None:
                raise ValueError("arrays require 1D/2D values, time and variable names")
            frame = pd.DataFrame(array.copy(), index=time, columns=variables)
        if not frame.columns.is_unique or not len(frame.columns):
            raise ValueError("variable labels must be nonempty and unique")
        for variable in frame.columns:
            text_field(variable, "variable")
            if frame[variable].dtype == object:
                present = frame[variable].dropna()
                if all(isinstance(item, Real) and not isinstance(item, (bool, np.bool_)) for item in present):
                    frame[variable] = pd.array(frame[variable], dtype="Float64")
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("analysis values must be real numeric columns")
        numeric = frame.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric).any():
            raise ValueError("infinite input values are invalid; declare missing values explicitly")
        axis = TimeAxis(frame.index, **({} if clock is None else clock))
        if not isinstance(metadata, dict) or set(metadata) != set(frame.columns):
            raise ValueError("metadata must describe exactly the input variables")
        for variable, record in metadata.items():
            if not isinstance(record, dict):
                raise ValueError("variable metadata must be mappings")
            unit_field(record.get("unit"))
            text_field(record.get("representation"), f"{variable}.representation")
        if not isinstance(source, dict):
            raise ValueError("source must declare its identity and segment or site/station/instrument")
        text_field(source.get("id"), "source.id")
        if "segment" in source:
            text_field(source["segment"], "source.segment")
        else:
            for field in ("site", "station", "instrument"):
                text_field(source.get(field), f"source.{field}")
        if "trusted_imputation" in source and type(source["trusted_imputation"]) is not bool:
            raise ValueError("trusted_imputation must be an explicit boolean declaration")
        finite = np.isfinite(numeric)
        if observed is not None and contribution is not None:
            raise ValueError("supply observed or contribution, not contradictory declarations")
        categories = np.where(finite, "unknown", "unavailable")
        if observed is not None:
            declared = aligned_array(observed, frame, "observed", boolean=True)
            if (declared & ~finite).any():
                raise ValueError("observed cells must have finite values")
            categories[declared] = "observed"
        if contribution is not None:
            categories = aligned_array(contribution, frame, "contribution").astype(str)
            if set(categories.ravel()) - CONTRIBUTIONS:
                raise ValueError("unknown contribution category")
            if np.any((categories != "unavailable") != finite):
                raise ValueError("finite values and available contribution categories must agree")
        # Fixed (sorted) flag order: set iteration varies with the process hash
        # seed and saved typed maps keep insertion order.
        flags = {name: np.zeros(frame.shape, dtype=bool) for name in sorted(QUALIFICATIONS)}
        flags["present"] = categories == "observed"
        flags["quality_accepted"] = categories == "observed"
        flags["absent"] = ~finite
        if qualification is not None:
            if not isinstance(qualification, dict) or set(qualification) - QUALIFICATIONS:
                raise ValueError("unknown qualification flag")
            for name, mask in qualification.items():
                flags[name] = aligned_array(mask, frame, name, boolean=True)
        if (flags["present"] & flags["absent"]).any():
            raise ValueError("original present and absent qualifications contradict")
        if (flags["quality_accepted"] & (flags["qc_rejected"] | flags["structural"])).any():
            raise ValueError("accepted qualification contradicts rejected/structural qualification")
        invalid_observed = flags["artificial"] | flags["qc_rejected"] | flags["structural"]
        invalid_observed |= flags["absent"] | flags["inserted"]
        if ((categories == "observed") & invalid_observed).any():
            raise ValueError("ineligible original qualifications cannot contribute as observations")
        if ((categories == "observed") & (~flags["present"] | ~flags["quality_accepted"])).any():
            raise ValueError("observation declarations require present, quality-accepted qualification")
        if (flags["artificial"] & finite & (categories != "imputed")).any():
            raise ValueError("finite artificial truth is ambiguous; supply an explicit imputed replacement")
        if (flags["structural"] & finite).any():
            raise ValueError("structural absence cannot contribute a finite value")
        if (categories == "imputed").any():
            imputation = source.get("imputation", {})
            if not isinstance(imputation, dict):
                raise ValueError("imputation provenance must be a mapping")
            for field in ("method_id", "plan_id"):
                text_field(imputation.get(field), f"source.imputation.{field}")
        lineages = {} if lineage is None else owned(lineage)
        if not isinstance(lineages, dict) or set(lineages) - set(frame.columns):
            raise ValueError("lineage keys must reference input variables")
        reconstructed = (categories == "imputed").astype(float)
        unknown = (categories == "unknown").astype(float)
        unknown_touched = categories == "unknown"
        touched = categories == "imputed"
        trusted = np.full(frame.shape, source.get("trusted_imputation", False), dtype=bool)
        operation = np.full(frame.shape, "identity", dtype=object)
        fraction_kind = np.full(frame.shape, "influence", dtype=object)
        for j, variable in enumerate(frame.columns):
            records = lineages.get(variable, [None] * len(frame))
            if not isinstance(records, (list, tuple)) or len(records) != len(frame):
                raise ValueError("lineage records must align with rows")
            for i, record in enumerate(records):
                if categories[i, j] == "derived":
                    if not isinstance(record, ContributorLineage):
                        raise ValueError("derived contributions require contributor lineage")
                    details = record.record
                    if details["unit"] != metadata[variable]["unit"]:
                        raise ValueError("lineage parent units must match the declared output unit")
                    reconstructed[i, j] = details["reconstructed_fraction"]
                    unknown[i, j] = details["unknown_fraction"]
                    unknown_touched[i, j] = details["unknown_origin"]
                    touched[i, j] = details["touched_by_imputation"]
                    operation[i, j] = details["operation"]
                    fraction_kind[i, j] = details["fraction_kind"]
                    trusted[i, j] = details.get("trusted_imputation", trusted[i, j])
                elif record is not None:
                    raise ValueError("aggregation lineage requires a derived contribution")
        self._frame, self._axis = frame, axis
        self._metadata, self._source = owned(metadata), owned(source)
        self._source_details = None
        if source_details is not None:
            if "details" in source:
                raise ValueError("source.details is reserved for the computed detail summary")
            self._source_details = _SourceDetails(frame, source_details)
            self._source["details"] = self._source_details.summary
        self._contribution, self._qualification, self._lineage = categories.copy(), flags, lineages
        self._reconstructed, self._unknown, self._touched = reconstructed, unknown, touched
        self._unknown_touched = unknown_touched
        self._trusted = trusted
        self._operation, self._fraction_kind = operation, fraction_kind
        self._identity = identity({"values": numeric, "time": axis.index, "clock": axis.record,
                                   "variables": list(frame.columns), "metadata": metadata,
                                   "source": self._source, "contribution": categories,
                                   "qualification": flags,
                                   "lineage": {v: [r.identity if r else None for r in rows]
                                               for v, rows in lineages.items()}})

    @property
    def values(self):
        return owned(self._frame)

    @property
    def time(self):
        return owned(self._axis)

    @property
    def metadata(self):
        return owned(self._metadata)

    @property
    def source(self):
        return owned(self._source)

    @property
    def source_details(self):
        """Defensive copies of full producer metadata and aligned cell tables, or None."""
        return None if self._source_details is None else self._source_details.record

    @property
    def contribution(self):
        return self._contribution.copy()

    @property
    def qualification(self):
        return owned(self._qualification)

    @property
    def lineage(self):
        return owned(self._lineage)

    @property
    def origin(self):
        return {"reconstructed": self._reconstructed.copy(), "unknown": self._unknown.copy(),
                "touched": self._touched.copy(), "unknown_touched": self._unknown_touched.copy(),
                "operation": self._operation.copy(), "fraction_kind": self._fraction_kind.copy()}

    @property
    def identity(self):
        return self._identity

    def eligible(self, policy):
        finite = self._contribution != "unavailable"
        if policy.values == "provided":
            return finite.copy()
        known_observed = finite & ~self._unknown_touched & (self._reconstructed == 0)
        # Nonlinear coverage cannot certify that a touched output is observed.
        known_observed &= ~self._touched
        if policy.values == "observed_only":
            return known_observed
        return finite & ~self._unknown_touched & (known_observed | self._trusted)


class ResultRecord:
    """Owned canonical result; configuration errors and missing data are distinct."""

    def __init__(self, value, *, compute_status, metric_status, use_status, support, units,
                 method_id, config_id, reference_id=None, reason=None, representation="raw"):
        for name, item in (("compute_status", compute_status), ("metric_status", metric_status),
                           ("use_status", use_status), ("method_id", method_id),
                           ("config_id", config_id), ("representation", representation)):
            text_field(item, name)
        if reference_id is not None:
            text_field(reference_id, "reference_id")
        if compute_status == "ok":
            if value is None:
                raise ValueError("computed results require a value")
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError("computed scalar results must be finite")
        elif value is not None or not isinstance(reason, str) or not reason.strip():
            raise ValueError("missing results require value=None and an explicit reason")
        if isinstance(units, dict):
            for unit in units.values():
                unit_field(unit)
        else:
            unit_field(units)
        if not isinstance(support, dict):
            raise ValueError("result support must be an explicit mapping")
        self._record = owned(dict(value=value, compute_status=compute_status,
                                  metric_status=metric_status, use_status=use_status,
                                  support=support, units=units, method_id=method_id,
                                  config_id=config_id, reference_id=reference_id, reason=reason,
                                  representation=representation))

    @property
    def value(self):
        return owned(self._record["value"])

    def summary(self):
        return owned(self._record)

    def describe(self):
        return self.summary()

    def to_frame(self):
        return pd.DataFrame([self.summary()])

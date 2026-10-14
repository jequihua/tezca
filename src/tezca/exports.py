"""Explicit numerical result export and calculation-free validated reload.

See docs/exports.md for the closed typed-cell format and scientific boundary.
Only adapters of existing result records run here, never an estimator or fitter.
"""

import csv
import functools
import hashlib
import importlib.metadata
import io
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from ._owned import owned
from ._owned import identity as scientific_identity
from ._result_codec import encode, decode, dumps, loads, digest, encoded_text
from ._compat_columns import COLUMNS as COMPAT_COLUMNS
from ._export_validation import validate_backend, validate_item, validate_keys, validate_source
from .data import AnalysisSeries, ResultRecord
from .individual import DailyResidual
from .reference import FrozenReference
from .regimes import RegimeReference
from .collective import PreparedCollective
from .fi1 import FI1Trajectory
from .graph_state import GraphGridResult
from .network import NetworkResult
from .regimes import RegimeResult
from .sensitivity import SensitivityReport


FORMAT = "tezca.results/1"
FAMILIES = (
    "individual_timeseries_metrics.csv", "spectral_peaks.csv", "example_spectra_long.csv",
    "site_collective_summary.csv", "pca_spectrum.csv", "site_mode_loadings.csv",
    "site_cross_spectral_pairs.csv", "cross_spectra_long.csv", "spectral_gapfill_sensitivity.csv",
    "site_matrix_audit.csv", "matrix_support.csv", "fi1_windows.csv", "dynamic_connectivity.csv",
    "network_edges.csv", "density_fisher.csv", "graph_state_sensitivity.csv", "regime_labels.csv")
KINDS = {
    "metric": (ResultRecord, (FAMILIES[0],)),
    "peaks": (ResultRecord, (FAMILIES[1],)),
    "spectrum": (ResultRecord, (FAMILIES[2],)),
    "pca": (ResultRecord, FAMILIES[3:6]),
    "coherence": (ResultRecord, FAMILIES[6:8]),
    "sensitivity": (SensitivityReport, (FAMILIES[8],)),
    "matrix": (PreparedCollective, FAMILIES[9:11]),
    "fi1": (FI1Trajectory, (FAMILIES[0], FAMILIES[11])),
    "network": (NetworkResult, FAMILIES[12:14]),
    "density": (ResultRecord, (FAMILIES[14],)),
    "graph": (GraphGridResult, (FAMILIES[15],)),
    "regimes": (RegimeResult, (FAMILIES[16],)),
}
COLUMNS = ["result_id", "row_key", "metric_id", "compute_status", "metric_status", "use_status",
           "reason", "representation", "units", "config_id", "reference_id", "source_ids", "payload"]
SCHEMA = dict(version=1, columns=COLUMNS, keys=["result_id", "row_key"],
              types=["str", "typed_identity", "str", "str_or_null", "str_or_null", "str_or_null",
                     "str_or_null", "str_or_null", "unit_symbol_or_map_or_null", "str_or_null",
                     "str_or_null", "list[str]", "typed_record"],
              order="input result insertion order, then native row/component/frequency order",
              payload="closed typed tree; exact family projection bound to result_details.json")
CONVENTIONS = dict(encoding="UTF-8", newline="LF", csv="RFC4180 quoting; each cell is a typed JSON node",
    float="binary64 hexadecimal; exact finite roundtrip; explicit nan/inf/-inf tokens",
    missing="null, na, nat and float nan are distinct; empty strings and NA strings are strings",
    integer="base10 string with unbounded integer precision", boolean="bool tag and JSON boolean",
    complex="complex tag with separate signed real/imaginary binary64 components",
    time="integer nanoseconds plus timezone; naive timestamps remain explicitly naive",
    labels="typed unique scalar/tuple labels; ambiguous, missing and MultiIndex labels refused",
    dataframe="object cells after reload; logical scalar types and axis order retained, no dtype inference",
    identity="no path or wall-clock timestamp; hashes establish integrity, not authenticity or rights",
    freshness="expected identities compare supplied snapshots only; absent expectations cannot prove external freshness")


def _records(frame):
    # iterrows/to_dict can coerce a large integer label through binary64.
    return [{c: frame.iloc[i, j] for j, c in enumerate(frame.columns)} for i in range(len(frame))]


def _matrix_audit(frame):
    rows = _records(frame)
    for row in rows:
        for station in row["stations"]:
            # Measurements are input data, not analysis result exports.
            station.pop("value", None)
            station.pop("observed_only_value", None)
    return rows


class ResultItem:
    """Owned snapshot of one public result, with explicit source bindings/labels.

    `kind` selects one of the finite adapters above. `labels` are descriptive
    context (site/station/segment etc.), never a replacement scientific record.
    `timeline` optionally saves the already computed regime timeline.
    """

    def __init__(self, kind, result, *, sources, labels=None, timeline=None):
        if kind not in KINDS or not isinstance(result, KINDS[kind][0]):
            raise ValueError("result type does not match supported export kind")
        if not isinstance(sources, (tuple, list)) or not sources or any(type(s) is not str or not s for s in sources) or len(set(sources)) != len(sources):
            raise ValueError("sources must be ordered unique source names")
        record = result.describe()
        if kind == "regimes":
            # Measured wall time is operational, not scientific identity.
            record.pop("cost_seconds", None)
        data = dict(kind=kind, sources=list(sources), labels=owned(labels or {}), record=record)
        if kind == "matrix":
            data["rows"] = _matrix_audit(result.audit)
        elif kind == "fi1":
            data["rows"] = _records(result.table)
        elif kind == "graph":
            data["rows"] = _records(result.table)
        elif kind == "network":
            data.update(rows=_records(result.summary), edges=_records(result.edges))
        elif kind == "regimes":
            data.update(rows=_records(result.rows), timeline=owned(timeline), reference=result.reference.describe())
        elif kind == "sensitivity":
            data.update(routes=_records(result.routes), contrasts=_records(result.contrasts))
        elif timeline is not None:
            raise ValueError("timeline is only supported for regime results")
        # Reject unsupported metadata before touching any destination.
        self._data = decode(encode(data))

    def describe(self):
        return owned(self._data)


def _row(name, item, key, payload, record=None, metric=None):
    r = item["record"] if record is None else record
    return dict(result_id=name, row_key=key, metric_id=metric or r.get("method_id", item["kind"]),
        compute_status=r.get("compute_status"), metric_status=r.get("metric_status"),
        use_status=r.get("use_status"), reason=r.get("reason"),
        representation=r.get("representation", r.get("basis")), units=r.get("units"),
        config_id=r.get("config_id"), reference_id=r.get("reference_id"), source_ids=item["sources"], payload=payload)


def _tables(items):
    tables = {f: [] for f in FAMILIES}
    for name, item in items.items():
        kind, r = item["kind"], item["record"]
        value = r.get("value")
        def add(family, key, payload, record=None, metric=None):
            tables[family].append(_row(name, item, key, payload, record, metric))
        if kind in {"metric", "density"}:
            add(KINDS[kind][1][0], 0, r)
        elif kind == "peaks":
            if value is not None:
                for i, p in enumerate(_records(value["table"])):
                    add(FAMILIES[1], i, p)
        elif kind == "spectrum":
            if value is not None:
                for i, f in enumerate(value["frequency_hz"]):
                    p = {k: v[i] for k, v in value.items() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(value["frequency_hz"]) and not k.startswith("physical_")}
                    if "frequency_support" in r["support"]:
                        p["qualification"] = r["support"]["frequency_support"][i]
                    add(FAMILIES[2], i, p)
            else:
                for i, p in enumerate(r["support"].get("frequency_support", [])):
                    add(FAMILIES[2], i, p, dict(r, **p))
        elif kind == "pca":
            add(FAMILIES[3], 0, r)
            if value is not None:
                for i, eigenvalue in enumerate(value["eigenvalues"]):
                    add(FAMILIES[4], i + 1, dict(component=i+1, eigenvalue=eigenvalue, fraction=value["fractions"][i]))
                    for j, feature in enumerate(r["support"]["requested"]["features"]):
                        add(FAMILIES[5], (i+1, feature), dict(component=i+1, feature=feature,
                            eigenvector=value["eigenvectors"][j, i], scaled_loading=value["scaled_loadings"][j, i],
                            eigenvector_definition=value["eigenvector_definition"], scaled_loading_definition=value["scaled_loading_definition"]))
        elif kind == "coherence":
            add(FAMILIES[6], 0, r)
            support = r["support"]
            # Runs and aggregate/pool remain separate, including unavailable summaries.
            groups = [("run", i, run) for i, run in enumerate(support.get("runs", []))]
            if "pooling" in support:
                pool = support["pooling"]
                groups.append(("pooled", 0, dict(pool.get("spectra") or {}, **(pool.get("inference") or {}))))
            if "aggregation" in support:
                groups.append(("equal_run_aggregate", 0, support["aggregation"]))
            for route, run, data in groups:
                for i, f in enumerate(data.get("frequency_hz", [])):
                    p = dict(route=route, run=run, frequency_hz=f)
                    for k, v in data.items():
                        if isinstance(v, (list, np.ndarray)) and len(v) == len(data["frequency_hz"]):
                            p[k] = v[i]
                    add(FAMILIES[7], (route, run, i), p)
        elif kind == "sensitivity":
            for category in ("routes", "contrasts"):
                for i, row in enumerate(item[category]):
                    add(FAMILIES[8], (category, i), dict(category=category, **row), row)
        elif kind == "matrix":
            add(FAMILIES[9], "selection", r)
            for i, row in enumerate(item["rows"]):
                add(FAMILIES[10], (row["feature"], i), row, dict(r, **row))
        elif kind in {"fi1", "network", "graph", "regimes"}:
            target = {"fi1": FAMILIES[11], "network": FAMILIES[12], "graph": FAMILIES[15], "regimes": FAMILIES[16]}[kind]
            if kind == "fi1":
                add(FAMILIES[0], "summary", r)
            for i, row in enumerate(item["rows"]):
                merged = dict(r, **row)
                if kind == "graph":
                    merged.update(row["result"])
                add(target, row["row_id"] if kind == "regimes" else i, row, merged)
            if kind == "network":
                for i, row in enumerate(item["edges"]):
                    add(FAMILIES[13], i, row, dict(r, **row))
    return tables


def _source(series):
    if isinstance(series, DailyResidual):
        base = _source(series._series)
        base.update(base_identity=base["identity"], identity=series.identity,
            representation=series.describe(),
            signal_identities={v: scientific_identity(dict(input=series.identity, variable=v, basis="daily_residual"))
                               for v in series.values.columns})
        return base
    if not isinstance(series, AnalysisSeries):
        raise ValueError("sources must contain AnalysisSeries or DailyResidual snapshots")
    source, metadata, source_details = series.source, series.metadata, series.source_details
    return dict(identity=series.identity, source=source, metadata=metadata, clock=series.time.record,
        signal_identities={v: scientific_identity(dict(input=series.identity, variable=v, basis=m["representation"]))
                           for v, m in metadata.items()},
        mask_hash=digest(encode(dict(qualification=series.qualification, contribution=series.contribution,
                                    origin=series.origin))),
        source_hash=digest(encode(source)),
        details_hash=None if source_details is None else digest(encode(source_details)))


def _csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([encoded_text(row[c]) for c in columns])
    return stream.getvalue().encode("utf-8")


def _compatibility(items, tables):
    """Preserve frozen layouts and explain every cell; no historical formula guess."""
    output, sidecar = {}, []
    direct = {"individual.mean_v1": "mean", "individual.median_v1": "median",
        "individual.sample_sd_v1": "sd", "individual.iqr_v1": "iqr",
        "individual.minimum_v1": "minimum", "individual.maximum_v1": "maximum",
        "individual.elapsed_day_ols_v1": "trend_per_day"}
    for family, columns in COMPAT_COLUMNS.items():
        rows = []
        # Each independent result stays separate: never merge different support/bases.
        for i, row in enumerate(tables[family]):
            item, payload = items[row["result_id"]], row["payload"]
            mapped = {c: None for c in columns}
            mapping = {}
            for c in columns:
                if c in item["labels"] and c in {"family", "site", "station", "segment_id", "variable"}:
                    mapped[c], mapping[c] = item["labels"][c], "caller_label"
            if "source" in mapped:
                mapped["source"], mapping["source"] = item["sources"][0], "source_binding"
            if family == FAMILIES[0]:
                field = direct.get(row["metric_id"])
                if field:
                    mapped[field], mapping[field] = payload["value"], row["metric_id"]
                if item["kind"] == "metric":
                    mapped["unit"], mapping["unit"] = row["units"], "result.units"
                    mapped["analysis_status"], mapping["analysis_status"] = row["compute_status"], "result.compute_status_only"
            # Independent fits, bands, harmonics, FI1, irregular/coherence and
            # corrected networks have no admitted historical formula identity.
            for c in columns:
                reason = ("not_requested_or_historical_formula_unverified_Q001" if c not in mapping else row["reason"])
                if c.endswith("72h") and item["kind"] == "fi1":
                    reason = "historical_FI1_formula_unverified_Q001"
                    if pd.Timedelta(item["record"]["window_settings"]["window"]) != pd.Timedelta("72h"):
                        reason = "non_72_hour_trajectory"
                sidecar.append(dict(table=family, row=i, field=c, result_id=row["result_id"], row_key=row["row_key"],
                    mapping=mapping.get(c), formula=row["metric_id"] if c in mapping else None,
                    compute_status=row["compute_status"] if c in mapping else "unavailable",
                    metric_status=row["metric_status"], use_status=row["use_status"], reason=reason,
                    canonical_table=family, parity="unverified_Q001"))
            rows.append(mapped)
        output["compat_" + family] = _csv_bytes(columns, rows)
    output["compatibility_status.json"] = encoded_text(sidecar).encode("utf-8")
    return output


def _payloads(details, compatibility, encoded, text=None):
    # `text` is the already checked canonical JSON of `encoded`, when available.
    tables = _tables(details["results"])
    output = {f: _csv_bytes(COLUMNS, rows) for f, rows in tables.items()}
    output["result_details.json"] = (dumps(encoded) if text is None else text).encode("utf-8")
    if compatibility:
        output.update(_compatibility(details["results"], tables))
    return tables, output


def _manifest(details, tables, payloads, encoded, compatibility, unavailable):
    # These exact text-key maps are already encoded for result_details.json.
    # Reuse their tagged subtrees rather than encoding every result again.
    parts = {key[1]: value for key, value in encoded[1]}
    identities = {key: digest(parts[key]) for key in ("sources", "references", "requested_policy", "resolved_policy", "backend")}
    identities["mask"] = digest(encode({n: item["mask_hash"] for n, item in details["sources"].items()}))
    identities["formula"] = digest(encode({n: dict(method=item["record"].get("method_id"),
        config=item["record"].get("config_id"), kind=item["kind"]) for n, item in details["results"].items()}))
    families = {}
    for family in FAMILIES:
        relevant = [name for name, item in details["results"].items() if family in KINDS[item["kind"]][1]]
        families[family] = dict(schema=SCHEMA, rows=len(tables[family]), result_ids=relevant,
            status="requested" if relevant else "unavailable" if family in unavailable else "not_requested",
            reason=unavailable.get(family, None if relevant else "no result requested"))
    m = dict(format=FORMAT, tezca_version=__version__, compatibility=compatibility,
        conventions=CONVENTIONS, families=families, identities=identities,
        result_hashes={key[1]: digest(value) for key, value in parts["results"][1]},
        source_identities={n: v["identity"] for n, v in details["sources"].items()},
        reference_identities={n: v["identity"] for n, v in details["references"].items()},
        results={n: dict(kind=i["kind"], sources=i["sources"], families=list(KINDS[i["kind"]][1]),
            compute_status=i["record"].get("compute_status"), metric_status=i["record"].get("metric_status"),
            use_status=i["record"].get("use_status"), reason=i["record"].get("reason"),
            reference_id=i["record"].get("reference_id")) for n, i in details["results"].items()},
        files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for name, data in payloads.items()},
        unavailable=unavailable, historical_parity="unverified_Q001", complete=True)
    m["bundle_id"] = digest(m)
    return m


def _validate_details(details):
    try:
        _check_details(details)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("invalid result schema: malformed record structure") from exc


def _check_details(details):
    if (not isinstance(details, dict) or set(details) != {"results", "sources", "references", "requested_policy", "resolved_policy", "backend"}
            or any(not isinstance(v, dict) for v in details.values())):
        raise ValueError("invalid details schema")
    # Producer-written envelope shapes are mandatory even when no result
    # happens to index a field; omitted raw cells are never reconstructed.
    validate_backend(details["backend"], __version__)
    for name, source in details["sources"].items():
        validate_source(source)
        if (type(name) is not str or not name or not isinstance(source, dict)
                or not isinstance(source.get("identity"), str)
                or not isinstance(source.get("signal_identities"), dict)
                or not isinstance(source.get("source"), dict)
                or source.get("source_hash") != digest(encode(source["source"]))):
            raise ValueError("invalid source schema/content hash")
        metadata = source["metadata"]
        residual = source.get("representation")
        if residual is not None:
            if (residual["basis"] != "daily_residual" or residual["source_id"] != source["base_identity"]
                    or scientific_identity(residual["fit"]) != residual["reference_id"]):
                raise ValueError("source representation/reference binding mismatch")
        expected_signals = {v: scientific_identity(dict(input=source["identity"], variable=v,
            basis="daily_residual" if residual is not None else m["representation"])) for v, m in metadata.items()}
        if source["signal_identities"] != expected_signals:
            raise ValueError("source/signal identity binding mismatch")
    for ref, value in details["references"].items():
        if (not isinstance(value, dict) or set(value) != {"identity", "record", "content_hash"}
                or not isinstance(value["record"], dict) or ref != value["identity"] or scientific_identity(value["record"]) != ref
                or digest(encode(value["record"])) != value["content_hash"]):
            raise ValueError("reference identity/content mismatch")
    references = {ref:value["record"] for ref,value in details["references"].items()}
    for name, item in details["results"].items():
        if type(name) is not str or not name or not isinstance(item, dict) or item.get("kind") not in KINDS:
            raise ValueError("invalid result identity/kind")
        validate_item(item, details["sources"], references)
    validate_keys(_tables(details["results"]))


@functools.cache
def _software():
    # Installed numerical library versions are fixed for a running process.
    return tuple((name, importlib.metadata.version(name)) for name in ("numpy", "pandas", "scipy"))


def export_results(destination, *, results, sources, references=None, requested_policy=None,
                   resolved_policy=None, backend=None, compatibility=False, unavailable=None):
    """Write a NEW directory; write the manifest last as the completion marker.

    A failed write deliberately leaves an incomplete, non-overwritable directory
    for diagnosis. The parent directory must already exist. No cleanup or analysis.
    """
    if type(compatibility) is not bool:
        raise ValueError("compatibility must be boolean")
    refs = {}
    for reference in references or []:
        if not isinstance(reference, (FrozenReference, RegimeReference)):
            raise ValueError("references must be public frozen numerical references")
        record = reference.describe()
        ref = reference.identity
        if ref in refs:
            raise ValueError("duplicate reference")
        refs[ref] = dict(identity=ref, record=record, content_hash=digest(encode(record)))
    # Validation and encoding only read each item's private canonical snapshot;
    # decoding below creates the independently owned objects used for output.
    details = dict(results={n: item._data for n, item in results.items() if isinstance(item, ResultItem)},
        sources={n: _source(source) for n, source in sources.items()}, references=refs,
        requested_policy=owned(requested_policy or {}), resolved_policy=owned(resolved_policy or {}),
        backend=dict(software=dict(_software()),
                     python=platform.python_version(), tezca=__version__, declared=owned(backend or {})))
    if len(details["results"]) != len(results):
        raise ValueError("results must contain ResultItem values")
    # Refuse malformed native snapshots before the expensive typed round-trip.
    # The subsequent canonical codec check still proves that normalization
    # preserves every tagged value and its structure before any file is written.
    _validate_details(details)
    encoded = encode(details)
    text = dumps(encoded)
    details = decode(encoded, text)
    unavailable = owned(unavailable or {})
    if any(k not in FAMILIES or type(v) is not str or not v for k, v in unavailable.items()):
        raise ValueError("unavailable must map family filenames to explicit reasons")
    tables, payloads = _payloads(details, compatibility, encoded, text)
    if any(unavailable.get(f) for f in FAMILIES if any(f in KINDS[i["kind"]][1] for i in details["results"].values())):
        raise ValueError("family cannot be both supplied and declared unavailable")
    manifest = _manifest(details, tables, payloads, encoded, compatibility, unavailable)
    path = Path(destination)
    path.mkdir()  # exclusive; includes existing empty directories and symlinks
    for name, data in payloads.items():
        with (path / name).open("xb") as stream:
            stream.write(data)
    with (path / "analysis_manifest.json").open("xb") as stream:
        stream.write(dumps(manifest).encode("utf-8"))
    return owned(manifest)


class LoadedResults:
    def __init__(self, manifest, details, tables):
        self._manifest, self._details, self._tables = owned(manifest), owned(details), owned(tables)

    @classmethod
    def _adopt(cls, manifest, details, tables):
        # The loader's freshly decoded objects have no other references, so it
        # need not copy them; every accessor still returns an independent copy.
        loaded = cls.__new__(cls)
        loaded._manifest, loaded._details, loaded._tables = manifest, details, tables
        return loaded

    @property
    def manifest(self):
        return owned(self._manifest)

    @property
    def details(self):
        return owned(self._details)

    def table(self, family):
        if family not in FAMILIES:
            raise ValueError("unknown canonical table")
        return pd.DataFrame(owned(self._tables[family]), columns=COLUMNS, dtype=object)


def load_results(directory, *, expected_bundle_id=None, expected_sources=None, expected_references=None):
    """Validate the entire closed bundle before returning owned numerical records.

    Expected source/reference mappings compare exactly (including selection).
    No expectations means internal consistency only, never external freshness.
    """
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bundle must be a real directory")
    manifest_path = path / "analysis_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("incomplete bundle: missing regular manifest")
    manifest = loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("tezca_version") != __version__:
        raise ValueError("unsupported result format/version")
    if type(manifest.get("compatibility")) is not bool:
        raise ValueError("invalid compatibility schema")
    expected_files = set(FAMILIES) | {"result_details.json"}
    if manifest["compatibility"]:
        expected_files |= {"compat_" + f for f in COMPAT_COLUMNS} | {"compatibility_status.json"}
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != expected_files:
        raise ValueError("invalid manifest file allowlist")
    # Validate ALL names/types before opening any manifest-selected result file.
    resolved_path = path.resolve()
    for name in expected_files:
        candidate = path / name
        if candidate.is_symlink() or not candidate.is_file() or candidate.resolve().parent != resolved_path:
            raise ValueError("missing or unsafe result file")
    if set(p.name for p in path.iterdir()) != expected_files | {"analysis_manifest.json"}:
        raise ValueError("unexpected bundle files")
    check = dict(manifest)
    bundle_id = check.pop("bundle_id", None)
    if digest(check) != bundle_id:
        raise ValueError("manifest identity mismatch")
    for expected, actual, label in ((expected_bundle_id, bundle_id, "bundle"),
            (expected_sources, manifest.get("source_identities"), "source"),
            (expected_references, manifest.get("reference_identities"), "reference")):
        if expected is not None and expected != actual:
            raise ValueError("stale " + label + " selection")
    def read_payload(name):
        data = (path / name).read_bytes()
        if manifest["files"][name] != dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)):
            raise ValueError("file hash/size mismatch: " + name)
        return data
    # Reject malformed numerical records before reading their table projections.
    # Successful loads still check every file and regenerate every projection.
    payloads = {"result_details.json": read_payload("result_details.json")}
    # The details text itself must be the canonical serialization of its value.
    text = payloads["result_details.json"].decode("utf-8")
    encoded = loads(text)
    details = decode(encoded, text)
    _validate_details(details)
    payloads.update((name, read_payload(name)) for name in expected_files if name != "result_details.json")
    tables, canonical = _payloads(details, manifest["compatibility"], encoded, text)
    if canonical != payloads:
        raise ValueError("table schema/key/result binding mismatch")
    rebuilt = _manifest(details, tables, canonical, encoded, manifest["compatibility"], manifest["unavailable"])
    if rebuilt != manifest:
        raise ValueError("manifest schema/scientific binding mismatch")
    return LoadedResults._adopt(manifest, details, tables)

"""Closed, non-executable value vocabulary for saved numerical results."""

import hashlib
import json
from json.encoder import encode_basestring as _quote
import math

import numpy as np
import pandas as pd


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def loads(value):
    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=invalid)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def _label(value):
    if value is None or value is pd.NA or value is pd.NaT:
        raise ValueError("missing labels are unsupported")
    if isinstance(value, tuple):
        return ["tuple", [_label(v) for v in value]]
    if isinstance(value, (str, int, bool, np.integer, np.bool_, pd.Timestamp)):
        return encode(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(value):
        return encode(value)
    raise ValueError("unsupported/lossy label type: " + type(value).__name__)


def _labels(values):
    # Most scientific metadata maps use only text keys. Their tagged identity
    # is injective, so JSON-serializing each key again adds no validation.
    if all(type(v) is str for v in values):
        if len(set(values)) != len(values):
            raise ValueError("duplicate typed labels")
        return [["str", v] for v in values]
    result = [_label(v) for v in values]
    if len({dumps(v) for v in result}) != len(result):
        raise ValueError("duplicate typed labels")
    # pandas/Python equate True, 1 and 1.0; do not create ambiguous identities.
    if len(set(values)) != len(values):
        raise ValueError("colliding labels")
    return result


def encode(value):
    # Primitive metadata dominates result snapshots. Handle exact built-in
    # types before the pandas/NumPy checks without coercing subclasses.
    kind = type(value)
    if kind is str:
        return ["str", value]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", str(value)]
    if kind is float:
        return ["float", "nan" if math.isnan(value) else value.hex()]
    if kind is complex:
        return ["complex", encode(value.real), encode(value.imag)]
    if value is None:
        return ["null"]
    if value is pd.NA:
        return ["na"]
    if value is pd.NaT:
        return ["nat"]
    if kind is dict:
        # Exact text-key dictionaries already have unique keys; their typed
        # labels are injective. Other key types still need collision checks.
        if all(type(key) is str for key in value):
            return ["map", [[["str", key], encode(v)] for key, v in value.items()]]
        keys = _labels(list(value))
        return ["map", [[key, encode(v)] for key, v in zip(keys, value.values())]]
    if kind in (tuple, list):
        return ["tuple" if kind is tuple else "list", [encode(v) for v in value]]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", str(value.value), str(value.tz) if value.tz else None]
    if isinstance(value, pd.Timedelta):
        return ["duration", str(value.value)]
    if isinstance(value, np.generic):
        if value.dtype.kind in "mM":
            raise ValueError("use pandas Timestamp/Timedelta for nanosecond time values")
        return encode(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufcUO":
            raise ValueError("unsupported array dtype: " + str(value.dtype))
        # tolist converts each numeric element directly, without a second
        # recursive encode through np.generic. It never combines column types.
        return ["array", str(value.dtype), list(value.shape), [encode(v) for v in value.reshape(-1).tolist()]]
    if isinstance(value, pd.MultiIndex):
        raise ValueError("MultiIndex is unsupported; use explicit tuple labels on an Index")
    if isinstance(value, pd.Index):
        return ["index", _labels(value.tolist()), encode(value.name),
                str(value.tz) if isinstance(value, pd.DatetimeIndex) else None]
    if isinstance(value, pd.DataFrame):
        return ["frame", encode(value.index), encode(value.columns),
                [[encode(cell) for cell in row[1:]] for row in value.itertuples(index=True, name=None)],
                encode(value.attrs)]
    if isinstance(value, pd.Series):
        return ["series", encode(value.index), encode(value.name), [encode(v) for v in value], encode(value.attrs)]
    raise ValueError("unsupported result metadata type: " + type(value).__name__)


def encoded_text(value):
    """Exactly `dumps(encode(value))` without building nested node lists.

    Exact built-in scalars and text-key/sequence containers follow encode's
    branches above; every other value is delegated to encode itself. Large
    temporary node trees otherwise dominate canonical checks via GC work.
    """
    kind = type(value)
    if kind is str:
        return '["str",' + _quote(value) + "]"
    if kind is bool:
        return '["bool",true]' if value else '["bool",false]'
    if kind is int:
        return '["int",' + _quote(str(value)) + "]"
    if kind is float:
        return '["float",' + _quote("nan" if math.isnan(value) else value.hex()) + "]"
    if value is None:
        return '["null"]'
    if kind is dict and all(type(key) is str for key in value):
        return '["map",[' + ",".join('[["str",' + _quote(key) + "]," + encoded_text(item) + "]"
                                     for key, item in value.items()) + "]]"
    if kind in (tuple, list):
        return ('["tuple",[' if kind is tuple else '["list",[') + ",".join(map(encoded_text, value)) + "]]"
    return dumps(encode(value))


def decode(node, text=None):
    """Accept only canonical encodings; never import a type named by a file.

    `text` is the exact JSON that `node` was parsed from or serialized to.
    Comparing against it avoids serializing a large node again; the decoded
    value must still re-encode to exactly that canonical text.
    """
    try:
        result = _decode(node)
        if encoded_text(result) != (dumps(node) if text is None else text):
            raise ValueError("noncanonical typed value")
        return result
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError("invalid typed value") from exc


def _decode(n):
    if not isinstance(n, list) or not n or type(n[0]) is not str:
        raise ValueError("invalid typed value")
    tag = n[0]
    if tag in {"null", "na", "nat"} and len(n) == 1:
        return {"null": None, "na": pd.NA, "nat": pd.NaT}[tag]
    if tag == "bool" and len(n) == 2 and type(n[1]) is bool:
        return n[1]
    if tag == "str" and len(n) == 2 and type(n[1]) is str:
        return n[1]
    if tag == "int" and len(n) == 2 and type(n[1]) is str:
        return int(n[1])
    if tag == "float" and len(n) == 2 and type(n[1]) is str:
        return float.fromhex(n[1])
    if tag == "complex" and len(n) == 3:
        return complex(_decode(n[1]), _decode(n[2]))
    if tag == "duration" and len(n) == 2:
        return pd.Timedelta(int(n[1]), unit="ns")
    if tag == "timestamp" and len(n) == 3:
        stamp = pd.Timestamp(int(n[1]), unit="ns", tz="UTC" if n[2] else None)
        return stamp.tz_convert(n[2]) if n[2] else stamp
    if tag in {"tuple", "list"} and len(n) == 2:
        values = [_decode(v) for v in n[1]]
        return tuple(values) if tag == "tuple" else values
    if tag == "map" and len(n) == 2:
        keys = [_decode(p[0]) for p in n[1]]
        _labels(keys)
        return dict(zip(keys, [_decode(p[1]) for p in n[1]]))
    if tag == "index" and len(n) == 4:
        values = [_decode(v) for v in n[1]]
        _labels(values)
        name = _decode(n[2])
        if n[3] is not None:
            return pd.DatetimeIndex(values, tz=n[3], name=name)
        return pd.Index(values, dtype=object, tupleize_cols=False, name=name)
    if tag == "array" and len(n) == 4:
        if type(n[1]) is not str or not isinstance(n[2],list) or not isinstance(n[3],list):
            raise ValueError("invalid array declaration")
        dtype = np.dtype(n[1])
        if dtype.kind not in "biufcUO" or dtype.fields or dtype.subdtype:
            raise ValueError("unsupported array dtype")
        if any(type(v) is not int or v < 0 for v in n[2]) or math.prod(n[2]) != len(n[3]):
            raise ValueError("invalid array shape")
        if dtype.kind == "U" and n[3]:
            # Unicode itemsize is an untrusted storage count, unlike fixed-size
            # numeric/object dtypes. Bound the whole allocation by supplied JSON
            # characters (four bytes per Unicode code point), including padding.
            if any(not isinstance(v,list) or len(v) != 2 or v[0] != "str" or type(v[1]) is not str for v in n[3]):
                raise ValueError("invalid Unicode array elements")
            if dtype.itemsize*len(n[3]) > 4*len(dumps(n)):
                raise ValueError("unsupported Unicode array storage exceeds payload")
        result = np.empty(len(n[3]), dtype=dtype)
        for i, value in enumerate(n[3]):
            result[i] = _decode(value)
        return result.reshape(n[2])
    if tag == "frame" and len(n) == 5:
        index, columns = _decode(n[1]), _decode(n[2])
        if len(n[3]) != len(index) or any(len(row) != len(columns) for row in n[3]):
            raise ValueError("frame dimensions disagree")
        result = pd.DataFrame([[_decode(v) for v in row] for row in n[3]], index=index, columns=columns, dtype=object)
        result.attrs = _decode(n[4])
        return result
    if tag == "series" and len(n) == 5:
        result = pd.Series([_decode(v) for v in n[3]], index=_decode(n[1]), name=_decode(n[2]), dtype=object)
        result.attrs = _decode(n[4])
        return result
    raise ValueError("unknown or malformed typed payload: " + tag)

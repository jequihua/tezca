"""Ownership and deterministic identities shared by the foundation records."""

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pandas as pd


def owned(value):
    # Scientific snapshots contain many immutable metadata scalars. Returning
    # exact built-ins avoids dispatching each one through pandas and deepcopy.
    if value is None or type(value) in (str, int, float, bool, complex, bytes):
        return value
    if isinstance(value, dict):
        return {key: owned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [owned(item) for item in value]
    if isinstance(value, tuple):
        return tuple(owned(item) for item in value)
    # pandas deep=True deliberately does not copy Python objects inside cells.
    if isinstance(value, pd.DataFrame):
        result = value.copy(deep=True)
        for position in range(result.shape[1]):
            column = result.iloc[:, position]
            if column.dtype == object:
                result.isetitem(position, pd.Series([deepcopy(v) for v in column],
                                                    index=result.index, dtype=object))
        result.attrs = deepcopy(value.attrs)
        return result
    if isinstance(value, pd.Series):
        result = value.copy(deep=True)
        if result.dtype == object:
            result = pd.Series([deepcopy(v) for v in result], index=result.index,
                               name=result.name, dtype=object)
        result.attrs = deepcopy(value.attrs)
        return result
    return deepcopy(value)


def _json_value(value):
    # Metadata leaves do not need pandas/NumPy dispatch. Preserve subclass
    # handling below and the existing textual form of nonfinite floats.
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float:
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": _json_value(value.tolist())}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, pd.DatetimeIndex):
        return [item.isoformat() for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def identity(value):
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_field(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    return value


def unit_field(value):
    text_field(value, "unit")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("unit must be an explicit symbol without whitespace; use '1' for unitless")
    return value

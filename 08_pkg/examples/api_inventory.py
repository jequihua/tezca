"""Check the companion API sheet against the imported (normally installed) Tezca.

This is a documentation check, not a package API or an optional-dependency probe.
It reads only Tezca's own public Python modules, never a dependency directory.
"""

import argparse
import ast
from dataclasses import fields, is_dataclass
import importlib
import inspect
import json
from pathlib import Path
import re
import sys


# These unprefixed implementation utilities have no supported user contract.
# Keep each exclusion explicit so a newly added public definition is not hidden.
EXCLUSIONS = {
    "tezca.data.aligned_array": "internal mask alignment/validation",
    "tezca.support.positive_duration": "internal duration validation",
    "tezca.support.positions_array": "internal position validation",
    "tezca.policy.terminal_fields": "internal policy migration flattening",
}


def _signature(value, *, instance_method=False):
    """Document actual calling forms, omitting annotations and implicit self."""
    try:
        signature = inspect.signature(value)
    except ValueError:
        if inspect.isclass(value) and issubclass(value, Exception):
            # These classes inherit the ordinary built-in exception constructor.
            return "(*args)"
        raise
    parameters = list(signature.parameters.values())
    if instance_method:
        parameters = parameters[1:]
    return str(signature.replace(
        parameters=[parameter.replace(annotation=inspect.Parameter.empty)
                    for parameter in parameters],
        return_annotation=inspect.Signature.empty,
    ))


def inventory():
    """Return {qualified_name: {kind, signature}} from source AST + live symbols.

    Properties/fields have signature None. Methods use instance calling forms.
    Public inherited Tezca members are included; imported aliases and built-in
    exception methods are not. Dataclass fields are immutable public accessors.
    Optional backends must remain unimported if absent at entry.
    """
    optional = {name: name in sys.modules for name in ("yaml", "sklearn", "ecofill")}
    import tezca

    root = Path(tezca.__file__).resolve().parent
    paths = sorted(root.glob("*.py")) + sorted((root / "adapters").glob("*.py"))
    result = {}
    excluded = set()
    for path in paths:
        if path.name.startswith("_"):
            continue
        module_name = "tezca." + ".".join(path.relative_to(root).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = importlib.import_module(module_name)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name.startswith("_"):
                continue
            qualified = module_name + "." + node.name
            if qualified in EXCLUSIONS:
                excluded.add(qualified)
                continue
            value = getattr(module, node.name)
            if value.__module__ != module_name:
                raise AssertionError(f"source definition does not resolve locally: {qualified}")
            is_class = isinstance(node, ast.ClassDef)
            result[qualified] = {"kind": "class" if is_class else "function",
                                 "signature": _signature(value)}
            if not is_class:
                continue
            for base in reversed(value.__mro__):
                if not base.__module__.startswith("tezca."):
                    continue
                for name, member in vars(base).items():
                    if name.startswith("_"):
                        continue
                    member_name = qualified + "." + name
                    if isinstance(member, property):
                        result[member_name] = {"kind": "property", "signature": None}
                    elif isinstance(member, (classmethod, staticmethod)):
                        result[member_name] = {"kind": "method",
                                              "signature": _signature(getattr(value, name))}
                    elif inspect.isfunction(member):
                        result[member_name] = {"kind": "method",
                                              "signature": _signature(member, instance_method=True)}
            if is_dataclass(value):
                for field in fields(value):
                    if not field.name.startswith("_"):
                        result[qualified + "." + field.name] = {"kind": "field", "signature": None}
    if excluded != set(EXCLUSIONS):
        raise AssertionError(f"stale API exclusions: {sorted(set(EXCLUSIONS) - excluded)}")
    result["tezca.reference.OutsideReferenceError.diagnostics"] = {
        "kind": "field", "signature": None,
    }
    newly_imported = [name for name, present in optional.items() if not present and name in sys.modules]
    if newly_imported:
        raise AssertionError(f"API inspection imported optional backends: {newly_imported}")
    return dict(sorted(result.items()))


def validate_cheatsheet(path):
    """Require one exact documented row per supported installed symbol; return it."""
    actual = inventory()
    documented = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| `(?P<name>tezca\.[\w.]+)(?P<signature>\(.*\))?` \|", line)
        if match is None:
            continue
        name = match.group("name")
        if name in documented:
            raise AssertionError(f"duplicate API row: {name}")
        documented[name] = match.group("signature")
    if set(actual) != set(documented):
        raise AssertionError({"undocumented": sorted(set(actual) - set(documented)),
                              "unknown": sorted(set(documented) - set(actual))})
    for name, entry in actual.items():
        if documented[name] != entry["signature"]:
            raise AssertionError(f"signature mismatch: {name}: {documented[name]!r} != {entry['signature']!r}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cheatsheet", nargs="?", type=Path)
    arguments = parser.parse_args()
    result = validate_cheatsheet(arguments.cheatsheet) if arguments.cheatsheet else inventory()
    print(json.dumps({"symbols": len(result), "inventory": result}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Explicit, source-bound transform ancestry for pair qualification."""

from ._owned import identity, owned, text_field
from .individual import _ColumnSignal


def signal_identity(data, variable):
    return _ColumnSignal(data, variable).identity


class PairLineage:
    """A declared graph; absent ancestors remain unknown, cycles are invalid."""
    def __init__(self, nodes):
        if not isinstance(nodes, dict):
            raise ValueError("lineage nodes must be a mapping")
        graph = {}
        for key, node in nodes.items():
            text_field(key, "node id")
            if not isinstance(node, dict) or set(node) - {"signal_id", "kind", "parents", "donors"}:
                raise ValueError("invalid lineage node fields")
            kind = node.get("kind", "unknown")
            if kind not in {"measurement", "copy", "deterministic", "reconstruction", "unknown"}:
                raise ValueError("unknown lineage kind")
            parents, donors = node.get("parents", []), node.get("donors")
            for name, items in (("parents", parents), ("donors", donors)):
                if items is not None and (not isinstance(items, (list, tuple)) or any(
                        not isinstance(v, str) or not v.strip() for v in items) or len(set(items)) != len(items)):
                    raise ValueError(f"{name} must be distinct explicit node/source identifiers")
            if parents is None or (kind == "measurement" and parents) or (kind == "copy" and len(parents) != 1) or (kind == "deterministic" and not parents):
                raise ValueError("lineage kind conflicts with parent declarations")
            signal_id = node.get("signal_id")
            if signal_id is not None:
                text_field(signal_id, "signal identity")
            graph[key] = dict(kind=kind, parents=sorted(parents), donors=None if donors is None else sorted(donors), signal_id=signal_id)
        visiting, done = set(), set()
        def visit(node):
            if node in visiting:
                raise ValueError("cyclic lineage")
            if node not in graph or node in done:
                return
            visiting.add(node)
            for parent in graph[node]["parents"]:
                visit(parent)
            visiting.remove(node)
            done.add(node)
        for node in graph:
            visit(node)
        self._record = graph

    def describe(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def audit_pair(first, second, *, variable_a, variable_b, lineage=None, nodes=None, target=None):
    signals = [_ColumnSignal(first, variable_a), _ColumnSignal(second, variable_b)]
    bindings = [s.identity for s in signals]
    graph = {} if lineage is None else lineage.describe() if isinstance(lineage, PairLineage) else None
    if graph is None:
        raise ValueError("lineage must be PairLineage")
    if nodes is not None and (not isinstance(nodes, (list, tuple)) or len(nodes) != 2 or any(not isinstance(v, str) for v in nodes)):
        raise ValueError("nodes must identify the ordered pair")
    nodes = [None, None] if nodes is None else list(nodes)
    target = nodes[0] if target is None else text_field(target, "target node")
    for node, binding in zip(nodes, bindings):
        if node in graph and graph[node]["signal_id"] != binding:
            raise ValueError("lineage signal identity does not match selected data/variable/basis")
    # Equal signal bindings and declared copies denote the same signal even
    # when an ancestry edge uses another name for it. Merge before traversal.
    groups = {key: {key} for key in graph}
    by_signal = {}
    for key, record in graph.items():
        peers = list(record["parents"]) if record["kind"] == "copy" else []
        if record["signal_id"] is not None:
            peers.append(by_signal.setdefault(record["signal_id"], key))
        for peer in peers:
            merged = groups[key] | groups.get(peer, {peer})
            for member in merged:
                groups[member] = merged
    ancestry, deterministic, donors, unknown = [], [], [], []
    for root in nodes:
        ancestors, deterministic_ancestors, donor_ids, missing = set(), set(), set(), set()
        visited = set()
        def walk(node, deterministic_path=True):
            for alias in sorted(groups.get(node, {node}), key=str):
                if (alias, deterministic_path) in visited:
                    continue
                visited.add((alias, deterministic_path))
                if alias not in graph:
                    missing.add(alias or "undeclared_node")
                    continue
                record = graph[alias]
                if record["kind"] == "unknown" or record["donors"] is None:
                    missing.add(alias)
                donor_ids.update(record["donors"] or [])
                for parent in record["parents"]:
                    equivalents = groups.get(parent, {parent})
                    ancestors.update(equivalents)
                    path = deterministic_path and record["kind"] in {"copy", "deterministic"}
                    if path:
                        deterministic_ancestors.update(equivalents)
                    walk(parent, path)
        walk(root)
        ancestry.append(sorted(ancestors))
        deterministic.append(sorted(deterministic_ancestors))
        donors.append(donor_ids)
        unknown.append(sorted(missing))
    shared = sorted(donors[0] & donors[1])
    reasons = []
    if bindings[0] == bindings[1] or nodes[0] is not None and nodes[1] in groups.get(nodes[0], {nodes[0]}):
        reasons.append("duplicate_signal")
    if nodes[0] in deterministic[1] or nodes[1] in deterministic[0]:
        reasons.append("copy_or_deterministic_transform")
    if any(target in parents for parents in deterministic):
        reasons.append("target_descendant")
    if any(unknown) or target is not None and target not in graph:
        reasons.append("unknown_lineage")
    if shared:
        reasons.append("shared_reconstruction_donors")
    record = dict(pair=[dict(variable=s.variable, signal_id=s.identity, source=s.source,
                             basis=s.basis, reference_id=s.reference_id) for s in signals],
                  nodes=nodes, target=target, aliases={k: sorted(v) for k, v in groups.items()},
                  ancestors=ancestry, deterministic_ancestors=deterministic,
                  unknown_ancestry=unknown, shared_donors=shared, reasons=reasons,
                  lineage_id=None if lineage is None else lineage.identity,
                  ranking_eligible=not reasons,
                  ranking_status="eligible_descriptive_pair" if not reasons else "excluded_or_sensitivity_only",
                  observed_only_required=bool(shared), numerical_correlation_used=False)
    record["audit_id"] = identity(record)
    return owned(record)

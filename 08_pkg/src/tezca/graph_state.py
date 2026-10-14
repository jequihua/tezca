"""Explicit square-root mass/density graph energy and frozen state geometry."""

from itertools import product
from math import prod

import numpy as np
import pandas as pd

from ._owned import identity, owned, text_field
from .data import ResultRecord
from .individual import _observations, _training_signal
from .pca import PCASpec, fit_pca_reference, _support, _stricter
from .reference import FrozenReference
from .support import ObservationPolicy, positions_array


def sqrt_mass_graph_energy(probabilities, edges, weights=None, *, measure="mass", volumes=None,
                           volume_units=None, conductance_units=None):
    """Each declared undirected edge contributes exactly once; no renormalization."""
    p = np.asarray(probabilities)
    if p.dtype.kind not in "iuf" or p.ndim != 1 or not len(p):
        raise ValueError("probabilities must be a nonempty real vector")
    p = p.astype(float)
    if not np.isfinite(p).all() or np.any(p < 0) or abs(float(p.sum())-1) > 1e-12:
        raise ValueError("finite nonnegative normalized masses required")
    e = np.asarray(edges)
    if not e.size:
        e = np.empty((0, 2), dtype=int)
    if e.dtype.kind not in "iu" or e.ndim != 2 or e.shape[1] != 2 or np.any(e < 0) or np.any(e >= len(p)):
        raise ValueError("edges must be pairs of in-range integer node indices")
    canonical = [tuple(sorted(pair)) for pair in e.tolist()]
    if any(a == b for a, b in canonical) or len(set(canonical)) != len(canonical):
        raise ValueError("self, duplicate or reversed-duplicate edges are invalid")
    if measure not in {"mass", "density"}:
        raise ValueError("measure must be mass or density")
    if measure == "density" and weights is None:
        raise ValueError("density requires explicit edge conductances")
    w = np.ones(len(e)) if weights is None else np.asarray(weights)
    if w.dtype.kind not in "iuf" or w.ndim != 1 or len(w) != len(e) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("finite nonnegative weight for each edge required")
    w = w.astype(float)
    v = None
    units = "1"
    if measure == "density":
        v = np.asarray(volumes, dtype=float)
        if v.shape != p.shape or not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError("density requires positive finite node volumes")
        units = text_field(conductance_units, "conductance units") + "/(" + text_field(volume_units, "volume units") + ")"
    elif volumes is not None or volume_units is not None or conductance_units is not None:
        raise ValueError("volume/conductance units belong to density mode")
    q = np.sqrt(p) if measure == "mass" else np.sqrt(p/v)
    with np.errstate(over="ignore", invalid="ignore"):
        contributions = 4*w*(q[e[:, 0]]-q[e[:, 1]])**2
    if not np.isfinite(q).all() or not np.isfinite(contributions).all() or not np.isfinite(contributions.sum()):
        raise ValueError("graph energy exceeds finite precision")
    neighbors = [set() for _ in p]
    for (a, b), weight in zip(e, w):
        if weight > 0:
            neighbors[a].add(b)
            neighbors[b].add(a)
    unseen, components = set(range(len(p))), 0
    while unseen:
        stack = [unseen.pop()]
        components += 1
        while stack:
            fresh = neighbors[stack.pop()] & unseen
            unseen.difference_update(fresh)
            stack.extend(fresh)
    result = dict(energy=float(contributions.sum()), probabilities=p, edges=e, weights=w,
                  volumes=v, node_signal=q, per_edge=contributions, measure=measure, units=units,
                  connected_components=components, isolated_mass=float(sum(p[i] for i, n in enumerate(neighbors) if not n)),
                  occupied_count=int(np.count_nonzero(p)), empty_count=int(np.count_nonzero(p == 0)),
                  method_id="graph.sqrt_mass_energy_v1" if measure == "mass" else "graph.sqrt_density_energy_v1")
    result["geometry_id"] = identity(dict(nodes=len(p), edges=e, weights=w, volumes=v, measure=measure, units=units))
    return owned(result)


def _edges(values):
    arrays = []
    for raw in values:
        raw = np.asarray(raw)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or len(raw) < 2:
            raise ValueError("each coordinate needs real ordered bin edges")
        edge = raw.astype(float)
        with np.errstate(over="ignore", invalid="ignore"):
            widths = np.diff(edge)
        if not np.isfinite(edge).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("finite strictly increasing edges and widths required")
        arrays.append(edge)
    return arrays


class GraphStateSpec:
    def __init__(self, *, features, basis, scaling="none", pca_components=None, ddof=1,
                 partition="equal_width", bins=3, edges=None, ties="raise", node_policy="full_grid",
                 outside="raise", pseudocount=0, max_states=10000, graph_edges=None, weights=None,
                 measure="mass", volume_units=None, conductance_units=None):
        representation = PCASpec(features=features, basis=basis, scaling=scaling, ddof=ddof).record
        if pca_components is not None and (type(pca_components) is not int or pca_components < 1):
            raise ValueError("requested PC count must be a positive integer")
        if partition not in {"equal_width", "quantile", "custom"} or ties not in {"raise", "merge"}:
            raise ValueError("explicit partition and tie policy required")
        if type(bins) is not int or bins < 2 or type(max_states) is not int or max_states < 1:
            raise ValueError("bins >=2 and max_states >=1 must be integers")
        if node_policy not in {"occupied", "reference", "full_grid"} or outside not in {"raise", "exclude"}:
            raise ValueError("invalid node/outside policy")
        if isinstance(pseudocount, (bool, np.bool_)) or not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative")
        realized = None if edges is None else _edges(edges)
        dimension = pca_components or len(representation["features"])
        if (partition == "custom") != (edges is not None) or realized is not None and len(realized) != dimension:
            raise ValueError("custom edges must explicitly match the requested coordinate dimension")
        if measure not in {"mass", "density"} or measure == "density" and weights is None:
            raise ValueError("density requires explicit conductances")
        if measure == "density":
            text_field(volume_units, "volume units")
            text_field(conductance_units, "conductance units")
        elif volume_units is not None or conductance_units is not None:
            raise ValueError("density units do not apply to mass energy")
        if weights is not None:
            w = np.asarray(weights)
            if w.dtype.kind not in "iuf" or w.ndim > 1 or not np.isfinite(w).all() or np.any(w < 0):
                raise ValueError("weights must be finite nonnegative scalar or edge vector")
        self._record = dict(features=representation["features"], basis=basis, scaling=scaling,
                            pca_components=pca_components, ddof=ddof, partition=partition, bins=bins,
                            edges=realized, ties=ties, node_policy=node_policy, outside=outside,
                            pseudocount=float(pseudocount), max_states=max_states,
                            graph_edges=owned(graph_edges), weights=owned(weights), measure=measure,
                            volume_units=volume_units, conductance_units=conductance_units)

    @property
    def record(self):
        return owned(self._record)

    @property
    def identity(self):
        return identity(self._record)


def _geometry(nodes, edges, settings):
    lookup = {tuple(node): i for i, node in enumerate(nodes)}
    pairs = settings["graph_edges"]
    if pairs is None:
        pairs = []
        for i, node in enumerate(nodes):
            for j in range(len(edges)):
                neighbor = list(node)
                neighbor[j] += 1
                if tuple(neighbor) in lookup:
                    pairs.append([i, lookup[tuple(neighbor)]])
    pairs = np.asarray(pairs, dtype=int) if np.asarray(pairs).size == 0 else np.asarray(pairs)
    weights = settings["weights"]
    weights = np.ones(len(pairs)) if weights is None else np.full(len(pairs), weights) if np.isscalar(weights) else np.asarray(weights)
    with np.errstate(over="ignore", invalid="ignore"):
        volumes = np.array([prod(float(edges[j][cell+1]-edges[j][cell]) for j, cell in enumerate(node)) for node in nodes])
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("cell volumes must be positive and finite")
    kwargs = dict(measure=settings["measure"])
    if settings["measure"] == "density":
        kwargs.update(volumes=volumes, volume_units=settings["volume_units"], conductance_units=settings["conductance_units"])
    primitive = sqrt_mass_graph_energy(np.full(len(nodes), 1/len(nodes)), pairs, weights, **kwargs)
    return dict(nodes=np.asarray(nodes, dtype=int), edges=primitive["edges"], weights=primitive["weights"],
                volumes=volumes, units=primitive["units"], geometry_id=primitive["geometry_id"])


class StateReference(FrozenReference):
    def __init__(self, record):
        record = owned(record)
        settings = GraphStateSpec(**record["spec"]).record
        if any(identity(record[key]) != identity(settings) for key in ("requested", "resolved")):
            raise ValueError("reference settings disagree with its specification")
        coordinate_units, output_units = _state_units(settings, record["support"]["units"])
        if record["coordinate_units"] != coordinate_units or record["output_units"] != output_units:
            raise ValueError("reference units differ from its declared representation/geometry")
        if record["fit_scope"] == "training_only" and settings["node_policy"] == "occupied":
            raise ValueError("frozen training reference cannot rebuild occupied nodes")
        if record["compute_status"] == "ok":
            edges = _edges(record["partition_edges"])
            realized = [len(edge)-1 for edge in edges]
            if record["realized_bins"] != realized:
                raise ValueError("reference bin counts disagree with its partition")
            if settings["partition"] == "custom":
                if identity(edges) != identity(settings["edges"]):
                    raise ValueError("reference partition differs from declared custom edges")
            elif any(n > settings["bins"] or settings["ties"] == "raise" and n != settings["bins"] for n in realized):
                raise ValueError("reference partition differs from declared bin count/tie policy")
            nodes = np.asarray(record["geometry"]["nodes"])
            if nodes.dtype.kind not in "iu" or nodes.ndim != 2 or nodes.shape[1] != len(edges) or not len(nodes) or len(nodes) > settings["max_states"]:
                raise ValueError("invalid reference node geometry")
            if len({tuple(n) for n in nodes}) != len(nodes) or any(np.any(nodes[:, j] < 0) or np.any(nodes[:, j] >= len(e)-1) for j, e in enumerate(edges)):
                raise ValueError("reference nodes must be unique cells in the partition")
            if settings["node_policy"] == "full_grid" and len(nodes) != prod(len(e)-1 for e in edges):
                raise ValueError("full-grid reference omits declared cells")
            geometry = _geometry(nodes, edges, settings)
            if identity(geometry) != identity(record["geometry"]):
                raise ValueError("reference geometry differs from its declared partition/graph")
            d, p = len(edges), len(settings["features"])
            transform = record["transform"]
            if np.asarray(transform["center"]).shape != (p,) or np.asarray(transform["scales"]).shape != (p,) or not np.isfinite(transform["center"]).all() or not np.isfinite(transform["scales"]).all() or np.any(np.asarray(transform["scales"]) <= 0):
                raise ValueError("invalid frozen scaling")
            if d != (settings["pca_components"] or p):
                raise ValueError("reference partition dimension mismatch")
            vectors = transform["vectors"]
            if settings["pca_components"] is None:
                if vectors is not None or record["pca_reference_id"] is not None:
                    raise ValueError("non-PCA representation cannot contain a PCA transform")
                if settings["scaling"] == "none" and (np.any(np.asarray(transform["center"]) != 0) or np.any(np.asarray(transform["scales"]) != 1)):
                    raise ValueError("unscaled non-PCA representation must preserve raw coordinates")
            else:
                if np.asarray(vectors).shape != (p, d) or not np.isfinite(vectors).all() or not np.allclose(np.asarray(vectors).T @ vectors, np.eye(d), atol=1e-10, rtol=0):
                    raise ValueError("invalid frozen PCA basis")
                pca = record["pca_reference"]
                expected_spec = PCASpec(features=settings["features"], basis=settings["basis"], scaling=settings["scaling"], ddof=settings["ddof"]).record
                if identity(pca) != record["pca_reference_id"] or identity(pca["spec"]) != identity(expected_spec):
                    raise ValueError("frozen PCA reference differs from its identity/specification")
                expected_transform = dict(center=pca["center"], scales=pca["scales"], vectors=pca["eigensystem"]["eigenvectors"][:, :d])
                if identity(transform) != identity(expected_transform):
                    raise ValueError("state transform differs from its frozen PCA reference")
        super().__init__(record)


def _coordinates(values, transform):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinates = (values-transform["center"])/transform["scales"]
        if transform["vectors"] is not None:
            coordinates = coordinates @ transform["vectors"]
    return coordinates


def _cells(coordinates, edges):
    cells = np.empty(coordinates.shape, dtype=int)
    inside = np.ones(len(coordinates), dtype=bool)
    for j, edge in enumerate(edges):
        values = coordinates[:, j]
        inside &= (values >= edge[0]) & (values <= edge[-1])
        cells[:, j] = np.searchsorted(edge, values, side="right")-1
        cells[values == edge[-1], j] = len(edge)-2
    return cells, inside


def _selection(data, settings, observations, within=None):
    # Population support is shared with PCA; raw state axes may have different
    # units because no unscaled covariance between those axes is formed here.
    return _support(data, {**settings, "scaling": "zscore"}, observations, within)


def _state_units(settings, input_units):
    d = settings["pca_components"] or len(settings["features"])
    coordinates = (["1"]*d if settings["scaling"] == "zscore" else
                   [next(iter(input_units.values()))]*d if settings["pca_components"] is not None else list(input_units.values()))
    volume = settings["volume_units"] if settings["measure"] == "density" else "*".join(f"({u})" for u in coordinates)
    energy = "1" if settings["measure"] == "mass" else settings["conductance_units"]+"/("+volume+")"
    fields = dict(energy=energy, per_edge=energy, cell_volumes=volume,
                  weights="1" if settings["measure"] == "mass" else settings["conductance_units"],
                  node_signal="1" if settings["measure"] == "mass" else "1/sqrt("+volume+")")
    fields.update({key: "1" for key in ("probabilities", "observed_masses", "smoothed_masses", "counts",
                  "connected_components", "isolated_mass", "occupied_count", "empty_count", "observed_occupied_count", "observed_empty_count")})
    return coordinates, fields


def fit_state_reference(data, *, spec, training_positions=None, observations=None, fit_scope="training_only"):
    if not isinstance(spec, GraphStateSpec) or fit_scope not in {"training_only", "self"}:
        raise ValueError("explicit graph spec and valid fit scope required")
    settings = spec.record
    if (fit_scope == "training_only") != (training_positions is not None) or fit_scope == "training_only" and settings["node_policy"] == "occupied":
        raise ValueError("training requires positions and reference/full_grid nodes; occupied is self-only")
    observations = _observations(observations)
    requested = np.arange(len(data.values)) if training_positions is None else positions_array(training_positions, len(data.values))
    training = _training_signal(data, settings["features"][0], requested)._data
    values, positions, support = _selection(training, settings, observations)
    support.update(position_scope="training_subset", original_positions=requested.tolist(), original_eligible_positions=requested[positions].tolist())
    record = dict(spec=settings, requested=settings, resolved=settings, fit_scope=fit_scope,
                  semantics="training_only" if fit_scope == "training_only" else "retrospective",
                  training_id=training.identity, support=support, compute_status="ok", reason=None,
                  partition_edges=None, geometry=None, transform=None, rank=None, pca_reference_id=None,
                  requested_states=None, method_id="graph.state_reference_v1", backend="numpy", backend_version=np.__version__)
    record["coordinate_units"], record["output_units"] = _state_units(settings, support["units"])
    if len(values) < max(2, observations.min_points):
        record.update(compute_status="insufficient_support", reason="too few complete training vectors")
        return StateReference(record)
    p, count = len(settings["features"]), settings["pca_components"]
    transform = dict(center=np.zeros(p), scales=np.ones(p), vectors=None)
    if count is not None:
        pca = fit_pca_reference(training, spec=PCASpec(features=settings["features"], basis=settings["basis"], scaling=settings["scaling"], ddof=settings["ddof"]),
                                observations=observations, fit_scope=fit_scope,
                                training_positions=list(range(len(training.values))) if fit_scope == "training_only" else None)
        fitted = pca.describe()
        record["pca_reference_id"] = pca.identity
        record["pca_reference"] = fitted
        if fitted["compute_status"] != "ok":
            record.update(compute_status=fitted["compute_status"], reason=fitted["reason"])
            return StateReference(record)
        rank = fitted["eigensystem"]["rank"]
        record["rank"] = rank
        if count > p or count > rank:
            record.update(compute_status="infeasible_dimension", reason="requested PCs exceed feature dimension or training rank")
            return StateReference(record)
        transform = dict(center=fitted["center"], scales=fitted["scales"], vectors=fitted["eigensystem"]["eigenvectors"][:, :count])
    elif settings["scaling"] == "zscore":
        with np.errstate(over="ignore", invalid="ignore"):
            transform.update(center=values.mean(axis=0), scales=values.std(axis=0, ddof=settings["ddof"]))
        if not np.isfinite(transform["center"]).all() or not np.isfinite(transform["scales"]).all() or np.any(transform["scales"] <= 0):
            record.update(compute_status="invalid_scale", reason="training scales must be finite and positive")
            return StateReference(record)
    coordinates = _coordinates(values, transform)
    record["transform"] = transform
    if not np.isfinite(coordinates).all():
        record.update(compute_status="invalid_projection", reason="coordinates exceed finite precision")
        return StateReference(record)
    # Bound the declared lattice before allocating even its fitted edge arrays.
    if settings["node_policy"] == "full_grid":
        states = prod(len(e)-1 for e in settings["edges"]) if settings["partition"] == "custom" else settings["bins"]**coordinates.shape[1]
        record["requested_states"] = states
        if states > settings["max_states"]:
            record.update(compute_status="resource_limit", reason="requested full grid exceeds max_states before partition/graph allocation")
            return StateReference(record)
    if settings["partition"] == "custom":
        edges = settings["edges"]
    else:
        edges = []
        for column in coordinates.T:
            if np.ptp(column) == 0:
                record.update(compute_status="degenerate_partition", reason="constant coordinate has no finite-width fitted partition")
                return StateReference(record)
            edge = np.linspace(column.min(), column.max(), settings["bins"]+1) if settings["partition"] == "equal_width" else np.quantile(column, np.linspace(0, 1, settings["bins"]+1))
            if np.any(np.diff(edge) == 0):
                if settings["ties"] == "raise":
                    raise ValueError("partition ties require explicit merge")
                edge = np.unique(edge)
            edges.append(edge)
        edges = _edges(edges)
    record["partition_edges"] = edges
    cells, inside = _cells(coordinates, edges)
    if np.any(~inside) and settings["outside"] == "raise":
        raise ValueError("training vectors fall outside declared partition")
    record["training_outside_positions"] = requested[positions[~inside]].tolist()
    record["realized_bins"] = [len(e)-1 for e in edges]
    if settings["node_policy"] == "full_grid":
        states = prod(record["realized_bins"])
        record["requested_states"] = states
        if states > settings["max_states"]:
            record.update(compute_status="resource_limit", reason="declared full grid exceeds max_states before allocation")
            return StateReference(record)
        nodes = list(product(*(range(n) for n in record["realized_bins"])))
    else:
        nodes = sorted({tuple(c) for c in cells[inside]})
        record["requested_states"] = len(nodes)
        if len(nodes) > settings["max_states"]:
            record.update(compute_status="resource_limit", reason="declared node universe exceeds max_states before graph allocation")
            return StateReference(record)
    if not nodes:
        record.update(compute_status="outside_reference", reason="no training vectors in declared state universe")
    else:
        record["geometry"] = _geometry(nodes, edges, settings)
    return StateReference(record)


def _apply(data, reference, observations):
    frozen = reference.describe()
    settings = frozen["spec"]
    values, positions, candidate = _selection(data, settings, observations)
    if candidate["units"] != frozen["support"]["units"]:
        raise ValueError("evaluation units must match frozen feature units")
    status, reason, value = frozen["compute_status"], frozen["reason"], None
    used, excluded, outside = np.array([], dtype=int), [], []
    if status == "ok":
        coordinates = _coordinates(values, frozen["transform"])
        if not np.isfinite(coordinates).all():
            status, reason = "invalid_projection", "evaluation coordinates exceed finite precision"
        else:
            cells, inside = _cells(coordinates, frozen["partition_edges"])
            lookup = {tuple(node): i for i, node in enumerate(frozen["geometry"]["nodes"])}
            represented = np.array([tuple(cell) in lookup for cell in cells], dtype=bool)
            accepted = inside & represented
            excluded, outside = positions[~accepted].tolist(), positions[~inside].tolist()
            if excluded and settings["outside"] == "raise":
                raise ValueError("evaluation vectors outside frozen partition or node universe")
            used = positions[accepted]
            if len(used) < observations.min_points:
                status, reason = "outside_reference" if excluded else "insufficient_support", "too few complete vectors in the frozen state universe"
            else:
                counts = np.zeros(len(lookup), dtype=int)
                for cell in cells[accepted]:
                    counts[lookup[tuple(cell)]] += 1
                raw = counts/counts.sum()
                with np.errstate(over="ignore", invalid="ignore"):
                    smoothed = (counts+settings["pseudocount"])/(counts.sum()+settings["pseudocount"]*len(counts))
                if not np.isfinite(smoothed).all() or abs(float(smoothed.sum())-1) > 1e-12:
                    status, reason = "invalid_probabilities", "pseudocount normalization exceeds finite precision"
                else:
                    geometry = frozen["geometry"]
                    kwargs = dict(measure=settings["measure"])
                    if settings["measure"] == "density":
                        kwargs.update(volumes=geometry["volumes"], volume_units=settings["volume_units"], conductance_units=settings["conductance_units"])
                    primitive = sqrt_mass_graph_energy(smoothed, geometry["edges"], geometry["weights"], **kwargs)
                    value = dict(**primitive, nodes=geometry["nodes"], cell_volumes=geometry["volumes"], counts=counts,
                                 observed_masses=raw, smoothed_masses=smoothed, observed_occupied_count=int(np.count_nonzero(counts)),
                                 observed_empty_count=int(np.count_nonzero(counts == 0)), partition_edges=frozen["partition_edges"],
                                 realized_bins=frozen["realized_bins"], node_policy=settings["node_policy"], pseudocount_universe="declared_nodes")
    _, _, evaluation = _selection(data, settings, observations, used)
    use = _stricter([evaluation["use_status"], frozen["support"]["use_status"], "sensitivity_only"])
    if len(used) < observations.min_points:
        use = _stricter([use, "insufficient_support"])
    if use == "unknown_provenance":
        if status == "ok":
            status, reason, value = "unknown_provenance", "contributing or training provenance is unknown", None
    support = dict(candidate=candidate, evaluation=evaluation, training=frozen["support"], reference_id=reference.identity,
                   training_id=frozen["training_id"], pca_reference_id=frozen["pca_reference_id"], fit_scope=frozen["fit_scope"],
                   requested=settings, resolved=settings, requested_states=frozen["requested_states"], rank=frozen["rank"],
                   eligible_positions=positions.tolist(), contributing_positions=used.tolist(),
                   excluded_positions=excluded, excluded_count=len(excluded), outside_positions=outside, outside_count=len(outside),
                   unrepresented_count=len(excluded)-len(outside), refitted=False, observations=observations.__dict__,
                   coordinate_units=frozen["coordinate_units"], partition_edges=frozen["partition_edges"],
                   backend="numpy", backend_version=np.__version__)
    return ResultRecord(value, compute_status=status, reason=reason, metric_status="exploratory_graph_roughness", use_status=use,
                        support=support, units=frozen["output_units"], method_id="graph.state_mass_v1" if settings["measure"] == "mass" else "graph.state_density_v1",
                        config_id=identity(dict(reference=reference.identity, settings=settings, observations=observations.__dict__, support=evaluation["support_id"])),
                        reference_id=reference.identity, representation=settings["basis"])


def graph_state_roughness(data, *, spec=None, reference=None, observations=None):
    if (reference is None) == (spec is None):
        raise ValueError("supply exactly one of spec or reference")
    observations = _observations(observations)
    if spec is not None:
        reference = fit_state_reference(data, spec=spec, observations=observations, fit_scope="self")
    elif not isinstance(reference, StateReference) or reference.describe()["spec"]["node_policy"] == "occupied":
        raise ValueError("a frozen reference must use reference/full_grid nodes")
    result = _apply(data, reference, observations)
    if observations.values != "observed_only":
        record = result.summary()
        record["support"]["observed_only_counterpart"] = _apply(data, reference,
            ObservationPolicy("observed_only", min_points=observations.min_points)).summary()
        return ResultRecord(**record)
    return result


class GraphGridResult:
    def __init__(self, rows):
        self._rows = owned(rows)

    @property
    def table(self):
        return owned(pd.DataFrame(self._rows))

    def describe(self):
        energies = [r["energy"] for r in self._rows if r["compute_status"] == "ok"]
        return owned(dict(rows=self._rows, configurations=len(self._rows),
                          median=float(np.median(energies)) if energies else None,
                          minimum=min(energies) if energies else None, maximum=max(energies) if energies else None,
                          interpretation="descriptive sensitivity range, not a confidence interval",
                          grid_id=identity(self._rows)))


def graph_state_grid(data, *, features, basis, scaling="zscore", ddof=1, max_states=10000,
                     observations=None, training_positions=None):
    rows = []
    for pcs in range(2, 6):
        for bins in range(3, 7):
            settings = GraphStateSpec(features=features, basis=basis, scaling=scaling, ddof=ddof,
                pca_components=pcs, bins=bins, partition="equal_width", node_policy="full_grid", outside="exclude", max_states=max_states)
            reference = fit_state_reference(data, spec=settings, training_positions=training_positions,
                fit_scope="self" if training_positions is None else "training_only", observations=observations)
            result = graph_state_roughness(data, reference=reference, observations=observations)
            record, frozen = result.summary(), reference.describe()
            value = result.value
            rows.append(dict(pcs=pcs, bins=bins, requested_states=bins**pcs, rank=frozen["rank"],
                             realized_bins=frozen.get("realized_bins"), reference_id=reference.identity,
                             config_id=record["config_id"], compute_status=record["compute_status"], reason=record["reason"],
                             metric_status=record["metric_status"], use_status=record["use_status"],
                             energy=None if value is None else value["energy"],
                             connected_components=None if value is None else value["connected_components"],
                             occupied_count=None if value is None else value["observed_occupied_count"],
                             empty_count=None if value is None else value["observed_empty_count"], result=record))
    return GraphGridResult(rows)

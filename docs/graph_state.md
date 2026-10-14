# Graph-state roughness

```python
from tezca.graph_state import (
    GraphStateSpec, fit_state_reference, graph_state_roughness, graph_state_grid,
    sqrt_mass_graph_energy,
)

spec = GraphStateSpec(features=["level", "temperature", "salinity"], basis="raw",
    scaling="zscore", pca_components=2, ddof=1, partition="quantile", bins=4,
    ties="merge", node_policy="full_grid", outside="exclude",
    pseudocount=0, max_states=10000)
reference = fit_state_reference(training, spec=spec, training_positions=positions)
result = graph_state_roughness(evaluation, reference=reference)
```

The method separates representation, partition, masses and geometry. Inputs are
owned `AnalysisSeries` or `DailyResidual`; features and raw/residual basis must be
explicit. Complete eligible vectors retain original aware instants, holes,
variable membership, units and contributor qualifications. No observations are
filled or compressed into a temporal estimator. The default observation policy
is observed-only. No rolling window API is implied by this population summary.

`sqrt_mass_graph_energy(p, edges, weights=None)` is the small numerical primitive:

    energy = 4 sum_{unique undirected edges u,v} w_uv*(sqrt(p_u)-sqrt(p_v))**2
           = 4 q.T @ L @ q, q=sqrt(p)

Masses must be finite, nonnegative and sum to one within absolute 1e-12. The
primitive does not renormalize. Edges are pairs of in-range integer indices;
self edges, duplicates and reversed duplicates raise. Weights are finite and
nonnegative; omitted mass weights are one. No silent symmetrization occurs.
The owned result includes masses, edges, weights, node signal, per-edge energy,
total energy, occupied/empty counts, connected components and isolated mass.
Connectivity uses positive-weight edges; zero weights connect nothing. Uniform
mass on a connected unit-weight graph has zero energy. Simultaneously relabeling
nodes/masses/edges preserves energy; changing partitions or PCA orientation need
not. Sample shuffling preserves fixed-representation occupancy, without implying
temporal stability, transitions, causality or a statistical Fisher matrix.

Density is explicit: `measure="density"`, positive `volumes`, explicit edge
conductances in `weights`, and nonempty `volume_units`/`conductance_units`. Its
signal is sqrt(p/V); energy units are conductance units divided by volume units,
under the distinct `graph.sqrt_density_energy_v1` identity. Mass mode uses
`graph.sqrt_mass_energy_v1` and dimensionless weights/energy. In one dimension,
bin widths as volumes and inverse center spacing as conductances recover the
histogram interior formula. Arbitrary multidimensional geometry is not claimed
to converge to continuum Fisher information.

Representation uses `scaling="none"` or `"zscore"` and optional positive integer
`pca_components`. No PCA leaves raw coordinates unchanged unless z-scaling was
requested. PCA reuses the full covariance eigensystem, explicit ddof0/1 and
training scaling; unscaled PCA requires matching feature units. Raw state axes
may have distinct units because no covariance is formed between them. Requested
PCs above feature dimension or numerical rank return `infeasible_dimension`;
there is no silent cap. Frozen references retain scaling, eigensystem/reference
identity, feature order, units, basis, selected training support and rank. A
training reference fits only its supplied positions; held-out values cannot
change it. Evaluation applies these coordinates and never refits them. PCA signs
and tied subspaces are scientific geometry choices; use a fixed reference for
comparisons. Incompatible feature order/units/basis raises.

Partitions are per-coordinate equal-width, quantile, or custom finite strictly
increasing `edges`, with the last edge included. Custom edge arrays must match
the coordinate dimension; no automatic clipping occurs. Quantile ties either
raise or explicitly merge; realized edges and counts remain in the reference
and result. A constant coordinate cannot define a fitted finite-width partition
and returns `degenerate_partition`; a declared custom partition may legitimately
contain constant data. Outside behavior is `raise` or counted `exclude` for both
partition extremes and unrepresented reference cells.

Node policy is a scientific choice:

| Policy | Node universe |
| --- | --- |
| `occupied` | Currently occupied cells, through the retrospective self-fit `spec` route only |
| `reference` | Cells occupied in training, fixed for every subsequent evaluation |
| `full_grid` | Every cell of the finite declared partition, including empty cells |

`fit_state_reference` requires explicit training positions with reference/full-grid
nodes. An occupied request cannot become a frozen training reference. The
`graph_state_roughness(data, spec=spec)` route self-fits and records retrospective
semantics. Reference evaluation preserves empty nodes, edges and volumes. New
cells outside a reference universe are counted, even if within the partition.
`StateReference` owns nested records and validates supplied geometry against its
declared node universe, edges, weights, partition, volumes and transform. Custom
edges must exactly match the stored partition; declared and realized bin counts
must obey the tie policy. Requested/resolved settings must agree with the spec.
Non-PCA references cannot contain projection matrices, and unscaled non-PCA
coordinates require zero centers and unit scales. PCA transforms must match the
bound frozen PCA record, including its settings and identity.

Default graph edges connect coordinate cells differing by exactly one along one
axis. `graph_edges` provides an explicit edge list indexed into the lexicographically
ordered node list; weights may be one explicit scalar for every edge or a vector
in edge order. Density mode requires explicit conductances; node volumes are
products of realized bin widths. State results use separate
`graph.state_mass_v1`/`graph.state_density_v1` identities and field-specific output
units, including energy, per-edge contributions, weights, cell volumes and node
signal. Coordinate units are recorded even for unavailable results.

`pseudocount` adds the declared nonnegative count to every node in the selected
universe. Counts, observed masses and smoothed masses are separate, along with
observed occupied/empty counts. No-data evaluation is unavailable even with a
pseudocount. On A-B-C with masses (.5,0,.5), full-grid energy is4; occupied-only
removes B, yielding energy0, two components and isolated mass1. Neither result
has a geometry-independent ecological interpretation.

`max_states` bounds nodes before graph allocation. For full grids, the declared
bin-count product is also checked before allocating fitted partitions, before
any possible quantile tie merge. Oversized requests return `resource_limit`;
the implementation never silently switches node policy or reduces bins.

Owned results preserve candidate and actual contributing support, training
support, outside/unrepresented positions/counts, source/reference bindings,
requested/resolved settings, backend/version and independent statuses. Metric
status is `exploratory_graph_roughness`. Use is `sensitivity_only`,
`insufficient_support`, or `unknown_provenance`; small reconstruction burden never
promotes this method to primary evidence. Strict training qualifications persist.
Explicit provided/trusted policies include an observed-only evaluation counterpart
on the same frozen geometry, with retained training qualifications or reasoned
unavailability. Nonlinear ancestry keeps conservative touched-cell support gates.

Compute statuses are `ok`, `insufficient_support`, `outside_reference`,
`unknown_provenance`, `infeasible_dimension`, `resource_limit`, `degenerate_partition`,
`invalid_scale`, `invalid_projection`, `invalid_probabilities`, plus inherited
PCA `zero_scale`, `invalid_covariance` and `degenerate_input`. Unavailable values
are None with reasons and inspectable support. Malformed settings, explicit
raise-policy violations and invalid supplied geometry raise; numerical errors
are not hidden by a broad support-failure catch.

`graph_state_grid(data, features=..., basis=..., max_states=...)` retains all16
PC2..5 × bins3..6 configurations. It uses equal-width full grids and optional
explicit training positions, returning an owned `.table` and `.describe()`.
Every row has requested PCs/bins/states, rank, actual bins when available,
status/reason, reference/config identity, connectivity, occupancy and full result.
Median/minimum/maximum across available energies are descriptive sensitivity
summaries, not confidence intervals. No configuration is silently duplicated by
clipping an infeasible dimension. Policy recipes remain unresolved in
`AnalysisPolicy.method_spec()`. Use this separate API with explicit independent
geometry and settings; doing so does not resolve a named recipe.

`examples/collective_workflow.py build OUTPUT.json` runs a bounded synthetic
installed path across preparation, training PCA/projection, network, report/modern
coherence and graph-grid APIs plus missing/unknown/lineage exclusions. It writes
only the caller-selected summary. A second process using `reload OUTPUT.json`
checks the example version/digest and reads saved values without importing the
analysis package or refitting. This example-local JSON format is not the canonical
result format available through `tezca.exports`. Historical parity, field validity
and reuse rights remain unverified; all graph outputs remain exploratory.

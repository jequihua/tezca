# Saved numerical results

`tezca.exports` exports existing analysis results and validates them in a fresh
process. It never analyzes data, fits a reference, fills a gap, selects a model,
or reconstructs an executable estimator. There is no rendering API or dependency.
Downstream presentation code can follow the repository's separate
[`docs/visualization_guidelines.md`](../../docs/visualization_guidelines.md).

```python
from tezca.individual import describe_series
from tezca.exports import ResultItem, export_results, load_results

results = {
    f"description-{i}": ResultItem("metric", r, sources=["observations"])
    for i, r in enumerate(describe_series(observations))
}
manifest = export_results(
    "new-result-directory", results=results, sources={"observations": observations},
    requested_policy={"question": "raw descriptions"},
    resolved_policy={"support": "all_runs"}, compatibility=True,
)
# Another process, with no input measurements or fitted models:
saved = load_results("new-result-directory", expected_bundle_id=manifest["bundle_id"])
metrics = saved.table("individual_timeseries_metrics.csv")
```

Supply the existing source objects under explicit names. A `ResultItem` captures
an owned snapshot; changing returned records or the caller's containers cannot
change it. Supported source snapshots are `AnalysisSeries` and `DailyResidual`.
Only source/clock/metadata and fingerprints are exported, not their measurement
arrays. Source-details tables remain represented by their digest and compact
source record; they are not copied into every reference. Collective audit station
measurement values are omitted; contributor selection, rejected contributors,
coverage, lineage, reasons and qualifications remain. Numerical transformations
that are results (e.g. PCA scores) remain saved results.

`ResultItem(kind, result, sources=[...], labels={...}, timeline=None)` admits a
closed set of adapters. Labels provide optional site, family, station, segment
and variable context; they do not replace scientific identities. Result names
must be unique nonempty strings. Each result stores its full native description,
including requested/resolved settings, units, support, references, fits and
diagnostics. Explicitly provide available frozen references in `references=[...]`.
Their numerical descriptions are stored; models and transformers are excluded.
Some self-fit APIs expose only a reference identity and sufficient saved result
geometry. Such identities remain exactly as reported; their presence does not
claim a recoverable executable reference. The default reference expectation map
enumerates only explicitly supplied reference snapshots.
When a matching snapshot is supplied, passive checks bind copied training
support, settings and numerical reference fields to it, including PCA's saved
eigensystem and graph geometry. Training-support schemas remain checked when
only the reference identity is available. Training datasets need not appear as
evaluation sources, and loading never refits or reconstructs them.

Run the installed example outside the checkout:

```text
python -I -B <path-to-result_exports.py> build <new-directory>
python -I -B <path-to-result_exports.py> reload <directory>
```

`08_pkg/examples/result_exports.py` uses actual public APIs for all families,
including all-null and unknown inputs, unavailable long-period power, a complete
FI1 schedule with skipped windows, every graph configuration, and native regime
labels. Its custom fixture clusterer is explicit; it does not qualify sklearn.

## Table schema and joins

Format `tezca.results/1` emits all 17 canonical CSV files, even for an empty
family, plus `analysis_manifest.json` and `result_details.json`. All canonical
tables have the following **ordered, fixed** columns. CSV cells contain the
closed typed JSON vocabulary below; `payload` is a structured record, not an
opaque Python object or a JSON string to evaluate.

| Column | Logical type and meaning |
| --- | --- |
| result_id | String; caller's unique result name; joins manifest results and detail snapshots |
| row_key | Typed scalar/tuple; unique within result and table; original ordering is retained |
| metric_id | String; saved versioned method, or adapter category for composite reports |
| compute_status | Saved computation status, or null where a composite has no aggregate status |
| metric_status | Saved scientific qualification, independently nullable |
| use_status | Saved provenance/use qualification, independently nullable |
| reason | Saved reason or null; row-specific reasons take precedence |
| representation | Raw/residual basis, or null where the native composite records it per member |
| units | Symbol or keyed unit mapping, or null where units are in the native row/reference |
| config_id | Saved configuration identity or null |
| reference_id | Saved reference identity or null; never a model deserialization instruction |
| source_ids | Ordered list of keys into the source snapshot map |
| payload | Typed native record with the concrete shapes below |

Order is caller result insertion order, then original native row, frequency,
component or feature order. The primary key is `(result_id, row_key)`; an ordinal
row key is a stable location within the bound immutable result, not a substitute
for native row/window labels in its payload. Every file's schema, row count and
result selection are in the manifest. Loader validation requires the exact
projection of saved details to match every table byte. Separately, passive
checks enforce adapter/record fields, native status/value/unit invariants,
array dimensions, row identities and unique primary keys. Agreement between
hashes and projections alone cannot make a malformed record valid.
Numerical adapters also check real scalar/array/table element types, nullable
diagnostics and boolean decisions, including retained coherence support arrays
when its outer result is unavailable. Dedicated peak, spectrum, PCA, density and
coherence adapters accept their existing method IDs. The generic `metric`
adapter accepts any `ResultRecord`, but a native method ID always receives the
same concrete validation as its dedicated adapter (see *Native schema dispatch*
below); other explicitly declared caller methods keep typed caller data.
Native result representations agree with their retained signal support. Signal
basis, source and clock declarations bind to the matching declared input snapshot,
including the separate residual-reference identity. Collective feature support
keeps the same binding. Native individual, spectral, FI1, density and harmonic
signal methods require these declarations even when all identifying fields have
been removed. Harmonic's explicit refused-spectrum route may retain a source-free
caller record, but only as an unavailable `unsupported_estimator` result.
Generic records use caller method names and retain their caller-defined
representation and typed metadata. For example, an array-only `fit_log_models`
dictionary can be saved under `caller.array_log_fit_v1`; claiming the native
`spectral.log_models_v1` result also requires its spectrum-derived signal support.
Native individual results retain their observation policy alongside requested
and resolved settings. Reload checks selection positions, instants, runs,
durations, selected/full burden, support/configuration identities, and the
qualification implied by the saved burden and policy. ACF retains the actual
pair endpoints and lag; trend retains elapsed-day geometry and its uncertainty
declaration. Selection availability remains distinct from metric availability:
one eligible observation can have a primary qualification while sample SD or a
trend remains unavailable. These checks do not reevaluate measurements or fit a
trend/correlation. The compact source snapshot does not contain the full input
time index or cells, so it cannot independently recreate their selection.
Network and regime descriptions and collective contributor source records bind
to their declared inputs too; missing-timestamp audit entries retain their native
absence of a source record. Sensitivity spectra bind source declarations and
basis/reference identities to each route's signature and saved preprocessing
reference. Interval subsets keep their own signal identities, with no attempted
reconstruction of the original measurements or daily transformation.

| Kind / public result | Canonical table(s) and payload shape |
| --- | --- |
| metric / ResultRecord | individual_timeseries_metrics: complete ResultRecord (`value`, three statuses, reason, units, support, method/config/reference, representation). Descriptives, trend, ACF, spectral models/bands and each harmonic diagnostic remain separate rows. |
| peaks / ResultRecord | spectral_peaks: stored `rank`, original frequency `position`, `frequency_hz`, `period_seconds`, `power`, prominence, cycles, basis. Rejected peaks and selection policy remain in details. |
| spectrum / ResultRecord | example_spectra_long: every `frequency_hz`/`psd` ordinate, including unavailable irregular ordinates and their frequency qualification. Padded and physical Welch grids remain separately saved in details. |
| pca / ResultRecord | site_collective_summary: full ResultRecord including PC90 and participation ratio; pca_spectrum: `component`, `eigenvalue`, `fraction`; site_mode_loadings: `component`, `feature`, `eigenvector`, `scaled_loading`, both literal definitions. Full covariance, scores, scaling, training support and all eigenvalues remain in details. |
| coherence / ResultRecord | site_cross_spectral_pairs: complete ordered-pair ResultRecord with audit, support, aggregation, effective averages and null/threshold scale; cross_spectra_long: `route`, `run`, `frequency_hz` and each aligned stored ordinate (sxx, syy, sxy_real, sxy_imag, raw coherence, phase, correction/inference where supplied). Run and pooled/equal-run outputs are distinct rows. |
| sensitivity / SensitivityReport | spectral_gapfill_sensitivity: `category` routes/contrasts plus the entire native row: route/draw, source/reference/signature, estimate, fit/spectrum, signed/absolute contrasts or incomparability. No completion or estimator winner is chosen. |
| matrix / PreparedCollective | site_matrix_audit: source membership, settings, selected features and complete preparation description; matrix_support: one row per feature/cell with start/end/label, station contributors/rejections, coverage and unresolved-cell reason. Station input measurements are excluded. |
| fi1 / FI1Trajectory | individual_timeseries_metrics: complete stored trajectory summary; fi1_windows: every scheduled start/end/label, FI1, entropy, eligibility, partition, distribution, support, absolute delta and nullable flag. Threshold and flag availability are saved in details. |
| network / NetworkResult | dynamic_connectivity: complete native window summary, denominators, signed-network method/threshold and separate valid-pair subset summary; network_edges: every original signed unique-pair row with eligibility, actual support, status and threshold. Undefined edges remain undefined. |
| density / ResultRecord | density_fisher: complete ResultRecord, distinct coordinate/domain/reference identities, value and saved density/derivative/boundary/tail diagnostics. |
| graph / GraphGridResult | graph_state_sensitivity: every original PC/bin configuration, feasible or infeasible, including full result/geometry/energy/support/status. |
| regimes / RegimeResult | regime_labels: original `row_id`, start/end, native and display label, noise/excluded/failed `status`, reason. Description/reference and optional already computed `timeline` retain training, memberships and gap barriers; no model object is saved. |

### Native schema dispatch

Validation dispatches on the declared method ID, never on which fields happen
to remain, so removing support cannot bypass a check. Native IDs form a closed
set: the individual descriptives, elapsed-day trend and exact ACF;
`spectral.welch_mean_v1`, `spectral.ls_projection_density_wosa_v1` and their
summaries `spectral.log_models_v1`, `spectral.peaks_v1`, `spectral.band_power_v1`
and `spectral.irregular_log_models_v1`; Fisher/Siegel harmonic diagnostics;
`fi1.ordered_mass_roughness_v1`; histogram/KDE density Fisher; PCA projection;
graph mass/density roughness; the four coherence routes; and the foundation
`fixed_edges_and_population_scaling_v1` reference summary. An unknown method in a
reserved namespace (`individual.`, `spectral.`, `harmonic.`, `fi1.`,
`density_fisher.`, `pca.`, `graph.`, `network.`, `coherence.`, `regimes.`,
`collective.`, `daily.`) is refused rather than treated as a caller record.
The same checks apply through the dedicated and generic `metric` adapters and to
nested native records: FI1 trajectory rows, network windows/edges, graph grid
rows and observed-only counterparts, coherence observed-only counterparts and
sensitivity route spectra.

For each native signal method, the required support is checked before any
conditional branch: the shared saved selection (positions, instants, runs,
durations, selected/full burden, selection and support identities), the saved
observation policy, requested/resolved settings, the configuration identity
derived from those saved values, the constant metric qualification, output
units derived from the bound source variable's unit, and the use qualification
implied by the saved burden and policy. Saved regular timestamps keep the
retained clock's grid. Every saved run start obeys the `TimeAxis` rule
`(t_ns - phase_ns) mod cadence_ns = 0`; this includes Welch candidate runs that
contribute no segment. So does every complete-case selected instant (PCA and
graph). The rule is evaluated in exact integer nanoseconds: equivalent offsets
compare by instant, distances beyond int64 stay exact, and no tolerance or grid
is built. Run ends and bound instants follow from their exact offsets to the
checked start. A nonempty regular input must retain its phase; only an empty
axis may lack one. Irregular clocks keep their existing rules. Producer-specific
rules include:

- Welch: estimator/policy by clock (`regular`/`all_runs`, or the irregular-clock
  `point`/`irregular` unavailable shape with `resolved=None`, zero segments and
  no regular run or success diagnostics), availability implied by segment/point
  minima, and power units (`unit^2/Hz` or `unit^2`). Saved `WelchSpec` settings
  keep their exact types and domains. Before any status is derived, regular run
  support is rebuilt in producer order from saved integer geometry, cadence and
  the declared taper. The sampling frequency is exactly 1e9 / cadence (ns).
  Candidate runs are ordered and in range, separated by a missing cell or clock
  gap, and bound to the contributing instants. The all-runs or longest-run
  policy (first longest run on ties) fixes the selected positions and runs, and
  sensor-length segments use that pre-segment selected count. Candidates
  partition into used runs (original run plus segment count, used-prefix stop
  and duration; a trailing incomplete hop is unused) and rejected runs with the
  producer reason. Segment count, starts, stops and run links follow
  `floor((b-a-L)/H)+1` per used run, and the contributing selection has one run
  per used prefix. The longest duration is the largest used-prefix duration.
  Taper energy/sum and the effective-average overlap term follow the declared
  taper and within-run overlap only; zero segments keep a null value and reason.
  Successful cycles hold one finite value per display ordinate, equal to its
  frequency times that longest duration. Constant, zero-spectrum and cycle
  diagnostics exist only for a successful spectrum. The three regular summaries
  inherit all of these checks before their own gates. Only counts, cadence and
  taper coefficients are evaluated (no periodogram), and each generated sequence
  is first bounded by a supplied list. Eligibility of candidate cells that
  contributed no segment cannot be checked against omitted raw cells, and a
  fully self-consistent forged chain is not detected without external expected
  identities. Irregular LSFT-WOSA uses its `irregular` estimator,
  requested/resolved settings and segment resolution.
- Spectral summaries inherit the complete upstream support, including its
  `metric`, and add `spectral_input_id` and `summary_settings`; their
  configuration identity is derived from those two saved values. An unavailable
  upstream spectrum fixes the summary status. Upstream primary use becomes
  sensitivity-only. Log fits and irregular log fits are unitless and both
  validate the saved fit value; peaks keep spectrum units and their prominence
  unit; band power removes `/Hz`. Summary values are never compared with
  spectrum arrays. Every saved setting is checked against its spec constructor
  before any availability branch, including for an unavailable upstream: exact
  types, numeric domains (for example nonnegative float `min_cycles`, integer
  `min_arm`/`min_ordinates` >= 3, increasing positive float64 bounds), enums,
  fixed declarations and derived band frequency bounds. A recomputed
  configuration identity does not make a setting legal. Each producer branch
  keeps exactly its own support fields. Unresolved spectra keep their full
  excluded grid; log fits bind their selection, inherited range and
  availability (three eligible ordinates); explicit candidates and arm counts
  bind `min_arm`. Peaks and band power compare saved rows, physical ordinate
  counts, low-frequency cycles and denominators with their period, cycle,
  count and ordinate gates. The irregular adapter's saved selection must match
  its gates over the saved per-ordinate frequency, power and occupied span. No
  fit, peak search or integral is recomputed.
  Status claims cannot choose which decidable gates apply: band bounds are
  checked before ordinate and cycle minima, with inclusive equality; regular
  fits also bind their saved frequency exclusions/intervals to the physical
  grid. An empty peak selection remains `ok`. A null band denominator is the
  retained density-degeneracy diagnostic; its omitted input PSD cannot be
  recovered or independently certified by the summary loader. The complete
  producer decisions and controls are retained in development-only evidence
  unavailable in this checkout. See the [qualification and evidence limits](../../04_delivery/qualification.md)
  for the public summary; no omitted input evidence is recovered by reload.
  Declared FFT/segment counts are compared with supplied vectors before any
  count-driven iteration. The same checks apply to unavailable and generic
  native records and retained nested coherence support.
- Harmonic: settings, null, backend and method declarations; the status implied
  by settings, basis, support and clock; Fourier geometry only on a successful
  representable raw route; Siegel `not_computed` without calibration or the
  calibration's own status; the saved decision equals `p_value <= alpha`, Fisher
  `g` equals the largest saved proportion and Siegel's critical value equals the
  calibration snapshot. The source-free refused-spectrum route stays
  unavailable and never primary.
- FI1 and density bind requested/resolved settings, edge/geometry identities,
  measurement units and backend to the saved fitted reference. Their saved
  evaluation observation policy and post-exclusion positions/burden determine
  the use qualification; unknown fitted provenance dominates. Failed KDE
  integration diagnostics remain legitimate unavailable support.

Composite records apply their producers' aggregation rules: network edge use is
the stricter of its two complete pair selections, windows and the outer record
take the stricter retained statuses and units bind to source metadata;
coherence use combines both contributing selections, the minimum-support rule
and unknown lineage, with explicit unavailable historical routes; collective
feature availability binds to the cell audit; regime use combines selection and
training provenance; sensitivity routes take their nested spectrum's use and
contrasts become unknown only from an unknown endpoint. PCA and graph records
derive their metric qualification, configuration identity and output/coordinate
units from retained settings and training units, with or without a reference
snapshot.

Every bundle also validates the producer-written envelope: each source snapshot
has exactly its compact source record, variable metadata (unit symbol and
representation), native clock record, signal identities and SHA-256
mask/source/details fingerprints (details may be null), plus the residual
representation for a daily residual; the backend has exactly its software,
Python, Tezca and declared entries; references have identity, record and
content hash. Fingerprints are syntax- and content-checked where their content
is saved; a mask fingerprint cannot be recomputed from omitted cells.

These are passive comparisons of saved structure, counts, constants, derived
identities, units and status rules. They do not rerun FFT/OLS/PCA/KDE,
clustering or calibration, refit references, recover omitted measurements or
prove that a fully self-consistent record came from its claimed source.

An empty numerical table does not erase the request: the manifest identifies all
result IDs and their independent statuses/reasons, and details preserve their
records. Unrequested families have `not_requested`; externally unavailable
families can be declared with `unavailable={filename: reason}`. A supplied family
cannot also be declared unavailable. No numeric placeholder rows are fabricated.
All-null variables appear as the original per-metric missing results. A valid
mean does not change a failed band or harmonic status. Composite native records
can have different statuses for fits, thresholds or valid-pair summaries; their
keyed nested statuses remain intact rather than being promoted by the outer row.

## Typed CSV and JSON conventions

Both formats use the same tagged nodes. JSON is UTF-8, compact and deterministic;
CSV uses RFC4180 quoting with LF records and the literal header above. A standard
CSV reader returns cells which `json.loads` parses into these nodes. A string
`NA`, an empty string, false, zero, missing and nan are distinct.

| Node | Type |
| --- | --- |
| `["null"]`, `["na"]`, `["nat"]` | None, pandas missing scalar, missing instant |
| `["str", "text"]`, `["bool", false]` | Exact string / boolean |
| `["int", "9007199254740993"]` | Decimal integer with no floating conversion |
| `["float", "0x1.0000000000001p+0"]` | Exact binary64 via `float.fromhex`; nan/inf/-inf explicitly encoded; negative zero retained |
| `["complex", real, imaginary]` | Two typed signed binary64 components |
| `["timestamp", "nanoseconds", "Europe/Berlin"]` | Nanosecond instant and timezone; null timezone means explicitly naive |
| `["duration", "nanoseconds"]` | Exact duration |
| `["list", nodes]`, `["tuple", nodes]` | Ordered collections |
| `["map", [[key, value], ...]]` | Ordered typed unique keys and values |
| `["array", dtype, shape, flat_nodes]` | Numeric/boolean/string/object ndarray; no structured or executable dtype |
| `["index", labels, name, timezone]` | Ordered unique scalar/tuple labels; mixed logical types preserved |
| `["frame", index, columns, rows, attrs]`, `["series", index, name, nodes, attrs]` | Owned pandas numerical records with object cells; logical values retained without dtype inference |

NumPy scalar types normalize to their logical Python type. Labels must be unique
and nonmissing; collisions such as integer 1 versus boolean true or float 1.0
are rejected. MultiIndex and unrecognized metadata classes/sets are rejected
before any write. Use explicit tuple labels on a plain Index instead. This is
not arbitrary object serialization: no pickle, import-by-name, eval or network.

Array shape counts must match supplied elements before allocation. Numeric and
object element storage has fixed native size. For nonempty Unicode arrays the
total decoded storage must not exceed four bytes per character in that array's
canonical tagged JSON payload, so a short value cannot demand an enormous
fixed-width buffer. Exact dtype retention is conditional on this bound: empty
arrays, modest padding and inferred-width strings of similar lengths keep their
dtype. Fixed-width storage is the element count times the longest element, so
the bound also refuses unpadded inferred-width arrays whose longest string
exceeds the mean element length by more than the per-element JSON overhead
(roughly ten code points in large arrays), as well as excessively padded
storage. Such arrays are refused at export (before creating a directory) and
at reload; use object string arrays for skewed or padded text. This is an
input-size bound, not a spectral-size limit or a change to the wire format.

## Compatibility and formula separation

With `compatibility=True`, eight `compat_*.csv` companions reproduce the exact
frozen column names/order from `sensor_table_schemas.json`, including all 49
individual columns. Canonical files keep their names and separate identity.
`compatibility_status.json` contains one typed record for **every** compatibility
cell, keyed by table/row/field and linked to the canonical result/row, with
mapping, versioned formula, independent statuses, reason and Q001 qualification.
Independent results with different support/configuration are never collapsed
into a single apparently uniform historical row.

Direct descriptive values and elapsed-day slopes map to their corresponding
fields with their explicit formula IDs and `unverified_Q001`. Source/context and
unit/status fields map as declarations. Unadmitted historical algorithms remain
null, with `not_requested_or_historical_formula_unverified_Q001`; no scientific
meaning is inferred solely from a matching name. This includes independent
spectral selection/fits/bands/harmonic methods, irregular comparisons and corrected
network quantities. Non-72-hour FI1 rows explicitly report
`non_72_hour_trajectory` for the three `*_72h` fields. Even a 72-hour result has
no recovered historical FI1 parity authority. Full independent numerical outputs
are retained in canonical files/details, so a null historical field loses none
of the scientific result. The compatibility mode is a schema migration aid,
not an exact historical-results reproduction. Keyed numeric parity remains Q001.

## Integrity, ownership and freshness

The destination must be a new directory with an existing parent. Existing files,
directories, partial bundles and symlinks are refused without changing their
bytes. Payloads use exclusive creation; the completion manifest is written last.
A failed write leaves an incomplete directory for diagnosis; retry into a new
destination. A partial manifest is invalid JSON or fails complete-bundle validation.
No destructive cleanup is performed.

Each `result_hashes` entry is SHA-256 of that result's canonical typed JSON.
The source, reference, requested/resolved policy and backend identities use the
same convention for their respective detail maps. The writer reuses the checked
typed snapshot for file content and manifest hashing; the hash definitions and
canonical bytes are unchanged. Canonical checks and table cells serialize a
value directly to exactly the compact JSON of its typed nodes, without building
a second node tree. `result_details.json` must be exactly the canonical text of
its decoded value: equal JSON with other whitespace or escaping is refused as
noncanonical before schema validation, even when its hash was updated.

Reload checks format/package version, the complete closed filename allowlist,
regular files (no symlinks), no extra files, exact byte sizes/hashes, duplicate
JSON keys, typed schema, result/source bindings, frozen reference content
identity and all canonical/compatibility projections. Invalid manifest paths
are refused before opening a payload. The details file is checked first so a
malformed numerical record is rejected before reading its table projections.
A successful reload still validates every file. CSV/JSON tampering remains detectable even
when an isolated file hash is updated, because the tables must agree with the
saved details. An attacker able to replace an entire self-consistent bundle can
replace its self-reported hashes: they do not authenticate an author.

Each composite validates its evaluation inputs: collective members and audit
contributors; network windows, pairs and edges; graph candidate/evaluation
support; regime source and variables; and every sensitivity route, including
each supplied completion. Declare all these sources in the item's `sources`
list. Frozen training support may refer to a separate training dataset; it does
not silently become an evaluation input. An explicitly constructed generic
`ResultRecord` with no input identity has only caller-declared provenance.
Supplying such a record does not prove it came from a scientific function.
Signal IDs are checked against their saved source ID, variable and basis.
Native reference links are checked against the saved representation, metric fit
or composite reference identity as appropriate. A residual's daily reference
and a later FI1/density/PCA reference remain distinct. These checks use metadata
and numerical reference hashes; they do not reconstruct omitted measurements or
invent reference snapshots that a native result never exposed.
PCA and graph results also validate retained training declarations when the
reference snapshot is omitted: fit scope, feature/unit order, original and subset
positions, eligible instants, per-feature support and qualification counts.
The training source need not occur among declared evaluation sources.
Retained burden records distinguish linear influence, nonlinear contributor
coverage and the primary-use gate. Their fractions, counts and reconstruction
durations keep native types and relationships, including missing duration on
irregular clocks. PCA and graph use qualifications retain the stricter saved
training/evaluation qualification; unknown graph training provenance prevents
a successful numerical output.
Training qualifications must also agree with their saved selected burden and
observation policy: unknown-origin counts, primary-use fractions and duration
restrictions cannot silently retain primary eligibility. Residual training keeps
the earlier daily fit's qualification and reconstruction burden.
The same complete-vector schema applies to PCA evaluation and graph candidate
and contributing selections, including retained observed-only counterparts.
Feature support cannot disappear; ordered positions, per-feature selections,
instants and score rows must agree, and feature order/units bind to declared
source metadata, including inside graph counterparts. Available PCA and graph
values require their declared minimum training and evaluation support. Graph
exclusion keeps its requested subset distinct from the full input support:
contributing, excluded and outside positions agree with their counts, and saved
node counts sum to the contributing population. Outside-partition exclusions
remain distinct from in-partition cells absent from a reference-node universe.
Unavailable fitting/projection can prevent partition evaluation altogether;
those records retain empty contributing/exclusion declarations.
The observed-only graph counterpart retains the minimum count while its native
policy resets imputation-fraction and reconstruction-duration limits to null.
Native PCA geometry checks also apply when a caller selects the generic metric
adapter. Graph payload partitions and realized bins agree with retained support;
mass/density methods, units and volume declarations agree, and the two saved
smoothed-probability fields must be identical. These bindings do not require an
optional reference snapshot or numerical refitting.

`expected_bundle_id` checks the whole scientific bundle, including output bytes
and selection. `expected_sources` and `expected_references` must be exact maps
matching the respective manifest identity maps; they detect a different supplied
source/reference selection, not changes in a remote data store. Without external
expected identities, internal consistency proves no external freshness. Hashes
also grant neither data rights nor scientific validation. Software/backend,
requested policy, resolved policy, source, mask, reference, formula and output
identities remain explicit. Scientific identity excludes paths and operational
timestamps; equivalent exports to two new destinations have identical bytes,
also from fresh processes with different Python hash seeds. Since S01 r33,
internal qualification flag maps (mask fingerprints, collective contributor
audits) use one fixed sorted order. Bundles written earlier kept a
seed-dependent order, so re-exporting them can change mask/source digests,
collective result hashes and the bundle identity; the earlier bundles still
reload unchanged and stale-selection checks remain exact.

Loaded tables/details/manifest are defensive copies. The loaded object exposes
no fitting, prediction or plotting capabilities. Future analysis must explicitly
use admitted inputs and its own scientific APIs.

Composite rows keep their native geometry: FI1 windows use aware ISO strings;
network windows and collective cells use aware timestamps. Validation checks
FI1 rows against the retained schedule and counts, network windows/ordered pairs
against declared denominators and duplicated summaries, matrix cells against
their declared grid and station coverage denominators, and graph configurations
against saved settings and numerical results. Unavailable graph fitting may
leave fitted rank/state counts absent while preserving the requested grid.
Sensitivity routes retain each completion's draw label and provenance, including
unstarted draws. Regime timeline rows match the saved regime rows; episode and
transition fields bind labels and geometry to those rows. Empty native tables
and unavailable values are retained. These checks compare saved structure and
metadata; they do not recompute estimates or fit models.
Episode membership and transition links respect excluded/noise barriers and
the declared maximum gap. Continuity uses the whole episode's covered extent,
including nested or overlapping feature intervals.

Overall FI1 availability must agree with its eligible-window count. Network
thresholds, window settings and observation policies retain native types and
agree across descriptions, window support and edges. Successful sensitivity
routes retain their spectrum, fitted model and matching saved estimate;
contrasts retain all declared endpoint pairs. Regime cluster-size mappings,
display mappings and fractions agree with the saved row counts and denominators.
Unavailable native routes may retain a spectrum without a fit, and an empty
regime denominator retains null fractions.
FI1 changes are available only between adjacent successful windows at the saved
stride; the first window and windows separated by unavailable support have no
delta or flag. Absolute changes and their summary scales are nonnegative.
Entropy permits the native mass-normalization roundoff tolerance of 1e-12.
Network correlations stay in [-1, 1], counts are integers bound to retained
eligible edges, and densities/connectivity stay in [0, 1]. Partial-pair summaries
retain their own denominator and availability.
Edge positions and instants agree with retained pair support and window bounds.
FI1/density evaluation counts bind to post-exclusion positions and available
distribution/histogram totals; eligible support before exclusion stays separate.
Collective contributor membership and totals use station availability reasons,
retaining audits for noncontributing stations with failed coverage or reduction.

Saved log-model fits require selection geometry, fitted candidates and explicit
preferred-model/AICc availability. Exact fits and insufficient AICc support
retain null comparisons. Sensitivity contrasts require compatible saved
signatures and available endpoints before retaining numerical deltas. Successful
Welch PSD vectors must be finite; irregular spectra may retain missing ordinates
at unsupported frequencies while other frequencies are available. Each irregular
ordinate agrees with its saved frequency-support value and availability.
Successful Welch display and physical grids retain every bin implied by their
saved FFT geometry. Equal-length grids retain identical power copies; odd FFT
lengths and noninteger padding ratios keep their native supported intervals.
Power ordinates, FI1, density information and graph energy cannot be negative.
PCA leading summaries agree with their retained arrays and count domains.
PCA arrays, FI1 distributions, density geometry and graph numerical arrays must
be finite. Native probability vectors are nonnegative and normalized; bin edges
are ordered. Graph nodes and edges must belong to the saved state grid, with no
duplicate nodes, duplicate undirected edges or self edges. Generic typed metadata
keeps the declared explicit nonfinite encoding.
Peak ranks are consecutive integers; positions, frequencies, periods and power
retain native domains and saved support. Band power retains its explicit fraction
status, including a null fraction at a zero denominator. Harmonic probabilities
and significance levels keep their native bounds, and saved fit residual sums
cannot be negative. KDE tail mass is probability-valued, and graph range summaries
bound the successful saved energies. These checks do not rerun the calculations.

Coherence inference preserves native null corrections and decisions for single
effective averages or unsupported null models. Available correction arrays and
decision vectors must align with the owning frequency grid, including inference
retained only in support. Required fields are checked separately for accepted
run spectra, pooled spectra/inference and report aggregates; successful values
must agree with their retained pool or aggregate. A report aggregation with
fewer than three supported
bins retains its proposed grid and empty, unavailable ordinate array; it does
not acquire synthetic estimates during export.
Successful pool/aggregate band means and available white-noise thresholds and
significance fractions must be finite numbers. Retained run, pool and aggregate
statuses must be native text values with a reason only when unavailable; an
unavailable outer result can still retain a successfully computed pool.
Coherence frequency grids are finite and ordered. A successful report aggregate
has finite raw coherence; constant-input runs and unavailable modern ordinates
may retain their native missing values.
Modern FFT display spacing is distinct from physical resolution when padding is
requested; all displayed frequencies survive round-trip, while inference stays
qualified to physical bins. Saved modern band positions agree with the requested
period band, successful bands have finite raw ordinates, and auto spectra cannot
be negative. Positive effective averages bind correction and threshold
availability, retaining unavailable inference at a single effective average.
Corrections and decisions are unavailable at padded, DC and Nyquist bins and at
missing raw ordinates. Their availability at physical bins, the physical-band
count, and the effective segment denominator agree with retained grid/run
support. Coherence means, fractions and available thresholds stay in [0, 1].
Finite modern phase angles stay in [-pi, pi] radians.

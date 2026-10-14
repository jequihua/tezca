# Optional EcoFill inputs

The explicit `tezca.adapters.ecofill` module converts public EcoFill objects or
result CSV/JSON exports into the existing owned `AnalysisSeries`. It performs
validation and mapping, with no reconstruction, model execution, source lookup
or automatic installation. Tezca's dependencies and optional extras are unchanged.
Importing Tezca or this adapter does not import EcoFill; only object conversion
requires it. Export loading works in the core-only Python 3.11 environment.

## Admitted compatibility

Object conversion admits EcoFill package/module version 0.0.1. Qualification is
limited to an admitted development snapshot (its compatibility record is
unavailable in this checkout), whose local wheel SHA-256 is
`97679bb0f22ffffadfd67434904d174985e2f283ce9b9a6c69cbd96754b40d9a`.
The architect's optional job checks that wheel before installing it. A matching
version string alone does not certify another build; no other release is
qualified. The public schema uses `time_policy="strict"` and regular consecutive
aware timestamps. Clock repair, missing-interval compression and schema migration
are not performed. EcoFill metadata requires Python >=3.12,<3.13; its object
job runs separately from Tezca's accepted Python 3.11 core/YAML jobs.

The loader admits result export `format_version` integer 1, rejecting booleans
and other versions. Format 1 does not identify an authenticated producer version
or provide table checksums. `source["ecofill"]["contract_version"]` identifies
the admitted interface, not proof of artifact origin or integrity. Structural
checks do not certify that an imputed value was produced by a particular method.
No local EcoFill wheel or external source code ships with Tezca.

## Conversion

```python
from tezca.adapters.ecofill import from_ecofill, load_ecofill_export

declarations = {
    "source": {"id": "caller-source-v1", "segment": "caller-segment"},
    "representation": "raw",
    "clock": {"civil_timezone": "UTC", "clock_resolution": "resolved"},
}
analysis = from_ecofill(public_series_or_result, **declarations)
reloaded = load_ecofill_export(export_directory, **declarations)
```

The caller supplies source identity (segment, or site/station/instrument),
representation (one string or a mapping for exactly the variables), civil
timezone and resolved-clock declaration. Storage timezone, cadence, variable
order and units come from the schema. Optional `clock.phase` must agree with
the grid; otherwise the first timestamp records its constant phase. Source
identifiers are never inferred from filenames. Reserved source keys `ecofill`,
`imputation`, `trusted_imputation` and `details` fail rather than overwriting declarations.

| Public input | Preserved analysis meaning |
| --- | --- |
| `SeriesData.available` | Sole SeriesData numerical input; finite cells must be donors |
| `ImputationResult.completed` | Sole result numerical input; every finite cell must be donor or imputed |
| `mask("donor")` | Observed current contribution; `observed` alone cannot expose hidden truth |
| `mask("imputed")` | Imputed current contribution, independently of original qualification |
| `raw_present`, `observed` | Original present, quality-accepted flags |
| `artificial`, `qc_rejected`, `structural`, `inserted`, `absent` | Each original qualification independently preserved |
| `schema` | Units, order, role and parent names; original schema saved in provenance |
| `methods`, `states`, `unresolved_reasons`, optional `quality` | Owned aligned `source_details["cells"]` tables, keyed by `method`, `state`, `reason`, optional `quality` |
| `provenance()` / public fingerprint and plan | Complete `source_details["provenance"]`, including fingerprint, schema, timestamps, plan, backend and nested metadata |

The plan ID hashes the complete normalized JSON plan. The imputation method ID
`ecofill:per_cell_methods_v1` refers to the preserved heterogeneous method table;
it does not replace that table with an arbitrary single algorithm. Schema parent
names are metadata, with validated acyclic relationships; they do not establish
contributor weights or create aggregation lineage. Artificial parents must hide
their observed descendants. Opaque fingerprints may depend on hidden truth,
so numerical input/eligibility invariance does not imply fingerprint invariance.

The adapter never reads `.raw`, private fields, or `raw.csv`. Imputation trust
defaults to false. Only explicit `trusted_imputation=True` enables supplied
imputed cells under the existing `ObservationPolicy("trusted_imputation")` gate.
Original QC rejection/absence/insertion can coexist with an imputed replacement;
finite structural cells are always invalid. Missing values remain missing and
all-null records remain representable. Integer measurements with missing values
retain signed/unsigned 64-bit precision; mixed object values that would round an
integer are rejected instead of silently changing it.

## Unreleased source-detail accessor migration

M001-S06 moves the former `analysis.source["ecofill"]["provenance"]` and
per-cell `cells` list into `analysis.source_details`. This development API
migration preserves their complete contents; no legacy list is retained.
Use an explicit detail read when the full history is needed:

```python
details = analysis.source_details  # fresh defensive copies
producer = details["provenance"]
methods = details["cells"]["method"]
method = methods.loc[timestamp, variable]
reasons = details["cells"]["reason"]
quality = details["cells"].get("quality")  # absent if the producer omitted it
```

Each cell field is a pandas DataFrame with the measurement frame's timestamp
and variable order. The returned tables share copied coordinates; internal
storage uses one aligned array per field and shared coordinates, with no dict
per cell. State/method/reason are strings or `None` (state is never missing).
Quality is finite float or `None`. Literal `NA` remains a string. A SeriesData
input has an empty cell-table mapping and retains its full public fingerprint,
schema and timestamps in producer provenance. Detail reads copy the entire
payload and can be expensive; source reads and reference fitting/description
do not read or copy it.

`analysis.source["ecofill"]` now contains only distribution, admitted contract
version, route and opaque data fingerprint. `analysis.source["details"]` holds
`representation_version=1`, the computed SHA-256 `digest`, row/variable counts
and sorted `cell_fields`. Trust and the deterministic imputation plan ID remain
in source. Producer timestamps, schema, plan gaps/events and backend records
reside solely in the separate detail payload. Frozen references retain the
originating series identity and this compact source summary, without detail
tables or hidden producer copies.

The digest is computed once when AnalysisSeries owns the details. Its canonical
input contains representation version 1, ISO-8601 timestamps in row order,
variable names in column order, complete normalized producer JSON, and each
cell field's object-array dtype, shape and row-major values. Mapping keys are
sorted; JSON uses compact separators, UTF-8, unescaped Unicode and no nonfinite
numbers. Missing table values (`None`, NaN or pandas NA) canonicalize to JSON
null; quality numbers canonicalize to floats. Optional quality absence stays
distinct from a present all-missing quality table. Method, reason, quality,
plan or backend changes affect both detail digest and series identity, even
with identical measurements. Callers cannot provide a replacement digest with
details. This content binding does not authenticate the producer, and previous
development hashes are not preserved. Route remains in source rather than the
detail digest, so equivalent object/export details have identical digests while
their series/reference identities may differ by route.

## Export parsing

Only `provenance.json`, `completed.csv`, `masks.csv`, and `cells.csv` are opened.
`raw.csv` can be missing or poisoned. The JSON `files` list is retained as metadata
and never followed. Loading does not write, repair or reorder an input bundle.
JSON rejects duplicate keys and nonfinite values, and validates required schema,
plan, timestamp, mask inventory and state-count records.

Completed and mask rows must have identical ordered, unique, aware timestamps
and exact declared headers. Timestamp offsets must agree with the declared
storage timezone. Mask headers are constructed from the declared inventory and
variable names, allowing names containing `__` without heuristic splitting.
Duplicate headers, missing/extra records and malformed row widths fail. Masks
accept only literal producer `True`/`False` encodings; blanks/truthy strings fail.
Numeric blanks represent missing values, and nonfinite numeric literals fail.

Cells align by timestamp and variable, permitting shuffled rows but requiring
exactly one row per key. Method and reason blanks become `None`; literal `NA`
remains text. State counts must match the table. Observed/imputed state labels
must agree with masks, imputed cells need a method, and unresolved finite cells
are rejected. Optional quality must be declared in provenance and present as a
finite numeric/missing column; it remains producer metadata, not calibrated
uncertainty. Input tables and nested returned records are defensively copied.

## Installed producer and fresh reload

This checkout contains `08_pkg/examples/ecofill_bridge.py`. Open PowerShell
in the source checkout root. The producer needs an operator-supplied environment containing
this candidate's Tezca wheel, its numerical dependencies and the admitted
EcoFill 0.0.1 snapshot on CPython 3.12.10. The candidate does not include that
EcoFill wheel or environment. Use the core-only Python 3.11 environment from the
[quickstart](../../04_delivery/guides/quickstart.md) for reload. Exact environment
evidence is in the [qualification record](../../04_delivery/qualification.md).

Choose unused sibling output names. These commands keep all generated outputs
outside the source checkout, capture the producer JSON and retain its
bundle identity for the next process:

```powershell
$EcoFillPython = Read-Host 'Path to the admitted EcoFill environment python.exe'
$ProducedText = & $EcoFillPython -I -B .\08_pkg\examples\ecofill_bridge.py produce ..\tezca-ecofill
if ($LASTEXITCODE -ne 0) { throw 'Production failed; preserve the outputs and inspect the error.' }
$Produced = $ProducedText | ConvertFrom-Json
$ExpectedBundleId = $Produced.result_bundle.bundle_id
$ExpectedBundleId
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\ecofill_bridge.py reload ..\tezca-ecofill --expected-bundle-id "$ExpectedBundleId"
```

Record the displayed bundle ID with your candidate/run record before closing the
PowerShell session. In a later session, restore `$ExpectedBundleId` from that
separately trusted record. The argument is the ID string, not a path to an
identity JSON file. Omitting it still validates the bundle internally but does
not check that it is the previously selected bundle.

**Retain both output directories together:**

| Output | Contents and use |
| --- | --- |
| `..\tezca-ecofill\` | EcoFill's public producer export. The adapter reads `provenance.json`, `completed.csv`, `masks.csv` and `cells.csv`; the producer also writes `raw.csv`, which Tezca never reads. |
| `..\tezca-ecofill-results\` | Tezca's numerical result bundle, with `analysis_manifest.json`, `result_details.json` and the canonical CSV tables. It retains observed-only and provided-value descriptions plus this synthetic example's complete cell provenance and masks in result labels. |

The general naming rule is `<directory>-results`, beside the supplied producer
directory. The example's `reload <directory>` **always reads both directories**;
passing only the producer export or moving one directory without its sibling
is insufficient. Do not pass the `-results` path to this command. The standalone
`load_ecofill_export()` API still reads only the producer's four required files;
the example additionally calls `load_results()` on the sibling bundle and checks
its saved provenance/masks against the converted export.

Production constructs public schema/data/plan/result objects with eight synthetic
hourly slots and supplied completions, checks preservation of those public inputs,
converts them and exports the descriptions. It does not run an imputation executor
or use real measurements. The artificial canary `999999` never enters analysis
or the numerical bundle. Original artificial/QC/structural/inserted/absent masks
remain separate. There are six finite cells, three imputed: provided-support
burden is `3/6`, while all eight timestamps remain present.

The producer JSON and fresh reload JSON should agree. They include complete
`source_details`, qualification/contribution masks, observed-only positions
`[0, 6, 7]`, and `result_bundle` with its bundle/source identities and 14 metrics.
For `observed_only:individual.mean_v1`, expect value `160/3`, positions `[0, 6, 7]`
and use status `primary_eligible`. For `provided:individual.mean_v1`, expect
value `290/6`, positions `[0, 1, 4, 5, 6, 7]` and `sensitivity_only` under the
declared `0.15` burden gate. The example removes only the adapter route marker
(`result_object` versus `result_export`) from its semantic comparison; ordinary
series/reference identities can differ because their source retains that route.

Reload uses core Tezca without EcoFill, analysis or fitting. A wrong
`--expected-bundle-id` is refused as a stale bundle selection; missing sibling
files, changed payloads and invalid schemas also fail. Keep the producer and
bundle unchanged while investigating. Both destinations must be new: the
producer is written before the numerical bundle, so a later refusal can leave
an incomplete pair. Preserve it and retry with an entirely new base name.
Hashes and a self-reported ID do not authenticate the producer or grant rights.
The existing optional witness checks fresh-process equality, hidden-truth
exclusion, wrong-ID refusal, no EcoFill/raw-file access and unchanged bytes of
both directories; see [qualification and evidence limits](../../04_delivery/qualification.md).

## Evidence boundary

Development tests and native verification commands are unavailable in this
checkout. The [qualification record](../../04_delivery/qualification.md) summarizes
accepted cumulative core and separate YAML/EcoFill/models receipt counts and
exact environments. The EcoFill job had four required tests on Python 3.12.
Use the standalone installed producer/reload commands above only with the admitted
optional environment; its wheel is not bundled. Other EcoFill versions/backends,
reconstruction algorithms and real-data admission remain outside this adapter's
qualification. The example uses the separate [Tezca result format](exports.md);
it is not a broader EcoFill compatibility claim.

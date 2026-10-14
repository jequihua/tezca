# Migration and current limits

[Start here](../README.md) · [Qualification evidence](../qualification.md)

This source projection keeps package version `0.1.0.dev0`. Use the source commit,
dirty flag, manifest hash and file hashes from `.front_repo_sync.json`, together
with your built wheel's raw SHA-256. The earlier wheel-only candidate and its
index are not included. Version or source commit alone cannot distinguish
uncommitted overlays or locally rebuilt wheel bytes.

## Moving an existing workflow

1. Declare units, source identity, time geometry and contribution masks before
   analyzing. Generic finite values are unknown until their origins are declared.
   Keep gaps and original qualification separate from supplied replacements.
2. Choose explicit method specifications and a raw or residual branch. Do not
   treat named recipes as fully executable historical pipelines.
3. Fit references only on the intended training data and retain their identities.
   Self-fitting on evaluation data is retrospective and must stay labelled so.
4. Export canonical results first. If another tool needs historical table names,
   request `compatibility=True` and read `compatibility_status.json` for each
   field's formula, qualification and missing-value reason.
5. Keep the producer's bundle/source/reference identities separately and provide
   them when loading in another process. Compare exact saved values before doing
   any new downstream transformation.

The old policy name `manglaria_v1_10_compat` requires an explicit migration target:
`manglaria_sensor_20260911_compat` or `manglaria_report_v1_10_compat`. The
[45-leaf migration table](../../08_pkg/docs/policy_migration.md) documents mapped,
changed and unresolved fields. `manglaria_research_v1` is separately identified.
Passing strict YAML validation records choices; it does not supply missing
historical methods or resolve an executable spec through `method_spec()`.

Compatibility exports preserve names/order, including the 49 individual columns.
Independent spectral, harmonic and irregular estimates and corrected network
formulas do not become historical results because their labels look familiar.
Unadmitted historical mappings remain null with explicit reasons. A non-72-hour
FI1 trajectory does not populate historical `*_72h` values as if it were 72 hours.
Use [canonical details](../../08_pkg/docs/exports.md) for the complete result.

Earlier development bundles may have seed-dependent qualification-map order.
Re-exporting equivalent old inputs with the current fixed order can change
mask/source digests and bundle identities; original old bundles still reload
unchanged. Preserve old identities rather than retroactively replacing them.

## What reload can and cannot establish

Reload validates saved bytes, schema, numerical domains, available internal
relationships and expected identities supplied by you. It does not execute a
model or reconstruct omitted source measurements, candidate eligibility,
raw-value evidence for a constant input, or omitted summary spectra. A coherent
replacement of all saved metadata may require an independently trusted identity
to detect. Integrity is not authenticity, freshness, scientific validation or
permission to use or share data.

Canonical exports support documented typed scalars, arrays, frames and labels.
MultiIndex and arbitrary Python objects are not supported. Fixed-width Unicode
arrays with excessive padding or highly uneven string lengths are explicitly
refused by a payload-relative allocation bound; object string arrays avoid that
fixed-width issue. See the [full typed encoding](../../08_pkg/docs/exports.md#typed-csv-and-json-conventions).
Loaded accessors return defensive copies; loaded results cannot predict, refit
or reconstruct an executable estimator.

## Evidence and authorization limits

The [qualification appendix](../qualification.md) separates local engineering
checks, scientific/domain evidence and authorization to distribute. Historical
parity needs pinned maintainer sources/configurations/inputs/outputs. Real-data
claims need approved fixtures, clock/unit/QC/lineage evidence and admitted
evaluation rules from the data steward. Reuse and sharing need explicit terms
from the owner and relevant rights holders. These remain open; synthetic
examples settle none of them.

Exact environments are recorded in qualification; declared Python/dependency
ranges are not a supported-platform matrix. Other platforms and untested
versions remain unqualified. Host symlink execution was unavailable in the
earlier qualification, and an earlier unexplained example identity difference
remains recorded. Tooling qualification is limited: a retained output pipe can
leave verifier reader cleanup waiting without a deadline. This carried tooling
issue is not repaired by the candidate's numerical tests. The detailed appendix
also preserves the prior baseline-recovery qualification limits.

Metrics do not by themselves establish ecosystem health, causality, long memory
or natural-gap reconstruction accuracy. Regimes are exploratory clusters. There
is no automatic completion, model serialization, plotting API, renderer or
notebook implementation. Future external result presentation should follow the
[visualization guidelines](../../docs/visualization_guidelines.md).

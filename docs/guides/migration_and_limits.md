# Compatibility and interpretation limits

The [45-leaf policy migration table](../policy_migration.md) separates supported,
derived and unresolved declarations. Named policy recipes may remain unresolved
even when a separate method API exists; use explicit scientifically justified specs.

Compatibility CSVs are migration aids. Historical formula equivalence remains
unverified: matching names, units or shapes does not establish numerical parity.
Keep [canonical details](../exports.md) with support, status, reason and identity.
Stable compatibility identifiers are preserved so old saved results still reload.
Their question-like suffixes are legacy wire vocabulary, not instructions to
consult another repository. Plain-language reasons explain the limitation.

Frozen references bind training data and settings. A clean evaluation does not
erase imputed/unknown training support. Separate observed and supplied results,
raw and residual bases, estimator disagreement and reconstruction effects.
Missing values are not zeros. Natural gaps have no observed truth RMSE.

Exported checksums establish byte consistency, not authorship, permission or
scientific validity. Expected identities must be trusted separately. No executable
model is loaded. Excessively padded/uneven fixed-width Unicode arrays can exceed
allocation bounds; use [object-string arrays](../exports.md#typed-csv-and-json-conventions).
Read [qualification](../qualification.md) before interpreting results on real data.

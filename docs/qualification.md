# Tested behavior and scientific limits

Tezca `0.1.0.dev0` has synthetic engineering controls, not field validation or
proof of equivalence to historical algorithms. The complete scientific regression
set contains 149 core tests including 127 required identities, plus YAML 3,
EcoFill 4 and models 6 controls. This checkout's smaller [portable test suite](contributing.md)
covers installed input/support, literal statistics/spectrum and export/reload.
It is not the complete historical regression suite.

The tested Windows x64 environment uses CPython 3.11.9, NumPy 2.4.6,
pandas 2.3.3, SciPy 1.17.1, python-dateutil 2.9.0.post0, pytz 2026.5,
tzdata 2026.5 and six 1.17.0. Builds use setuptools 84.0.0, wheel 0.48.0
and packaging 26.3. YAML uses PyYAML 6.0.3. Models controls use
scikit-learn 1.8.0, joblib 1.5.3 and threadpoolctl 3.6.0.
EcoFill object controls use the tested 0.0.1 artifact on CPython 3.12.10.
Declared dependency ranges do not qualify every combination or platform.

The standalone checks cover local, editable, wheel and a commit-pinned local Git
install with normal isolated builds. They exercise observed/supplied delivery,
fresh expected-identity reload, stdlib reading, 18 families, 261 exact calling forms
and literal controls: eigenvalues `[4,1]`, power `0.5`, native labels, frozen
training support, skipped windows, unknown provenance and withheld bands.
The new GitHub revision's HTTPS install remains to be checked after it is available.
The package does not import optional or plotting libraries during core use.

| Evidence | Scope and limit |
| --- | --- |
| Definitions | Literal FI1, Gaussian Fisher, graph energy and covariance controls test specified mathematics, not ecological interpretation. |
| Signal controls | Synthetic periodic, phase/delay, break and white-noise controls exercise selected estimators and assumptions, not field calibration. |
| Saved contracts | Checksums, typed values, identities and no-analysis reload establish consistency. They do not authenticate authors or reconstruct omitted raw values. |
| Historical equivalence | Unverified. Comparisons require original pinned helpers, exact inputs, configuration and outputs with predeclared tolerances. Matching historical column names is not evidence of parity. |
| Real data | Unverified. Access/retention, units, clocks, QC, deployment/lineage, splits and tolerances need domain validation before trials. Examples are synthetic. |
| Reconstruction sensitivity | Matched estimator/support contrasts are sensitivity, not calibrated uncertainty. Natural gaps have no observed hidden-truth RMSE. |

Reload cannot recover omitted raw cells, summary PSD or eligibility of omitted
noncontributing candidates. Coherent replacement requires a separately trusted
expected identity. No executable model is deserialized.
Excessively padded/uneven fixed-width Unicode arrays can exceed export allocation
bounds; use the [documented object-string representation](exports.md#typed-csv-and-json-conventions).
No automatic completion, plotting API or notebook implementation is included.
Historical Windows symlink checks and a past example identity discrepancy remain
limits of earlier observations. No new scientific or platform claim follows from installation.

No license has been selected. Local access does not establish permission to reuse
external code/data or redistribute Tezca. Determine relevant rights before sharing.

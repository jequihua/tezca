# Candidate clusters and support timelines

`tezca.regimes` groups explicitly selected feature vectors. Its labels describe
candidate clusters, not ecological states or change points. Core imports,
custom factories, comparison and timelines work without scikit-learn. Native
KMeans/DBSCAN and StandardScaler use the declared optional `tezca[models]` extra;
an absent extra raises an actionable ImportError, without installation or fallback.

Construct `RegimeFeatures(data, features=..., definitions=..., row_ids=...,
intervals=...)` from an `AnalysisSeries`. Feature selection and original unique
string row IDs are mandatory. Each definition has `method_id`, `reference_id`
(explicit None is allowed), and a nonempty `window` mapping. Ordered feature
names, units and raw/residual basis come from the source metadata. Definition,
reference and window changes alter schema identity. No numeric-column discovery
or automatic legacy FI feature exists. Finite values with unknown provenance
remain unknown; the caller must retain the source contribution and qualification
layers when constructing derived feature observations.

Row starts are the source's ordered, unique aware timestamps. `intervals` gives
aligned exclusive ends greater than starts; omission declares point observations
with zero covered duration. Overlapping windows are supported. Reversed or
duplicate starts, reversed intervals and invalid IDs fail. Schema definitions
record the window method; row geometry records actual support.

```python
spec = ClusterSpec("kmeans", n_clusters=2, n_init=10,
                   random_state=74003, algorithm="lloyd")
fit = cluster_regimes(training_features, clusterer=spec, preprocess=None,
                      fit_scope="training_only", training_ids=training_row_ids)
assigned = assign_regimes(evaluation_features, reference=fit.reference)
timeline = summarize_regime_timeline(assigned, max_gap="2h", noise_policy="barrier")
```

`preprocess` is required, including explicit None. Native StandardScaler is
cloned unfitted; custom preprocessing uses the checked factory protocol below.
Mixed units require declared preprocessing or a nonempty `metric_interpretation`
that explains the supplied coordinate geometry. Only selected eligible training
rows fit preprocessing and clustering. Default `fit_scope="self"` is explicitly
retrospective; `training_only` requires unique declared IDs. Non-training rows
remain in the fit result as exclusions. Assignment transforms evaluation rows
through the frozen transformer and calls genuine `predict`, never fit or refit.
Evaluation schema/order/units/definitions must equal the training schema.

The reference separately retains `preprocess_fit_support` (every row supplied to
transformer fitting) and `support` (rows retained for clustering). A row dropped
by the transformer can still affect its fitted parameters; its unknown origin
or reconstruction burden remains in preprocessing support. Reference/result
qualification includes both supports and survives subsequent assignment.

Declarative KMeans requires positive integer `n_clusters` and `n_init`, plus an
integer `random_state`. Initialization, algorithm, maximum iterations and
tolerance are saved. DBSCAN requires positive `eps`, integer `min_samples` and
an explicit metric; K and randomness arguments are rejected. Supplied native
KMeans/DBSCAN objects use independently cloned unfitted estimators, with their
resolved configuration recorded. DBSCAN native -1 is noise and has no future
assignment. All-noise is a valid result with zero clusters. Native capabilities
follow [KMeans](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.KMeans.html),
[DBSCAN](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.DBSCAN.html)
and [clone](https://scikit-learn.org/1.8/modules/generated/sklearn.base.clone.html).

`CustomClusterer(factory, config=..., version=..., assignment=False,
sample_weight=False, soft=False, precomputed=False, randomness=False,
noise_labels=(), soft_meaning=None)` declares a small adapter. The factory must
produce independently owned unfitted instances; shared instances and conventional
fitted state are refused. Factories are called twice to check independence and
must not rely on invocation side effects. The estimator implements `fit_predict`
or `fit` plus `labels_`. Assignment requires `predict`. Arbitrary negative labels
are ordinary groups unless explicitly named in `noise_labels`. Configuration
and version are required and serializable; unsupported requests fail explicitly.
Factories remain responsible for supplying fresh custom state beyond conventional
fitted attributes. No sklearn base class is required.

`CustomPreprocess(factory, config=..., version=..., row_map=False)` implements
`fit(DataFrame)` and `transform(DataFrame)`. Transform returns a finite numeric
DataFrame whose index names original row IDs. With row_map=False the IDs/order
must be unchanged; with True an explicitly checked unique subset/permutation is
allowed. Dropped rows remain `excluded_quality` with a transformer-map reason.
Array-only, foreign and duplicate identity maps fail. Native preprocessing is
limited to StandardScaler, whose row-preserving behavior is known; custom
transformers must use this identity protocol. Assignment also validates the
transformed feature columns.

Sample weights require a supported adapter and a Series exactly aligned to
every original row ID. Finite nonnegative values, positive total and positive
selected total are required. The actual transformed-row weights are forwarded;
their original mapping binds the reference identity. Supported precomputed mode
requires a DataFrame with the same ordered row and column IDs, finite nonnegative
entries, symmetry and zero diagonal (absolute tolerance 1e-10). Both axes are
restricted to eligible training IDs. Scaling is bypassed and assignment in this
mode is explicitly refused. Missing matrices and unsupported modes raise.

Soft output is available only to a custom adapter declaring a nonempty
method-specific `soft_meaning` and implementing `predict_membership`. Finite
two-dimensional row-aligned output is retained against support row IDs. Neither
KMeans distances nor arbitrary membership strengths become calibrated probabilities.

`RegimeResult.rows` retains every original row with native/display labels,
timestamps/intervals, reason and one status: `assigned`, `noise`,
`excluded_missing`, `excluded_quality`, or `failed`. Display labels have no
ordinal meaning and may differ between separate result tables. Native scalar
types are preserved: integer1 differs from string"1", numeric noise labels remain
numeric, and large integers are not rounded when excluded rows carry None.
Empty row tables retain the same columns, allowing unavailable zero-row candidates
to remain in comparisons with denominator0. Missing values
are not noise. `describe()` retains independent compute/metric/use status,
source/schema/reference identities, exact training/evaluation support, each
feature's burden, configuration, cluster sizes and fractions with denominators.
Training unknown origin remains unknown on qualified evaluation data.

Compute statuses are `ok`, `insufficient_support`, `fit_failed`, and
`not_converged`. Too few usable rows produce insufficiency. A custom backend
may raise `RegimeFitError` for an expected numerical fit/assignment failure;
programming errors and invalid configuration propagate. Backend warnings and a
reached KMeans iteration limit are explicit nonconvergence, retaining diagnostics
and failed rows without an algorithm switch. Metric status is
`exploratory_candidate_regimes`; use is `sensitivity_only`, `unknown_provenance`
or `insufficient_support`. Native KMeans alone supplies meaningful centroids,
inertia and iterations. DBSCAN/custom do not invent centroids.

The frozen `reference` exposes owned `describe()`, `fitted_model` and
`fitted_preprocess` snapshots. Modifying these accessors cannot mutate future
assignment. Fitted portable serialization is not implemented. Requested settings,
source support, weights, distances and numerical outputs bind scientific identity;
measured elapsed cost does not. New summaries carry compact provenance without
embedding producer source_details.

`compare_regime_models([(name, result), ...], noise_policy="include"|"exclude")`
accepts an explicit finite list of already computed candidates. It returns
owned candidate and agreement tables; no extra model is fitted and no winner is
selected. Adjusted Rand agreement uses common original row IDs, compatible schemas
and matching actual row geometry, ignoring label permutations. It reports exact
comparison denominators and retained failed candidates. All-noise and one-group
or singleton-per-row partitions have unavailable silhouette; ordinary Euclidean
and validated precomputed distances have the usual descriptive silhouette where
defined. Other metrics report unavailable separation. Noise inclusion is explicit;
cluster counts always exclude native noise. Model costs are measured diagnostics.

`summarize_regime_timeline(result, max_gap=..., noise_policy="include"|"barrier")`
returns raw rows, episodes and transitions. Excluded/failed rows are barriers;
native labels retain their scalar types and exact values in all three tables,
including both transition endpoints when integer, float and string labels mix.
Noise remains a separate status and is included as an episode or acts as an
explicit barrier. A gap larger than max_gap beyond the current episode's covered end breaks
continuity. No transition is invented across a barrier/gap. Episode duration is
the union of actual covered intervals, with elapsed span separately named.
Intervals [0,2h),[1,3h) contribute 3h, not 4h. Point rows have zero coverage.
Transition bounds span contributing feature supports; they are not precise change
points. Labels are never smoothed and temporal/overlapping rows are not independent
replicate evidence.

The fixed controls recovered all three synthetic chronological blocks (adjusted
Rand 1.0). Stationary K2/K3 partitions had sizes 99/93 and 69/76/47, agreement
0.463066834764 and 100/140 episodes. Continuous seasonal K2/K3 partitions had
sizes 96/96 and 64/64/64, agreement 0.330395064441 and 5/7 apparent episodes.
These are descriptive outcomes, with no expectation that stationary data yields
one group and no ecological interpretation of seasonal episodes. Seeds, counts,
candidates and settings are fixed before evaluating results, with no result tuning.

## Installed models example and result reload

This checkout includes `examples/sensitivity_regimes.py`
and its required `examples/reconstruction_sensitivity.py` sibling. Keep both
scripts together. They remain together under `examples/`.
The example traverses actual-time spectra, four spectral routes (five rows
because one route has two completion draws), frozen scaling/assignment, DBSCAN
noise/refusal, timeline barriers and insufficient/unknown/incomparable paths.

Open PowerShell in the source checkout root. Writing needs a
Python 3.11 environment with Tezca installed,
numerical dependencies and the models extra. The exercised optional stack is
CPython 3.11.9, scikit-learn 1.8.0, joblib 1.5.3 and threadpoolctl 3.6.0; see
[qualification](qualification.md) for the complete environment.
Do not substitute an arbitrary backend version as if it were qualified. The two
reading commands use a core-only environment from the
[quickstart](quickstart.md), with no sklearn or EcoFill.

Choose unused output names and run each command as a separate Python process:

```powershell
$ModelsPython = Read-Host 'Path to the models environment python.exe'
& $ModelsPython -I -B .\examples\sensitivity_regimes.py --write ..\tezca-models-summary.json
```

After the write succeeds:

```powershell
python -I -B .\examples\sensitivity_regimes.py --validate ..\tezca-models-summary.json
python -I -B .\examples\sensitivity_regimes.py --reload-results ..\tezca-models-summary.json
```

**Retain both outputs together:**

| Output | Contents and use |
| --- | --- |
| `..\tezca-models-summary.json` | Example-local summary, its SHA-256 digest, literal controls and `summary.result_bundle` expected bundle/source/reference identities plus saved regime evidence. Keep this record separately trusted. |
| `..\tezca-models-summary-results\` | Numerical bundle with `analysis_manifest.json`, `result_details.json` and canonical CSV tables, including spectra, sensitivity, native regime labels and the saved timeline. |

The bundle is the sibling `<summary-stem>-results`: `--write summary.json`
creates `summary-results/`. Both reading options take the **summary JSON path**,
not the bundle directory. Moving or renaming one artifact without its matching
sibling prevents bundle reload. The writer exports the bundle before creating
the summary with exclusive creation; neither path may already exist. A failure
can leave one output without the other, so preserve it and choose a new pair
for a retry.

The two read commands do different jobs. `--validate` uses only the Python
standard library, checks the summary digest and literal/support/reference/row
links, and prints that it validated without Tezca or refitting. It **does not
read or validate the numerical bundle**. `--reload-results` checks the summary
digest, derives the sibling bundle path, and calls `load_results()` with all
three expected identity maps/values saved in `summary.result_bundle`. It compares
the loaded native labels, reference identities, frozen preprocessing state,
training rows and complete saved timeline evidence with the summary. Its success
message ends with `without sklearn or refitting`. No new spectrum, assignment,
clustering or fitted-model reconstruction occurs on this route.

Useful expected synthetic values are density `32.0` at `0.125` Hz, integrated
power `0.5` over eight segments, unscaled KMeans centers `[-2.0, 2.0]` and inertia
`4.0` within numerical tolerance. DBSCAN has two clusters, one noise row with
native label `-1`, and an explicitly refused future assignment. The frozen
StandardScaler retains mean `[0.0]`, scale `[sqrt(14/3)]` and training IDs
`train0` through `train5`; future and timeline results retain that same frozen
reference ID. The timeline keeps `timeline2` as `excluded_missing`, with no
native label, and preserves timestamps at hours `0, 1, 2, 10, 11, 12`, three
episodes and one transition. Missing and gap rows never become continuous
episodes. Summary statuses remain `insufficient_support`, `unknown_provenance`
and `incomparable` for the respective unavailable controls.

An altered summary with its original digest is refused as `digest mismatch`.
Numerical payload tampering, a missing sibling and wrong expected bundle/source/
reference identities fail the bundle route. Do not edit checksums or remove
identity comparisons to force a result. A fully replaced, internally consistent
summary and bundle still need an external trusted identity to authenticate their
selection; hashes alone do not prove authorship. The existing installed models
witness checks saved preprocessing, native noise/excluded labels, reference and
timeline preservation, with analysis/assignment patched to fail during reload
and optional imports denied.

General [Tezca result exports](exports.md) preserve numerical labels and timelines
without executable model serialization. These synthetic examples establish no
ecological truth or rights to distribute; Historical numerical parity, field validation and reuse rights remain external gates.
Tezca has no plotting API.

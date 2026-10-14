# Complete public API cheat sheet

Use this page to find a function, its actual calling form and the decisions you
need to make before calling it. Start with the [package guide](../README.md) for
installation and the [installed delivery example](../examples/delivery.py) for
the complete analysis → export → fresh-process reload journey.

## Read the calling forms

Every table names the defining module. For example,
`from tezca.data import AnalysisSeries` imports the constructor documented as
`tezca.data.AnalysisSeries(...)`. Import submodules explicitly; `import tezca`
alone exposes the deliberately small root API (`tezca.__version__`).

Constructor rows create an instance. A method row such as
`tezca.data.ResultRecord.summary()` means `result.summary()` on that instance;
implicit `self`/`cls` and type annotations are omitted, but parameter order,
keyword-only `*`, defaults and `**parameters` are exact. Properties and fields
have **no parentheses**. Class method `AnalysisPolicy.from_dict(data)` is called
on the class. Built-in exception calling forms accept ordinary `*args`.

The descriptions give essential inputs, returned types and important choices.
Module notes apply to all its rows, including trivial record/identity accessors.
Unless explicitly stated, calls run locally in memory, write no files and need
only the declared core dependencies (NumPy, pandas and SciPy). No function
automatically installs optional packages, fills observations or renders figures.
Time durations accept pandas-compatible fixed-duration values where documented;
positions are indices into the original rows, never compressed sample numbers.

Owned copies protect the public data/result interface, including nested objects.
Immutable strings/numbers need no copy. Direct constructors for produced
reference/result containers are listed for completeness; prefer their named
producer or `load_results`. Constructing a container from arbitrary records does
not establish scientific validity, provenance or bundle validation.

## Find a task

| Task | APIs and detailed guide |
| --- | --- |
| [Declare data, clocks and eligible support](#data-and-support) | `tezca.data`, `tezca.support` · [guide](foundation.md) |
| [Inspect recipes and reuse frozen reference transforms](#policies-and-references) | `tezca.policy`, `tezca.reference` · [guide](foundation.md) |
| [Describe one signal and remove an explicitly fitted daily pattern](#individual-dynamics) | `tezca.individual` · [guide](individual.md) |
| [Estimate regular spectra and summarize them](#regular-spectra) | `tezca.spectral`, `tezca.spectral_summary` · [guide](spectral.md) |
| [Work with actual irregular sample times](#irregular-spectra) | `tezca.irregular` · [guide](irregular.md) |
| [Measure ordered masses and continuous-density roughness](#fisher-and-density) | `tezca.fi1`, `tezca.density_fisher` · [guide](fi1.md) |
| [Run qualified harmonic diagnostics](#harmonic-diagnostics) | `tezca.harmonic` · [guide](harmonic.md) |
| [Prepare collective inputs, PCA and networks](#collective-pca-networks) | `tezca.collective`, `tezca.pca`, `tezca.network` · [guide](collective.md) |
| [Audit pair ancestry and compare frequency coupling](#pair-coherence) | `tezca.pair_audit`, `tezca.coherence`, `tezca.coherence_modern` · [guide](coherence.md) |
| [Inspect frozen graph-state geometry](#graph-state) | `tezca.graph_state` · [guide](graph_state.md) |
| [Compare supplied completions without inventing them](#reconstruction-sensitivity) | `tezca.sensitivity` · [guide](sensitivity.md) |
| [Fit candidate regimes and preserve gap barriers](#regimes) | `tezca.regimes` · [guide](regimes.md) |
| [Save, validate and read numerical results](#exports-and-reload) | `tezca.exports` · [guide](exports.md) |
| [Bridge admitted EcoFill objects and saved exports](#ecofill) | `tezca.adapters.ecofill` · [guide](ecofill.md) |

## Small runnable example

Run this whole block in an environment with the local wheel built from this checkout installed.
It needs no optional packages or input files and writes no files. It demonstrates
observed data, a frozen training reference, a withheld unknown-provenance result,
defensive ownership, and two independently inspectable numerical primitives.

```python
import pandas as pd
from tezca.data import AnalysisSeries
from tezca.fi1 import discrete_information
from tezca.graph_state import sqrt_mass_graph_energy
from tezca.individual import describe_series
from tezca.reference import ReferenceSpec, fit_reference, apply_reference
from tezca.support import ObservationPolicy, select_support

frame = pd.DataFrame(
    {"level": [0., 2., 4., 3., 3., 3.]},
    index=pd.date_range("2000-01-01", periods=6, freq="h", tz="UTC"),
)
metadata = {"level": {"unit": "m", "representation": "raw"}}
source = {"id": "cheatsheet-synthetic-v1", "segment": "synthetic"}
clock = {"mode": "regular", "cadence": "1h", "civil_timezone": "UTC"}
data = AnalysisSeries(frame, metadata=metadata, source=source,
                      observed=[[True]] * 6, clock=clock)
observations = ObservationPolicy("observed_only")
selection = select_support(data, observations=observations)
assert selection.positions.tolist() == [0, 1, 2, 3, 4, 5]
mean = describe_series(data, observations=observations)[0]
assert mean.value == 2.5
assert mean.summary()["compute_status"] == "ok"

reference = fit_reference(
    data, spec=ReferenceSpec([0, 2, 4]), training_positions=[0, 1, 2],
    observations=observations,
)
evaluated = apply_reference(data, reference=reference, observations=observations)
assert evaluated.value["counts"]["level"].tolist() == [1, 5]
assert evaluated.summary()["reference_id"] == reference.identity
assert reference.describe()["scaling"]["mean"].tolist() == [2.0]
assert evaluated.summary()["use_status"] == "sensitivity_only"

unknown = AnalysisSeries(frame, metadata=metadata, source=source, clock=clock)
withheld = describe_series(unknown)[0]
assert withheld.value is None
assert withheld.summary()["compute_status"] == "insufficient_support"
provided = describe_series(unknown, observations=ObservationPolicy("provided"))[0]
assert provided.value == 2.5
assert provided.summary()["use_status"] == "unknown_provenance"

copy = data.values
copy.iloc[0, 0] = 999
assert data.values.iloc[0, 0] == 0
assert discrete_information([0.5, 0.5])["fi1"] == 0
assert sqrt_mass_graph_energy([0, 1], [[0, 1]])["energy"] == 4
print({"mean_m": mean.value, "unknown_status": provided.summary()["use_status"],
       "frozen_counts": evaluated.value["counts"]["level"].tolist()})
```

Expected output is `{'mean_m': 2.5, 'unknown_status': 'unknown_provenance',
'frozen_counts': [1, 5]}`. The unknown series contains the same numbers as the
observed series; the different status reflects missing provenance, not different
arithmetic. The reference example deliberately evaluates the small source again
to expose exact counts; use a separate evaluation interval for a prospective
scientific comparison.

<a id="data-and-support"></a>

## Declare data, clocks and eligible support

Start with declared units, source identity and a timezone-aware clock. Most analysis functions use `ObservationPolicy('observed_only')` when `observations=None`; the policy constructor itself defaults to `provided`. Explicitly select the intended policy. Data and result accessors return defensive copies unless a row says otherwise. Arrays use original row positions. All functions in this group use only the declared core NumPy/pandas/SciPy stack.

Detailed contracts and status meanings: [foundation guide](foundation.md).

### `tezca.data`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.data.AnalysisSeries(values, *, metadata, source, time=None, variables=None, clock=None, observed=None, contribution=None, qualification=None, lineage=None, source_details=None)` | Build an owned input from a numeric pandas DataFrame or array; arrays also need time and variable names. Declare per-variable unit/representation, source identity and clock. Choose observed mask OR contribution names; finite undeclared cells remain unknown. Returns AnalysisSeries. |
| `tezca.data.AnalysisSeries.contribution` | Property → copied aligned ndarray of observed/imputed/derived/unavailable/unknown contribution names. |
| `tezca.data.AnalysisSeries.eligible(policy)` | ObservationPolicy → new boolean per-cell eligibility array; does not modify, fill or compress the source. |
| `tezca.data.AnalysisSeries.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.data.AnalysisSeries.lineage` | Property → copied per-variable row-aligned ContributorLineage objects. |
| `tezca.data.AnalysisSeries.metadata` | Property → copied per-variable unit/representation mapping. |
| `tezca.data.AnalysisSeries.origin` | Property → copied arrays for reconstructed/unknown fractions, touched flags, operation and fraction kind. |
| `tezca.data.AnalysisSeries.qualification` | Property → copied mapping of all original qualification-mask arrays; completion never promotes original flags. |
| `tezca.data.AnalysisSeries.source` | Property → copied compact source/provenance mapping and source-details digest when present. |
| `tezca.data.AnalysisSeries.source_details` | Property → copied producer JSON and aligned state/method/reason/quality tables, or None; separate from the compact source summary. |
| `tezca.data.AnalysisSeries.time` | Property → owned TimeAxis copy; inspect its index/record without changing the source. |
| `tezca.data.AnalysisSeries.values` | Property → defensive-copy numeric DataFrame; includes missing cells and original row order. |
| `tezca.data.ContributorLineage(contributor_ids, origins, *, unit, operation, weights=None, parents=None, details=None, observed_only_value=None, trusted_imputation=None)` | Describe one derived cell from unique contributor IDs/origins, units and operation. Linear weights record influence; nonlinear reducers retain coverage/touched ancestry. Optional parents/details/observed-only value are copied. Returns ContributorLineage; performs no aggregation. |
| `tezca.data.ContributorLineage.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.data.ContributorLineage.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.data.ResultRecord(value, *, compute_status, metric_status, use_status, support, units, method_id, config_id, reference_id=None, reason=None, representation='raw')` | Construct an owned value/support/status record; usually obtain it from a method. Missing results require value=None and a reason. Units and method/config IDs are explicit. Direct construction does not establish scientific validity or validate an exported bundle. |
| `tezca.data.ResultRecord.describe()` | Return the same owned complete mapping as summary(); no analysis or fitting. |
| `tezca.data.ResultRecord.summary()` | Return complete copied record: value, three independent statuses, support, units, reason, representation and scientific identities. |
| `tezca.data.ResultRecord.to_frame()` | Return one-row DataFrame containing the copied complete result record, including nested values/support. |
| `tezca.data.ResultRecord.value` | Property → defensive copy of the scalar, arrays, tables or mapping value; None for unavailable computation. |

### `tezca.support`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.support.ObservationPolicy(values='provided', min_points=1, max_imputed_fraction=None, max_reconstructed_duration=None)` | Choose eligible values: provided (constructor default), observed_only or trusted_imputation. Optional reconstruction fraction/duration limits govern use status, not filling. Returns immutable policy. Analysis functions normally default to observed_only when observations=None. |
| `tezca.support.ObservationPolicy.max_imputed_fraction` | Immutable field → optional finite fraction limit in [0,1]; None disables this use gate. |
| `tezca.support.ObservationPolicy.max_reconstructed_duration` | Immutable field → optional duration string bounding reconstructed blocks; unknown geometry can withhold use. |
| `tezca.support.ObservationPolicy.min_points` | Immutable field → minimum eligible-point count (int, default 1). |
| `tezca.support.ObservationPolicy.values` | Immutable field → provided/observed_only/trusted_imputation string selected at construction. |
| `tezca.support.SupportSelection(record)` | Advanced record container; normally returned by select_support. Owns the supplied selection mapping; construction alone does not derive or certify support. |
| `tezca.support.SupportSelection.positions` | Property → owned integer ndarray of original row positions; a union is not permission to concatenate separated runs. |
| `tezca.support.SupportSelection.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.support.TimeAxis(index, *, mode='irregular', cadence=None, phase=None, storage_timezone=None, civil_timezone='UTC', clock_resolution='resolved', coverage_seconds=None, max_coverage_seconds=None)` | Validate ordered, unique timezone-aware DatetimeIndex. Default irregular retains actual instants; regular requires cadence and preserves holes. Explicit coverage_seconds/max_coverage_seconds can justify irregular duration. Returns an owned clock; never rounds or fills. |
| `tezca.support.TimeAxis.cadence` | Property → immutable pandas Timedelta for regular clocks, otherwise None. |
| `tezca.support.TimeAxis.duration(positions)` | Original integer positions → duration in seconds, or None without justified irregular coverage. Regular duration counts selected cells, not elapsed gap coverage. |
| `tezca.support.TimeAxis.index` | Property → owned timezone-aware DatetimeIndex copy, preserving actual instants and holes. |
| `tezca.support.TimeAxis.mode` | Property → immutable 'regular' or 'irregular' string. |
| `tezca.support.TimeAxis.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.support.TimeAxis.resolve_lag(requested)` | one_step/one_hour/twenty_four_hours or duration → requested/realized-lag/status mapping. Requires regular clock; nonintegral cadence multiples retain unsupported_exact_lag. |
| `tezca.support.pair_support(first, second, *, variable_a, variable_b, observations=None)` | Two AnalysisSeries and variable names → owned pair-support mapping on common eligible instants, separate burdens and union touched coverage. Does not manufacture observations or compress gaps. |
| `tezca.support.reconstruction_burden(series, positions=None, *, denominator=None)` | AnalysisSeries and optional original positions/denominator → mapping of counts, influence/coverage fractions, touched cells and longest reconstructed blocks. Default full support uses declared grid coverage; irregular durations may remain unknown. |
| `tezca.support.select_support(series, *, observations=None, policy='all_runs', estimator='regular', window=None, step=None, anchor=None, complete=True, label='start', closure='left', within=None)` | AnalysisSeries plus ObservationPolicy → SupportSelection with original positions, runs/windows, counts and burden. Policies: all_runs, longest_run, complete_case, windows, irregular. Estimator compatibility is checked; windows need durations and left closure. Never concatenate separated runs for regular analysis. |
| `tezca.support.use_status(series, positions, observations, *, metric_status='primary_with_support')` | Series, selected positions and ObservationPolicy → status string: primary_eligible, sensitivity_only, unknown_provenance, insufficient_support or unverified_gap_geometry. Combines relevant burden and metric qualification; no ecological certification. |

<a id="policies-and-references"></a>

## Inspect recipes and reuse frozen reference transforms

Use a recipe to inspect choices and derive observation/support policy. `AnalysisPolicy.method_spec()` refuses unresolved scientific recipes; choose an explicit method-specific spec below. `RECIPES` in `tezca.policy` lists recipe names. YAML load/dump alone need the YAML extra. Fit on declared training rows, then pass `reference=` to evaluation to prevent refitting; a `spec=` self-fit intentionally has a different identity.

Detailed contracts and status meanings: [foundation guide](foundation.md).

### `tezca.policy`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.policy.AnalysisPolicy(recipe, settings=None, *, operational=None)` | Select a named recipe and validated settings; operational labels are separate from scientific identity. Returns an inspectable policy with unresolved requirements. A recipe declaration is not an executable estimator. |
| `tezca.policy.AnalysisPolicy.describe()` | Return copied recipe/requested/resolved settings, identities, operational labels, unresolved requirements, parity status and migration evidence. |
| `tezca.policy.AnalysisPolicy.from_dict(data)` | Class method: validated resolved-policy envelope → AnalysisPolicy. Rejects unknown schema/fields; no YAML dependency. |
| `tezca.policy.AnalysisPolicy.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.policy.AnalysisPolicy.method_spec(method)` | Known method name → UnresolvedPolicyError listing required choices; unknown name → ValueError. No current named recipe resolves these estimator choices automatically. |
| `tezca.policy.AnalysisPolicy.migration` | Property → defensive-copy migration rows explaining legacy-to-current choices. |
| `tezca.policy.AnalysisPolicy.observations()` | Return ObservationPolicy resolved from declared value/support and primary-use burden settings. |
| `tezca.policy.AnalysisPolicy.requested_identity` | Property → immutable requested-settings digest; distinct requests can resolve to the same scientific identity. |
| `tezca.policy.AnalysisPolicy.settings` | Property → defensive-copy resolved settings mapping; does not execute unresolved methods. |
| `tezca.policy.AnalysisPolicy.support_options()` | Return copied keyword mapping for select_support, including window completeness/label only when applicable. |
| `tezca.policy.AnalysisPolicy.to_dict()` | Return copied schema-versioned resolved-policy envelope; retain describe() separately for request/migration history. |
| `tezca.policy.UnresolvedPolicyError(*args)` | ValueError subclass raised by method_spec for an unresolved named method. Ordinary exception arguments are messages; inspect policy.describe()['unresolved'] and choose explicit admitted method specs. |
| `tezca.policy.dump_policy_yaml(policy)` | AnalysisPolicy → YAML string using optional PyYAML. Serializes resolved settings; retain describe() separately for original requests/migration evidence. Writes no file. |
| `tezca.policy.load_policy_yaml(text, *, target=None)` | YAML text → AnalysisPolicy. Requires optional PyYAML only when called; target is mandatory for legacy input. Rejects duplicate keys, aliases/merges, unsafe constructors and invalid settings; no file or network access. |
| `tezca.policy.migrate_legacy(data, *, target)` | Legacy settings mapping plus explicit target recipe → AnalysisPolicy with migration rows and requested/resolved identities. Rejects unknown/invalid settings rather than guessing a target. |

### `tezca.reference`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.reference.FrozenReference(record)` | Advanced owned record container with deterministic identity. Prefer fit_reference or a method-specific fit function; wrapping an arbitrary record is not saved-result validation. |
| `tezca.reference.FrozenReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.reference.FrozenReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.reference.OutsideReferenceError(diagnostics)` | ValueError subclass carrying copied outside-edge diagnostics. Raised when frozen evaluation leaves admitted edges under outside='raise'; explicit exclusion is a different requested analysis. |
| `tezca.reference.OutsideReferenceError.diagnostics` | Public dict field copied at construction with below/above counts and positions. This exception field is mutable; unlike record properties, repeated access returns the same diagnostic mapping. |
| `tezca.reference.ReferenceSpec(edges, *, scale=True, min_points=2, outside='raise', support='all_runs')` | Declare increasing raw-coordinate histogram edges and optional population scaling (ddof=0). Default outside='raise'; explicit 'exclude' retains excluded-row diagnostics. Returns a spec for the shared representation primitive, not a Fisher estimator. |
| `tezca.reference.ReferenceSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.reference.ReferenceSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.reference.apply_reference(series, *, reference=None, spec=None, observations=None)` | AnalysisSeries → ResultRecord with scaled frame and freshly evaluated histogram counts/probabilities. Supply exactly one frozen reference (no refit) or spec (retrospective self-fit). Checks variable order, units, representation and outside policy. |
| `tezca.reference.fit_reference(series, *, spec, observations=None, training_positions=None, fit_scope='training_only')` | AnalysisSeries + ReferenceSpec → FrozenReference containing training mean/population SD, edges, source/support identities and statuses. Default training_only requires training_positions; self explicitly uses evaluation data. |

<a id="individual-dynamics"></a>

## Describe one signal and remove an explicitly fitted daily pattern

Input is an `AnalysisSeries` or, where noted, a `DailyResidual`. Raw and `daily_residual` are different representations; do not relabel one as the other. Daily fitting requires resolved civil time and training coverage. No optional backend is needed. Sample SD (ddof=1) and population variance (ddof=0) are distinct quantities.

Detailed contracts and status meanings: [individual guide](individual.md).

### `tezca.individual`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.individual.DailyReference(record)` | Advanced frozen daily-fit record container, inheriting identity/describe. Prefer fit_daily_reference; record construction does not fit or validate a persisted reference. |
| `tezca.individual.DailyReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.individual.DailyReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.individual.DailyResidual(series, values, reference, evaluation)` | Advanced owned residual container from original series, residual values, DailyReference and evaluation diagnostics. Prefer apply_daily_reference; original qualification and training burden remain distinct. |
| `tezca.individual.DailyResidual.describe()` | Return copied source/reference identity, fitted daily description and evaluation diagnostics; no refit. |
| `tezca.individual.DailyResidual.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.individual.DailyResidual.reference_id` | Property → immutable identity string of the fitted DailyReference. |
| `tezca.individual.DailyResidual.values` | Property → copied residual DataFrame; unavailable slots stay missing, no reconstruction. |
| `tezca.individual.DailySpec(*, civil_timezone, fold_policy='daily_median', min_dates=2)` | Declare civil timezone, repeated-hour fold treatment (daily_median) and minimum dates. Returns DailySpec; a daily reference needs resolved clock/phase and enough distinct dates per slot. |
| `tezca.individual.DailySpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.individual.DailySpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.individual.apply_daily_reference(series, *, reference=None, spec=None, observations=None)` | AnalysisSeries → DailyResidual after subtracting a frozen civil daily pattern. Supply exactly one reference (no refit) or spec (self-fit). Missing/unsupported slots remain unavailable. |
| `tezca.individual.autocorrelation(data, *, variable, lag='one_step', observations=None, support='all_runs')` | Series/residual and variable → exact-lag Pearson ResultRecord with eligible pair positions/count. one_step is default; duration lags must exactly match regular cadence. Gaps never become adjacent samples. |
| `tezca.individual.describe_series(data, *, observations=None, support=None)` | AnalysisSeries or DailyResidual → list of seven ResultRecords per variable: mean, median, sample SD, IQR, minimum, maximum, population variance. Default support is all_runs for regular clocks, irregular otherwise; missing results retain reasons. |
| `tezca.individual.estimate_trend(data, *, variable, observations=None, support=None, uncertainty='unavailable')` | Series/residual and variable → ResultRecord of OLS slope per elapsed day. Actual timestamps determine elapsed time; default uncertainty is unavailable. iid_ols_compatibility is explicitly qualified, not dependence-robust inference. |
| `tezca.individual.fit_daily_reference(series, *, spec, training_positions=None, observations=None, fit_scope='training_only')` | AnalysisSeries + DailySpec → DailyReference from only explicit training_positions under training_only. self is retrospective. Stores slot/date support, clock convention, fit burden and source identities. |

<a id="regular-spectra"></a>

## Estimate regular spectra and summarize them

Welch needs a regular clock and enough uninterrupted support. FFT padding adds display samples, not physical frequency resolution. Frequency is Hz; PSD density carries squared input units per Hz. Descriptive fits/peaks/bands use [the full summary status guide](spectral_summary.md); they are sensitivity-only. No optional backend is needed. Array primitives omit provenance and cannot replace source-bound result methods.

Detailed contracts and status meanings: [spectral guide](spectral.md).

### `tezca.spectral`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.spectral.WelchSpec(nperseg=None, *, recipe='research_explicit', overlap=None, nfft=None, taper='hann', periodic=True, detrend='constant', scaling='density', onesided=True, average='mean', support='all_runs', min_segments=1, min_nperseg=64, max_nperseg=4096, length_divisor=8)` | Declare Welch segment/FFT/taper/detrend/support choices. research_explicit requires nperseg; overlap=None resolves to half the segment length. sensor_length resolves its declared length rule. Default mean one-sided density uses Hann/constant detrend. |
| `tezca.spectral.WelchSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.spectral.WelchSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.spectral.power_spectrum(data, *, variable, spec, observations=None)` | Series/residual + variable + WelchSpec → ResultRecord with display/physical frequency grids, PSD and support. Regular eligible runs stay separate; segment support, effective averages and missing outcomes remain visible. |

### `tezca.spectral_summary`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.spectral_summary.BandSpec(bounds, *, coordinates='hours', min_cycles=2, min_ordinates=3, basis=None)` | Declare two band endpoints (hours by default, or frequency coordinates), minimum cycles/physical ordinates and optional basis. Returns band spec; out-of-support bands are withheld rather than extrapolated. |
| `tezca.spectral_summary.BandSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.spectral_summary.ModelSpec(*, interval_hz=None, candidates_hz=None, min_arm=3, log_base=10, tie_tolerance=1e-12, exact_rss_tolerance=1e-24, basis=None)` | Choose supported fit interval/candidate break frequencies, minimum arm size, logarithm base and exact tie/RSS tolerances. Optional basis must match raw/residual input. Returns explicit log-model spec. |
| `tezca.spectral_summary.ModelSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.spectral_summary.PeakSpec(*, period_seconds=None, min_cycles=2, max_peaks=5, prominence='linear', basis=None)` | Choose period range in seconds, minimum cycles and peak cap; default ranks up to five by linear prominence. basis can enforce raw/residual consistency. Returns peak spec; padding grants no extra physical resolution. |
| `tezca.spectral_summary.PeakSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.spectral_summary.band_power(spectrum, *, spec)` | Welch ResultRecord + BandSpec → ResultRecord of supported band power/fraction with inherited support and status. Numerical zero and unavailable remain distinct. |
| `tezca.spectral_summary.fit_log_models(frequency_hz, power, *, spec, supported_range_hz=None)` | Frequency/power arrays + ModelSpec → mapping with single/broken log-model fits, working AICc values, `delta_aicc`, `preferred`, exclusions and status. Comparison can be unavailable as explained below. Array primitive has no source provenance; nonpositive/invalid ordinates cannot silently become usable. |
| `tezca.spectral_summary.integrate_band(frequency_hz, density, *, physical_frequency_hz, duration_seconds, spec, supported_range_hz=None)` | Density/display and physical-frequency arrays, duration and BandSpec → mapping with trapezoidal power/fraction, support and status. Physical ordinates control resolution; array primitive has no source provenance. |
| `tezca.spectral_summary.rank_peaks(frequency_hz, power, *, duration_seconds, spec, supported_range_hz=None)` | Frequency/power arrays, actual duration and PeakSpec → mapping of ranked local peaks and support/status. Duration/cycle and supported-range gates precede ranking. This primitive does not establish source provenance. |
| `tezca.spectral_summary.spectral_models(spectrum, *, spec)` | Welch ResultRecord + ModelSpec → ResultRecord of descriptive log-model fits, inheriting source, basis and support. Retains unavailable upstream and model-comparison outcomes; no historical-parity claim. |
| `tezca.spectral_summary.spectral_peaks(spectrum, *, spec)` | Welch ResultRecord + PeakSpec → ResultRecord of ranked peaks with inherited support/identity and unavailable status paths. Uses saved spectrum; does not refit Welch. |

The log-model comparison uses AICc (corrected Akaike information criterion)
under a working Gaussian log-residual model. It is descriptive, with no process
calibration. Fewer than three eligible ordinates gives `insufficient_support`.
An `exact_fit` or `insufficient_aicc_points` fit has null AICc; a break with too
few points on either arm has `insufficient_arm_points`. If either model lacks
AICc or no broken fit is usable, `preferred` and `delta_aicc` stay null and the
overall comparison is `unavailable`. Otherwise it is
`descriptive_working_aicc`. See the [spectral-summary status guide](spectral_summary.md#status-and-rejection-vocabulary)
for conditions and next actions; a computed fit does not guarantee a model choice.

<a id="irregular-spectra"></a>

## Work with actual irregular sample times

Explicit actual-time projection-density LSFT-WOSA is independent of the unavailable historical backend. `f_ref_hz` is a declared density scale, not an inferred Nyquist frequency. Keep per-frequency rank, cycle and support failures. No optional backend is needed.

Detailed contracts and status meanings: [irregular guide](irregular.md).

### `tezca.irregular`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.irregular.IrregularSpec(*, interval, segment, hop, anchor, frequency_hz, f_ref_hz, requested_band_hz, taper='boxcar', detrend='none', min_points=3, min_span='1ns', min_cycles=0, rank_tolerance=1e-10, weights=None)` | Declare actual-time interval, segment/hop/anchor, frequency grid, f_ref_hz scale and requested band. Explicit rank/span/cycle/weight choices define the projection-density method; no inferred Nyquist frequency or filling. |
| `tezca.irregular.IrregularSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.irregular.IrregularSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.irregular.irregular_spectral_models(spectrum, *, spec, min_ordinates=3, min_cycles=2)` | Irregular-spectrum ResultRecord + ModelSpec → ResultRecord using actual supported ordinates separated by physical resolution. Minimum ordinates/cycles default to 3/2; dense display grids do not earn resolution. |
| `tezca.irregular.irregular_spectrum(data, *, variable, spec, observations=None)` | Series/residual + variable + IrregularSpec → ResultRecord of LSFT-WOSA projection density on actual eligible instants, with frequency-specific support and overlap moments. Independent definition; exact historical parity is unresolved. |

<a id="fisher-and-density"></a>

## Measure ordered masses and continuous-density roughness

Discrete FI1 and [density Fisher](density_fisher.md) have different definitions and units. FI1Spec choices are `explicit`, `local_fd_v1` (independent local rule) and the unavailable `sensor_pinned` route. Density coordinates are raw/standardized; histogram interior omits boundary contributions, while KDE declares bandwidth, integration bounds and tail limits. Frozen references must retain compatible units/representations. No optional backend is needed.

Detailed contracts and status meanings: [fi1 guide](fi1.md).

### `tezca.fi1`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.fi1.FI1Reference(record)` | Advanced frozen partition record container with inherited identity/describe. Prefer fit_fi1_reference; evaluation recomputes occupancies while retaining fitted edges. |
| `tezca.fi1.FI1Reference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.fi1.FI1Reference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.fi1.FI1Spec(edges=None, *, partition='explicit', pseudocount=None, outside='raise', min_points=2)` | Declare ordered edges or admitted fitted partition, pseudocount, outside policy and minimum points. Explicit partition needs edges; pseudocount=None resolves the selected partition's declared rule. FI1 measures discrete mass roughness, not continuous density Fisher. |
| `tezca.fi1.FI1Spec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.fi1.FI1Spec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.fi1.FI1Trajectory(table, record)` | Advanced owned trajectory container from a DataFrame and description mapping. Prefer fi1_windows; all scheduled unavailable rows remain part of the output. |
| `tezca.fi1.FI1Trajectory.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.fi1.FI1Trajectory.table` | Property → defensive-copy DataFrame of all requested windows, estimates/statuses and adjacent flags. |
| `tezca.fi1.adjacent_flags(times, values, *, step='6h')` | Aligned times/values → mapping of adjacent-stride differences, strict median + 3 unscaled MAD threshold and flags. Default stride 6h; missing/skipped strides break comparisons. |
| `tezca.fi1.discrete_information(probabilities)` | Normalized ordered mass vector → dict with fi1 and entropy_nats. Computes adjacent square-root-mass differences without ghost boundary bins. Stateless array primitive; no source/support claims. |
| `tezca.fi1.fi1_discrete(data, *, variable, spec=None, reference=None, observations=None)` | Series/residual + variable → ResultRecord with masses, counts, FI1 and entropy. Exactly one explicit spec (self-fit) or frozen reference; outside values follow declared policy. |
| `tezca.fi1.fi1_windows(data, *, variable, anchor, spec=None, reference=None, observations=None, window='72h', step='6h', label='start', fit_scope='global_self')` | Series/residual, variable and explicit anchor → FI1Trajectory of complete scheduled windows, default 72h/6h. Choose spec or reference; global_self and within-window fitting are distinct from frozen training. Gaps retain unavailable rows. |
| `tezca.fi1.fit_fi1_reference(data, *, variable, spec, training_positions, observations=None, fit_scope='training_only')` | Series/residual + variable + FI1Spec + training_positions → frozen FI1Reference. Default training_only records training support; no evaluation-driven edge changes. |

### `tezca.density_fisher`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.density_fisher.DensityReference(record)` | Advanced frozen histogram/KDE record container with identity/describe. Prefer fit_density_reference; it retains fitted geometry and coordinate-scale semantics. |
| `tezca.density_fisher.DensityReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.density_fisher.DensityReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.density_fisher.HistogramSpec(edges=None, *, binning='explicit', bins=None, ties=None, pseudocount=0, outside='raise', coordinates='raw', measurement='continuous', min_points=2, boundary='interior')` | Declare explicit/equal-width/quantile edges, bin/tie/pseudocount/outside choices and raw/standardized coordinates. Returns a histogram-interior spec; omitted boundary contributions prevent a full-domain Fisher claim. |
| `tezca.density_fisher.HistogramSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.density_fisher.HistogramSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.density_fisher.KDESpec(*, bandwidth, bounds, integration='full_bounded_tails', points=1025, atol=1e-08, rtol=1e-06, tail_tolerance=1e-08, coordinates='raw', measurement='continuous', min_points=2, boundary='none', physical_support='real_line')` | Declare positive absolute Gaussian bandwidth or 'scott', finite bounds and full_bounded_tails/finite_partial integration. Odd points/tolerances, raw/standardized coordinates and measurement assumptions remain explicit; real-line support with no boundary correction. |
| `tezca.density_fisher.KDESpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.density_fisher.KDESpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.density_fisher.density_fisher(data, *, variable, spec=None, reference=None, observations=None)` | Series/residual + variable and exactly one spec/reference → ResultRecord with coordinate-aware estimate, integration diagnostics and status. References remain frozen; failed tail/tolerance/support gates retain missing values. |
| `tezca.density_fisher.fit_density_reference(data, *, variable, spec, training_positions, observations=None, fit_scope='training_only')` | Series/residual + variable + HistogramSpec/KDESpec + training_positions → DensityReference. Default training_only fixes scaling/geometry on admitted training support. |
| `tezca.density_fisher.gaussian_kde_geometry(samples, points, *, bandwidth)` | Finite samples and evaluation points + positive absolute bandwidth → dict of points, Gaussian mixture density, analytic derivative and Fisher integrand arrays. No fitting, integration or provenance inference. |
| `tezca.density_fisher.histogram_interior(probabilities, edges)` | Normalized masses and increasing edges → dict with interior Fisher approximation, widths/centers/heights and omitted-boundary statement. Not a continuous full-domain Fisher estimator. |
| `tezca.density_fisher.legacy_density_v16()` | Return HistogramSpec for independently specified historical choices: 20 quantile bins, merged ties, standardized rounded-continuous input, minimum 100 points. Exact legacy backend parity remains unverified. |

<a id="harmonic-diagnostics"></a>

## Run qualified harmonic diagnostics

Fisher-g and Siegel diagnostics assume a complete ordinary raw rectangular periodogram and an explicitly declared iid Gaussian white-noise null. They do not establish ecological periodicity or support a Welch substitution. Siegel calibration runs a seeded finite simulation; record its identity and Monte Carlo uncertainty. No optional backend is needed.

Detailed contracts and status meanings: [harmonic guide](harmonic.md).

### `tezca.harmonic`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.harmonic.HarmonicSpec(*, alpha=0.05, lambda_=0.6, center=True, backend='independent', estimator='rectangular_periodogram', taper='boxcar', nfft=None, null='iid_gaussian_white_noise')` | Specify independent raw rectangular-periodogram diagnostics and assumed iid Gaussian white-noise null. Defaults alpha=.05, lambda=.6 and centering. Alternate declared unsupported choices produce explicit method refusals, never a Welch fallback. |
| `tezca.harmonic.HarmonicSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.harmonic.HarmonicSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.harmonic.SiegelCalibration(*, m, alpha=0.05, lambda_=0.6, replications=19999, seed=2026101101)` | Construct a seeded Monte Carlo calibration from m Fourier ordinates, alpha/lambda and replication count. Returns owned sorted null statistics and calibration identity; actually runs the simulation (default 19999). Arbitrary imported calibration records are unsupported. |
| `tezca.harmonic.SiegelCalibration.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.harmonic.SiegelCalibration.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.harmonic.calibrate_siegel(*, m, alpha=0.05, lambda_=0.6, replications=19999, seed=2026101101)` | Keyword simulation parameters → SiegelCalibration; same seeded simulation as its constructor. No hidden download/cache; record seed, replication count and finite Monte Carlo uncertainty. |
| `tezca.harmonic.compatibility_fields(results)` | Harmonic result mapping → dict with six historical fields plus mandatory semantic/status sidecar. Column names preserve client compatibility without claiming historical backend parity. |
| `tezca.harmonic.fisher_critical(m, alpha=0.05)` | Integer m>=2 and alpha (default .05) → numeric Fisher-g critical value by root finding. It is a null-specific threshold, not an ecological decision rule. |
| `tezca.harmonic.fisher_tail(g, m)` | g in [0,1] and integer m>=2 → exact-inclusion-exclusion Fisher-g tail as float. Applies only to the declared ordinary Fourier white-noise null. |
| `tezca.harmonic.fourier_proportions(samples, *, center=True)` | Complete finite one-dimensional samples → dict with ordinary Fourier powers/proportions, used bins, n/m and removed mean. Default centers; constants/invalid support are refused. No PSD/Welch substitute. |
| `tezca.harmonic.harmonic_diagnostics(data, *, variable, spec, calibration=None, observations=None)` | Series/residual + variable + HarmonicSpec and optional calibration → dict of separate Fisher and Siegel ResultRecords. Checks raw-periodogram eligibility, provenance and null; unavailable choices remain explicit. |
| `tezca.harmonic.siegel_statistic(proportions, *, alpha=0.05, lambda_=0.6)` | One normalized Fourier-proportion vector or row matrix → scalar/array of Siegel excess statistics. alpha/lambda default .05/.6; calibration must use matching choices. |
| `tezca.harmonic.siegel_tail(statistic, calibration)` | Observed statistic and matching SiegelCalibration → finite-simulation plus-one tail probability. Retains the calibration's assumptions; does not infer a new null. |

<a id="collective-pca-networks"></a>

## Prepare collective inputs, PCA and networks

Explicitly align/aggregate collection members; no missing-value reconstruction is provided. Pass `prepared.series` to [PCA](pca.md) or [networks](network.md), not the PreparedCollective wrapper. Retain raw/daily_residual basis, feature order, unit and reference identities. PCA uses NumPy/SciPy rather than sklearn. Network missingness is `pairwise` or `listwise`; a missing pair is not a zero correlation.

Detailed contracts and status meanings: [collective guide](collective.md).

### `tezca.collective`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.collective.CollectionMember(series, *, variable, station, family, feature, semantics)` | Wrap an AnalysisSeries variable with explicit station/family/feature and state, accumulation, rate or angle semantics. Returns member declaration; source clocks remain separate. |
| `tezca.collective.CollectionMember.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.collective.FeatureSpec(name, members, reducer='mean', min_coverage=1.0, station_reducer=None, spatial_group=None, priority=(), output_unit=None, rate_period=None)` | Name selected member IDs and explicit reducer/coverage rule. Default mean requires complete coverage; optional median station aggregation needs spatial_group, priority orders all sources, integral needs output_unit/rate_period. Returns immutable spec. |
| `tezca.collective.FeatureSpec.members` | Immutable field → ordered tuple of declared member IDs. |
| `tezca.collective.FeatureSpec.min_coverage` | Immutable field → source-cell coverage requirement in [0,1], default 1.0. |
| `tezca.collective.FeatureSpec.name` | Immutable field → output feature name (string). |
| `tezca.collective.FeatureSpec.output_unit` | Immutable field → explicitly declared output unit for integration, optional elsewhere. |
| `tezca.collective.FeatureSpec.priority` | Immutable field → source-priority tuple naming every member, or empty tuple. |
| `tezca.collective.FeatureSpec.rate_period` | Immutable field → rate time-unit duration required only for integral reduction. |
| `tezca.collective.FeatureSpec.reducer` | Immutable field → mean/sum/rate_mean/integral/circular_mean, default mean. |
| `tezca.collective.FeatureSpec.spatial_group` | Immutable field → declared grouping string paired with station_reducer, or None. |
| `tezca.collective.FeatureSpec.station_reducer` | Immutable field → explicit 'median' for station aggregation, otherwise None. |
| `tezca.collective.PreparedCollective(series, audit, observed_only, record, labels)` | Advanced container for start-cell AnalysisSeries, audit, observed counterpart, description and labels. Prefer prepare_collective; direct construction does not perform alignment or establish admissible site data. |
| `tezca.collective.PreparedCollective.audit` | Property → defensive-copy DataFrame of per-cell coverage, reduction and lineage audit. |
| `tezca.collective.PreparedCollective.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.collective.PreparedCollective.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.collective.PreparedCollective.matrix` | Property → defensive-copy DataFrame with requested display labels; these can differ from analysis cell starts. |
| `tezca.collective.PreparedCollective.observed_only` | Property → defensive-copy DataFrame produced independently from observed contributors using the same reducers/coverage rules. |
| `tezca.collective.PreparedCollective.series` | Property → defensive-copy AnalysisSeries using target-cell starts for analysis support. |
| `tezca.collective.prepare_collective(collection, *, features, start, end, cadence, label='start', observations=None)` | Mapping of member IDs to CollectionMember plus FeatureSpecs and [start,end) target cadence → PreparedCollective. Aggregates left-closed cells with explicit reducers/coverage/priority; no filling. Output labels may differ from analysis cell starts. |
| `tezca.collective.report_site_gate(prepared, *, coverage_basis='observed_only')` | PreparedCollective → mapping for report-style screen: >=2 families, >=30 common covered days, >=.65 variable coverage and >=5 retained variables. Default observed-only coverage; numeric screening does not admit domain data. |

### `tezca.pca`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.pca.PCAReference(record)` | Advanced owned fitted center/scales/full-eigensystem record with identity/describe. Prefer fit_pca_reference; wrapping records does not create trusted training evidence. |
| `tezca.pca.PCAReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.pca.PCAReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.pca.PCASpec(*, features, basis, scaling, ddof, center=True)` | Declare ordered features, raw/residual basis, scaling (none/zscore), covariance ddof and centering. Returns explicit PCA spec; full eigenvalue spectrum is retained. |
| `tezca.pca.PCASpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.pca.PCASpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.pca.fit_pca_reference(data, *, spec, training_positions=None, fit_scope='training_only', observations=None)` | AnalysisSeries/DailyResidual + PCASpec → PCAReference trained on complete vectors; for PreparedCollective pass .series. Default training_only requires explicit positions; self is retrospective. Records variable units/order, support and all eigenvalues. |
| `tezca.pca.pca_summary(data, *, spec, observations=None)` | Data + PCASpec → self-fit ResultRecord with full eigensystem, participation ratio and PC90. Retrospective identity differs from training projection; no sklearn dependency. |
| `tezca.pca.project_pca(data, *, reference, observations=None)` | Compatible evaluation data + frozen PCAReference → ResultRecord of complete-vector scores, projection covariance and support. Never refits training scale/basis; eigenvector sign/tied-subspace ambiguity remains. |

### `tezca.network`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.network.NetworkResult(summary, edges, record)` | Advanced owned summary/edges/record container; normally returned by correlation_network. Table ownership does not certify an arbitrary hand-constructed network. |
| `tezca.network.NetworkResult.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.network.NetworkResult.edges` | Property → defensive-copy DataFrame of signed unique edges, actual pair support and eligibility. |
| `tezca.network.NetworkResult.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.network.NetworkResult.summary` | Property → defensive-copy DataFrame of every requested window summary and status, including unavailable windows. |
| `tezca.network.NetworkSpec(*, features, basis, missingness='pairwise', min_points=3, threshold=0.3, window=None, step=None, complete=True, label='start', include_partial=False, historical=False, valid_pairs=False)` | Declare features, raw/daily_residual basis, pairwise/listwise missingness and minimum points/threshold. Optional window/step defines retained rolling rows; partial, historical and valid-pair summaries are explicit choices. |
| `tezca.network.NetworkSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.network.NetworkSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.network.correlation_network(data, *, spec, observations=None, start=None, end=None)` | Data + NetworkSpec and optional interval → NetworkResult with all requested summary rows, signed unique edges and actual pair support. Constants/undefined pairs do not become zero-weight edges. |
| `tezca.network.report_network_spec(*, features, basis, missingness='pairwise', min_points=3, historical=False, valid_pairs=False)` | Explicit feature/basis choices → NetworkSpec for complete 168h windows, 24h steps and abs(r)>=.3. Keeps missingness/support/denominator options explicit; not a site-validity guarantee. |

<a id="pair-coherence"></a>

## Audit pair ancestry and compare frequency coupling

Pair audits preserve ancestry and source bindings. Report-style equal-run coherence and [modern pooled coherence](coherence_modern.md) are distinct estimators. Modern phase is arg(conj(X)*Y): a delayed second signal has negative phase. Gaussian-white thresholds are approximate, pointwise and assumption-dependent. No optional backend is needed; historical backend routes remain unavailable.

Detailed contracts and status meanings: [coherence guide](coherence.md).

### `tezca.pair_audit`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.pair_audit.PairLineage(nodes)` | Declare node mapping with signal IDs, measurement/copy/deterministic/reconstruction/unknown kinds, parents and donors. Rejects cycles; absent ancestors remain unknown. Returns owned ancestry graph. |
| `tezca.pair_audit.PairLineage.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.pair_audit.PairLineage.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.pair_audit.audit_pair(first, second, *, variable_a, variable_b, lineage=None, nodes=None, target=None)` | Two series/residuals and variable names → dict auditing self/copy/deterministic/shared-ancestry and donor/target concerns. Optional PairLineage and node/target bindings must match actual signals; absent evidence stays unknown. |
| `tezca.pair_audit.signal_identity(data, variable)` | AnalysisSeries/DailyResidual and variable → deterministic signal identity string for explicit lineage binding. Does not infer donor independence. |

### `tezca.coherence`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.coherence.ReportCoherenceSpec(*, basis, period_band, taper, detrend, nperseg=None, min_nperseg=64, max_nperseg=256, min_segments=4, periodic=True, route='independent_report')` | Declare basis, period band (duration endpoints), taper and detrend. Fixed nperseg or bounded 64–256 rule, default minimum four segments. independent_report is separately identified from unavailable historical route. |
| `tezca.coherence.ReportCoherenceSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.coherence.ReportCoherenceSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.coherence.aggregate_report_runs(runs, *, period_band)` | Compatible accepted run dictionaries + period band → owned dict of equal-run corrected coherence on a zero-anchored coarsest physical grid. No extrapolation; array-record primitive cannot certify provenance. |
| `tezca.coherence.report_coherence(first, second, *, variable_a, variable_b, spec, observations=None, lineage=None, nodes=None, target=None)` | Two series/residuals, variables and ReportCoherenceSpec → ResultRecord of report-style equal-run coherence. Preserves pair audit, run support, exclusions and observed-only counterpart when required; raw/residual choice is explicit. |

### `tezca.coherence_modern`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.coherence_modern.ModernCoherenceSpec(*, basis, period_band, nperseg, hop, taper, detrend, periodic=True, nfft=None, alpha=0.05, null='gaussian_white', weighting='equal_segments', route='independent_modern')` | Declare basis/band, fixed segment length/hop, taper and detrend. Default equal-segment pooling, alpha=.05, gaussian_white null; optional positive run weights are explicit. Returns spec with saved phase/correction conventions. |
| `tezca.coherence_modern.ModernCoherenceSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.coherence_modern.ModernCoherenceSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.coherence_modern.coherence_inference(raw, *, n_eff, alpha, null='gaussian_white')` | Raw coherence vector (NaN allowed), n_eff and alpha → dict with correction, thresholds, significant flags and inference status. n_eff<=1 withholds inference; colored/unspecified null has no admitted threshold. |
| `tezca.coherence_modern.effective_averages(run_segment_weights, *, window, hop)` | One positive segment-weight array per contiguous run, taper and hop → dict of normalized weights, overlap penalty, approximate n_eff and degrees of freedom. Cross-gap overlaps are excluded; this is not ecological replication. |
| `tezca.coherence_modern.modern_coherence(first, second, *, variable_a, variable_b, spec, observations=None, lineage=None, nodes=None, target=None, run_weights=None)` | Two series/residuals + variables + spec → ResultRecord with pooled autospectra/cross-spectra, phase and qualified inference. Public run_weights use accepted run start timestamps. Pair ancestry and unavailable/observed-only paths are retained. |
| `tezca.coherence_modern.pool_coherence_runs(runs, *, spec, run_weights=None)` | Compatible density-normalized run dictionaries + ModernCoherenceSpec → owned pooling dict with spectra, phase, inference and support. Primitive weights use run IDs; identical cadence/grid/taper/hop is required. No provenance certification. |

<a id="graph-state"></a>

## Inspect frozen graph-state geometry

Mass and density energies differ: density uses node volumes and declared conductances/units. Declare geometry, feature order, scaling/PCA, outside policy and finite resource bounds before fitting. Evaluation with a training reference never rebuilds its geometry. Grid summaries retain every requested configuration. No optional backend is needed.

Detailed contracts and status meanings: [graph_state guide](graph_state.md).

### `tezca.graph_state`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.graph_state.GraphGridResult(rows)` | Advanced owned list-of-row container; normally returned by graph_state_grid. Every requested successful or infeasible PC/bin configuration stays visible. |
| `tezca.graph_state.GraphGridResult.describe()` | Return copied rows, configuration count, descriptive median/minimum/maximum and grid identity. Range is sensitivity across settings, not a confidence interval. |
| `tezca.graph_state.GraphGridResult.table` | Property → defensive-copy DataFrame including all feasible/infeasible requested grid rows. |
| `tezca.graph_state.GraphStateSpec(*, features, basis, scaling='none', pca_components=None, ddof=1, partition='equal_width', bins=3, edges=None, ties='raise', node_policy='full_grid', outside='raise', pseudocount=0, max_states=10000, graph_edges=None, weights=None, measure='mass', volume_units=None, conductance_units=None)` | Declare ordered features/basis, scaling/PCA, partition/edges, tie/outside/node policies and finite max_states. Defaults full grid, mass energy and no scaling. Density energy additionally needs explicit geometric conductance/volume units. |
| `tezca.graph_state.GraphStateSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.graph_state.GraphStateSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.graph_state.StateReference(record)` | Advanced frozen state-geometry record. Prefer fit_state_reference; constructor checks record geometry but does not recover omitted data or certify a saved bundle. |
| `tezca.graph_state.StateReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.graph_state.StateReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.graph_state.fit_state_reference(data, *, spec, training_positions=None, observations=None, fit_scope='training_only')` | Data + GraphStateSpec → StateReference fitting only declared training rows by default. Freezes scale/PCA/partition/graph geometry and support; insufficient or infeasible fits remain status records. |
| `tezca.graph_state.graph_state_grid(data, *, features, basis, scaling='zscore', ddof=1, max_states=10000, observations=None, training_positions=None)` | Data + features/basis → GraphGridResult covering all 16 admitted PC/bin choices, with defaults zscore and ddof=1. Retains infeasible dimensions/configurations; no best-grid selection or silent clipping. |
| `tezca.graph_state.graph_state_roughness(data, *, spec=None, reference=None, observations=None)` | Data + exactly one spec or frozen StateReference → ResultRecord of mass/density graph energy and occupancies. Self-fit and fixed-geometry evaluation remain distinct; outside cells are refused or explicitly excluded. |
| `tezca.graph_state.sqrt_mass_graph_energy(probabilities, edges, weights=None, *, measure='mass', volumes=None, volume_units=None, conductance_units=None)` | Normalized masses, unique undirected edges and optional weights → dict with 4*sum(weight*(sqrt mass difference)^2), per-edge values and geometry. Density mode requires positive volumes/conductances and units; no renormalization. |

<a id="reconstruction-sensitivity"></a>

## Compare supplied completions without inventing them

Provide independently generated completion series with exact retained values/masks/lineage and input/draw/method/plan identities. Complete-reference Welch, complete-reference irregular, masked irregular and each completion Welch are separate routes. For natural gaps use `complete=None`; there is no observed hidden truth. Contrasts are descriptive, not causal reconstruction effects. No optional backend is needed.

Detailed contracts and status meanings: [sensitivity guide](sensitivity.md).

### `tezca.sensitivity`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.sensitivity.SensitivityReport(routes, contrasts, description)` | Advanced owned route/contrast/description container. Prefer spectral_sensitivity; constructor does not execute routes or infer causal effects. |
| `tezca.sensitivity.SensitivityReport.contrasts` | Property → defensive-copy DataFrame of signed/absolute matched contrasts and explicit incompatibility reasons. |
| `tezca.sensitivity.SensitivityReport.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.sensitivity.SensitivityReport.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.sensitivity.SensitivityReport.routes` | Property → defensive-copy DataFrame of every declared route/draw and its result/support/status. |
| `tezca.sensitivity.compare_estimates(baseline, candidate, *, contrast)` | Baseline/candidate estimate mappings with explicit scientific signatures + contrast label → owned contrast dict. Only comparable available values get signed/absolute deltas; missing/mismatched cases retain reasons. |
| `tezca.sensitivity.compare_metric_results(baseline, candidate, *, baseline_interval, candidate_interval, design='fixed')` | Existing FI1/PCA/graph/coherence ResultRecords + explicit baseline/candidate intervals → DataFrame of comparable metrics and qualifications. Fixed reference is default; allowed_training_refit declares representation change. Uses invariants for PCA rather than arbitrary eigenvector signs. |
| `tezca.sensitivity.compare_networks(baseline, candidate)` | Two NetworkResults → dict of edge and summary DataFrames plus setting differences. Aligns true window endpoints/node IDs and signed edges; missing edges are not invented zeros. |
| `tezca.sensitivity.gap_geometry(masked)` | Masked AnalysisSeries → dict of missing positions, blocks and regular durations. Irregular coverage/adjacency stays unknown; missing cells remain distinct from observed zeros. |
| `tezca.sensitivity.spectral_sensitivity(complete, masked, completions, *, variables, welch, irregular, model, fit_frequency_hz, min_cycles, daily_spec=None, training=None, training_positions=None, design='fixed', masked_training=None)` | Complete empirical reference (or None for natural gaps), masked data and independent supplied draws + explicit Welch/irregular/model/common-fit specs → SensitivityReport. Keeps W0/I0/Im/Wc routes; optional daily transform uses separate nonoverlapping training, fixed or explicitly allowed training-refit design. |
| `tezca.sensitivity.summarize_draws(report)` | SensitivityReport → DataFrame of separately computed completion metrics, means/spread and exclusions. Between-draw spread is descriptive, not calibrated uncertainty. |
| `tezca.sensitivity.validate_completions(masked, completions, *, complete=None)` | Masked AnalysisSeries, mapping of draw IDs to supplied completed series (or None) and optional complete reference → None on success, ValueError on mismatch. Checks exact retained cells, masks, lineage and source-bound imputation identity; performs no completion. |

<a id="regimes"></a>

## Fit candidate regimes and preserve gap barriers

Use declared feature definitions and stable original row IDs. `preprocess` is explicit: `None` for no transformation, an unfitted sklearn StandardScaler, or CustomPreprocess (see detailed guide). KMeans/DBSCAN execution and native scaling need the models extra; custom factories with preprocess=None need only core. Factories return independent unfitted objects; assignment requires a declared capability. Native labels are nominal, noise/exclusions are distinct, and transitions do not cross unsupported gaps. No executable model serialization is offered.

Detailed contracts and status meanings: [regimes guide](regimes.md).

### `tezca.regimes`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.regimes.CapabilityError(*args)` | ValueError subclass for an unsupported backend capability, such as assignment or weights. Ordinary message arguments; choose a backend that explicitly supports the requested operation. |
| `tezca.regimes.ClusterSpec(method, **parameters)` | Construct optional native kmeans or dbscan spec. KMeans requires n_clusters/n_init/random_state; DBSCAN requires eps/min_samples/metric. Rejects unrelated parameters; constructor needs no sklearn, execution requires the models extra. |
| `tezca.regimes.ClusterSpec.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.regimes.ClusterSpec.record` | Property → defensive-copy mapping of the declared/resolved record and settings; changing the copy does not change the instance. |
| `tezca.regimes.CustomClusterer(factory, *, config, version, assignment=False, sample_weight=False, soft=False, precomputed=False, randomness=False, noise_labels=(), soft_meaning=None)` | Declare a factory returning independent unfitted model instances, serializable config/version and explicit capabilities. Default no assignment/weights/soft/precomputed/randomness; soft needs meaning and noise labels stay native. Core custom route with preprocess=None needs no sklearn. |
| `tezca.regimes.CustomPreprocess(factory, *, config, version, row_map=False)` | Declare an independent unfitted transformer factory, serializable config/version and optional explicit row-map capability. Returns adapter; factory executes only during analysis. No implicit row dropping or optional imports. |
| `tezca.regimes.RegimeFeatures(data, *, features, definitions, row_ids, intervals=None)` | AnalysisSeries + ordered features, definitions, unique string row IDs and optional interval ends → owned feature/schema view. Definitions identify method/reference/window and preserve original support. |
| `tezca.regimes.RegimeFeatures.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.regimes.RegimeFeatures.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.regimes.RegimeFeatures.schema` | Property → copied ordered feature-definition list with units, basis, method/reference and window identities. |
| `tezca.regimes.RegimeFeatures.values` | Property → copied selected-feature DataFrame indexed by declared original row IDs. |
| `tezca.regimes.RegimeFitError(*args)` | RuntimeError subclass for expected numerical failure raised by a custom backend. Becomes an explicit failed analysis outcome; unexpected programming exceptions propagate. |
| `tezca.regimes.RegimeReference(record, model, preprocess)` | Advanced container owning copied scientific record, fitted model and preprocessing object. Prefer cluster_regimes result.reference. In-memory assignment only; result export never serializes executable models. |
| `tezca.regimes.RegimeReference.describe()` | Return defensive-copy description mapping with this object's saved settings/support/provenance/status fields; no refitting or file access. |
| `tezca.regimes.RegimeReference.fitted_model` | Property → owned copy of fitted in-memory backend. Never an exported/reloaded executable model. |
| `tezca.regimes.RegimeReference.fitted_preprocess` | Property → owned copy of fitted in-memory preprocessing object (or None when absent). |
| `tezca.regimes.RegimeReference.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.regimes.RegimeResult(rows, description, reference, coordinates, distances=None, cost_seconds=0)` | Advanced row/description/reference/coordinate container with optional distances and reported cost. Prefer cluster_regimes/assign_regimes; wrapping records does not establish a fitted-model contract. |
| `tezca.regimes.RegimeResult.describe()` | Return copied method/config/support/capability description plus identity and reported cost_seconds. |
| `tezca.regimes.RegimeResult.identity` | Property → deterministic immutable identity string for this object's scientific record; integrity/identity does not establish authenticity or scientific validity. |
| `tezca.regimes.RegimeResult.reference` | Property → the same RegimeReference instance, whose public record/model accessors return defensive copies. No automatic model export or reconstruction. |
| `tezca.regimes.RegimeResult.rows` | Property → fresh DataFrame preserving original row IDs, native labels, exclusions/noise and status/reason. Mixed strings/large integers remain exact. |
| `tezca.regimes.assign_regimes(features, *, reference, observations=None, soft=False)` | Compatible RegimeFeatures + frozen RegimeReference → RegimeResult using supported assignment without refitting. Requires assignment capability; DBSCAN refuses rather than inventing predict. soft remains explicit. |
| `tezca.regimes.cluster_regimes(features, *, clusterer, preprocess, observations=None, fit_scope='self', training_ids=None, sample_weight=None, soft=False, distances=None, metric_interpretation=None)` | RegimeFeatures + declared clusterer/preprocess → RegimeResult with labels, exclusions, support and reference. Default self-fit; training scope needs explicit IDs. Requested weights/soft/distances are capability-checked; built-in specs need models, custom route is core. |
| `tezca.regimes.compare_regime_models(candidates, *, noise_policy)` | Finite (name, RegimeResult) candidates + noise policy → dict with candidate/adjusted-Rand-agreement DataFrames. Preserve all candidates and label identities; no winner or temporal-independent confidence claim. |
| `tezca.regimes.summarize_regime_timeline(result, *, max_gap, noise_policy)` | RegimeResult + explicit max_gap/noise_policy → dict of row, episode and transition DataFrames. Uses union interval duration; preserves gap barriers, noise and exclusions with no smoothing. |

<a id="exports-and-reload"></a>

## Save, validate and read numerical results

Only export writes files; parent destination must already exist and the child bundle must be new. `load_results` reads/validates without running analysis, fitting or model reconstruction. Hashes establish internal integrity, not authenticity, freshness or data rights. Supply expected identities from a trusted caller record when identity matters. Direct result/reference constructors are advanced containers; they do not replace validated reload.

Detailed contracts and status meanings: [exports guide](exports.md).

### `tezca.exports`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.exports.LoadedResults(manifest, details, tables)` | Advanced owned manifest/details/tables container. Use load_results for validation; direct construction from dictionaries bypasses bundle checks and does not reconstruct executable analysis objects. |
| `tezca.exports.LoadedResults.details` | Property → defensive-copy complete decoded numerical/scientific detail mapping; no executable fitted objects. |
| `tezca.exports.LoadedResults.manifest` | Property → defensive-copy validated manifest including identities, checksums, availability and backend/version records. |
| `tezca.exports.LoadedResults.table(family)` | Canonical family filename string → new DataFrame with canonical columns and owned nested records. Unknown family → ValueError; absent numerical results stay missing. |
| `tezca.exports.ResultItem(kind, result, *, sources, labels=None, timeline=None)` | Snapshot one supported result under an explicit kind, source aliases and optional descriptive labels/already-computed regime timeline. Returns owned item; the result's scientific record remains authoritative. |
| `tezca.exports.ResultItem.describe()` | Return copied kind/result/source bindings/labels and any already-computed timeline snapshot. |
| `tezca.exports.export_results(destination, *, results, sources, references=None, requested_policy=None, resolved_policy=None, backend=None, compatibility=False, unavailable=None)` | NEW destination, named ResultItems and source bindings → owned manifest and explicit CSV/JSON/array bundle files. Parent must exist; manifest written last. Optional references/policies/backend/compatibility/unavailable declarations are preserved; refuses overwrites and leaves failed partial output for diagnosis. |
| `tezca.exports.load_results(directory, *, expected_bundle_id=None, expected_sources=None, expected_references=None)` | Saved directory plus optional expected bundle/source/reference identities → validated LoadedResults. Validates complete closed schema, versions and byte/content bindings without fitting. Expectations enforce caller identity; without them only internal consistency is established. |

<a id="ecofill"></a>

## Bridge admitted EcoFill objects and saved exports

Importing the adapter loads no EcoFill, YAML or sklearn. Real object conversion needs the admitted optional EcoFill implementation; format-1 export loading remains standalone core. Source, representation, clock and imputation trust are explicit caller declarations. Unknown provenance never becomes observed; hidden truth is excluded.

Detailed contracts and status meanings: [ecofill guide](ecofill.md).

### `tezca.adapters.ecofill`

| Import-qualified calling form or accessor | Purpose, input → output, choices and limits |
| --- | --- |
| `tezca.adapters.ecofill.from_ecofill(value, *, source, representation, clock, trusted_imputation=False)` | Admitted real EcoFill SeriesData/ImputationResult + caller source/representation/clock declarations → AnalysisSeries from available/completed values and public provenance. Optional EcoFill needed at call; excludes hidden raw truth and requires explicit imputation trust. |
| `tezca.adapters.ecofill.load_ecofill_export(directory, *, source, representation, clock, trusted_imputation=False)` | EcoFill format-1 directory + caller source/representation/clock declarations → AnalysisSeries using four fixed admitted filenames. Core path needs no EcoFill and never reads raw.csv; validates aligned masks/details and preserves unknown provenance. |

## Errors and unavailable outcomes

A configuration/type/unit/clock/binding problem generally raises `ValueError`
before analysis; unsupported optional imports raise an actionable `ImportError`.
Catch the named `UnresolvedPolicyError`, `OutsideReferenceError` and
`CapabilityError` when their documented conditions are expected. A custom
clusterer may raise `RegimeFitError` for a known numerical failure; do not hide
programming errors with a broad catch. Export also reports filesystem failures
and refuses existing/incomplete destinations; preserve failed output and choose
a new destination.

Valid but insufficient inputs return explicit missing outcomes, not zeros.
Always read all three layers: `compute_status` says whether a numeric result was
produced, `metric_status` qualifies the definition, and `use_status` qualifies
the support/provenance for use. The vocabulary is method-specific; inspect
`reason` and the linked method page, rather than treating every non-`ok` string
as the same condition. Examples include insufficient support/cycles/resolution,
unknown provenance, degenerate input/partition, outside-reference exclusion,
unavailable historical backend, unsupported null and failed regime fitting.
The [spectral summary status table](spectral_summary.md) distinguishes supported
numerical estimates from undefined model comparisons and withheld summaries.

## Inventory coverage and exclusions

This sheet covers **261** currently supported module functions, class
constructors, public Tezca methods, properties, inherited frozen-reference
accessors, immutable dataclass fields and the public outside-reference diagnostic
field. The count includes accessors, not just top-level imports.

The stdlib [inventory checker](../examples/api_inventory.py) discovers definitions
from the imported package's public source modules, resolves live symbols and
compares each row's calling form using `inspect.signature`. It checks the
installed package supplied by the active interpreter, not a presumed checkout.
Imports must preserve absent YAML, sklearn and EcoFill backends. With the
working directory set to the checkout root, run
`python -I -B examples/api_inventory.py docs/api_cheatsheet.md`; it prints a JSON
inventory on success and raises on a missing, duplicate, unknown or mismatched
row. 

Explicit exclusions: `tezca.data.aligned_array`,
`tezca.support.positive_duration`, `tezca.support.positions_array` and
`tezca.policy.terminal_fields` are implementation utilities for alignment,
validation and migration traversal; none is a supported analysis entry point.
Imported library objects, imported aliases of an API defined elsewhere, private
underscore modules/classes/functions, magic methods and inherited built-in
exception methods are not additional Tezca APIs. Public result/reference
constructors were **included**, even when only their producer is recommended.
Factory backend protocols are explained in the [regime guide](regimes.md);
methods on caller-owned third-party backends are not Tezca definitions.
Noncallable constants are outside the callable inventory: useful examples are
`tezca.__version__`, `tezca.policy.RECIPES`, and `tezca.exports.FAMILIES` /
`FORMAT` for canonical output-family names and saved-format identity.

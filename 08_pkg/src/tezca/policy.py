"""Three inspectable recipes, strict validation and explicit legacy migration.

Future-method settings are declarations, not implemented estimators. YAML is
imported only when a caller explicitly requests a YAML operation.
"""

import importlib
import math

from ._owned import identity, owned
from .support import ObservationPolicy, positive_duration


RECIPES = ("manglaria_sensor_20260911_compat", "manglaria_report_v1_10_compat",
           "manglaria_research_v1")
OLD_PROFILE = "manglaria_v1_10_compat"

_COMMON = {
    "individual": {
        "min_analysis_points": 256, "anomaly": "median_minute_of_day", "trend_unit": "per_day",
        "acf_lags": ["one_step", "one_hour", "twenty_four_hours"],
        "psd": {"method": "welch", "max_nperseg": 4096, "min_nperseg": 64,
                "target_segments": 8, "beta_sign": "beta_equals_negative_loglog_slope",
                "peak_prominence_log10": 0.08, "max_peaks": 3,
                "bands_hours": {"circadian": [20, 28], "semidiurnal": [10, 14], "synoptic": [48, 240]},
                "segment_rule": "floor_n_over_target_length"},
        "fi1": {"label": "discrete_first_order_fisher_index_historical", "window_hours": 72,
                "step_hours": 6, "abrupt_k": 3}},
    "collective": {"cadence": "1h", "min_family_overlap_days": 30,
                   "min_variable_coverage": 0.65, "min_variables": 5, "standardize": True,
                   "pca": {"full_spectrum": True, "pc90": True,
                           "effective_dimension": "participation_ratio", "export_modes": 3},
                   "graph": {"window_hours": 168, "step_hours": 24, "correlation_threshold_abs": 0.3}},
    "coherence": {"estimator": "pinned_sensor_helper", "min_raw_welch_segments_per_run": 4,
                  "overlap_fraction": 0.5, "alpha": 0.05, "exclude_direct_target_transforms": True,
                  "run_pooling": "pinned_sensor_unverified"},
    "interpretation": {"primary_use_max_imputed_fraction": 0.15,
                       "burden_convention": "relevant_support_v1",
                       "metric_status": {"psd_beta": "primary_with_support", "fi1_discrete": "historical_sensitivity_index",
                                         "pca_full_spectrum": "primary_with_support",
                                         "functional_correlation": "exploratory_association",
                                         "coherence": "exploratory_frequency_coupling",
                                         "graph_state_fisher": "exploratory_resolution_sensitive",
                                         "density_fisher": "estimator_specific_unresolved",
                                         "regime_labels": "exploratory_candidate_regimes",
                                         "harmonic_diagnostics": "null_dependent_diagnostic"}},
    "preprocessing": {"individual_descriptive": "raw", "individual_psd": "daily_residual",
                      "individual_fi1": "daily_residual", "collective_pca": "raw",
                      "collective_network": "daily_residual", "collective_coherence": "daily_residual"},
    "support": {"selection": "longest_run", "estimator": "regular", "values": "provided",
                "min_points": 1, "complete_windows": None, "window_label": None, "window_hours": None,
                "step_hours": None, "max_reconstructed_duration": None},
}

# Each unresolved item is preserved in descriptions and identities. The foundation
# does not choose missing scientific definitions or claim pinned numerical parity.
_UNRESOLVED = {
    "preparation": ["two_family_gate", "station_segment_and_reducer", "source_priority_local_remote",
                    "coverage_observed_or_completed", "pair_metadata_rules"],
    "support": ["maximum_reconstruction_duration", "window_anchor_and_endpoint", "observed_only_comparison"],
    "daily_cycle": ["reference_period", "civil_timezone", "minimum_replicates", "fallback"],
    "trend": ["dependence_and_p_value_method"],
    "psd": ["taper", "overlap", "detrending", "scaling", "fit_frequency_limits", "broken_model_criterion"],
    "harmonic": ["backend", "null", "alpha", "lambda"],
    "fi1": ["binning", "pseudocount", "entropy_base", "adjacency_and_mad_rule"],
    "graph": ["normalization", "state_geometry"],
    "pca": ["physical_features", "basis", "scaling", "ddof", "training_support"],
    "network": ["physical_features", "basis", "missingness", "minimum_pair_support", "window_boundaries"],
    "coherence": ["taper", "nperseg_bounds", "phase_sign", "effective_average_formula",
                  "bias_correction", "threshold_scale", "null_and_multiplicity"],
    "lsft_wosa": ["backend", "normalization", "frequency_grid", "segments_and_overlap"],
    "density_fisher": ["coordinates", "estimator", "boundaries", "partition_and_support"],
    "regimes": ["features", "preprocess", "clusterer_and_parameters", "assignment_capability"],
    "delivery": ["figure_version", "export_version"],
}


class UnresolvedPolicyError(ValueError):
    pass


def terminal_fields(mapping, prefix=""):
    result = {}
    if not isinstance(mapping, dict):
        raise ValueError("policy root must be a mapping")
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError("policy keys must be strings")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(terminal_fields(value, path))
        else:
            result[path] = owned(value)
    return result


def _merge(base, requested, path=""):
    if not isinstance(requested, dict):
        raise ValueError(f"{path or 'settings'} must be a mapping")
    result = owned(base)
    for key, value in requested.items():
        if key not in base:
            raise ValueError(f"unknown policy field: {path + '.' if path else ''}{key}")
        field = f"{path}.{key}" if path else key
        if isinstance(base[key], dict):
            result[key] = _merge(base[key], value, field)
        else:
            result[key] = owned(value)
    return result


def _number(value, path, *, minimum=0, maximum=None, integer=False, inclusive_min=True, inclusive_max=True):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} requires a finite {'integer' if integer else 'number'}, not a boolean")
    if value < minimum or (not inclusive_min and value == minimum):
        raise ValueError(f"{path} is below its valid range")
    if maximum is not None and (value > maximum or (not inclusive_max and value == maximum)):
        raise ValueError(f"{path} is above its valid range")


def _validate(settings):
    leaves = terminal_fields(settings)
    expected = terminal_fields(_COMMON)
    if set(leaves) != set(expected):
        raise ValueError("resolved settings must contain exactly the documented schema fields")
    positive_integer = {"individual.min_analysis_points", "individual.psd.max_nperseg",
                        "individual.psd.min_nperseg", "individual.psd.target_segments",
                        "individual.psd.max_peaks", "collective.min_variables", "collective.pca.export_modes",
                        "coherence.min_raw_welch_segments_per_run", "support.min_points"}
    positive = {"individual.fi1.window_hours", "individual.fi1.step_hours", "collective.graph.window_hours",
                "collective.graph.step_hours", "collective.min_family_overlap_days"}
    fractions = {"collective.min_variable_coverage", "collective.graph.correlation_threshold_abs",
                 "interpretation.primary_use_max_imputed_fraction"}
    for path in positive_integer:
        _number(leaves[path], path, minimum=1, integer=True)
    for path in positive:
        _number(leaves[path], path, inclusive_min=False)
    for path in fractions:
        _number(leaves[path], path, maximum=1)
    for path in ("individual.psd.peak_prominence_log10", "individual.fi1.abrupt_k"):
        _number(leaves[path], path)
    _number(leaves["coherence.overlap_fraction"], "coherence.overlap_fraction", maximum=1, inclusive_max=False)
    _number(leaves["coherence.alpha"], "coherence.alpha", maximum=1, inclusive_min=False, inclusive_max=False)
    for path, default in expected.items():
        if type(default) is bool and type(leaves[path]) is not bool:
            raise ValueError(f"{path} must be a boolean")
    enums = {
        "individual.anomaly": {"median_minute_of_day", "none"}, "individual.trend_unit": {"per_day"},
        "individual.psd.method": {"welch"}, "individual.psd.beta_sign": {"beta_equals_negative_loglog_slope"},
        "individual.psd.segment_rule": {"floor_n_over_target_length", "minimum_raw_segments_with_resolution"},
        "individual.fi1.label": {"discrete_first_order_fisher_index_historical"},
        "collective.pca.effective_dimension": {"participation_ratio"},
        "coherence.estimator": {"pinned_sensor_helper", "report_run_average_corrected", "research_pooled_spectra"},
        "coherence.run_pooling": {"pinned_sensor_unverified", "average_run_coherences", "pool_auto_cross_spectra"},
        "interpretation.burden_convention": {"relevant_support_v1"},
        "support.selection": {"longest_run", "all_runs", "windows", "irregular", "complete_case"},
        "support.estimator": {"regular", "irregular", "point", "complete_case"},
        "support.values": {"provided", "observed_only", "trusted_imputation"},
    }
    for path in expected:
        if path.startswith("preprocessing."):
            enums[path] = {"raw", "daily_residual", "raw_and_daily_residual", "unresolved"}
        elif path.startswith("interpretation.metric_status."):
            enums[path] = {"primary_with_support", "historical_sensitivity_index", "exploratory_association",
                           "exploratory_frequency_coupling", "exploratory_resolution_sensitive",
                           "estimator_specific_unresolved", "exploratory_candidate_regimes", "null_dependent_diagnostic"}
    for path, choices in enums.items():
        if not isinstance(leaves[path], str) or leaves[path] not in choices:
            raise ValueError(f"{path} must be one of {sorted(choices)}")
    lags = leaves["individual.acf_lags"]
    if not isinstance(lags, list) or not lags or any(not isinstance(item, str) for item in lags) or len(set(lags)) != len(lags):
        raise ValueError("acf_lags requires a nonempty list of unique exact labels/durations")
    for lag in lags:
        if lag not in {"one_step", "one_hour", "twenty_four_hours"}:
            positive_duration(lag, "acf_lag")
    for path in ("individual.psd.bands_hours.circadian", "individual.psd.bands_hours.semidiurnal",
                 "individual.psd.bands_hours.synoptic"):
        endpoints = leaves[path]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError(f"{path} must have two hour endpoints")
        for value in endpoints:
            _number(value, path, inclusive_min=False)
        if endpoints[0] >= endpoints[1]:
            raise ValueError(f"{path} endpoints must increase")
    positive_duration(leaves["collective.cadence"], "collective.cadence")
    if leaves["individual.psd.min_nperseg"] > leaves["individual.psd.max_nperseg"]:
        raise ValueError("min_nperseg must not exceed max_nperseg")
    if leaves["individual.anomaly"] == "none" and any(leaves[path] in {"daily_residual", "raw_and_daily_residual"}
                                                       for path in ("preprocessing.individual_psd", "preprocessing.individual_fi1")):
        raise ValueError("residual individual branches require the declared anomaly transform")
    pooling = {"pinned_sensor_helper": "pinned_sensor_unverified", "report_run_average_corrected": "average_run_coherences",
               "research_pooled_spectra": "pool_auto_cross_spectra"}
    if pooling[leaves["coherence.estimator"]] != leaves["coherence.run_pooling"]:
        raise ValueError("coherence estimator and run pooling are incompatible")
    window_fields = (leaves["support.window_hours"], leaves["support.step_hours"])
    support = settings["support"]
    complete, label = support["complete_windows"], support["window_label"]
    if complete is not None and type(complete) is not bool:
        raise ValueError("support.complete_windows must be a boolean or null")
    if label is not None and (not isinstance(label, str) or label not in {"start", "end"}):
        raise ValueError("support.window_label must be start, end or null")
    if leaves["support.selection"] == "windows":
        for value in window_fields:
            _number(value, "support window/step hours", inclusive_min=False)
        support["complete_windows"] = True if complete is None else complete
        support["window_label"] = "start" if label is None else label
    else:
        if any(value is not None for value in window_fields) or complete is False or label == "end":
            raise ValueError("support window controls require windows selection")
        # Legacy neutral defaults have one canonical, nonoperative representation.
        support["complete_windows"] = support["window_label"] = None
    if leaves["support.estimator"] == "regular" and leaves["support.selection"] in {"irregular", "complete_case"}:
        raise ValueError("regular estimator requires contiguous runs/windows")
    if leaves["support.selection"] == "irregular" and leaves["support.estimator"] not in {"point", "irregular"}:
        raise ValueError("irregular selection requires an irregular or point estimator")
    ObservationPolicy(values=leaves["support.values"], min_points=leaves["support.min_points"],
                      max_imputed_fraction=leaves["interpretation.primary_use_max_imputed_fraction"],
                      max_reconstructed_duration=leaves["support.max_reconstructed_duration"])


def _defaults(recipe):
    result = owned(_COMMON)
    if recipe == RECIPES[1]:
        result["support"]["selection"] = "all_runs"
        result["preprocessing"].update(individual_psd="unresolved", individual_fi1="unresolved",
                                        collective_coherence="unresolved")
        result["coherence"].update(estimator="report_run_average_corrected", run_pooling="average_run_coherences")
    elif recipe == RECIPES[2]:
        result["support"].update(selection="windows", window_hours=72, step_hours=6)
        result["preprocessing"].update(individual_psd="raw_and_daily_residual",
                                        collective_pca="raw_and_daily_residual")
        result["individual"]["psd"]["segment_rule"] = "minimum_raw_segments_with_resolution"
        result["coherence"].update(estimator="research_pooled_spectra", run_pooling="pool_auto_cross_spectra")
    return result


class AnalysisPolicy:
    def __init__(self, recipe, settings=None, *, operational=None):
        if recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {RECIPES}; the old profile requires explicit migration")
        requested = {} if settings is None else settings
        resolved = _merge(_defaults(recipe), requested)
        _validate(resolved)
        operational = {} if operational is None else operational
        if not isinstance(operational, dict) or set(operational) - {"created_at", "run_label"}:
            raise ValueError("operational accepts only created_at and run_label")
        if any(not isinstance(value, str) for value in operational.values()):
            raise ValueError("operational metadata values must be strings")
        self._recipe, self._settings = recipe, resolved
        self._requested, self._operational = owned(requested), owned(operational)
        self._migration = []
        self._id = identity({"schema_version": 1, "recipe": recipe, "settings": resolved,
                             "unresolved": _UNRESOLVED, "parity": "unverified"})
        self._requested_id = identity({"recipe": recipe, "settings": requested})

    @property
    def identity(self):
        return self._id

    @property
    def requested_identity(self):
        return self._requested_id

    @property
    def settings(self):
        return owned(self._settings)

    @property
    def migration(self):
        return owned(self._migration)

    def describe(self):
        return {"recipe": self._recipe, "requested": owned(self._requested), "resolved": self.settings,
                "requested_id": self._requested_id, "resolved_id": self._id,
                "operational": owned(self._operational), "unresolved": owned(_UNRESOLVED),
                "parity": "unverified", "capabilities": ["observation_policy", "support_configuration"],
                "migration": self.migration}

    def to_dict(self):
        return {"schema_version": 1, "recipe": self._recipe, "settings": self.settings,
                "operational": owned(self._operational)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - {"schema_version", "recipe", "settings", "operational"}:
            raise ValueError("unknown policy envelope fields or nonmapping root")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported resolved policy schema_version")
        return cls(data.get("recipe"), data.get("settings"), operational=data.get("operational"))

    def observations(self):
        support = self._settings["support"]
        return ObservationPolicy(support["values"], support["min_points"],
                                 self._settings["interpretation"]["primary_use_max_imputed_fraction"],
                                 support["max_reconstructed_duration"])

    def support_options(self):
        support = self._settings["support"]
        result = {"observations": self.observations(), "policy": support["selection"],
                  "estimator": support["estimator"]}
        if support["selection"] == "windows":
            result.update(window=f"{support['window_hours']}h", step=f"{support['step_hours']}h",
                          complete=support["complete_windows"], label=support["window_label"])
        return result

    def method_spec(self, method):
        if method not in _UNRESOLVED:
            raise ValueError(f"unknown scientific method: {method}")
        raise UnresolvedPolicyError(f"{method} has unresolved requirements {_UNRESOLVED[method]}; "
                                    "use an explicit method specification; this named recipe does not resolve these choices")


def _legacy_template():
    legacy = {"format_version": 1, "profile": OLD_PROFILE}
    for key in ("individual", "collective", "coherence", "interpretation"):
        legacy[key] = owned(_COMMON[key])
    del legacy["individual"]["psd"]["segment_rule"]
    legacy["collective"]["anomaly"] = "median_minute_of_day"
    del legacy["coherence"]["estimator"]
    del legacy["coherence"]["run_pooling"]
    legacy["coherence"]["corrected"] = True
    del legacy["interpretation"]["burden_convention"]
    for field in ("density_fisher", "regime_labels", "harmonic_diagnostics"):
        del legacy["interpretation"]["metric_status"][field]
    return legacy


def migrate_legacy(data, *, target):
    if target not in RECIPES:
        raise ValueError("old profile requires an explicit named migration target")
    template = _legacy_template()
    source = _merge(template, data)
    if set(terminal_fields(data)) != set(terminal_fields(template)):
        raise ValueError("legacy migration requires the complete 45-leaf source schema")
    if type(source["format_version"]) is not int or source["format_version"] != 1 or source["profile"] != OLD_PROFILE:
        raise ValueError("unsupported legacy format/profile")
    if source["coherence"]["corrected"] is not True:
        raise ValueError("legacy corrected must be true; choose a separately explicit coherence specification")
    if (not isinstance(source["collective"]["anomaly"], str)
            or source["collective"]["anomaly"] not in {"median_minute_of_day", "none"}):
        raise ValueError("invalid collective anomaly")
    settings = _defaults(target)
    for key in ("individual", "collective", "coherence", "interpretation"):
        old = owned(source[key])
        if key == "collective":
            old.pop("anomaly")
        if key == "coherence":
            old.pop("corrected")
        settings[key] = _merge(settings[key], old, key)
    if source["collective"]["anomaly"] == "none":
        for branch in ("collective_network", "collective_coherence"):
            settings["preprocessing"][branch] = "raw"
    policy = AnalysisPolicy(target, settings)
    resolved = terminal_fields(policy.settings)
    rows = []
    for path, value in sorted(terminal_fields(source).items()):
        destination, disposition, reason = path, "mapped", None
        if path == "format_version":
            destination, disposition, reason = "schema_version", "changed", "separate resolved schema identity"
            final = 1
        elif path == "profile":
            destination, disposition, reason = "recipe", "changed", "explicit target resolves ambiguous routes"
            final = target
        elif path == "coherence.corrected":
            destination, disposition = "coherence.estimator", "unresolved"
            reason = "explicit estimator route; pinned backend/correction/threshold definitions remain unverified"
            final = resolved[destination]
        elif path == "collective.anomaly":
            destination, disposition = "preprocessing", "changed"
            reason = "branch-specific PCA, network and coherence preprocessing"
            final = {k: v for k, v in policy.settings["preprocessing"].items() if k.startswith("collective_")}
        else:
            final = resolved[path]
            if path == "individual.psd.target_segments":
                disposition, reason = "changed", f"interpreted by {resolved['individual.psd.segment_rule']}"
            elif path == "interpretation.primary_use_max_imputed_fraction":
                disposition, reason = "changed", "versioned rule applies to relevant support and reconstructed blocks"
        rows.append({"source_path": path, "input_value": owned(value), "disposition": disposition,
                     "destination": destination, "resolved_value": owned(final), "reason": reason})
    policy._migration = rows
    policy._requested = owned(data)
    policy._requested_id = identity({"legacy": data, "target": target})
    return policy


def _yaml_module():
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError as error:
        if error.name != "yaml":
            raise
        raise ImportError("YAML operations require the optional extra: install tezca[yaml]") from error


def load_policy_yaml(text, *, target=None):
    yaml = _yaml_module()
    class StrictSafeLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    raise ValueError("YAML merge keys are not allowed")
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("YAML policy keys must be strings")
                if key in result:
                    raise ValueError(f"duplicate YAML key: {key}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result
    try:
        if any(isinstance(event, yaml.events.AliasEvent) for event in yaml.parse(text)):
            raise ValueError("YAML aliases are not allowed in policy data")
        data = yaml.load(text, Loader=StrictSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("invalid or unsafe policy YAML") from error
    if not isinstance(data, dict):
        raise ValueError("YAML policy root must be a mapping")
    if "profile" in data:
        return migrate_legacy(data, target=target)
    if target is not None:
        raise ValueError("migration target applies only to the legacy profile")
    return AnalysisPolicy.from_dict(data)


def dump_policy_yaml(policy):
    return _yaml_module().safe_dump(policy.to_dict(), sort_keys=True)

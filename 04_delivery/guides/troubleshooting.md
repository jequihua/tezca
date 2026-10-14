# Troubleshooting

[Start here](../README.md) · [Quickstart](quickstart.md) · [Bug template](bug_report.md)

Keep the original command, full exception and projection and built-wheel identities. Preserve failed
outputs; retry with a new destination after understanding the cause.

| Symptom | What to check and do |
| --- | --- |
| `py` or Python 3.11 is unavailable | Ask the local operator for the admitted Python environment. Other interpreter/platform combinations are not qualified by the candidate. |
| pip cannot find NumPy, pandas, SciPy or another requirement | Check the approved offline wheel directory and Python/architecture compatibility. The local wheel does not contain dependencies. Keep `--no-index`; have the operator supply the missing approved wheel. |
| Import uses a different Tezca than expected | Use the environment's explicit `Scripts\python.exe`, run outside the checkout, and check `import tezca; print(tezca.__version__, tezca.__file__)`. Keep the resulting local path out of shared reports. |
| Version or format mismatch at reload | Compare the projection file hashes, installed version and bundle format. Use the producing candidate's supported loader. Do not edit version fields in the manifest. |
| Naive, duplicate, off-grid or unresolved timestamps are refused | Supply unique ordered timezone-aware instants. Resolve daylight-saving ambiguity from source evidence, declare the regular cadence/phase or a genuine irregular clock, and preserve holes. Do not round timestamps just to satisfy the validator. |
| A method rejects its input type or representation | Consult its signature: methods accept specific `AnalysisSeries`, residual, spectrum or feature objects. Preserve declared units and raw/residual basis. Convert unsupported metadata deliberately before export. |
| `AnalysisPolicy.method_spec()` raises `UnresolvedPolicyError` | The recipe is a declaration with unresolved scientific choices. Use the explicit method constructor shown in [Choose a method](choose_methods.md). Supplying one does not make `method_spec()` resolve or certify historical parity. |
| Output directory or identity file already exists | Use a fresh name. Never overwrite an earlier bundle, partial write or trusted identity record. |
| Hash, schema or reference checks fail | Keep the bundle and error intact. Recheck file transfer/selection and the independently trusted expected identities. Do not “repair” hashes or remove the expectations to force success. |
| The right-looking bundle fails expected identity checks | Source, mask, reference or configuration differences matter even when a few values agree. Locate the intended bundle using the saved candidate/run record; make a new analysis only if scientifically warranted. |
| No number appears for a band, fit, pair or window | Inspect the independent status and reason. Use the [interpretation guide](understand_results.md); a preserved refusal is not automatically a bug. |
| A Unicode array is refused at export | Highly uneven or padded fixed-width strings can exceed the codec's allocation bound. Use an object string array for such text; see [the exact convention](../../08_pkg/docs/exports.md#typed-csv-and-json-conventions). |
| Unsupported metadata, labels or objects fail export | Use supported typed values and unique nonmissing labels. Do not stringify collisions such as integer `1` versus string `"1"`; do not use pickle or executable model serialization. |

## Optional paths

Core installation and the main delivery example need no PyYAML, scikit-learn or
EcoFill. Their absence must not prevent a core export/reload.

For YAML policy loading, ask the operator for the approved PyYAML wheel and
dependencies. For native clustering ask for the approved models stack. From the
checkout root, install only the selected extra into the isolated test
environment, using the same explicit `$DependencyWheels` value as the quickstart:

```powershell
..\tezca-test-env\Scripts\python.exe -m pip install --no-index --find-links "$DependencyWheels" "..\tezca-built-wheels\tezca-0.1.0.dev0-py3-none-any.whl[yaml]"
```

Replace `[yaml]` with `[models]` only for a models test. The [qualification
record](../qualification.md) names the exact optional versions exercised. An
ImportError for an uninstalled extra is expected; Tezca neither installs it nor
switches algorithms automatically.

EcoFill public-object tests use a separate admitted EcoFill 0.0.1 snapshot on
Python 3.12. That is not a promise about arbitrary EcoFill installations or a
command to fetch one. Obtain the specifically approved environment from the
operator. `tezca.adapters.ecofill.load_ecofill_export` instead reads the admitted
format-1 export with core dependencies and no EcoFill import. It deliberately
ignores raw/hidden truth. See [adapter requirements](../../08_pkg/docs/ecofill.md).

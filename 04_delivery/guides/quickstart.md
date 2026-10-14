# Install, analyze, save and reload

[Start here](../README.md) · [API cheat sheet](../../08_pkg/docs/api_cheatsheet.md)

Run these Windows PowerShell commands from the source checkout root. This is
an unreleased source projection, distinct from the earlier wheel-only candidate.
No prebuilt wheel or dependency store is included. The package is not installed
from PyPI by name. Use the prepared Python 3.11 runtime and existing build tools;
other platforms/versions are unqualified. See the [qualification record](../qualification.md).

## 1. Build and install locally

Your operator supplies an approved offline dependency wheel directory including
setuptools, wheel and their dependencies. Choose unused sibling environment and
output names. The source keeps `pyproject.toml` and `08_pkg/src` together.

```powershell
py -3.11 -m venv ..\tezca-test-env
$DependencyWheels = Read-Host 'Path to the approved local dependency wheels directory'
..\tezca-test-env\Scripts\python.exe -m pip install --no-index --find-links "$DependencyWheels" 'setuptools>=68' wheel
..\tezca-test-env\Scripts\python.exe -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir ..\tezca-built-wheels .
..\tezca-test-env\Scripts\python.exe -m pip install --no-index --find-links "$DependencyWheels" ..\tezca-built-wheels\tezca-0.1.0.dev0-py3-none-any.whl
..\tezca-test-env\Scripts\python.exe -m pip check
..\tezca-test-env\Scripts\python.exe -I -B -c "import tezca; print(tezca.__version__)"
```

Generated build metadata is ignored. Keep any built wheel's raw SHA-256 separately;
a rebuild is not claimed byte-identical to the earlier development candidate.

The last command prints `0.1.0.dev0`. `pip check` should report no broken
requirements. If installation cannot find a dependency, ask the operator for
the missing compatible wheel; the command intentionally does not contact an
index. You do not need to activate the environment or install optional packages.

## 2. Run the observed-data example

```powershell
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\delivery.py analyze ..\tezca-first-run
```

The generator uses only small synthetic inputs. Its explicit public calls cover
individual summaries, spectra, collective preparation, PCA, networks, coherence,
graph-state roughness, sensitivity and candidate regimes. It also preserves
missing/unknown inputs and skipped windows. No download or real measurement is
needed. The example's custom regime labels exercise the core export path;
they do not qualify scikit-learn clustering.

The new `..\tezca-first-run` directory contains 17 canonical CSV tables,
`analysis_manifest.json`, `result_details.json`, eight `compat_*.csv` files and
`compatibility_status.json`. The manifest accounts for all 18 table/manifest
families. The sibling `..\tezca-first-run.identity.json` stores the expected
bundle, source and reference identities for later comparison. Keep that identity
file separate from the result directory. Both names must be new.

The useful synthetic checks are a cosine's integrated Welch power near `0.5`
and PCA eigenvalues near `[4.0, 1.0]`. PCA describes variance along two learned
axes; those eigenvalues are properties of this constructed matrix, not a field
threshold. The requested 48–240 hour band is missing with an explicit reason:
its interval exceeds this spectrum's supported frequencies. It is not zero
power and does not indicate an absence of slow variation.

## 3. Read the saved result in a new process

Each command below starts a separate Python process after the producer exits.

```powershell
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\delivery.py reload ..\tezca-first-run --expected-identity ..\tezca-first-run.identity.json
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\read_bundle_stdlib.py ..\tezca-first-run
```

The first command uses Tezca's validated loader. It reads saved values and
checks the expected identities without fitting or analyzing again. The second
uses only Python's standard library to verify file hashes and read representative
CSV/JSON values; it is a small independent consumer, not a replacement for the
full schema validator. Its process does not import Tezca.

The loaded PCA eigenvalues should match the saved values exactly. A hash catches
changed bytes; it does not authenticate the author. A separately trusted identity
file detects selecting a different valid bundle. Replacing both a bundle and its
expected-identity file defeats that external comparison.

## 4. Compare an explicitly supplied completion

```powershell
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\delivery.py analyze ..\tezca-completed-run --completion supplied
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\delivery.py reload ..\tezca-completed-run --expected-identity ..\tezca-completed-run.identity.json
```

The example supplies replacement values separately and declares their provenance.
Tezca does not infer or generate an imputation. Inspect the completion identity,
contributing imputed fraction and the frozen training-reference identities,
not just changes in a summary number. [Understand results](understand_results.md)
explains the independent computation, scientific and use statuses.

An existing output destination is refused to protect your earlier work. Keep
it and use a new name. Compatibility CSVs are migration aids; their historical
fields may intentionally be null. See [migration and limitations](migration_and_limits.md)
before comparing them with old reports.

## 5. A small standalone smoke and the API inventory

```powershell
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\toolchain_smoke.py
..\tezca-test-env\Scripts\python.exe -I -B .\08_pkg\examples\api_inventory.py .\08_pkg\docs\api_cheatsheet.md
```

The first checks arrays, frames and a known sinusoid. The second checks all 261
public API calling forms against the installed package. Development scientific
tests and the native ledger/verifier are excluded; these commands are standalone.

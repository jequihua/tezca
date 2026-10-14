# Analyze, save and reload

[Installation](installation.md) · [API sheet](api_cheatsheet.md)

From a checkout root with Tezca installed in your Python >=3.11 environment:

```text
python --version
python -m pip install .
python -m pip check
python -I -B examples/toolchain_smoke.py
python -I -B examples/api_inventory.py docs/api_cheatsheet.md
python -I -B examples/delivery.py analyze ../tezca-first-run
python -I -B examples/delivery.py reload ../tezca-first-run --expected-identity ../tezca-first-run.identity.json
python -I -S -B examples/read_bundle_stdlib.py ../tezca-first-run
python -I -B examples/delivery.py analyze ../tezca-completed-run --completion supplied
python -I -B examples/delivery.py reload ../tezca-completed-run --expected-identity ../tezca-completed-run.identity.json
python -I -S -B examples/read_bundle_stdlib.py ../tezca-completed-run
```

Both output names must be new; existing destinations are refused. The inputs are
deterministic synthetic generators. The observed route preserves missing values;
the supplied route declares explicit replacement values and their provenance.
Neither route downloads data or performs automatic reconstruction.

Each result directory holds 17 canonical CSV tables, `analysis_manifest.json`,
`result_details.json`, eight compatibility CSVs and `compatibility_status.json`.
The manifest accounts for all 18 table/manifest families. Keep the sibling
expected-identity JSON separately for selecting the same bundle in future.

The cosine's integrated Welch power is approximately `0.5`; PCA eigenvalues are
approximately `[4.0, 1.0]`, participation ratio `25/17` and PC90 `2`. These describe
the constructed inputs, not field thresholds. The 48–240 hour band is withheld
with `outside_frequency_support`: "band/denominator would require extrapolation".
An unavailable band is not zero power. The supplied route's burden is `4/96`.

Every reload starts a new process and validates saved values and expected
identities without analyzing or fitting. The stdlib reader independently checks
27 saved payloads without importing Tezca; it is not the full schema validator.
Hashes establish byte consistency, not author authenticity or scientific validity.
Changing both a bundle and its expected identity defeats external selection.

Keep sibling helpers with [delivery.py](../examples/delivery.py).
See [result interpretation](guides/understand_results.md), [exports](exports.md),
[testing](contributing.md) and [scientific limits](qualification.md).

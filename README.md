# Tezca

Tezca is a Python package for environmental time-series analysis. It describes
individual signals, regular and irregular spectra, collective structure, networks,
coherence and exploratory regimes. Results retain support, provenance, method
settings and reusable reference identities, and can be saved for validated reload.

## Install

Use Python 3.11 or newer in your own virtual environment. From this checkout:

```text
python --version
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell), then run:

```text
python -m pip install .
python -m pip check
```

For editable development use `python -m pip install -e .`.
Optional YAML policies use `python -m pip install ".[yaml]"`; native clustering
models use `python -m pip install ".[models]"`.
With Git installed, install directly from GitHub using:

```text
python -m pip install "tezca @ git+https://github.com/jequihua/tezca.git"
```

See [installation](docs/installation.md) for commit pinning and offline use.
Installation provides the import package. Get examples, docs and tests from a
checkout; they are not installed commands. No PyPI release is asserted.

## A real API example

```python
import numpy as np
import pandas as pd
from tezca.data import AnalysisSeries
from tezca.individual import describe_series

signal = AnalysisSeries(
    [1., 2., 3., 4.], variables=["x"],
    time=pd.date_range("2000-01-01", periods=4, freq="h", tz="UTC"),
    metadata={"x": {"unit": "m", "representation": "raw"}},
    source={"id": "small-observed-example", "segment": "synthetic"},
    observed=np.ones(4, dtype=bool), clock={"mode": "regular", "cadence": "1h"},
)
mean = next(r for r in describe_series(signal) if r.summary()["method_id"] == "individual.mean_v1")
print(mean.value)  # 2.5
```

A finite replacement value is not automatically an observation. Declare supplied
values and their provenance explicitly; Tezca never completes gaps implicitly.
Missing results retain a reason and support rather than becoming zero. Fit
reference transforms on declared training data and reuse them without refitting.

Follow the [quickstart](docs/quickstart.md) for both observed and supplied workflows,
all 18 export families, expected-identity reload and the independent stdlib reader.
The [API sheet](docs/api_cheatsheet.md) covers every public calling form.
Use the [method guide](docs/guides/choose_methods.md), [result interpretation](docs/guides/understand_results.md)
and [qualification and limitations](docs/qualification.md) before drawing scientific conclusions.
Tezca has no plotting dependency or API; see [downstream visualization guidance](docs/visualization_guidelines.md).

Version: `0.1.0.dev0`. Historical numerical equivalence and real-data validity
remain unverified. No license has been selected; local availability does not
establish redistribution permission. See [contributing](docs/contributing.md)
and [bug reports](docs/guides/bug_report.md).

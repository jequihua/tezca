# Contributing and portable tests

Use Python >=3.11 and [editable installation](installation.md):

```text
python -m pip install -e .
python -m pip check
python -I -B -m unittest discover -s tests -v
```

The stdlib unittest suite imports the installed package and needs only core
dependencies. It checks owned inputs, provenance/support gates, literal mean and
Welch power, export/expected-identity reload and a withheld unsupported band.
It uses synthetic data and temporary results, with no downloads or private paths.
This portable suite is deliberately smaller than the full scientific regression
history. Preserve public signatures, statuses, serialized identifiers, support and
numerics when proposing changes. Keep optional dependencies optional.
Run the [delivery workflow](quickstart.md) for all-family coverage.
Report issues using the [bug template](guides/bug_report.md).
No license has been selected; agree reuse terms before contributing external work.

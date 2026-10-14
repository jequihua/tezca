# Troubleshooting

| Symptom | Action |
| --- | --- |
| Python or pip missing | Install/select Python >=3.11, create an environment and check `python --version`. See [installation](../installation.md). |
| Dependency unavailable | Check Python/architecture compatibility and network access, or provide complete compatible wheels for optional offline use. |
| Tezca cannot be imported | Install it into the same interpreter you run; use `python -m pip` and `python -m pip check`. |
| Unknown provenance | A finite value is not an observation. Supply explicit observation/replacement masks and provenance. |
| Missing band or spectrum | Inspect the recorded reason, supported frequency range, contiguous runs, cycles and resolution. Never replace missing power with zero. |
| Output exists | Preserve it and choose a new destination. |
| Reload fails | Inspect missing/changed files, schema and expected identity. Do not repair hashes to conceal changed results. |
| Uneven Unicode strings refused | Use the documented object-string encoding in [exports](../exports.md#typed-csv-and-json-conventions). |
| Optional backend missing | Install the relevant YAML/models extra or a compatible EcoFill environment; core routes remain usable. |

See [bug reports](bug_report.md), [result interpretation](understand_results.md)
and [qualification](../qualification.md). No implicit clock repair, completion
or model substitution is performed.

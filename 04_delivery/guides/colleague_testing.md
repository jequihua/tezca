# A short testing session

[Start here](../README.md) · [Bug report template](bug_report.md)

These tasks are prepared for a separately authorized test session. The current
candidate is local and unreleased; this guide does not grant permission to share
it or admit real data. Use synthetic data first. Record the source commit, manifest/file hashes and dirty flag from
`.front_repo_sync.json`, plus your built wheel checksum, commands and versions.
The earlier candidate index is unavailable in this source checkout. Keep each run in a new directory.

| Task | What to do | What to record |
| --- | --- | --- |
| Install and run | Follow the [quickstart](quickstart.md) from the source checkout root in a fresh environment. | Interpreter/platform, package versions, command exit and printed result summary; check power near 0.5 and PCA eigenvalues near 4 and 1. |
| Read a refusal | Find the unavailable long-period band and the all-null/unknown results. | Result ID, compute/metric/use statuses, reason and actual support; missing must remain distinct from zero. |
| Compare completions | Run the supplied-completion command into a second destination. | Completion identities, contribution burden, raw/residual basis and frozen-reference identity; note any unsupported comparison rather than selecting a winner. |
| Examine gaps | Inspect FI1 windows and the regime timeline. | Skipped windows, noise/excluded rows and episode boundaries; a gap must not silently become continuous support. |
| Restart and reload | Run the documented loader and independent standard-library reader after the producer exits. | Expected identity record, matching saved values, file counts/hashes and any exception. No analysis/refitting should be necessary. |
| Protect earlier work | Repeat an export to the same destination. | The refusal and evidence that the previous files are intact. Use a new destination afterward. |
| Try EcoFill integration | In the admitted Python 3.12/EcoFill environment, run the [producer example](../../08_pkg/docs/ecofill.md#installed-producer-and-fresh-reload); then reload in core Python 3.11. Keep both `<directory>` and `<directory>-results`. | Exact backend version, separately recorded expected bundle ID, observed/supplied means and positions, full masks/provenance, identical producer/reload JSON and unchanged outputs. |
| Try native models | In the admitted models environment, follow the [models example](../../08_pkg/docs/regimes.md#installed-models-example-and-result-reload). Keep the summary JSON and sibling `<summary-stem>-results`; run both `--validate` and `--reload-results` in fresh core processes. | Exact backend version, summary digest, bundle/source/reference identities, frozen scaling/training rows, native labels, noise/exclusions and gap-safe timelines. Summary-only validation does not check the numerical bundle. |

## Your own authorized inputs

After the steward has authorized the inputs and their local use, begin with a
small copy. Supply a numeric pandas frame, timezone-aware unique timestamps,
explicit units/representation, source identity and aligned observation or
contribution masks to `AnalysisSeries`. Declare a regular cadence/phase or an
irregular clock based on evidence. Keep artificially hidden truth out of analysis.
Start with [descriptions](../../08_pkg/docs/individual.md) and inspect support
before adding methods. Do not send the data to the package developer as part of
this exercise, and do not treat a successful import as domain qualification.

## What warrants a report?

Report a reproducible contradiction with the documented contract: changed saved
values after reload, lost timestamps/masks, an overwrite, silent gap filling,
unexpected optional dependency in core, wrong identity accepted, or a public
call/signature that differs from the cheat sheet. Include a confusing refusal
when its message does not identify the relevant input or next action. A clearly
documented unsupported method, missing optional package, unavailable band or
unknown provenance is expected; report it if the reason contradicts the input.

Use the [bug template](bug_report.md). Reduce the case to invented values and
portable relative paths. Never include credentials, private measurements,
personal identifiers, proprietary source dumps or resolved machine paths.
Keep private diagnostics locally and arrange any necessary evidence transfer
separately with the owner. The template is a local record, not an instruction
to post to a public tracker.

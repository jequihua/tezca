# Tezca issue report template

[Testing guide](colleague_testing.md) · [Troubleshooting](troubleshooting.md)

Copy this template into a local text file and replace the prompts. Use synthetic
values and relative paths. Redact usernames, absolute paths, secrets and private
measurements from exceptions before sharing through an authorized channel.

```text
Title: one concrete failure

Source commit (.front_repo_sync.json; base identity only):
Source dirty flag:
Projection manifest SHA-256 and relevant file SHA-256 values:
Wheel filename, raw SHA-256 and size:
Tezca version:
Python version / OS / architecture:
NumPy / pandas / SciPy versions:
Optional backend and version, if relevant:

Task I was trying to complete:
Command and working directory (relative to the candidate):
Minimal runnable reproducer with invented values:

Input declaration:
  units, representation, source alias:
  timestamps/timezone, cadence/phase or irregular mode:
  observed/contribution/qualification masks:
  supplied completion and lineage declarations, if any:

Settings:
  method import and explicit spec:
  observation/support policy:
  raw or residual basis:
  self or training-only reference; training positions/IDs:
  export/reload options; expected identities from my separate record:

Expected behavior and relevant documentation:
Actual value, compute_status, metric_status, use_status and reason:
First complete exception/trace (redacted):
Does it repeat in a fresh process with this candidate? Exact command/result:
Existing output preserved? New output name used for retry?:
Small synthetic result ID / manifest identity, if applicable:
```

Version and digest values identify the tested artifact; do not replace them with
“latest”. A matching hash does not prove a source's authenticity or data rights.
If no safe synthetic reproducer is available, describe the shape, settings and
failure without attaching private inputs.

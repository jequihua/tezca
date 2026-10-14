# Installation

Use Python >=3.11 and check `python --version`. Create an environment with
`python -m venv .venv` and activate it: `.venv\Scripts\Activate.ps1` in
PowerShell, or `source .venv/bin/activate` in a POSIX shell.
From the repository root, choose a route:

```text
python -m pip install .
python -m pip install -e .
python -m pip install ".[yaml]"
python -m pip install ".[models]"
python -m pip check
python -I -c "import tezca; print(tezca.__version__)"
```

The first installs normally; editable installation follows changes to `src/tezca`.
YAML adds PyYAML for policy loading; models adds scikit-learn for built-in estimators.
Core/custom-estimator routes need neither extra. NumPy, pandas and SciPy are
runtime requirements. The declared ranges do not establish every version/platform.

For a GitHub install, Git must be available (`git --version`):

```text
python -m pip install "tezca @ git+https://github.com/jequihua/tezca.git"
```

For repeatable work replace `FULL_PUBLIC_COMMIT` with the full commit you selected:

```text
python -m pip install "tezca @ git+https://github.com/jequihua/tezca.git@FULL_PUBLIC_COMMIT"
```

No PyPI release by name is claimed. A wheel or VCS install supplies importable
Tezca; checkout examples/docs/tests are not installed scripts. Obtain a checkout
to follow the [quickstart](quickstart.md). Builds use normal pip isolation.

## Optional offline supply

Supply compatible dependency and build wheels yourself in `WHEEL_DIRECTORY`:

```text
python -m pip install --no-index --find-links WHEEL_DIRECTORY .
```

The directory must include NumPy, pandas, SciPy and transitive requirements,
plus setuptools>=68, wheel and packaging for isolated builds. Offline supply is
optional. Keep isolation enabled. A missing wheel fails installation rather than
contacting an index. Inspect [tested environments](qualification.md).

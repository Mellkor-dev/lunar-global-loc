# Contributing

Create a focused branch from `main`, keep generated experiment data out of the
repository, and open a pull request with a clear description of the change.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q LargeScale_Implement/tests
```

Use small, descriptive commits. Pull requests should pass the test workflow
and receive review before merge. Do not commit credentials, local editor
configuration, virtual environments, generated DEMs, scans, plots, or result
files. Large required inputs belong in Git LFS and must be documented.

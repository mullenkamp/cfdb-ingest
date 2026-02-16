# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfdb-ingest is a Python package for converting file types (netCDF4/HDF5, GRIB2, etc.) from various organizations/model outputs (e.g., WRF, ERA5/ECMWF) into the cfdb format, standardizing variable names and attributes/metadata to CF conventions.

Requires Python >= 3.11. Uses UV for environment management and Hatchling as the build backend.

## Development Commands

```bash
# Setup environment
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest cfdb_ingest/tests/test_foo.py

# Run a single test
uv run pytest cfdb_ingest/tests/test_foo.py::test_name

# Linting/formatting
uv run lint

# Serve docs locally
hatch run docs-serve
```

## Code Style

- **Line length**: 120
- **Formatter**: Black (skip string normalization)
- **Linter**: Ruff with extensive rule set (see `[tool.ruff]` in pyproject.toml)
- **Relative imports banned** — use absolute imports only (`from cfdb_ingest.module import ...`)
- **Indent**: 4 spaces, UTF-8, LF line endings

## Architecture

- `cfdb_ingest/` — main package
- `cfdb_ingest/tests/` — test directory with conftest.py
- `conda/meta.yaml` — conda recipe for distribution
- `docs/` — MkDocs Material documentation (mkdocstrings for API reference, Google-style docstrings)

## CI/CD

Three GitHub Actions workflows in `.github/workflows/`:
- **build.yml** — lint + test (Python 3.10–3.12) + publish to PyPI on tag push
- **test.yml** — tests on `dev` branch pushes and PRs to `main`
- **documentation.yml** — builds and deploys MkDocs to GitHub Pages on push to `main`
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
```

## Code Style

- **Line length**: 120
- **Formatter**: Black (skip string normalization)
- **Linter**: Ruff with extensive rule set (see `[tool.ruff]` in pyproject.toml)
- **Relative imports banned** — use absolute imports only (`from cfdb_ingest.module import ...`)
- **Indent**: 4 spaces, UTF-8, LF line endings

## Architecture

- `cfdb_ingest/` — main package
  - `base.py` — `H5Ingest` abstract base class handling file management, time/space filtering, variable resolution, cfdb output writing, and rechunkit-optimized reads
  - `wrf.py` — `WrfIngest(H5Ingest)` for WRF wrfout files: CRS parsing, wind rotation, 3D level interpolation, variable transforms
  - `cli.py` — Typer CLI entry point
- `cfdb_ingest/tests/` — test directory with conftest.py (WRF test data fixtures pointing to `/home/mike/data/wrf/tests/`)
- `conda/meta.yaml` — conda recipe for distribution
- `docs/` — MkDocs Material documentation (mkdocstrings for API reference, Google-style docstrings)

## Adding New Variables

When adding a new cfdb variable (new `cfdb_name` in `WRF_VARIABLE_MAPPING`):

1. **cfdb_ingest side** — add the mapping entry in `WRF_VARIABLE_MAPPING` (`wrf.py`), any needed transform method, and dispatch case in `_read_variable`
2. **cfdb side** — the variable's dtype/encoding, name mapping, and CF attributes must be registered in **three dicts** in `/home/mike/git/cfdb/cfdb/utils.py`:
   - `default_dtype_params` — encoding (precision, offset, dtype_encoded, fillvalue)
   - `default_var_params` — cfdb display name
   - `default_attrs` — CF standard_name, long_name, units
3. **cfdb side** — add the variable name to the `@create_data_var_methods(var_names=(...))` decorator in `/home/mike/git/cfdb/cfdb/creation.py` so template creation methods are generated

If step 2–3 are skipped, things still work at runtime — `_create_cfdb_data_var` falls back to `creator.generic()` which creates a plain float32 variable without CF attributes or optimized encoding. Tests that only check shapes/values will pass, but the output won't have proper metadata or compression.

## Variable Mapping Pattern

`WRF_VARIABLE_MAPPING` maps WRF variable keys to cfdb variables. Each entry has:
- `cfdb_name` — the cfdb short name (must match a key in cfdb's `default_dtype_params`/`default_var_params`/`default_attrs` for full metadata support)
- `source_vars` — list of WRF HDF5 dataset names needed
- `transform` — name of a transform (dispatched in `_read_variable`), or `None` for direct reads
- `height` — float (meters above ground) for surface vars, or `'levels'` for 3D variables requiring vertical interpolation to user-specified `target_levels`

Variables sharing the same `cfdb_name` (e.g., `T2` and `T` both map to `air_temp`) are automatically merged into a single cfdb data variable spanning all their height levels.

## CI/CD

Three GitHub Actions workflows in `.github/workflows/`:
- **build.yml** — lint + test (Python 3.10–3.12) + publish to PyPI on tag push
- **test.yml** — tests on `dev` branch pushes and PRs to `main`
- **documentation.yml** — builds and deploys MkDocs to GitHub Pages on push to `main`

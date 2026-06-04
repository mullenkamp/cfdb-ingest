# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cfdb-ingest is a Python package for converting netCDF4/HDF5 files from various meteorological sources (WRF, ERA5) into the cfdb format, standardizing variable names and attributes/metadata to CF conventions.

Requires Python >= 3.10. Uses UV for environment management and Hatchling as the build backend.

## Development Commands

```bash
uv sync                                          # Setup environment
uv run pytest                                    # Run all tests
uv run pytest cfdb_ingest/tests/test_wrf.py      # Run WRF tests
uv run pytest cfdb_ingest/tests/test_era5.py     # Run ERA5 tests
uv run pytest cfdb_ingest/tests/test_era5.py::TestConvertSurface::test_2m_variable  # Single test
```

## Code Style

- **Line length**: 120
- **Formatter**: Black (skip string normalization)
- **Linter**: Ruff with extensive rule set (see `[tool.ruff]` in pyproject.toml)
- **Relative imports banned** -- use absolute imports only (`from cfdb_ingest.module import ...`)

## Architecture

- `cfdb_ingest/`
  - `base.py` -- `H5Ingest` abstract base class. Handles variable classification (surface/level/soil), named height coordinates, cfdb dataset creation, and three population strategies (rechunkit, per-timestep, batch). Subclass hooks: `file_glob_pattern`, `x_coord_name`/`y_coord_name`, `_init_source_metadata()`, `_create_spatial_coords()`.
  - `wrf.py` -- `WrfIngest(H5Ingest)` for WRF wrfout files. CRS parsing from MAP_PROJ, wind rotation via COSALPHA/SINALPHA, 3D eta-to-height/pressure interpolation, 53 variable mappings with ~30 transforms.
  - `era5.py` -- `Era5Ingest(H5Ingest)` for NCAR ERA5 NetCDF files. One-variable-per-file handling with `_var_file_map`/`_var_time_map`. EPSG:4326 lat/lon grid. 94 variable mappings. Z disambiguation (pressure-level vs invariant). Split/combined output modes.
  - `cli.py` -- Typer CLI with `wrf`, `era5`, and `cfdb-to-int` commands.
  - `cfdb_to_int.py` -- cfdb to WPS intermediate file conversion.
- `cfdb_ingest/tests/`
  - `test_wrf.py` -- 92 WRF conversion tests using subsetted real data in `tests/data/`
  - `test_era5.py` -- 43 ERA5 conversion tests using synthetic data in `tests/data/era5/`
  - `create_test_data.py` -- generates subsetted WRF test files from full wrfout via ncks
  - `create_era5_test_data.py` -- generates synthetic ERA5 test files via h5py

## Named Height Coordinates

Surface variables at specific heights use named coordinates (`height_0m`, `height_2m`, `height_10m`, `height_100m`) so they can coexist with pressure-level variables in a combined dataset:

- Variables at the same height share a coordinate (e.g. VAR_2T and VAR_2D both use `height_2m`)
- When a multi-level vertical coord exists (pressure/height), it gets `axis='Z'`; named height coords get `axis=None`
- When the same cfdb_name exists at both surface and pressure levels (e.g. `air_temperature`), the surface variant is suffixed: `air_temperature_2m`

## Variable Mapping Pattern

Both `WRF_VARIABLE_MAPPING` and `ERA5_VARIABLE_MAPPING` follow the same structure:

```python
'VAR_KEY': {
    'cfdb_name': 'air_temperature',  # cfdb-vars registered name
    'source_vars': ['VAR_2T'],       # source HDF5 dataset names needed
    'transform': None,               # transform method name, or None
    'height': 2.0,                   # float (m), 'levels', or 'soil'
}
```

- `height: <float>` -- surface variable at fixed height, gets `height_{int}m` coordinate
- `height: 'levels'` -- variable on vertical levels (pressure or height), shared multi-value coordinate
- `height: 'soil'` -- soil variable using depth coordinate

An entry may also declare optional `fallback_source_vars` (and an optional `fallback_transform`). When the primary `source_vars` are missing from a file but the fallback sources are present, `_init_variables()` in `base.py` promotes the entry to use the fallback sources/transform. This lets one cfdb output be served either by a native field (emitted directly by newer WRF builds) or computed from 3D fields on older `wrfout` files. Used by WRF `SLP`, `PWAT`, `PWAT_TR`, `VIMF_U`, `VIMF_V`.

Variable metadata (dtype, encoding, CF attributes) comes from the [cfdb-vars](https://github.com/mullenkamp/cfdb-vars) package. Template creation methods (`ds.create.data_var.<cfdb_name>()`) auto-apply the registered metadata. Unregistered names fall back to generic float32.

## Adding a New Source

To add support for a new data source (e.g. GFS, HRRR):

1. Create `cfdb_ingest/new_source.py` with a class inheriting from `H5Ingest`
2. Set class attributes: `file_glob_pattern`, `x_coord_name`/`y_coord_name` (if not x/y)
3. Override `_init_source_metadata()` if the file structure differs from "first file is representative"
4. Override `_init_time()` and `_init_variables()` if the time/variable structure differs
5. Implement abstract methods: `_parse_crs()`, `_parse_time()`, `_parse_spatial_coords()`, `_get_variable_mapping()`, `_read_variable()`
6. Override `_populate_*()` methods if the base class file iteration pattern doesn't fit
7. Add a CLI command in `cli.py`
8. Add any new variables to cfdb-vars
9. Export from `__init__.py`

## CI/CD

Three GitHub Actions workflows in `.github/workflows/`:
- **build.yml** -- lint + test (Python 3.10-3.12) + publish to PyPI on tag push
- **test.yml** -- tests on `dev` branch pushes and PRs to `main`
- **documentation.yml** -- builds and deploys MkDocs to GitHub Pages on push to `main`

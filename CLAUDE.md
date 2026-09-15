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
  - `ifs.py` -- `IfsIngest` for ECMWF IFS open-data forecast GRIB2 (one cycle per instance; standalone, no h5py machinery). Two-pass read (headers, then per (variable, level) decode/clip) into a `grid_forecast` dataset via `ForecastWriter`. 31 variable mappings; eccodes is the optional `ifs` extra (`eccodeslib` for CCSDS packing).
  - `forecast.py` -- the `grid_forecast` rules shared by every source: `(forecast_reference_time, forecast_period, level, y, x)` layout, explicit init step in minutes, lead `units`, `open_target` (path or open handle; remote-backed paths refused), `place_init` (new / backfill / overwrite), completion marker `attrs['complete_inits']`, `forecast_chunk_shape`, and `ForecastWriter` (one chunk-row buffered per (variable, level), written once).
  - `thermo.py` -- RH diagnostics (Thompson RSLF from q/t; Clausius-Clapeyron from T/Td). **Relative humidity is a 0-1 fraction everywhere in cfdb** (cfdb-vars >= 0.2.4: precision 0.001, units '1'); the WPS exporter multiplies by 100.
  - `cli.py` -- Typer CLI with `wrf` (incl. `--forecast`), `era5`, `ifs`, and `cfdb-to-int` (incl. `--init`) commands.
  - `cfdb_to_int.py` -- cfdb to WPS intermediate file conversion, for `grid` and `grid_forecast` (one init) datasets. Matches variables by canonical stored name (height suffix stripped), handles 4-D surface variables, refuses two candidates for one WPS field, reads chunk-aligned.
- `cfdb_ingest/tests/`
  - `test_wrf.py` -- 92 WRF conversion tests using subsetted real data in `tests/data/`
  - `test_era5.py` -- 43 ERA5 conversion tests using synthetic data in `tests/data/era5/`
  - `create_test_data.py` -- generates subsetted WRF test files from full wrfout via ncks
  - `create_era5_test_data.py` -- generates synthetic ERA5 test files via h5py
  - `create_ifs_test_data.py` -- generates a synthetic IFS cycle as real GRIB2 (eccodes) with every production quirk (dateline seam, CCSDS, `soilLayer` indices, `sithick` bitmap, 0 h-only orography, accumulated fields); closed-form values exported for assertions. Generated into a session temp dir by the `ifs_cycle_*` fixtures (deterministic, sub-second) -- nothing binary is committed.
  - `test_ifs.py`, `test_forecast.py`, `test_wrf_forecast.py`, `test_cfdb_to_int.py`, `test_base_helpers.py` -- forecast mode, the exporter (round-trips through `wps_int_reader.py`, a minimal WPS intermediate-format reader), and the shared helpers

## Naming rules that changed in 0.4.0 (release note)

- Surface variables that need a height suffix are stored under the **full** cfdb-vars name (`air_temperature_2m`, not `air_temp_2m`) and carry the template's attrs/dtype; a name that appears at more than one height gets a suffix at every height (`u_wind_10m` / `u_wind_100m` -- previously the last height silently won).
- `relative_humidity` is a **fraction** at 0.001 resolution in every source (WRF already stored a fraction, but the precision-1 template quantised it to 0.1; ERA5 `R` is now divided by 100; cfdb-vars 0.2.4 carries the precision-3 template). The exporter writes percent. cfdb-vars 0.2.4 also lifts the −0.9 m floor on `terrain_height`/`geopotential_height` and adds `wind_gust`.
- ERA5 variables now receive the cfdb-vars templates (packed dtype + CF attrs); before 0.4.0 the full-name mapping entries never matched the short-name templates and every ERA5 variable was generic float32 with no attrs.
- `cfdb-to-int` previously exported no surface field at all (4-D surface variables were skipped) and no 3-D `TT` (short-name tables never matched stored names), and put `SOILHGT` at level 1.0 instead of 200100.

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

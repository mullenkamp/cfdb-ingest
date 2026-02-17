# cfdb-ingest

## Project Overview

`cfdb-ingest` is a Python package designed to convert various meteorological and oceanographic file formats (specifically WRF output, ERA5, etc.) into the `cfdb` format. It standardizes variable names and metadata to adhere to Climate and Forecast (CF) conventions, facilitating efficient data storage and retrieval.

## Architecture

*   **Core Logic:** Located in `cfdb_ingest/`.
    *   `base.py`: Contains `H5Ingest`, an abstract base class for file management, spatial/temporal filtering, and `cfdb` writing.
    *   `wrf.py`: Implements `WrfIngest` (subclass of `H5Ingest`) for processing WRF output files (handling CRS, wind rotation, interpolation).
    *   `cli.py`: The command-line interface entry point using `typer`.
*   **Dependencies:** `cfdb`, `h5py`, `numpy`, `pyproj`, `geointerp`, `rechunkit`, `typer`.
*   **Build System:** `hatchling` (backend), managed by `uv`.

## Development

### Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync
```

### Testing

Tests are located in `cfdb_ingest/tests/`.

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=cfdb_ingest
```

**Note:** Tests may require specific data fixtures located at `/home/mike/data/wrf/tests/` (referenced in `conftest.py`).

### Linting & Formatting

The project uses `ruff` for linting and `black` for formatting.

```bash
# Run linting and style checks
uv run lint
```

### Documentation

Documentation is built with `mkdocs` and `mkdocstrings`.

```bash
# Serve docs locally
uv run mkdocs serve

# Build docs
uv run mkdocs build
```

## CLI Usage

The package provides a CLI tool `cfdb-ingest`.

```bash
# General help
uv run cfdb-ingest --help

# Convert WRF files
uv run cfdb-ingest wrf --help
uv run cfdb-ingest wrf input_file.nc output.cfdb --variables T2,U10 --start-date 2023-01-01
```

## Key Workflows

### Adding New Variables

To add a new variable mapping (e.g., for WRF):

1.  **In `cfdb-ingest`:** Update `WRF_VARIABLE_MAPPING` in `cfdb_ingest/wrf.py` with the `cfdb_name`, source variables, and any necessary transforms.
2.  **In `cfdb` (separate repo):**
    *   Register the variable in `cfdb/utils.py` (dictionaries: `default_dtype_params`, `default_var_params`, `default_attrs`).
    *   Add the variable name to `@create_data_var_methods` in `cfdb/creation.py`.

Refer to `CLAUDE.md` for more detailed architectural guidelines.

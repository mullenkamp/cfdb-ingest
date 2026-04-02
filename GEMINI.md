# cfdb-ingest

## Project Overview

`cfdb-ingest` is a Python package designed to convert various meteorological and oceanographic file formats (specifically WRF output, ERA5, etc.) into the `cfdb` format. It standardizes variable names and metadata to adhere to Climate and Forecast (CF) conventions, facilitating efficient data storage and retrieval.

## Architecture

*   **Core Logic:** Located in `cfdb_ingest/`.
    *   `base.py`: Contains `H5Ingest`, an abstract base class for file management, spatial/temporal filtering, and `cfdb` writing. Includes `_multi_rechunker` for synchronized multi-variable processing.
    *   `wrf.py`: Implements `WrfIngest` (subclass of `H5Ingest`) for processing WRF output files (handling CRS, wind rotation, interpolation).
    *   `era5.py`: Implements `Era5Ingest` for processing ERA5 products. Supports native VIMF calculation and synchronized rechunking across multiple source files.
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

## Implementation Learnings & Constraints

*   **Synchronized Rechunking:** When computing derived variables from multiple source files (like ERA5 VIMF), use the `_multi_rechunker` helper. It leverages `rechunkit`'s deterministic yielding to process data in large, synchronized temporal blocks, avoiding redundant reads of shared variables (e.g., reading `Q` once for both `VIMF_U` and `VIMF_V`).
*   **HDF5 Chunk Caching:** For per-timestep transformation loops, inject `max_mem` into the `h5py.File(..., rdcc_nbytes=max_mem)` constructor. This enables the C-level HDF5 chunk cache, preventing severe performance degradation from chunk thrashing when multiple variables share the same underlying disk chunks.
*   **cfdb Indexing:** `cfdb` data variables use unit-length slices for integer indexing. When writing 2D data to a specific time/level, always ensure the data is reshaped to 4D `(1, 1, ny, nx)` to match the expected unit-slice shape: `dv[(t, z, slice(None), slice(None))] = data[np.newaxis, np.newaxis, ...]`.
*   **ERA5 Disambiguation:** ERA5 products often use the same internal variable names (e.g., `'Z'` for both geopotential and terrain height). Always use the mapping keys (`'Z_PL'`, `'Z_INV'`) to lookup file entries and times, rather than raw source names, to avoid file-lookup collisions in combined datasets.

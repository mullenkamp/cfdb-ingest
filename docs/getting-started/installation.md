# Installation

Requires Python >= 3.10.

```bash
pip install cfdb-ingest
# or
uv add cfdb-ingest
```

This installs two CLI commands:

- `cfdb-ingest` -- the main CLI with `wrf`, `era5`, `ifs`, and `cfdb-to-int` subcommands
- `cfdb-to-int` -- standalone command for converting cfdb datasets to WPS intermediate files

The IFS (GRIB2) source needs the `ifs` extra, which brings in `eccodes` and the bundled `eccodeslib` library (the open-data files use CCSDS packing, which system builds of libeccodes may lack):

```bash
pip install 'cfdb-ingest[ifs]'
```

# Development

## Setup environment

We use [UV](https://docs.astral.sh/uv/) to manage the development environment and production build.

```bash
uv sync
```

## Run tests

```bash
uv run pytest
```

## Lint and format

```bash
uv run ruff check .
uv run black --check --diff .
```

To auto-format:

```bash
uv run black .
uv run ruff check --fix .
```

## Build docs locally

```bash
uv run mkdocs serve
```

## Keeping variable docs in sync

The variable mapping dicts (`WRF_VARIABLE_MAPPING`, `ERA5_VARIABLE_MAPPING`) are the source of
truth for supported variables. A helper checks that every mapped variable is documented in the
reference pages with a matching cfdb name:

```bash
uv run python scripts/check_var_docs.py          # report drift (exit 1 if stale)
uv run python scripts/check_var_docs.py --rows   # also print paste-ready rows for new variables
```

This runs automatically in CI via `cfdb_ingest/tests/test_docs_sync.py`, so adding a variable to a
mapping without documenting it will fail the test suite. After adding a variable, run with `--rows`
to get a starting table row, paste it into the matching `docs/reference/*-variables.md` page, and
fill in the source/transform/description columns.

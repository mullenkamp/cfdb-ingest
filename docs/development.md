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

#!/usr/bin/env python3
"""Keep the variable reference docs in sync with the mapping dicts.

``WRF_VARIABLE_MAPPING`` and ``ERA5_VARIABLE_MAPPING`` are the single source of
truth for which variables each source supports. This helper renders reference
table rows from those dicts and checks that every mapped variable is documented
(with a matching cfdb name) in the corresponding ``docs/reference`` page.

Run directly for a report::

    uv run python scripts/check_var_docs.py            # exit non-zero if docs are stale
    uv run python scripts/check_var_docs.py --rows     # also print paste-ready rows for gaps

A pytest test (``cfdb_ingest/tests/test_docs_sync.py``) calls :func:`check`, so CI
fails automatically when a variable is added to a mapping but not to the docs.

Note: this checks variable *coverage* and the *cfdb name* of each row. The free-text
``Source Vars`` / ``Transform`` / ``Description`` columns are left for humans to
curate; use ``--rows`` to get a starting skeleton for any newly added variable.
"""
from __future__ import annotations

import argparse
import pathlib
import re

from cfdb_ingest.era5 import ERA5_VARIABLE_MAPPING
from cfdb_ingest.wrf import WRF_VARIABLE_MAPPING

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / 'docs' / 'reference'

SOURCES = {
    'WRF': (WRF_VARIABLE_MAPPING, DOCS / 'wrf-variables.md'),
    'ERA5': (ERA5_VARIABLE_MAPPING, DOCS / 'era5-variables.md'),
}

# First two cells of a table row: | `KEY` | `cfdb_name` | ...
# Keys are upper-case; a trailing range like `STL1-4` expands to STL1..STL4.
_ROW = re.compile(r'^\|\s*`([A-Z][A-Z0-9_]*(?:-\d+)?)`\s*\|\s*`([a-z0-9_]+)`', re.M)
_RANGE = re.compile(r'^([A-Z]+?)(\d+)-(\d+)$')


def _expand(key: str) -> list[str]:
    """Expand a collapsed range key (``STL1-4`` -> STL1..STL4)."""
    m = _RANGE.match(key)
    if not m:
        return [key]
    base, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    return [f'{base}{i}' for i in range(lo, hi + 1)]


def documented(md_path: pathlib.Path) -> dict[str, str]:
    """Return ``{key: cfdb_name}`` parsed from a reference page's table rows."""
    out: dict[str, str] = {}
    for key, cfdb_name in _ROW.findall(md_path.read_text()):
        for k in _expand(key):
            out[k] = cfdb_name
    return out


def _humanize_transform(name: str | None) -> str:
    if name is None:
        return 'computed'
    return name.replace('_3d', ' (3D)').replace('_2d', ' (2D)').replace('_', ' ')


def _transform_label(info: dict) -> str:
    if 'fallback_source_vars' in info:
        return f"native; fallback = {_humanize_transform(info.get('fallback_transform'))}"
    return 'direct' if info.get('transform') is None else _humanize_transform(info['transform'])


def render_row(key: str, info: dict) -> str:
    """Render a paste-ready markdown table row for one mapping entry.

    Surface/soil-by-fixed-height entries get a ``Height`` column; ``levels`` and
    ``soil`` entries use the 4-column (no height) shape used by the 3D tables.
    """
    cfdb_name = info['cfdb_name']
    srcs = ', '.join(info['source_vars'])
    if 'fallback_source_vars' in info:
        srcs += f" _(fallback: {', '.join(info['fallback_source_vars'])})_"
    height = info['height']
    transform = _transform_label(info)
    if isinstance(height, (int, float)):
        label = f'{int(height)} m' if float(height).is_integer() else f'{height} m'
        return f'| `{key}` | `{cfdb_name}` | {label} | {srcs} | {transform} |'
    return f'| `{key}` | `{cfdb_name}` | {srcs} | {transform} |'


def check() -> list[str]:
    """Return a list of human-readable drift problems (empty == in sync)."""
    problems: list[str] = []
    for name, (mapping, md_path) in SOURCES.items():
        if not md_path.exists():
            problems.append(f'{name}: reference page missing: {md_path}')
            continue
        docs = documented(md_path)
        for key, info in mapping.items():
            if key not in docs:
                problems.append(f'{name}: `{key}` is in the mapping but not documented in {md_path.name}')
            elif docs[key] != info['cfdb_name']:
                problems.append(
                    f"{name}: `{key}` documents cfdb name `{docs[key]}` but mapping says `{info['cfdb_name']}`"
                )
        for key in docs:
            if key not in mapping:
                problems.append(f'{name}: `{key}` is documented in {md_path.name} but not in the mapping')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', action='store_true', help='print paste-ready rows for undocumented keys')
    args = parser.parse_args(argv)

    problems = check()
    if not problems:
        print('Variable reference docs are in sync with the mapping dicts.')
        return 0

    print('Variable reference docs are OUT OF SYNC:\n')
    for p in problems:
        print(f'  - {p}')

    if args.rows:
        print('\nPaste-ready rows for undocumented keys (adapt source/transform/description as needed):')
        for name, (mapping, md_path) in SOURCES.items():
            docs = documented(md_path) if md_path.exists() else {}
            missing = [k for k in mapping if k not in docs]
            if missing:
                print(f'\n# {name} ({md_path.name})')
                for k in missing:
                    print(render_row(k, mapping[k]))
    else:
        print('\nRe-run with --rows to print paste-ready table rows for the missing keys.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Guard: every mapped variable must be documented in the reference pages.

Loads the standalone ``scripts/check_var_docs.py`` helper so the drift logic lives
in one place, and fails if any variable mapping has drifted from ``docs/reference``.
"""
import importlib.util
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_HELPER = REPO_ROOT / 'scripts' / 'check_var_docs.py'

_spec = importlib.util.spec_from_file_location('check_var_docs', _HELPER)
check_var_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_var_docs)


def test_variable_docs_in_sync():
    problems = check_var_docs.check()
    assert not problems, 'Variable reference docs are out of sync:\n' + '\n'.join(problems)

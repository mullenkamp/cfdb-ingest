"""
cfdb_ingest.forecast_archive: the shared archive protocol (lifted from ifs-download). Everything here
runs on local datasets or a fake push handle; the S3 behaviour (attach, push, lock tickets) is covered
by ifs-download's integration test against the real bucket.
"""

import signal
import subprocess
import sys
import textwrap

import cfdb
import numpy as np
import pytest
from cfdb import dtypes

from cfdb_ingest import forecast as fc

pytest.importorskip('ebooklet')  # the `archive` extra; must precede the module import below
fa = pytest.importorskip('cfdb_ingest.forecast_archive')

A, B, C = '2026-09-13T00:00', '2026-09-13T12:00', '2026-09-14T12:00'
LEADS = np.array([0, 3, 6], dtype='int32')


def _write_init(ds, init, value):
    idx = fc.place_init(ds, init, step_minutes=360)['index'] if len(ds[fc.FRT].data) else 0
    ds['air_temperature_2m'][(idx, slice(0, 3), 0, slice(None), slice(None))] = np.full((3, 3, 4), value, 'float32')
    fc.mark_init_complete(ds, init)
    return idx


@pytest.fixture
def three_init_file(tmp_path):
    """Inits A, B (12 h later) and C (36 h after A): axis A, B, 3 empty slots, C."""
    p = tmp_path / 'three.cfdb'
    with cfdb.open_dataset(str(p), 'n', dataset_type='grid_forecast') as ds:
        fc.create_forecast_coords(ds, np.datetime64(A, 'm'), LEADS, step_minutes=360)
        ds.create.coord.generic('longitude', data=np.arange(4.0), axis='x', step=True)
        ds.create.coord.generic('latitude', data=np.arange(3.0), axis='y', step=True)
        ds.create.coord.generic('height_2m', data=np.array([2.0]), axis=None)
        ds.create.data_var.generic('air_temperature_2m', (fc.FRT, fc.LEAD, 'height_2m', 'latitude', 'longitude'),
                                   dtype=dtypes.dtype('float32'), chunk_shape=(1, 3, 1, 3, 4))
        ds['air_temperature_2m'][(0, slice(0, 3), 0, slice(None), slice(None))] = np.full((3, 3, 4), 1.0, 'float32')
        fc.mark_init_complete(ds, np.datetime64(A, 'm'))
        _write_init(ds, np.datetime64(B, 'm'), 2.0)
        _write_init(ds, np.datetime64(C, 'm'), 3.0)
    return p


# ---------------------------------------------------------------- opening


def test_open_target_local_and_refuse_create(tmp_path, three_init_file):
    with fa.open_target(None, tmp_path / 'absent.cfdb', 'r') as ds:
        assert ds is None
    with fa.open_target(None, three_init_file, 'r') as ds:
        assert fa.done_inits(ds) == [A, B, C]
        assert fa.refuse_create(ds, allow_create=False) is False
    with fa.open_target(None, tmp_path / 'new.cfdb', 'c') as ds:
        with pytest.raises(fa.WouldCreate):
            fa.refuse_create(ds, allow_create=False)
        assert fa.refuse_create(ds, allow_create=True) is True
    assert fa.done_inits(None) == []


def test_open_target_url_requires_read_flag(tmp_path):
    with pytest.raises(ValueError, match='flag must be r'):
        with fa.open_target('https://example.invalid/x.cfdb', tmp_path / 'c.cfdb', 'c'):
            pass


# ---------------------------------------------------------------- retention (pure)


def _inits(*specs):
    return [np.datetime64(s, 'm') for s in specs]


def test_inits_to_drop_boundary_is_inclusive():
    inits = _inits('2026-08-01T00', '2026-08-15T00', '2026-08-31T00', '2026-09-14T00')
    cutoff, dropped = fa.inits_to_drop(inits, keep_days=30, min_keep=1)
    assert cutoff == np.datetime64('2026-08-15T00', 'm') and dropped == _inits('2026-08-01T00')
    cutoff, dropped = fa.inits_to_drop(inits, keep_days=29, min_keep=1)
    assert cutoff == np.datetime64('2026-08-31T00', 'm') and dropped == _inits('2026-08-01T00', '2026-08-15T00')


def test_inits_to_drop_aligns_to_a_complete_init_and_min_keep():
    inits = _inits('2026-09-13T00', '2026-09-14T00', '2026-09-14T12')
    cutoff, dropped = fa.inits_to_drop(inits, keep_days=1, min_keep=1)
    assert cutoff == np.datetime64('2026-09-14T00', 'm') and dropped == _inits('2026-09-13T00')
    cutoff, dropped = fa.inits_to_drop(inits, keep_days=1, min_keep=2)
    assert dropped == _inits('2026-09-13T00')
    assert fa.inits_to_drop(inits, keep_days=1, min_keep=3) == (None, [])
    assert fa.inits_to_drop(['2026-09-14T12', '2026-09-13T00', '2026-09-13T00'], keep_days=1, min_keep=1)[1] == _inits(
        '2026-09-13T00'
    )


def test_inits_to_drop_protect_and_guards():
    inits = _inits('2026-08-01T00', '2026-08-20T00', '2026-09-01T00', '2026-09-14T00')
    cutoff, dropped = fa.inits_to_drop(inits, keep_days=7, protect=['2026-08-20T00'], min_keep=1)
    assert cutoff == np.datetime64('2026-08-20T00', 'm') and dropped == _inits('2026-08-01T00')
    assert fa.inits_to_drop(inits, keep_days=0) == (None, [])
    assert fa.inits_to_drop([], keep_days=5) == (None, [])
    with pytest.raises(ValueError):
        fa.inits_to_drop(inits, keep_days=1, min_keep=0)


# ---------------------------------------------------------------- retention (applied)


def test_apply_retention_truncates_and_prunes_marker(three_init_file):
    with fa.open_target(None, three_init_file, 'w') as ds:
        cutoff, dropped = fa.apply_retention(ds, keep_days=1, min_keep=1, dry_run=True)
        assert (str(cutoff), dropped) == (B, [A])
        assert fa.done_inits(ds) == [A, B, C]
        cutoff, dropped = fa.apply_retention(ds, keep_days=1, min_keep=1, label='wrf-fc retention')
        assert (str(cutoff), dropped) == (B, [A])
        assert fa.done_inits(ds) == [B, C]
        axis = ds[fc.FRT].data
        assert str(axis[0]) == B and len(axis) == 5
        assert 'wrf-fc retention' in ds.attrs.data['history']
    with fa.open_target(None, three_init_file, 'r') as ds:
        assert float(np.squeeze(ds['air_temperature_2m'][0, 0, 0, 0, 0].data)) == 2.0
        assert fc.missing_chunks(ds, np.datetime64(B, 'm')) == []


def test_apply_retention_drops_an_unmarked_init_positionally(three_init_file):
    """An init that never finished (unmarked) older than the cutoff goes with the marked ones."""
    with fa.open_target(None, three_init_file, 'w') as ds:
        fc.unmark_init_complete(ds, np.datetime64(B, 'm'))
        assert fa.done_inits(ds) == [A, C]
        cutoff, dropped = fa.apply_retention(ds, keep_days=1, min_keep=1)  # horizon = B: A is older
        assert (str(cutoff), dropped) == (C, [A])  # computed on the complete inits only ...
        assert str(ds[fc.FRT].data[0]) == C  # ... but the positional cut removed the unmarked B too


# ---------------------------------------------------------------- push protocol on a fake handle


class _Result:
    def __init__(self, failures):
        self.failures = failures


class _FakeDS:
    """Records the order of pushes and the marker at each push -- the two-commit invariant."""

    def __init__(self, fail_first=0):
        self.attrs = _Attrs()
        self.attrs['complete_inits'] = []
        self.pushes = []
        self._fail = fail_first

    def push(self, force_push=False):
        self.pushes.append((list(self.attrs.data.get('complete_inits', [])), force_push))
        if self._fail > 0:
            self._fail -= 1
            return _Result({'k1': 'boom'})
        return _Result({})


class _Attrs:
    def __init__(self):
        self.data = {}

    def __setitem__(self, k, v):
        self.data[k] = v


def test_push_and_mark_is_two_commits_and_never_advertises_early():
    ds = _FakeDS()
    ds.attrs['complete_inits'] = [A]
    fa.push_and_mark(ds, [B])
    assert ds.pushes == [([A], False), ([A, B], False)]  # chunks first with B unmarked, then the marks


def test_push_and_mark_retries_once_then_raises():
    ds = _FakeDS(fail_first=1)
    fa.push_and_mark(ds, [B])
    assert [f for _, f in ds.pushes] == [False, True, False]
    ds = _FakeDS(fail_first=2)
    with pytest.raises(fa.PushFailed):
        fa.push_and_mark(ds, [B])
    assert ds.attrs.data['complete_inits'] == []  # B stays unmarked after the failed chunk push


# ---------------------------------------------------------------- process guards


def test_sidecar_round_trip(tmp_path):
    assert fa.read_sidecar(tmp_path) is None
    fa.write_sidecar(tmp_path, 'push', [B], attempt=2)
    assert fa.read_sidecar(tmp_path) == {'stage': 'push', 'inits': [B], 'attempt': 2}
    fa.clear_sidecar(tmp_path)
    assert fa.read_sidecar(tmp_path) is None


def test_run_lock_is_exclusive(tmp_path):
    script = textwrap.dedent(f'''
        import time, sys
        from cfdb_ingest import forecast_archive as fa
        with fa.run_lock({str(tmp_path)!r}):
            print('held', flush=True); time.sleep(3)
    ''')
    child = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'held'
        with pytest.raises(RuntimeError, match='holds'):
            with fa.run_lock(tmp_path, holder='the other job'):
                pass
    finally:
        child.kill()
        child.wait()
    with fa.run_lock(tmp_path):
        pass


def test_sigterm_handler_turns_into_system_exit(tmp_path):
    script = textwrap.dedent('''
        import os, signal, sys
        from cfdb_ingest import forecast_archive as fa
        fa.install_sigterm_handler()
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except SystemExit as e:
            print('exit', e.code, flush=True)
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True).stdout
    assert out.strip() == f'exit {128 + signal.SIGTERM}'


def test_remove_local_deletes_the_sidecar_index(tmp_path):
    p = tmp_path / 'x.cfdb'
    p.write_bytes(b'')
    (tmp_path / 'x.cfdb.remote_index').write_bytes(b'')
    fa.remove_local(p)
    assert not p.exists() and not (tmp_path / 'x.cfdb.remote_index').exists()
    fa.remove_local(p)  # idempotent

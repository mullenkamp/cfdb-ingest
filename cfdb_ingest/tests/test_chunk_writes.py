"""
Every output chunk is written exactly once per convert() call (cfdb-ingest 0.6.0).

cfdb's store is log-structured: a chunk written twice leaves a dead copy on disk and, for an S3-backed
dataset, a second upload. Before 0.6.0 the cross-file rechunk blocks were aligned to the first RAW input
frame, so dropping a leading frame (``start_date``) wrote the first output chunk once per timestep and
every later chunk twice. Counted here at the booklet ``set`` call that stores a chunk.
"""
import collections

import booklet
import cfdb
import numpy as np
import pytest

import cfdb_ingest.base as base
from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.era5 import Era5Ingest
from cfdb_ingest.tests.test_era5 import PL_DIR
from cfdb_ingest.tests.conftest import WRF_FILE_1, WRF_FILE_2
from cfdb_ingest.wrf import WrfIngest


@pytest.fixture
def chunk_writes(monkeypatch):
    """Counter of booklet set() calls per chunk key ('<var>!<t>.<...>')."""
    counts = collections.Counter()
    orig = booklet.main.VariableLengthValue.set

    def spy(self, key, value, *a, **k):
        if isinstance(key, str) and '!' in key:
            counts[key] += 1
        return orig(self, key, value, *a, **k)

    monkeypatch.setattr(booklet.main.VariableLengthValue, 'set', spy)
    return counts


def _data_keys(counts, var):
    return {k: n for k, n in counts.items() if k.startswith(var + '!')}


@pytest.mark.parametrize('variable,cfdb_name', [('T2', 'air_temperature'), ('RAIN', 'precipitation')])
def test_each_chunk_written_once_dropped_first_frame(tmp_path, chunk_writes, variable, cfdb_name):
    """Single-source (T2) and accumulation (RAIN) paths on the real d01 fixtures.
    Catches: rechunk blocks aligned to the raw input instead of the output chunk grid."""
    files = [WRF_FILE_1, WRF_FILE_2]
    w = WrfIngest(files)
    out = tmp_path / 'drop.cfdb'
    w.convert(out, variables=[variable], chunk_shape=(12, 1, len(w.y), len(w.x)), start_date=str(w.times[1]))
    keys = _data_keys(chunk_writes, cfdb_name)
    assert keys and all(n == 1 for n in keys.values()), keys

    # and the values are those of the full conversion, shifted by one frame
    WrfIngest(files).convert(tmp_path / 'full.cfdb', variables=[variable], chunk_shape=(12, 1, len(w.y), len(w.x)))
    with cfdb.open_dataset(tmp_path / 'full.cfdb') as a, cfdb.open_dataset(out) as b:
        np.testing.assert_array_equal(a[cfdb_name].data[1:], b[cfdb_name].data)


def test_each_chunk_written_once_multi_source(tmp_path, chunk_writes):
    """PREC_ACC (two sources, the multi-rechunker path)."""
    files = syn.write_run(tmp_path / 'run', '2026-01-01T00', 71, 6, 5, frames_per_file=24,
                          end_frame_file=False)
    w = WrfIngest(files)
    out = tmp_path / 'p.cfdb'
    w.convert(out, variables=['PREC_ACC'], chunk_shape=(12, 1, 6, 5), start_date=str(w.times[5]))
    keys = _data_keys(chunk_writes, 'precipitation')
    assert keys and all(n == 1 for n in keys.values()), keys


@pytest.mark.parametrize('first,second', [
    ((0, 29), (30, 71)),    # append whose first output lands part-way into a chunk (pad = 6)
    ((40, 71), (0, 39)),    # prepend: negative origin, chunk grid anchored at the first build
    ((48, 71), (0, 23)),    # prepend across a placeholder gap
])
def test_extend_writes_each_chunk_once(tmp_path, chunk_writes, first, second):
    """Grid extend: every call writes each touched output chunk exactly once.
    Catches: a pad/anchor that ignores the target's origin (values stay right, writes double)."""
    init = np.datetime64('2026-03-01T00', 'm')
    h = np.timedelta64(1, 'h')
    files = syn.write_run(tmp_path / 'run', str(init), 72, 6, 5, frames_per_file=24)
    out = tmp_path / 'x.cfdb'
    kw = dict(variables=['PREC_ACC'], extend=True, time_label='start', squeeze_height=True, chunk_shape=(12, 3, 3))
    WrfIngest(files).convert(out, start_date=str(init + first[0] * h), end_date=str(init + first[1] * h), **kw)
    chunk_writes.clear()
    WrfIngest(files).convert(out, start_date=str(init + second[0] * h), end_date=str(init + second[1] * h), **kw)
    keys = _data_keys(chunk_writes, 'precipitation')
    assert keys and all(n == 1 for n in keys.values()), keys


def test_extend_with_stored_chunk_shape_writes_once(tmp_path, chunk_writes):
    """chunk_shape=None on an existing target: the stored chunk decides the write alignment.
    Catches: aligning to the default (1, ny, nx) instead (every chunk written once per timestep)."""
    init = np.datetime64('2026-03-01T00', 'm')
    h = np.timedelta64(1, 'h')
    files = syn.write_run(tmp_path / 'run', str(init), 72, 6, 5, frames_per_file=24)
    out = tmp_path / 'n.cfdb'
    kw = dict(variables=['PREC_ACC'], extend=True, time_label='start', squeeze_height=True)
    WrfIngest(files).convert(out, start_date=str(init), end_date=str(init + 23 * h), chunk_shape=(24, 3, 3), **kw)
    chunk_writes.clear()
    WrfIngest(files).convert(out, start_date=str(init + 24 * h), end_date=str(init + 71 * h), **kw)
    keys = _data_keys(chunk_writes, 'precipitation')
    assert keys and all(n == 1 for n in keys.values()), keys


def test_time_chunked_source_read_whole_chunks(monkeypatch, tmp_path):
    """A source chunked 12 frames deep in time (the ERA5 fixtures) is read 12 frames per call, as 0.5.1 did.
    Catches: the time-mapped source declaring a time chunk of 1 (one decompression per frame)."""
    sizes = collections.Counter()
    orig = base._ConcatTimeSource.__call__

    def spy(self, slices):
        sizes[slices[0].stop - slices[0].start] += 1
        return orig(self, slices)

    monkeypatch.setattr(base._ConcatTimeSource, '__call__', spy)
    Era5Ingest(PL_DIR).convert(tmp_path / 'e.cfdb', variables=['T'], vertical_coord='pressure',
                               target_levels=[100000.0, 85000.0, 50000.0], chunk_shape=(24, 3, 10, 12))
    assert set(sizes) == {12}, dict(sizes)

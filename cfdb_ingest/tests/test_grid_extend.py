"""
Grid ``extend`` mode, interval-start labels and ``squeeze_height`` (cfdb-ingest 0.6.0).

A long record is built in time bands over many convert() calls. The reference for every value is the
closed form of ``wrf_synthetic`` or a one-shot build of the same frames, never the code under test.
"""
import cfdb
import h5py
import numpy as np
import pytest

import cfdb_ingest.base as base
from cfdb_ingest import grid, wrf_synthetic as syn
from cfdb_ingest.wrf import WrfIngest

NY, NX = 6, 5
KW = dict(variables=['PREC_ACC'], extend=True, time_label='start', squeeze_height=True)
INIT = np.datetime64('2026-03-01T00', 'm')
H = np.timedelta64(1, 'h')


def _run(tmp_path, days=4, name='run'):
    """One run from INIT, daily files, frames at leads 0 .. 24*days."""
    return syn.write_run(tmp_path / name, str(INIT), 24 * days, NY, NX, frames_per_file=24)


def _expected(label):
    """PREC_ACC over [label, label + 1h) of the run started at INIT: the frame at label + 1h."""
    lead = int((np.datetime64(label, 'm') + H - INIT) / H)
    return syn.expected('PREC_ACC_NC', lead, NY, NX) + syn.expected('PREC_ACC_C', lead, NY, NX)


def _read(path):
    with cfdb.open_dataset(path) as ds:
        return ds['time'].data.astype('datetime64[m]'), ds['precipitation'].data


def _build(files, path, windows, chunk_t=24):
    for s, e in windows:
        WrfIngest(files).convert(path, start_date=str(s), end_date=str(e), chunk_shape=(chunk_t, 3, 3), **KW)
    return _read(path)


# ---------------------------------------------------------------- interval-start labels

def test_labels_are_interval_starts(tmp_path):
    """label = frame - PREC_ACC_DT, value = the frame's accumulation; the lead-0 frame is dropped.
    Catches: an off-by-one shift, or keeping WRF's zero-initialised lead-0 frame as the hour before init."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'l.cfdb'
    WrfIngest(files).convert(out, variables=['PREC_ACC'], time_label='start', squeeze_height=True)
    times, v = _read(out)
    np.testing.assert_array_equal(times, INIT + np.arange(24) * H)  # frames 01..24 -> labels 00..23
    assert times[0] == INIT  # nothing labelled before the run began
    for k in (0, 5, 23):
        np.testing.assert_allclose(v[k], _expected(times[k]), atol=0.006)
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].attrs['time_label'] == 'interval_start'
        assert ds['time'].attrs['interval_minutes'] == 60
        assert ds['precipitation'].attrs['cell_methods'] == 'time: sum (interval: 60 minutes)'
        assert ds['precipitation'].coord_names == ('time', 'y', 'x')


def test_start_end_select_labels(tmp_path):
    """start_date/end_date select LABELS: a band [b0, b1) is exactly b0 .. b1-1h (needs frame b1)."""
    files = _run(tmp_path, days=2)
    out = tmp_path / 's.cfdb'
    b0 = INIT + 5 * H
    WrfIngest(files).convert(out, start_date=str(b0), end_date=str(b0 + 23 * H), **KW)
    times, _ = _read(out)
    np.testing.assert_array_equal(times, b0 + np.arange(24) * H)


def test_default_labels_unchanged(tmp_path):
    """time_label='end' (default) keeps WRF's frame times and writes no label attrs."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'e.cfdb'
    WrfIngest(files).convert(out, variables=['PREC_ACC'])
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].data.astype('datetime64[m]')[0] == INIT
        assert 'time_label' not in ds['time'].attrs.data
        assert ds['precipitation'].coord_names == ('time', 'height_0m', 'y', 'x')


def test_lead0_dropped_per_file_across_runs(tmp_path):
    """A band spanning two independent runs: each file's OWN run start decides its lead-0 frame.
    Catches: evaluating the rule against the first file's run only."""
    vars_ = ('PREC_ACC_NC', 'PREC_ACC_C')
    a = syn.write_wrfout(tmp_path / 'wrfout_d03_a.nc', '2026-03-01T01', 23, NY, NX,
                         simulation_start='2026-03-01T00', variables=vars_)
    # the second run starts at 03-02T00 and its file begins with its own lead-0 frame
    b = syn.write_wrfout(tmp_path / 'wrfout_d03_b.nc', '2026-03-02T00', 24, NY, NX,
                         simulation_start='2026-03-02T00', variables=vars_)
    out = tmp_path / 'r.cfdb'
    WrfIngest([a, b]).convert(out, variables=['PREC_ACC'], time_label='start', squeeze_height=True)
    times, _ = _read(out)
    # run 1 frames 01..23 -> labels 00..22; run 2's lead-0 frame (03-02T00) dropped; 01..23 -> 00..22
    assert np.datetime64('2026-03-01T23:00') not in times
    assert times[0] == INIT and times[-1] == np.datetime64('2026-03-02T22:00')


@pytest.mark.parametrize('kw,match', [
    (dict(variables=['PREC_ACC', 'T2'], time_label='start'), 'accumulations only'),
    (dict(variables=['PREC_ACC'], time_label='start', dataset_type='grid_forecast'), 'grid mode only'),
    (dict(variables=['PREC_ACC'], extend=True, dataset_type='grid_forecast'), 'grid mode only'),
    (dict(variables=['PREC_ACC'], squeeze_height=True, dataset_type='grid_forecast'), 'grid mode only'),
    (dict(variables=['PREC_ACC'], time_label='middle'), "'end' or 'start'"),
    (dict(variables=['PREC_ACC'], chunk_shape=(24, 3, 3)), 'squeeze_height'),
])
def test_option_refusals(tmp_path, kw, match):
    files = _run(tmp_path, days=1)
    with pytest.raises(ValueError, match=match):
        WrfIngest(files).convert(tmp_path / 'x.cfdb', **kw)


def test_prec_acc_dt_mismatch_refused(tmp_path):
    files = _run(tmp_path, days=1)
    for f in files:
        with h5py.File(f, 'r+') as h5:
            h5.attrs['PREC_ACC_DT'] = np.float32(180.0)
    with pytest.raises(ValueError, match='PREC_ACC_DT=180'):
        WrfIngest(files).convert(tmp_path / 'x.cfdb', variables=['PREC_ACC'], time_label='start')


def test_non_wrf_sources_refuse(tmp_path, monkeypatch):
    """The options are WRF-only; a source that does not implement them refuses instead of half-working."""
    files = _run(tmp_path, days=1)
    monkeypatch.setattr(WrfIngest, '_supports_grid_extend', False)
    with pytest.raises(ValueError, match='implemented for WRF'):
        WrfIngest(files).convert(tmp_path / 'x.cfdb', variables=['PREC_ACC'], extend=True)


# ---------------------------------------------------------------- extend: split build == one-shot build

@pytest.mark.parametrize('windows', [
    # middle band, then prepend, prepend, append (band-aligned)
    [(24, 47), (0, 23), (48, 71), (72, 95)],
    # non-aligned edges
    [(30, 59), (0, 29), (60, 95)],
    # later band first, earlier band across a placeholder gap, then the gap
    [(72, 95), (0, 23), (24, 71)],
    # ascending, re-running a band on the way (idempotent)
    [(0, 23), (24, 47), (24, 47), (48, 95)],
    # a window whose last block holds ONE frame (label 48): exercises the single-timestep writer at an offset
    [(0, 23), (24, 48), (49, 95)],
])
def test_split_equals_one_shot(tmp_path, windows):
    """Catches: a wrong write offset / chunk anchor after an append or a prepend (negative origin)."""
    files = _run(tmp_path, days=4)
    one_t, one_v = _build(files, tmp_path / 'one.cfdb', [(INIT, INIT + 95 * H)])
    split_t, split_v = _build(files, tmp_path / 'split.cfdb', [(INIT + a * H, INIT + b * H) for a, b in windows])
    np.testing.assert_array_equal(split_t, one_t)
    np.testing.assert_array_equal(split_v, one_v)
    for k in (0, 47, 95):
        np.testing.assert_allclose(one_v[k], _expected(one_t[k]), atol=0.006)


def test_gap_is_a_placeholder(tmp_path):
    """Extending across a gap auto-fills unwritten placeholder slots: no chunks, read as NaN.
    Catches: refusing gaps, or writing NaN chunks into them."""
    files = _run(tmp_path, days=4)
    out = tmp_path / 'g.cfdb'
    _build(files, out, [(INIT + 72 * H, INIT + 95 * H)])
    r = WrfIngest(files).convert(out, start_date=str(INIT), end_date=str(INIT + 23 * H),
                                 chunk_shape=(24, 3, 3), **KW)
    assert r['status'] == 'prepend' and r['gap_filled'] == 48
    with cfdb.open_dataset(out) as ds:
        times = ds['time'].data.astype('datetime64[m]')
        np.testing.assert_array_equal(times, INIT + np.arange(96) * H)  # evenly spaced slots
        assert np.isnan(ds['precipitation'][24:72].data).all()
        missing = grid.missing_chunks(ds)
        # first build anchors absolute 0 at INIT+72h; after the prepend the origin is -72, so the gap slots
        # (positions 24..71) are absolute -48..-1: exactly the two 24-h chunk rows starting -48 and -24
        assert missing and {start[0] for _, start in missing} == {-48, -24}
        # one placeholder time on its own (catches an end-exclusive search in missing_chunks)
        assert grid.missing_chunks(ds, INIT + 30 * H, INIT + 30 * H)


def test_partial_window_written_and_reported(tmp_path):
    """Frames missing from the input are unwritten slots, reported in the result; re-running the window
    once the files exist fills them. Catches: shifting later frames into the holes, or not reporting."""
    files = _run(tmp_path, days=3)
    one_t, one_v = _build(files, tmp_path / 'one.cfdb', [(INIT, INIT + 71 * H)])
    out = tmp_path / 'p.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)])
    without_day2 = [f for f in files if '03-02' not in f.name]
    # the 03-02 file holds frames 03-02T00..23 = labels 03-01T23..03-02T22; 23 of them are in the window
    r = WrfIngest(without_day2).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 71 * H),
                                        chunk_shape=(24, 3, 3), **KW)
    assert len(r['missing_frames']) == 23
    assert r['missing_frames'][0] == str(INIT + 24 * H) and r['missing_frames'][-1] == str(INIT + 46 * H)
    t, v = _read(out)
    np.testing.assert_array_equal(t, one_t)
    assert np.isnan(v[24:47]).all()
    np.testing.assert_array_equal(v[47:], one_v[47:])
    # the files turn up: re-run the window
    _build(files, out, [(INIT + 24 * H, INIT + 71 * H)])
    np.testing.assert_array_equal(_read(out)[1], one_v)


def test_trailing_missing_frames_reported(tmp_path):
    """Missing frames at the END of the window (the last files absent) are reported too.
    Catches: the window's end taken from the frames present instead of end_date."""
    files = _run(tmp_path, days=3)
    present = [f for f in files if '03-03' not in f.name and '03-04' not in f.name]
    out = tmp_path / 'tr.cfdb'
    r = WrfIngest(present).convert(out, start_date=str(INIT), end_date=str(INIT + 71 * H),
                                   chunk_shape=(24, 3, 3), **KW)
    assert len(r['missing_frames']) == 25 and r['missing_frames'][-1] == str(INIT + 71 * H)
    t, v = _read(out)
    assert t[-1] == INIT + 71 * H and np.isnan(v[47:]).all()


def test_straddle_and_off_step_refused(tmp_path):
    files = _run(tmp_path, days=4)
    out = tmp_path / 's.cfdb'
    _build(files, out, [(INIT + 24 * H, INIT + 47 * H)])
    with pytest.raises(ValueError, match='extends both'):
        WrfIngest(files).convert(out, start_date=str(INIT), end_date=str(INIT + 71 * H),
                                 chunk_shape=(24, 3, 3), **KW)
    (tmp_path / 'off').mkdir()
    off = syn.write_wrfout(tmp_path / 'off' / 'wrfout_d03_x.nc', '2026-03-03T00:30', 3, NY, NX,
                           simulation_start='2026-03-01T00:30')
    with pytest.raises(ValueError, match='off the target time grid'):
        WrfIngest([off]).convert(out, chunk_shape=(24, 3, 3), **KW)


def test_crash_then_rerun(tmp_path, monkeypatch):
    """A crash part-way through a window leaves partly written chunks; re-running the window completes it
    exactly (writes are idempotent overwrites). Catches: a re-run that cannot overwrite placed times."""
    files = _run(tmp_path, days=4)
    one_t, one_v = _build(files, tmp_path / 'one.cfdb', [(INIT, INIT + 95 * H)], chunk_t=24)
    out = tmp_path / 'c.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)], chunk_t=24)
    orig = base.H5Ingest._write_data_var_block
    calls = {'n': 0}

    def flaky(self, *a, **k):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('simulated crash')
        return orig(self, *a, **k)

    monkeypatch.setattr(base.H5Ingest, '_write_data_var_block', flaky)
    with pytest.raises(RuntimeError, match='simulated crash'):
        WrfIngest(files).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 95 * H),
                                 chunk_shape=(24, 3, 3), **KW)
    monkeypatch.setattr(base.H5Ingest, '_write_data_var_block', orig)
    assert np.isnan(_read(out)[1]).any()
    _build(files, out, [(INIT + 24 * H, INIT + 95 * H)], chunk_t=24)
    t, v = _read(out)
    np.testing.assert_array_equal(t, one_t)
    np.testing.assert_array_equal(v, one_v)


def test_step_none_target_refused(tmp_path):
    """A target whose time axis has no step (step=True on one frame) cannot hold placeholder gaps."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'n.cfdb'
    w = WrfIngest(files)
    with cfdb.open_dataset(out, flag='n') as ds:
        ds.create.coord.time(data=np.array([INIT], dtype='datetime64[m]'), step=True)
        assert ds['time'].step is None
        ds.create.coord.generic('y', data=w.y.astype('float64'), axis='y', step=True)
        ds.create.coord.generic('x', data=w.x.astype('float64'), axis='x', step=True)
        ds.create.crs.from_user_input(w.crs, x_coord='x', y_coord='y')
    with pytest.raises(ValueError, match='has no step'):
        WrfIngest(files).convert(out, chunk_shape=(24, 3, 3), **KW)


def test_new_target_has_numeric_step(tmp_path):
    files = _run(tmp_path, days=1)
    out = tmp_path / 'one_frame.cfdb'
    WrfIngest(files).convert(out, start_date=str(INIT), end_date=str(INIT), chunk_shape=(24, 3, 3), **KW)
    with cfdb.open_dataset(out) as ds:
        assert len(ds['time'].data) == 1 and ds['time'].step == 60


def test_existing_chunk_shape_enforced(tmp_path):
    files = _run(tmp_path, days=2)
    out = tmp_path / 'k.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)], chunk_t=24)
    with pytest.raises(ValueError, match='chunk_shape'):
        WrfIngest(files).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 47 * H),
                                 chunk_shape=(12, 3, 3), **KW)
    # chunk_shape=None extends with the stored one
    WrfIngest(files).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 47 * H), **KW)
    with cfdb.open_dataset(out) as ds:
        assert ds['precipitation'].chunk_shape == (24, 3, 3)


def test_label_mode_mismatch_refused(tmp_path):
    files = _run(tmp_path, days=2)
    out = tmp_path / 'm.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)])
    with pytest.raises(ValueError, match='interval_start'):
        WrfIngest(files).convert(out, variables=['PREC_ACC'], extend=True, squeeze_height=True,
                                 start_date=str(INIT + 25 * H), end_date=str(INIT + 48 * H), chunk_shape=(24, 3, 3))


def test_time_bands_cover_and_align():
    anchor = np.datetime64('1990-07-01T00', 'm')
    bands = grid.time_bands(anchor, '1990-06-20T05', '1990-07-02T00', 24, 60)
    assert bands[0][0] <= np.datetime64('1990-06-20T05') and bands[-1][1] >= np.datetime64('1990-07-02T00')
    assert all(((b0 - anchor) / H) % 24 == 0 for b0, _ in bands)
    assert all(b1 - b0 == 24 * H for b0, b1 in bands)
    assert [b0 for b0, _ in bands] == sorted(b0 for b0, _ in bands)


def test_existing_encoding_enforced(tmp_path, monkeypatch):
    """Extend refuses to write a variable whose stored encoding differs from the incoming template
    (e.g. after a cfdb-vars precision change). Catches: dropping the strict dtype check."""
    files = _run(tmp_path, days=2)
    out = tmp_path / 'd.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)])
    orig = base._resolve_var_template

    def other_precision(cfdb_name, chunk_shape, dtype=None):
        name, params, attrs = orig(cfdb_name, chunk_shape, dtype)
        return name, {**params, 'dtype': cfdb.dtypes.dtype('float32', precision=1, min_value=0, max_value=5000)}, attrs

    monkeypatch.setattr(base, '_resolve_var_template', other_precision)
    with pytest.raises(ValueError, match='is encoded as'):
        WrfIngest(files).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 47 * H),
                                 chunk_shape=(24, 3, 3), **KW)


def test_variable_mismatch_refused_before_axis_touched(tmp_path):
    """A refused call leaves the target as it was. Catches: placing times before the variable checks."""
    files = _run(tmp_path, days=3)
    out = tmp_path / 'v.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)], chunk_t=24)
    with pytest.raises(ValueError, match='chunk_shape'):
        WrfIngest(files).convert(out, start_date=str(INIT + 48 * H), end_date=str(INIT + 71 * H),
                                 chunk_shape=(12, 3, 3), **KW)
    with cfdb.open_dataset(out) as ds:
        assert len(ds['time'].data) == 24


def test_rain_extend_split_equals_one_shot(tmp_path):
    """RAIN (differenced totals) built in two non-aligned windows equals a one-shot build.
    Catches: pad rows taken into the increment (the first increment would be the running total)."""
    files = _run(tmp_path, days=4)
    kw = dict(variables=['RAIN'], extend=True, squeeze_height=True, chunk_shape=(24, 3, 3))
    for name, windows in (('one', [(1, 95)]), ('split', [(1, 30), (31, 95)])):
        for a, b in windows:
            WrfIngest(files).convert(tmp_path / f'{name}.cfdb', start_date=str(INIT + a * H),
                                     end_date=str(INIT + b * H), **kw)
    t1, v1 = _read(tmp_path / 'one.cfdb')
    t2, v2 = _read(tmp_path / 'split.cfdb')
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)
    lead1 = (syn.expected('RAINNC', 1, NY, NX) - syn.expected('RAINNC', 0, NY, NX)
             + syn.expected('RAINC', 1, NY, NX) - syn.expected('RAINC', 0, NY, NX))
    np.testing.assert_allclose(v1[0], lead1, atol=0.006)


def test_three_hourly_extend(tmp_path):
    """A non-hourly record: step and labels follow PREC_ACC_DT (180 min). Catches: an assumed hourly step."""
    files = syn.write_run(tmp_path / 'r3', str(INIT), 96, NY, NX, frames_per_file=8, hour_step=3)
    kw = dict(KW, chunk_shape=(8, 3, 3))
    one = tmp_path / 'one3.cfdb'
    WrfIngest(files).convert(one, start_date=str(INIT), end_date=str(INIT + 93 * H), **kw)
    split = tmp_path / 'split3.cfdb'
    for a, b in ((45, 93), (0, 42)):
        WrfIngest(files).convert(split, start_date=str(INIT + a * H), end_date=str(INIT + b * H), **kw)
    t1, v1 = _read(one)
    t2, v2 = _read(split)
    np.testing.assert_array_equal(t1, INIT + np.arange(32) * 3 * H)
    np.testing.assert_array_equal(t2, t1)
    np.testing.assert_array_equal(v2, v1)
    with cfdb.open_dataset(one) as ds:
        assert ds['time'].step == 180


def test_anchor_after_unaligned_prepend(tmp_path):
    """The chunk grid stays anchored at the first time ever created, after a prepend of a non-multiple of
    the chunk. Catches: an anchor computed with the origin's sign flipped."""
    files = _run(tmp_path, days=2)
    out = tmp_path / 'a.cfdb'
    _build(files, out, [(INIT + 30 * H, INIT + 47 * H), (INIT, INIT + 29 * H)])
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].origin == -30
        assert grid.time_anchor(ds) == INIT + 30 * H
        bands = grid.time_bands(grid.time_anchor(ds), INIT, INIT + 48 * H, 24, 60)
        assert bands[0][0] == INIT - 18 * H


def test_gap_filled_counts_on_append(tmp_path):
    files = _run(tmp_path, days=3)
    out = tmp_path / 'g2.cfdb'
    _build(files, out, [(INIT, INIT + 23 * H)])
    r = WrfIngest(files).convert(out, start_date=str(INIT + 48 * H), end_date=str(INIT + 71 * H),
                                 chunk_shape=(24, 3, 3), **KW)
    assert r['status'] == 'append' and r['gap_filled'] == 24


def test_step_not_inferred_from_two_frames(tmp_path):
    """Two frames around a hole do not define the step: the smallest spacing of all frames does.
    Catches: a 2-frame window taken as a 2-hour step, hiding the missing frame."""
    a = syn.write_wrfout(tmp_path / 'wrfout_d03_a.nc', '2026-03-01T01', 2, NY, NX,
                         simulation_start='2026-03-01T00', variables=('T2',))
    b = syn.write_wrfout(tmp_path / 'wrfout_d03_b.nc', '2026-03-01T04', 1, NY, NX,
                         simulation_start='2026-03-01T00', variables=('T2',))
    out = tmp_path / 's.cfdb'
    r = WrfIngest([a, b]).convert(out, variables=['T2'], extend=True, squeeze_height=True,
                                  start_date='2026-03-01T02', end_date='2026-03-01T04')
    assert r['missing_frames'] == ['2026-03-01T03:00']
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].step == 60 and len(ds['time'].data) == 3


def test_frames_off_their_run_grid_refused(tmp_path):
    """PREC_ACC frames must lie on SIMULATION_START_DATE + k*PREC_ACC_DT of their run."""
    p = syn.write_wrfout(tmp_path / 'wrfout_d03_off.nc', '2026-03-01T01:00', 3, NY, NX,
                         simulation_start='2026-03-01T00:30', variables=('PREC_ACC_NC', 'PREC_ACC_C'))
    with pytest.raises(ValueError, match='not on their run'):
        WrfIngest(p).convert(tmp_path / 'o.cfdb', variables=['PREC_ACC'], time_label='start')

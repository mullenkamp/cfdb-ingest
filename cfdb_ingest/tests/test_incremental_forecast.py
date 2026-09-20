"""
An init built up over several convert() calls (0.5.0): ``leads=`` at creation, ``mark_complete=False``
for partial calls, ``forecast.missing_chunks`` as the completeness check, per-lead chunks, and the
stateless ``PREC_ACC`` precipitation source -- all on generated wrfout (``cfdb_ingest.wrf_synthetic``).
"""

import cfdb
import numpy as np
import pytest
from cfdb import dtypes

from cfdb_ingest import forecast as fc
from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.wrf import WrfIngest

NY, NX = 12, 10
INIT = np.datetime64('2026-09-19T00', 'm')
LEAD_HOURS = 48
AXIS = np.arange(0, LEAD_HOURS + 1, dtype='int32')  # 49 leads
CHUNK = (1, 1, 1, NY, NX)
VARS = ['T2', 'PSFC', 'WIND10', 'PREC_ACC']
CFDB_NAMES = ['air_temperature', 'surface_pressure', 'wind_speed', 'precipitation']  # no level vars -> bare names


@pytest.fixture(scope='module')
def run_files(tmp_path_factory):
    d = tmp_path_factory.mktemp('run')
    return syn.write_run(d, INIT, LEAD_HOURS, NY, NX)  # 24 + 24 + 1 frames


def _ingest(target, files, **kw):
    kw.setdefault('variables', VARS)
    kw.setdefault('leads', AXIS)
    kw.setdefault('chunk_shape', CHUNK)
    kw.setdefault('mark_complete', False)
    return WrfIngest(files).convert(target, dataset_type='grid_forecast', forecast_step_minutes=360, **kw)


def _field(ds, name, lead):
    return np.squeeze(np.asarray(ds[name][0, int(lead), 0, :, :].data))


# ---------------------------------------------------------------- the generator itself


def test_generator_shape_and_closed_forms(run_files):
    assert [p.name for p in run_files] == [
        'wrfout_d02_2026-09-19_00_00_00.nc', 'wrfout_d02_2026-09-20_00_00_00.nc', 'wrfout_d02_2026-09-21_00_00_00.nc'
    ]
    ing = WrfIngest(run_files)
    assert ing._simulation_start == INIT
    assert len(ing.times) == 49
    # increments and totals are consistent: RAIN differenced == PREC_ACC summed (except lead 0)
    for lead in (1, 24, 25, 48):
        inc = syn.expected('PREC_ACC_NC', lead, NY, NX) + syn.expected('PREC_ACC_C', lead, NY, NX)
        tot = (syn.expected('RAINNC', lead, NY, NX) + syn.expected('RAINC', lead, NY, NX)
               - syn.expected('RAINNC', lead - 1, NY, NX) - syn.expected('RAINC', lead - 1, NY, NX))
        np.testing.assert_allclose(inc, tot, atol=1e-4)
    assert syn.base('PREC_ACC_NC', 0) == 0.0 and syn.base('PREC_ACC_C', 0) == 0.0


# ---------------------------------------------------------------- incremental ingest


def test_init_built_file_by_file(tmp_path, run_files):
    p = tmp_path / 'fc.cfdb'
    day1, day2, last = run_files

    r1 = _ingest(p, day1)
    assert r1['status'] == 'new' and r1['n_leads'] == 24 and r1['complete'] is False
    with cfdb.open_dataset(str(p)) as ds:
        np.testing.assert_array_equal(ds[fc.LEAD].data, AXIS)  # the FULL axis, not this call's 24 leads
        assert ds['precipitation'].chunk_shape == CHUNK
        assert fc.complete_inits(ds) == []
        missing = fc.missing_chunks(ds, INIT)
        assert len(missing) == (49 - 24) * len(CFDB_NAMES)
        assert all(start[1] >= 24 for _, start in missing)  # exactly the leads not yet written
        np.testing.assert_allclose(_field(ds, 'air_temperature', 5), syn.expected('T2', 5, NY, NX), atol=0.06)  # packed to 0.1 K

    r2 = _ingest(p, day2)
    assert r2['status'] == 'backfill' and r2['complete'] is False  # no overwrite= needed
    with cfdb.open_dataset(str(p)) as ds:
        assert fc.complete_inits(ds) == []
        missing = fc.missing_chunks(ds, INIT)
        assert {start[1] for _, start in missing} == {48}
        # day-1 chunks untouched, day-2 chunks placed by lead value
        np.testing.assert_allclose(_field(ds, 'air_temperature', 5), syn.expected('T2', 5, NY, NX), atol=0.06)
        np.testing.assert_allclose(_field(ds, 'surface_pressure', 30), syn.expected('PSFC', 30, NY, NX), atol=1.0)
        # the boundary frame: precip at lead 24 is the increment over (23, 24], with no previous-frame state
        want = syn.expected('PREC_ACC_NC', 24, NY, NX) + syn.expected('PREC_ACC_C', 24, NY, NX)
        np.testing.assert_allclose(_field(ds, 'precipitation', 24), want, atol=0.011)  # packed to 0.01 mm
        assert np.isfinite(_field(ds, 'precipitation', 24)).all()
        np.testing.assert_allclose(_field(ds, 'precipitation', 0), syn.expected('PREC_ACC_NC', 0, NY, NX) * 0, atol=1e-6)

    r3 = _ingest(p, last, mark_complete=True)  # the single end-of-run frame; lead_step None path
    assert r3['n_leads'] == 1 and r3['complete'] is True
    with cfdb.open_dataset(str(p)) as ds:
        assert fc.missing_chunks(ds, INIT) == []
        assert fc.complete_inits(ds) == [str(INIT)]
        np.testing.assert_allclose(_field(ds, 'wind_speed', 48),
                                   np.hypot(syn.expected('U10', 48, NY, NX), syn.expected('V10', 48, NY, NX)), atol=0.06)


def test_partial_call_never_marks_and_a_complete_init_is_immutable(tmp_path, run_files):
    p = tmp_path / 'fc.cfdb'
    _ingest(p, run_files[0])
    _ingest(p, run_files[0])  # re-ingesting a partial init is a back-fill, not a refusal
    _ingest(p, run_files, mark_complete=True)
    with pytest.raises(ValueError, match='immutable'):
        _ingest(p, run_files[1])  # now complete: refused without overwrite=True
    _ingest(p, run_files[1], overwrite=True)


def test_writer_buffers_only_this_calls_span(tmp_path, run_files):
    p = tmp_path / 'fc.cfdb'
    captured = {}
    orig_init = fc.ForecastWriter.__init__

    def spy(self, *a, **kw):
        orig_init(self, *a, **kw)
        captured['span'] = self.span
        captured['stored'] = self.n_lead_stored

    fc.ForecastWriter.__init__ = spy
    try:
        _ingest(p, run_files[1])
    finally:
        fc.ForecastWriter.__init__ = orig_init
    assert captured == {'span': 24, 'stored': 49}


def test_leads_argument_guards(tmp_path, run_files):
    p = tmp_path / 'fc.cfdb'
    with pytest.raises(ValueError, match='not on leads='):
        _ingest(p, run_files[1], leads=np.arange(0, 24))  # this call's leads 24..47 are off the axis
    with pytest.raises(ValueError, match='not regularly spaced'):
        _ingest(p, run_files[0], leads=[0, 1, 2, 4])
    _ingest(p, run_files[0])
    with pytest.raises(ValueError, match='fixed at creation'):
        _ingest(p, run_files[1], leads=np.arange(0, 72))
    with pytest.raises(ValueError, match="grid_forecast"):
        WrfIngest(run_files[0]).convert(tmp_path / 'g.cfdb', variables=['T2'], leads=AXIS)
    with pytest.raises(ValueError, match="grid_forecast"):
        WrfIngest(run_files[0]).convert(tmp_path / 'g.cfdb', variables=['T2'], mark_complete=False)


def test_default_still_marks_complete(tmp_path, run_files):
    r = WrfIngest(run_files).convert(tmp_path / 'fc.cfdb', variables=['T2'], dataset_type='grid_forecast')
    assert r['complete'] is True
    with cfdb.open_dataset(str(tmp_path / 'fc.cfdb')) as ds:
        assert fc.complete_inits(ds) == [str(INIT)]
        np.testing.assert_array_equal(ds[fc.LEAD].data, AXIS)


# ---------------------------------------------------------------- PREC_ACC vs RAIN


def test_prec_acc_matches_rain_except_lead0(tmp_path, run_files):
    WrfIngest(run_files).convert(tmp_path / 'a.cfdb', variables=['PREC_ACC'])
    WrfIngest(run_files).convert(tmp_path / 'b.cfdb', variables=['RAIN'])
    with cfdb.open_dataset(str(tmp_path / 'a.cfdb')) as da, cfdb.open_dataset(str(tmp_path / 'b.cfdb')) as db:
        pa = np.squeeze(np.asarray(da['precipitation'][:, 0, :, :].data))
        pb = np.squeeze(np.asarray(db['precipitation'][:, 0, :, :].data))
        assert np.isnan(pb[0]).all() and not np.isnan(pa[0]).any() and (pa[0] == 0).all()
        np.testing.assert_allclose(pa[1:], pb[1:], atol=0.015)  # two 0.01 mm roundings


def test_prec_acc_with_bucketed_totals(tmp_path):
    d = tmp_path / 'run'
    files = syn.write_run(d, INIT, 30, NY, NX, bucket_mm=5.0, end_frame_file=False)
    WrfIngest(files).convert(tmp_path / 'a.cfdb', variables=['PREC_ACC'])
    WrfIngest(files).convert(tmp_path / 'b.cfdb', variables=['RAIN'])
    with cfdb.open_dataset(str(tmp_path / 'a.cfdb')) as da, cfdb.open_dataset(str(tmp_path / 'b.cfdb')) as db:
        pa = np.squeeze(np.asarray(da['precipitation'][:, 0, :, :].data))
        pb = np.squeeze(np.asarray(db['precipitation'][:, 0, :, :].data))
        np.testing.assert_allclose(pa[1:], pb[1:], atol=0.015)


def test_rain_and_prec_acc_together_refused(tmp_path, run_files):
    with pytest.raises(ValueError, match="both write cfdb 'precip'"):
        WrfIngest(run_files).convert(tmp_path / 'a.cfdb', variables=['RAIN', 'PREC_ACC'])


# ---------------------------------------------------------------- missing_chunks on a bare dataset


def _bare(path, chunk):
    ds = cfdb.open_dataset(str(path), 'n', dataset_type='grid_forecast')
    fc.create_forecast_coords(ds, INIT, [0, 3, 6], step_minutes=360)
    ds.create.coord.generic('longitude', data=np.arange(4.0), axis='x', step=True)
    ds.create.coord.generic('latitude', data=np.arange(3.0), axis='y', step=True)
    ds.create.coord.generic('height_2m', data=np.array([2.0]), axis=None)
    for name in ('a', 'b'):
        ds.create.data_var.generic(name, (fc.FRT, fc.LEAD, 'height_2m', 'latitude', 'longitude'),
                                   dtype=dtypes.dtype('float32'), chunk_shape=chunk)
    return ds


@pytest.mark.parametrize('chunk', [(1, 1, 1, 3, 4), (1, 3, 1, 3, 4), (1, 1, 1, 2, 2)])
def test_missing_chunks_enumerates_every_chunk_of_the_init(tmp_path, chunk):
    with _bare(tmp_path / 'x.cfdb', chunk) as ds:
        n_lead_chunks = -(-3 // chunk[1])
        n_tiles = -(-3 // chunk[3]) * -(-4 // chunk[4])
        assert len(fc.missing_chunks(ds, INIT)) == 2 * n_lead_chunks * n_tiles
        ds['a'][0, :, 0, :, :] = np.ones((3, 3, 4), dtype='float32')
        missing = fc.missing_chunks(ds, INIT)
        assert {v for v, _ in missing} == {'b'}
        ds['b'][0, 0:1, 0, :, :] = np.ones((1, 3, 4), dtype='float32')
        missing = fc.missing_chunks(ds, INIT)
        assert all(start[1] > 0 for _, start in missing) if chunk[1] == 1 else missing == []
    with pytest.raises(ValueError, match='not on the stored axis'):
        with _bare(tmp_path / 'y.cfdb', chunk) as ds:
            fc.missing_chunks(ds, INIT + np.timedelta64(6, 'h'))


def test_missing_chunks_honours_a_truncated_origin(tmp_path):
    """Chunk keys live in absolute index space; after truncate the origin is non-zero."""
    p = tmp_path / 'x.cfdb'
    with _bare(p, (1, 1, 1, 3, 4)) as ds:
        ds['a'][0, :, 0, :, :] = np.ones((3, 3, 4), dtype='float32')
        ds['b'][0, :, 0, :, :] = np.ones((3, 3, 4), dtype='float32')
        later = INIT + np.timedelta64(12, 'h')
        ds[fc.FRT].append(np.array([later], dtype='datetime64[m]'))  # auto-fills the 06z slot: 'later' is index 2
        assert fc.init_index(ds, later) == 2
        ds['a'][2, :, 0, :, :] = np.ones((3, 3, 4), dtype='float32')
    with cfdb.open_dataset(str(p), 'w') as ds:
        ds[fc.FRT].truncate(start=INIT + np.timedelta64(12, 'h'))
    with cfdb.open_dataset(str(p)) as ds:
        later = INIT + np.timedelta64(12, 'h')
        assert fc.init_index(ds, later) == 0
        assert {v for v, _ in fc.missing_chunks(ds, later)} == {'b'}
        assert len(fc.missing_chunks(ds, later)) == 3


def test_coarse_frames_leave_gaps_missing_not_nan(tmp_path):
    """3-hourly frames on the hourly axis: the gaps must stay ABSENT (missing_chunks sees them), never be
    written as NaN chunks, and an interleaving call must not clobber them (review ifs-forecast-cycle-code-1)."""
    d = tmp_path / 'run'
    d.mkdir()
    even = syn.write_wrfout(d / 'a.nc', INIT + np.timedelta64(24, 'h'), 9, NY, NX, hour_step=3, simulation_start=INIT, variables=('T2',))
    p = tmp_path / 'fc.cfdb'
    _ingest(p, even, variables=['T2'])
    with cfdb.open_dataset(str(p)) as ds:
        missing = {s[1] for _, s in fc.missing_chunks(ds, INIT)}
        assert missing == set(range(0, 24)) | {25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47}
        assert np.isfinite(_field(ds, 'air_temperature', 27)).all()
    # a second call supplying the odd hours 25, 28, ... (3-hourly, offset by 1) fills its own leads only
    odd = syn.write_wrfout(d / 'b.nc', INIT + np.timedelta64(25, 'h'), 8, NY, NX, hour_step=3, simulation_start=INIT, variables=('T2',))
    _ingest(p, odd, variables=['T2'])
    with cfdb.open_dataset(str(p)) as ds:
        for lead in (24, 25, 27, 28, 46):
            np.testing.assert_allclose(_field(ds, 'air_temperature', lead), syn.expected('T2', lead, NY, NX), atol=0.06)
        assert {s[1] for _, s in fc.missing_chunks(ds, INIT) if s[1] >= 24} == {26, 29, 32, 35, 38, 41, 44, 47}

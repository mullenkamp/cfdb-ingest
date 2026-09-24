"""PREC_ACC (WRF's windowed precipitation accumulators) in cfdb-ingest 0.6.0: negatives clipped to 0."""
import cfdb
import h5py
import numpy as np
import pytest

from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.checks import check_encodable
from cfdb_ingest.wrf import WrfIngest

NY, NX = 6, 5
POKE = (2, 1, 3)  # (time index, j, i) set to a small negative


def _file_with_negative(tmp_path):
    p = syn.write_wrfout(tmp_path / 'wrfout_d03_2026-01-01_00_00_00.nc', '2026-01-01T01', 6, NY, NX,
                         simulation_start='2026-01-01T00')
    with h5py.File(p, 'r+') as h5:
        nc = h5['PREC_ACC_NC'][:]
        c = h5['PREC_ACC_C'][:]
        nc[POKE] = -0.01
        c[POKE] = 0.0
        h5['PREC_ACC_NC'][:] = nc
        h5['PREC_ACC_C'][:] = c
    return p


def _expected_sum(t):
    lead = t + 1
    return syn.expected('PREC_ACC_NC', lead, NY, NX) + syn.expected('PREC_ACC_C', lead, NY, NX)


def test_grid_negative_clipped(tmp_path):
    """Catches: PREC_ACC mapped to the unclipped 'sum' transform (the < 0.6.0 behaviour)."""
    p = _file_with_negative(tmp_path)
    out = tmp_path / 'g.cfdb'
    WrfIngest(p).convert(out, variables=['PREC_ACC'])
    with cfdb.open_dataset(out) as ds:
        v = np.squeeze(ds['precipitation'].data)
    assert v[POKE] == 0.0
    for t in range(6):
        want = _expected_sum(t)
        if t == POKE[0]:
            want[POKE[1:]] = 0.0
        np.testing.assert_allclose(v[t], want, atol=0.006)


def test_forecast_negative_clipped(tmp_path):
    p = _file_with_negative(tmp_path)
    out = tmp_path / 'f.cfdb'
    WrfIngest(p).convert(out, variables=['PREC_ACC'], dataset_type='grid_forecast', mark_complete=False)
    with cfdb.open_dataset(out) as ds:
        v = np.squeeze(ds['precipitation'].data)
    assert v[POKE] == 0.0


# ---------------------------------------------------------------- out-of-range values refused



@pytest.fixture
def precip_var(tmp_path):
    ds = cfdb.open_dataset(tmp_path / 'edge.cfdb', flag='n')
    ds.create.coord.generic('i', data=np.arange(4.0), step=True)
    yield ds.create.data_var.precip(('i',))
    ds.close()


@pytest.mark.parametrize('value,ok', [
    (0.0, True), (654.35, True), (-0.99, True), (np.nan, True),
    (654.36, False), (700.0, False), (-1.0, False), (np.inf, False), (-np.inf, False),
])
def test_check_encodable_edges(precip_var, value, ok):
    """The precipitation template (uint16, precision 2, offset -1, fill code 0) stores -0.99 .. 654.35.
    The edges were measured against cfdb itself: 654.36 and -1.0 read back NaN."""
    block = np.array([1.0, value, 2.0, np.nan], dtype='float32')
    if ok:
        check_encodable(precip_var, block)
        precip_var[:] = block
        got = precip_var.data
        np.testing.assert_allclose(got[:3], block[:3], atol=0.006)
    else:
        with pytest.raises(ValueError, match='outside the storable range'):
            check_encodable(precip_var, block)


def test_grid_out_of_range_refused(tmp_path):
    """Catches: removing the check from the grid write path (cfdb would store NaN silently)."""
    p = _file_with_negative(tmp_path)
    with h5py.File(p, 'r+') as h5:
        nc = h5['PREC_ACC_NC'][:]
        nc[4, 2, 2] = 700.0
        h5['PREC_ACC_NC'][:] = nc
    with pytest.raises(ValueError, match='outside the storable range'):
        WrfIngest(p).convert(tmp_path / 'g.cfdb', variables=['PREC_ACC'])


def test_forecast_out_of_range_refused(tmp_path):
    """Catches: removing the check from ForecastWriter.put."""
    p = _file_with_negative(tmp_path)
    with h5py.File(p, 'r+') as h5:
        nc = h5['PREC_ACC_NC'][:]
        nc[4, 2, 2] = 700.0
        h5['PREC_ACC_NC'][:] = nc
    with pytest.raises(ValueError, match='outside the storable range'):
        WrfIngest(p).convert(tmp_path / 'f.cfdb', variables=['PREC_ACC'], dataset_type='grid_forecast',
                             mark_complete=False)


# ---------------------------------------------------------------- RAIN across independent runs refused

def _two_runs(tmp_path):
    """Two daily files from two independent (cold-start) runs, as in a stitched hindcast."""
    vars_ = ('PREC_ACC_NC', 'PREC_ACC_C', 'RAINNC', 'RAINC')
    a = syn.write_wrfout(tmp_path / 'wrfout_d03_2026-01-02_00_00_00.nc', '2026-01-02T00', 24, NY, NX,
                         simulation_start='2026-01-01T00', variables=vars_)
    b = syn.write_wrfout(tmp_path / 'wrfout_d03_2026-01-03_00_00_00.nc', '2026-01-03T00', 24, NY, NX,
                         simulation_start='2026-01-02T00', variables=vars_)
    return [a, b]


def test_rain_across_runs_refused(tmp_path):
    """Catches: removing the refusal (the first interval of the second run would be stored as 0)."""
    with pytest.raises(ValueError, match='restart at every cold start'):
        WrfIngest(_two_runs(tmp_path)).convert(tmp_path / 'r.cfdb', variables=['RAIN'])


def test_prec_acc_across_runs_allowed(tmp_path):
    WrfIngest(_two_runs(tmp_path)).convert(tmp_path / 'p.cfdb', variables=['PREC_ACC'])
    with cfdb.open_dataset(tmp_path / 'p.cfdb') as ds:
        assert ds['precipitation'].shape[0] == 48


def test_float64_edge_refused(precip_var):
    """cfdb casts to the decoded dtype (float32) before encoding: float64 654.35497 -> 654.355 -> missing.
    Catches: checking in the block's own dtype."""
    with pytest.raises(ValueError, match='outside the storable range'):
        check_encodable(precip_var, np.array([1.0, 654.35497], dtype='float64'))


def test_rain_window_inside_one_run_allowed(tmp_path):
    """The multi-run refusal looks at the requested window's frames, not every file passed in."""
    files = _two_runs(tmp_path)
    out = tmp_path / 'w.cfdb'
    WrfIngest(files).convert(out, variables=['RAIN'], start_date='2026-01-03T01', end_date='2026-01-03T23')
    with cfdb.open_dataset(out) as ds:
        assert ds['precipitation'].shape[0] == 23
        assert not np.isnan(ds['precipitation'].data).any()


def test_rain_first_increment_not_seeded_across_runs(tmp_path):
    """A window starting at a run's first frame: the frame before it belongs to the previous run, so the
    first increment is unknown (NaN), not a clipped cross-run difference (0)."""
    files = _two_runs(tmp_path)
    out = tmp_path / 's.cfdb'
    WrfIngest(files).convert(out, variables=['RAIN'], start_date='2026-01-03T00', end_date='2026-01-03T23')
    with cfdb.open_dataset(out) as ds:
        v = ds['precipitation'].data
    assert np.isnan(v[0]).all() and not np.isnan(v[1:]).any()

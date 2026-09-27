"""convert(clip_nonneg=True): quantities that cannot be negative are floored at 0; signed ones are untouched (0.6.2)."""

import cfdb
import h5py
import numpy as np
import pytest

from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.wrf import WrfIngest

START = np.datetime64('2026-09-26T12', 'm')
NY, NX = 6, 5
KEYS = ['Q2', 'SWDOWN', 'TSK', 'HFX']


def _wrfout(tmp_path, n_times):
    """A synthetic wrfout whose frame 0 carries a WRF-style undershoot: Q2 -0.004, SWDOWN -5, a NaN Q2 cell,
    and a negative (signed, legitimate) HFX of -250 W/m2."""
    f = syn.write_wrfout(tmp_path / f'wrfout_d02_{n_times}.nc', START, n_times, NY, NX,
                         simulation_start=START, variables=('T2', 'PSFC', 'Q2', 'SWDOWN', 'TSK'))
    with h5py.File(f, 'r+') as h:
        q2 = h['Q2'][:]; q2[0, 1, 2] = -0.004; q2[0, 3, 3] = np.nan; h['Q2'][:] = q2
        sw = h['SWDOWN'][:]; sw[0, 2, 1] = -5.0; h['SWDOWN'][:] = sw
        hfx = np.full(q2.shape, 40.0, dtype='float32'); hfx[0, 4, 4] = -250.0
        d = h.create_dataset('HFX', data=hfx)
        for k, v in h['T2'].attrs.items():
            d.attrs[k] = v
    return f


def _convert(f, path, forecast, **kw):
    extra = dict(dataset_type='grid_forecast', forecast_reference_time=START, leads=list(range(0, 3))) if forecast else {}
    WrfIngest(f).convert(path, variables=KEYS, **extra, **kw)
    return cfdb.open_dataset(str(path))


def _frame0(ds, name):
    v = ds[name]
    key = (0,) * (v.ndims - 2) + (slice(None), slice(None))  # first time / lead, the height axis, the full grid
    return np.squeeze(np.asarray(v[key].data, dtype='float64'))


@pytest.mark.parametrize('n_times', [1, 3])  # 1 frame: the single-timestep writer; 3: the coalesced block writer
@pytest.mark.parametrize('forecast', [False, True])  # direct write vs the ForecastWriter
def test_clip_nonneg_floors_only_the_flagged_variables(tmp_path, n_times, forecast):
    f = _wrfout(tmp_path, n_times)
    with h5py.File(f, 'r') as h:
        raw = {k: h[k][0].astype('float64') for k in ('Q2', 'SWDOWN', 'TSK', 'HFX')}
    with _convert(f, tmp_path / 'out.cfdb', forecast, clip_nonneg=True) as ds:
        q2, sw = _frame0(ds, 'mixing_ratio'), _frame0(ds, 'shortwave_radiation')
        assert q2[1, 2] == 0.0 and sw[2, 1] == 0.0                     # floored
        assert np.isnan(q2[3, 3])                                       # NaN kept, not turned into 0
        keep = np.ones((NY, NX), bool); keep[1, 2] = keep[3, 3] = False
        np.testing.assert_allclose(q2[keep], raw['Q2'][keep], atol=1e-6)  # every other cell untouched
        hfx = _frame0(ds, 'sensible_heat_flux')
        assert hfx[4, 4] == pytest.approx(-250.0, abs=0.06)          # signed: never clipped
        assert 'skin_temperature' in ds.data_var_names and 'soil_temperature' not in ds.data_var_names
        np.testing.assert_allclose(_frame0(ds, 'skin_temperature'), raw['TSK'], atol=0.006)


@pytest.mark.parametrize('forecast', [False, True])
def test_without_clip_nonneg_the_raw_undershoot_is_refused_not_stored(tmp_path, forecast):
    """The default keeps raw values; a negative mixing ratio does not fit its 0-based template, so the pre-write
    check refuses it rather than storing it as missing."""
    f = _wrfout(tmp_path, 3)
    with pytest.raises(ValueError, match='mixing_ratio|shortwave_radiation'):
        _convert(f, tmp_path / 'out.cfdb', forecast)

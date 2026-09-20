"""WrfIngest forecast mode: one WRF run's wrfout -> a cfdb grid_forecast dataset."""

import cfdb
import numpy as np
import pytest

from cfdb_ingest import forecast as fc
from cfdb_ingest.tests.conftest import WRF_FILE_1, WRF_FILE_2
from cfdb_ingest.wrf import WrfIngest

LEVELS = [100000.0, 85000.0]
VARS = ['T2', 'T', 'PSFC']
# Both fixtures are segments of ONE run initialised 2023-02-11_00 (SIMULATION_START_DATE):
# file 1 holds valid times 02-12T00..23 (leads 24..47), file 2 02-13T00..23 (leads 48..71).
RUN_INIT = np.datetime64('2023-02-11T00', 'm')


def _squeeze(a):
    return np.squeeze(np.asarray(a))


def _grid(path, files, **kw):
    WrfIngest(files).convert(path, variables=VARS, target_levels=LEVELS, vertical_coord='pressure', **kw)
    return cfdb.open_dataset(str(path))


def _forecast(path_or_ds, files, **kw):
    return WrfIngest(files).convert(
        path_or_ds, variables=VARS, target_levels=LEVELS, vertical_coord='pressure', dataset_type='grid_forecast', **kw
    )


def test_forecast_equals_grid_values(tmp_path):
    """The lead placement is proven by byte-equality against the grid-mode conversion."""
    result = _forecast(tmp_path / 'f.cfdb', WRF_FILE_1)
    assert result['status'] == 'new' and result['n_leads'] == 24 and result['init'] == '2023-02-11T00:00'
    assert result['chunk_writes'] == 4  # T2 (1 level) + T (2 levels) + PSFC (1): one row each
    with _grid(tmp_path / 'g.cfdb', WRF_FILE_1) as g, cfdb.open_dataset(str(tmp_path / 'f.cfdb')) as f:
        assert f.dataset_type == 'grid_forecast'
        assert f['air_temperature_2m'].coord_names == (fc.FRT, fc.LEAD, 'height_2m', 'y', 'x')
        assert f['air_temperature'].coord_names == (fc.FRT, fc.LEAD, 'pressure', 'y', 'x')
        assert f[fc.FRT].data[0] == RUN_INIT and f[fc.FRT].step == 360
        np.testing.assert_array_equal(f[fc.LEAD].data, np.arange(24, 48))
        assert f[fc.LEAD].attrs['units'] == 'h'
        for name in ('air_temperature_2m', 'air_temperature', 'surface_pressure'):
            np.testing.assert_array_equal(_squeeze(f[name][0, :, :, :, :].data), _squeeze(g[name][:, :, :, :].data))
        assert fc.complete_inits(f) == ['2023-02-11T00:00']
        assert f['air_temperature_2m'].chunk_shape == (1, 24, 1, 111, 99)
    # a fresh cfdb file always holds one reclaimable metadata block (grid mode too); no data chunk is rewritten
    with cfdb.open_dataset(str(tmp_path / 'f.cfdb'), 'w') as f:
        assert f.prune() <= 1


def test_explicit_init_and_start_filter(tmp_path):
    p = tmp_path / 'f.cfdb'
    _forecast(p, WRF_FILE_1, forecast_reference_time='2023-02-12T00')
    with cfdb.open_dataset(str(p)) as f:
        np.testing.assert_array_equal(f[fc.LEAD].data, np.arange(0, 24))
    # re-ingest the same init from 06Z onward: leads are placed by VALUE (6..23), not from 0
    _forecast(p, WRF_FILE_1, forecast_reference_time='2023-02-12T00', start_date='2023-02-12T06:00', overwrite=True)
    with _grid(tmp_path / 'g.cfdb', WRF_FILE_1) as g, cfdb.open_dataset(str(p)) as f:
        got = _squeeze(f['air_temperature_2m'][0, :, 0, :, :].data)
        exp = _squeeze(g['air_temperature_2m'][:, 0, :, :].data)
        np.testing.assert_array_equal(got[6:], exp[6:])
        np.testing.assert_array_equal(got[:6], exp[:6])  # untouched leads keep the first ingest's values


def test_two_segments_one_init_written_once(tmp_path):
    p = tmp_path / 'f.cfdb'
    result = _forecast(p, [WRF_FILE_1, WRF_FILE_2])
    assert result['n_leads'] == 48
    assert result['chunk_writes'] == 4  # per-file source blocks, still one write per chunk-row
    with cfdb.open_dataset(str(p)) as f:
        np.testing.assert_array_equal(f[fc.LEAD].data, np.arange(24, 72))
        t2 = _squeeze(f['air_temperature_2m'][0, :, 0, :, :].data)
        assert np.isfinite(t2).all()


def test_second_init_appends_with_autofill(tmp_path):
    p = tmp_path / 'f.cfdb'
    _forecast(p, WRF_FILE_1, forecast_reference_time='2023-02-12T00')
    result = _forecast(p, WRF_FILE_2, forecast_reference_time='2023-02-13T00')
    assert result == {
        'init': '2023-02-13T00:00',
        'init_index': 4,
        'status': 'new',
        'autofilled': 3,
        'n_leads': 24,
        'variables': VARS,
        'chunk_writes': 4,
        'complete': True,
    }
    with cfdb.open_dataset(str(p)) as f:
        assert len(f[fc.FRT].data) == 5
        assert fc.complete_inits(f) == ['2023-02-12T00:00', '2023-02-13T00:00']
        assert np.isnan(_squeeze(f['surface_pressure'][2, :, 0, :, :].data)).all()  # auto-filled slot
        assert np.isfinite(_squeeze(f['surface_pressure'][4, :, 0, :, :].data)).all()
    # a complete init is immutable without overwrite
    with pytest.raises(ValueError, match='immutable'):
        _forecast(p, WRF_FILE_2, forecast_reference_time='2023-02-13T00')
    assert _forecast(p, WRF_FILE_2, forecast_reference_time='2023-02-13T00', overwrite=True)['status'] == 'overwrite'


def test_open_handle_target_is_not_closed(tmp_path):
    p = tmp_path / 'f.cfdb'
    _forecast(p, WRF_FILE_1, forecast_reference_time='2023-02-12T00')
    ds = cfdb.open_dataset(str(p), 'w')
    _forecast(ds, WRF_FILE_2, forecast_reference_time='2023-02-13T00', forecast_step_minutes=1440)
    # still open; the 1440-min step was ignored because the axis (step 360) already exists -> 3 auto-filled slots
    assert len(ds[fc.FRT].data) == 5
    ds.close()


def test_mismatched_grid_and_wrong_type_refused(tmp_path):
    p = tmp_path / 'f.cfdb'
    _forecast(p, WRF_FILE_1, forecast_reference_time='2023-02-12T00')
    with pytest.raises(ValueError, match='does not match'):
        WrfIngest(WRF_FILE_2).convert(
            p,
            variables=VARS,
            target_levels=LEVELS,
            vertical_coord='pressure',
            dataset_type='grid_forecast',
            forecast_reference_time='2023-02-13T00',
            bbox=(168.0, -45.0, 172.0, -41.0),
        )
    g = tmp_path / 'g.cfdb'
    _grid(g, WRF_FILE_1).close()
    with pytest.raises(ValueError, match="expected 'grid_forecast'"):
        _forecast(g, WRF_FILE_2, forecast_reference_time='2023-02-13T00')


def test_accumulation_lead0_is_nan(tmp_path):
    p = tmp_path / 'f.cfdb'
    WrfIngest(WRF_FILE_1).convert(
        p, variables=['RAIN'], dataset_type='grid_forecast', forecast_reference_time='2023-02-12T00'
    )
    with cfdb.open_dataset(str(p)) as f:
        rain = _squeeze(f['precipitation'][0, :, 0, :, :].data)
        assert np.isnan(rain[0]).all()
        assert np.isfinite(rain[1:]).all() and (rain[1:] >= 0).all()


def test_irregular_leads_refused(tmp_path):
    with pytest.raises(ValueError, match='before the forecast_reference_time'):
        _forecast(tmp_path / 'f.cfdb', WRF_FILE_1, forecast_reference_time='2023-02-12T12')

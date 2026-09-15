"""cfdb -> WPS intermediate export, for grid (WRF) and grid_forecast (IFS) datasets."""

import os

import cfdb
import numpy as np
import pytest
from cfdb import dtypes

from cfdb_ingest import thermo
from cfdb_ingest.cfdb_to_int import convert_cfdb_to_int
from cfdb_ingest.cli import WPS_PRESET_VARS
from cfdb_ingest.tests.conftest import WRF_FILE_1
from cfdb_ingest.tests.wps_int_reader import fields_by_name, read_int_file
from cfdb_ingest.wrf import WrfIngest

LEVELS = [100000.0, 85000.0, 50000.0]
MISSING = -1.0e30


def _sq(a):
    return np.squeeze(np.asarray(a, dtype='float32'))


# ---------------------------------------------------------------- grid (WRF) round trip


@pytest.fixture(scope='module')
def wrf_wps_cfdb(tmp_path_factory):
    path = tmp_path_factory.mktemp('wps') / 'w.cfdb'
    WrfIngest(WRF_FILE_1).convert(
        path,
        variables=WPS_PRESET_VARS,
        target_levels=LEVELS,
        vertical_coord='pressure',
        start_date='2023-02-12T00:00',
        end_date='2023-02-12T06:00',
    )
    return path


def test_grid_export_writes_every_wps_field(tmp_path, wrf_wps_cfdb, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = convert_cfdb_to_int(
        wrf_wps_cfdb, output_prefix='WRF', start_date='2023-02-12', end_date='2023-02-12T06', hour_interval=6
    )
    assert sorted(p.name for p in written) == ['WRF:2023-02-12_00', 'WRF:2023-02-12_06']
    recs = read_int_file(tmp_path / 'WRF:2023-02-12_06')
    by = fields_by_name(recs)
    for wps in ('TT', 'UU', 'VV', 'HGT', 'RH', 'SPECHUMD'):  # GHT is renamed HGT by the writer
        for p in LEVELS:
            assert (wps, p) in by, f'{wps} @ {p}'
    for wps in (
        'PSFC',
        'SKINTEMP',
        'TT',
        'UU',
        'VV',
        'DEWPT',
        'RH',
        'LANDSEA',
        'SST',
        'SEAICE',
        'SNOWH',
        'SNOW',
        'SM000010',
        'SM010040',
        'SM040100',
        'SM100200',
        'ST000010',
        'ST010040',
        'ST040100',
        'ST100200',
    ):
        assert (wps, 200100.0) in by, wps
    assert ('PMSL', 201300.0) in by and ('SOILHGT', 200100.0) in by
    assert recs[0]['hdate'] == '2023-02-12_06:00:00'
    assert recs[0]['proj']['iproj'] == 3 and abs(recs[0]['proj']['dx'] - 27.0) < 1e-3  # Lambert, km
    assert recs[0]['ny'] == 111 and recs[0]['nx'] == 99

    with cfdb.open_dataset(str(wrf_wps_cfdb)) as ds:
        t_idx = int(np.where(np.asarray(ds['time'].data) == np.datetime64('2023-02-12T06', 'm'))[0][0])
        # values round-trip within float32
        np.testing.assert_allclose(by[('TT', 85000.0)], _sq(ds['air_temperature'][t_idx, 1, :, :].data), rtol=1e-6)
        np.testing.assert_allclose(by[('PSFC', 200100.0)], _sq(ds['surface_pressure'][t_idx, 0, :, :].data), rtol=1e-6)
        np.testing.assert_allclose(
            by[('SKINTEMP', 200100.0)], _sq(ds['soil_temperature'][t_idx, 0, :, :].data), rtol=1e-6
        )
        np.testing.assert_allclose(by[('SM010040', 200100.0)], _sq(ds['soil_moisture'][t_idx, 1, :, :].data), rtol=1e-6)
        # RH: fraction in cfdb, percent in WPS
        rh_cfdb = _sq(ds['relative_humidity'][t_idx, 1, :, :].data)
        np.testing.assert_allclose(by[('RH', 85000.0)], rh_cfdb * 100.0, rtol=1e-5)
        assert 1.0 < np.nanmax(by[('RH', 85000.0)]) <= 100.0
        rh2 = by[('RH', 200100.0)]
        assert 1.0 < rh2.max() <= 100.0


def test_grid_export_valid_time_filter(tmp_path, wrf_wps_cfdb, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = convert_cfdb_to_int(wrf_wps_cfdb, start_date='2023-02-12T01', end_date='2023-02-12T05', hour_interval=2)
    assert sorted(p.name for p in written) == ['WRF:2023-02-12_01', 'WRF:2023-02-12_03', 'WRF:2023-02-12_05']
    with pytest.raises(ValueError, match='only applies to grid_forecast'):
        convert_cfdb_to_int(wrf_wps_cfdb, init='2023-02-12T00')


# ---------------------------------------------------------------- grid_forecast (IFS)

pytest.importorskip('eccodes')
from cfdb_ingest.ifs import IFS_WPS_PRESET_KEYS, IfsIngest  # noqa: E402

BBOX = (160.0, -50.0, 190.0, -30.0)


@pytest.fixture(scope='module')
def ifs_cfdb(tmp_path_factory, ifs_cycle_dir, ifs_cycle_b):
    path = tmp_path_factory.mktemp('ifs') / 'ifs.cfdb'
    keys = IFS_WPS_PRESET_KEYS + ['U100', 'V100', 'TP']  # the extras bundle: 100 m winds coexist with 10 m
    IfsIngest(ifs_cycle_dir).convert(path, bbox=BBOX, variables=keys)
    IfsIngest(ifs_cycle_b).convert(path, bbox=BBOX, variables=keys)
    return path


def test_forecast_export_one_init(tmp_path, ifs_cfdb, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'wps'
    out.mkdir()
    written = convert_cfdb_to_int(ifs_cfdb, output_prefix=str(out / 'IFS'), init='2026-09-13T12', hour_interval=3)
    assert sorted(p.name for p in written) == ['IFS:2026-09-13_12', 'IFS:2026-09-13_15', 'IFS:2026-09-13_18']
    recs = read_int_file(out / 'IFS:2026-09-13_15')
    by = fields_by_name(recs)
    assert recs[0]['hdate'] == '2026-09-13_15:00:00'
    assert recs[0]['proj']['iproj'] == 0
    assert recs[0]['proj']['startlat'] == -50.0 and recs[0]['proj']['startlon'] == 160.0
    assert recs[0]['proj']['deltalat'] == 5.0 and recs[0]['proj']['deltalon'] == 5.0
    for wps in ('TT', 'UU', 'VV', 'HGT', 'RH', 'SPECHUMD'):
        for p in (50000.0, 85000.0, 100000.0):
            assert (wps, p) in by
    for wps in (
        'PSFC',
        'SKINTEMP',
        'TT',
        'UU',
        'VV',
        'DEWPT',
        'RH',
        'LANDSEA',
        'SST',
        'SEAICE',
        'SNOWH',
        'SNOW',
        'SM000007',
        'SM007028',
        'SM028100',
        'SM100289',
        'ST000007',
        'ST007028',
        'ST028100',
        'ST100289',
    ):
        assert (wps, 200100.0) in by, wps
    assert ('SOILHGT', 200100.0) in by and ('PMSL', 201300.0) in by
    with cfdb.open_dataset(str(ifs_cfdb)) as ds:
        # init 12z is index 2 (auto-filled 06z between); lead 3 h is index 1
        np.testing.assert_allclose(by[('TT', 100000.0)], _sq(ds['air_temperature'][2, 1, 2, :, :].data), rtol=1e-6)
        np.testing.assert_allclose(
            by[('SKINTEMP', 200100.0)], _sq(ds['skin_temperature'][2, 1, 0, :, :].data), rtol=1e-6
        )
        np.testing.assert_allclose(
            by[('RH', 200100.0)], _sq(ds['relative_humidity_2m'][2, 1, 0, :, :].data) * 100.0, rtol=1e-5
        )
        np.testing.assert_allclose(by[('SOILHGT', 200100.0)], _sq(ds['terrain_height'][2, 1, 0, :, :].data), rtol=1e-6)
        assert (by[('SOILHGT', 200100.0)] < 0).any()  # negative orography reaches WPS
        lsm = by[('LANDSEA', 200100.0)]
        assert set(np.unique(lsm)) <= {0.0, 1.0}
        sst = by[('SST', 200100.0)]
        assert (sst[lsm == 1.0] == MISSING).all() and (sst[lsm == 0.0] > 200.0).all()
        # UU/VV at the surface are the 10 m winds, never the 100 m ones that share the canonical name
        assert 'u_wind_100m' in ds.data_var_names
        np.testing.assert_allclose(by[('UU', 200100.0)], _sq(ds['u_wind_10m'][2, 1, 0, :, :].data), rtol=1e-6)
        assert not np.allclose(by[('UU', 200100.0)], _sq(ds['u_wind_100m'][2, 1, 0, :, :].data))
        assert sum(1 for r in recs if r['field'] == 'UU' and r['xlvl'] == 200100.0) == 1


def test_forecast_export_rh2_from_dewpoint(tmp_path, ifs_cycle_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'no_rh2.cfdb'
    keys = [k for k in IFS_WPS_PRESET_KEYS if k != 'RH2']
    IfsIngest(ifs_cycle_dir).convert(path, bbox=BBOX, variables=keys)
    convert_cfdb_to_int(path, output_prefix='IFS', init='2026-09-13T00', hour_interval=3)
    by = fields_by_name(read_int_file(tmp_path / 'IFS:2026-09-13_03'))
    with cfdb.open_dataset(str(path)) as ds:
        t = _sq(ds['air_temperature_2m'][0, 1, 0, :, :].data)
        td = _sq(ds['dew_point_temperature'][0, 1, 0, :, :].data)
    np.testing.assert_allclose(by[('RH', 200100.0)], thermo.rh_from_t_td(t, td) * 100.0, rtol=1e-5)


def test_forecast_export_refusals(tmp_path, ifs_cfdb, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='init= is required'):
        convert_cfdb_to_int(ifs_cfdb)
    with pytest.raises(ValueError, match='not in the dataset'):
        convert_cfdb_to_int(ifs_cfdb, init='2026-09-14T00')
    # an init that is on the axis but not complete (the auto-filled 06z slot) is refused before any file exists
    with pytest.raises(ValueError, match='not marked complete'):
        convert_cfdb_to_int(ifs_cfdb, init='2026-09-13T06')
    assert not [n for n in os.listdir(tmp_path) if n.startswith('WRF:')]


def test_forecast_export_valid_time_filter(tmp_path, ifs_cfdb, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = convert_cfdb_to_int(ifs_cfdb, init='2026-09-13T00', start_date='2026-09-13T03', hour_interval=3)
    assert sorted(p.name for p in written) == ['WRF:2026-09-13_03', 'WRF:2026-09-13_06']


# ---------------------------------------------------------------- edge layouts


def test_legacy_3d_surface_and_duplicate_skintemp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'legacy.cfdb'
    with cfdb.open_dataset(str(path), 'n') as ds:
        ds.create.coord.time(data=np.array(['2026-01-01T00', '2026-01-01T06'], dtype='datetime64[m]'), step=360)
        ds.create.coord.lat(data=np.arange(-45.0, -42.0, 1.0))
        ds.create.coord.lon(data=np.arange(170.0, 174.0, 1.0))
        ds.create.crs.from_user_input(4326, x_coord='longitude', y_coord='latitude')
        psfc = ds.create.data_var.generic(
            'surface_pressure', ('time', 'latitude', 'longitude'), dtype=dtypes.dtype('float32')
        )
        block = np.full((2, 3, 4), 101000.0, dtype='float32')
        block[1, 0, 0] = np.nan
        psfc[:] = block
    written = convert_cfdb_to_int(path, hour_interval=6)
    assert len(written) == 2
    by = fields_by_name(read_int_file(tmp_path / 'WRF:2026-01-01_06'))
    out = by[('PSFC', 200100.0)]
    assert out.shape == (3, 4) and out[0, 0] == MISSING and (out.ravel()[1:] == 101000.0).all()  # NaN -> WPS sentinel
    # two candidates for one WPS field are refused, not silently first-matched
    with cfdb.open_dataset(str(path), 'w') as ds:
        for name in ('skin_temperature', 'soil_temperature'):
            dv = ds.create.data_var.generic(name, ('time', 'latitude', 'longitude'), dtype=dtypes.dtype('float32'))
            dv[:] = np.full((2, 3, 4), 290.0, dtype='float32')
    with pytest.raises(ValueError, match='SKINTEMP'):
        convert_cfdb_to_int(path, hour_interval=6)
    # one quantity at two heights, neither the WPS surface height, is a real ambiguity
    with cfdb.open_dataset(str(path), 'w') as ds:
        for name in ('air_temperature_50m', 'air_temperature_100m'):
            dv = ds.create.data_var.generic(name, ('time', 'latitude', 'longitude'), dtype=dtypes.dtype('float32'))
            dv[:] = np.full((2, 3, 4), 280.0, dtype='float32')
    with pytest.raises(ValueError, match='several heights'):
        convert_cfdb_to_int(path, hour_interval=6)

"""Tests for the source-independent helpers in cfdb_ingest.base (grouping, naming, data-var creation)."""

import cfdb
import numpy as np
import pytest

from cfdb_ingest import base
from cfdb_ingest.tests.conftest import WRF_FILE_1
from cfdb_ingest.wrf import WrfIngest

MAPPING = {
    'T': {'cfdb_name': 'air_temp', 'source_vars': ['T'], 'transform': None, 'height': 'levels'},
    'T2': {'cfdb_name': 'air_temp', 'source_vars': ['T2'], 'transform': None, 'height': 2.0},
    'U10': {'cfdb_name': 'u_wind', 'source_vars': ['U10'], 'transform': None, 'height': 10.0},
    'U100': {'cfdb_name': 'u_wind', 'source_vars': ['U100'], 'transform': None, 'height': 100.0},
    'PSFC': {'cfdb_name': 'surface_pressure', 'source_vars': ['PSFC'], 'transform': None, 'height': 0.0},
    'SMOIS': {'cfdb_name': 'soil_moisture', 'source_vars': ['SMOIS'], 'transform': None, 'height': 'soil'},
}


def test_group_variables_by_name_and_height():
    level, surface, soil, region = base.group_variables(MAPPING, list(MAPPING))
    assert level == {'air_temp': ['T']}
    assert soil == {'soil_moisture': ['SMOIS']}
    assert region == {}
    # conflict with a level field -> full name + suffix; two heights of one name -> both suffixed;
    # a lone surface field keeps its bare (short) name
    assert surface == {
        'air_temperature_2m': (2.0, ['T2']),
        'u_wind_10m': (10.0, ['U10']),
        'u_wind_100m': (100.0, ['U100']),
        'surface_pressure': (0.0, ['PSFC']),
    }


def test_resolve_variable_keys():
    assert base.resolve_variable_keys(MAPPING, None) == list(MAPPING)
    assert base.resolve_variable_keys(MAPPING, ['air_temp']) == ['T', 'T2']
    assert base.resolve_variable_keys(MAPPING, ['U100', 'PSFC']) == ['U100', 'PSFC']
    with pytest.raises(ValueError, match='Unknown variable'):
        base.resolve_variable_keys(MAPPING, ['nope'])


def test_split_height_suffix():
    assert base.split_height_suffix('air_temperature_2m') == ('air_temperature', '_2m')
    assert base.split_height_suffix('u_wind_100m') == ('u_wind', '_100m')
    assert base.split_height_suffix('snow_depth') == ('snow_depth', '')


def _grid(path):
    ds = cfdb.open_dataset(str(path), 'n')
    ds.create.coord.time(data=np.array(['2026-01-01T00', '2026-01-01T01'], dtype='datetime64[m]'), step=60)
    ds.create.coord.generic('y', data=np.arange(3.0), axis='y', step=True)
    ds.create.coord.generic('x', data=np.arange(4.0), axis='x', step=True)
    ds.create.coord.generic('height_2m', data=np.array([2.0]), axis=None)
    return ds


def test_create_cfdb_data_var_names_and_templates(tmp_path):
    coords = ('time', 'height_2m', 'y', 'x')
    with _grid(tmp_path / 'g.cfdb') as ds:
        dv = base.create_cfdb_data_var(ds, 'air_temperature_2m', coords, (1, 1, 3, 4))
        assert dv.name == 'air_temperature_2m'
        assert dv.attrs['units'] == 'K' and dv.attrs['standard_name'] == 'air_temperature'
        dv = base.create_cfdb_data_var(ds, 'u_wind_100m', coords, (1, 1, 3, 4))
        assert dv.name == 'u_wind_100m' and dv.attrs['standard_name'] == 'eastward_wind'
        # relative humidity: fraction at 0.001 resolution, units '1' (cfdb-vars >= 0.2.4 template)
        rh = base.create_cfdb_data_var(ds, 'relative_humidity_2m', coords, (1, 1, 3, 4))
        rh[0, 0, :, :] = np.full((3, 4), 0.123, dtype='float32')
        assert rh.attrs['units'] == '1'
        np.testing.assert_allclose(np.asarray(rh[0, 0, :, :].data), 0.123, atol=5e-4)
        # dtype override keeps attrs; unknown names get generic float32 plus caller attrs
        t = base.create_cfdb_data_var(ds, 'skin_temp', ('time', 'y', 'x'), (1, 3, 4), dtype='float32')
        assert t.name == 'skin_temperature' and t.attrs['units'] == 'K'
        g = base.create_cfdb_data_var(
            ds,
            'wind_gust',
            ('time', 'y', 'x'),
            (1, 3, 4),
        )
        assert g.name == 'wind_gust' and g.attrs['standard_name'] == 'wind_speed_of_gust'  # registry template
        u = base.create_cfdb_data_var(ds, 'unknown_thing', ('time', 'y', 'x'), (1, 3, 4), attrs={'units': 'x'})
        assert u.attrs['units'] == 'x'  # caller attrs for names cfdb-vars does not know
        # append mode: an existing name is reused, a coordinate mismatch is refused
        assert base.create_cfdb_data_var(ds, 'air_temp_2m', coords, (1, 1, 3, 4)).name == 'air_temperature_2m'
        with pytest.raises(ValueError, match='coords'):
            base.create_cfdb_data_var(ds, 'air_temperature_2m', ('time', 'y', 'x'), (1, 3, 4))


def test_wrf_surface_names_carry_attrs(tmp_path):
    """The suffixed surface variables are stored under full names with template attrs (regression)."""
    out = tmp_path / 'w.cfdb'
    WrfIngest(WRF_FILE_1).convert(
        out,
        variables=['T2', 'T', 'U10', 'U', 'RH2', 'RH'],
        target_levels=[100000.0, 85000.0],
        vertical_coord='pressure',
        start_date='2023-02-12T00:00',
        end_date='2023-02-12T02:00',
    )
    with cfdb.open_dataset(str(out)) as ds:
        names = set(ds.data_var_names)
        assert {'air_temperature', 'air_temperature_2m', 'u_wind_10m', 'relative_humidity_2m'} <= names
        assert 'air_temp_2m' not in names
        assert ds['air_temperature_2m'].attrs['units'] == 'K'
        assert ds['u_wind_10m'].attrs['standard_name'] == 'eastward_wind'
        rh = np.asarray(ds['relative_humidity_2m'][0, 0, :, :].data)
        finite = rh[np.isfinite(rh)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0
        assert len(np.unique(np.round(finite, 3))) > 11  # not quantised to 0.1 any more

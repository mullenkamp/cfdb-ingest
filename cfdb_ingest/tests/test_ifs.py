"""IfsIngest: ECMWF IFS open-data GRIB2 -> cfdb grid_forecast, against the synthetic cycle fixture."""

import sys

import cfdb
import numpy as np
import pytest

from cfdb_ingest import forecast as fc
from cfdb_ingest import thermo
from cfdb_ingest import ifs_synthetic as syn

pytest.importorskip('eccodes')

from cfdb_ingest.ifs import (  # noqa: E402
    IFS_SOIL_DEPTHS,
    IFS_VARIABLE_MAPPING,
    IFS_WPS_PRESET_KEYS,
    IfsIngest,
    required_messages,
)

BBOX = (160.0, -50.0, 190.0, -30.0)  # crosses the dateline seam of the source grid
INIT_A = np.datetime64('2026-09-13T00', 'm')


def _sq(a):
    return np.squeeze(np.asarray(a))


def _grids(ds):
    lon = np.asarray(ds['longitude'].data)
    lat = np.asarray(ds['latitude'].data)
    return np.meshgrid(lon, lat)


# ---------------------------------------------------------------- metadata


def test_import_hint_without_eccodes(monkeypatch, ifs_cycle_dir):
    monkeypatch.setitem(sys.modules, 'eccodes', None)
    with pytest.raises(ImportError, match=r"cfdb-ingest\[ifs\]"):
        IfsIngest(ifs_cycle_dir)


def test_index_metadata(ifs_cycle_dir):
    ing = IfsIngest(ifs_cycle_dir)
    assert ing.init == INIT_A
    np.testing.assert_array_equal(ing.leads, [0, 3, 6])
    np.testing.assert_array_equal(ing.pressure_levels, [50000.0, 85000.0, 100000.0])
    assert set(IFS_VARIABLE_MAPPING) <= set(ing.variables)  # every mapping row is available
    assert ing.resolve_variables(['2t', 'air_temp']) == ['T2', 'T']  # request order, dedup
    assert ing._lon_axis[0] == 0.0 and np.all(np.diff(ing._lon_axis) > 0)  # seam rolled away


def test_mixed_inits_refused(ifs_cycle_dir, ifs_cycle_b):
    with pytest.raises(ValueError, match='mix forecast inits'):
        IfsIngest([ifs_cycle_dir, ifs_cycle_b])


# ---------------------------------------------------------------- conversion


@pytest.fixture(scope='module')
def converted(tmp_path_factory, ifs_cycle_dir):
    path = tmp_path_factory.mktemp('out') / 'ifs.cfdb'
    result = IfsIngest(ifs_cycle_dir).convert(path, bbox=BBOX)
    return path, result


def test_convert_layout(converted):
    path, result = converted
    assert result['status'] == 'new' and result['n_leads'] == 3 and result['init'] == '2026-09-13T00:00'
    with cfdb.open_dataset(str(path)) as ds:
        assert ds.dataset_type == 'grid_forecast'
        assert ds[fc.FRT].step == 360 and ds[fc.LEAD].attrs['units'] == 'h'
        np.testing.assert_array_equal(ds['longitude'].data, np.arange(160.0, 191.0, 5.0))
        np.testing.assert_array_equal(ds['latitude'].data, np.arange(-50.0, -29.0, 5.0))
        assert ds['air_temperature'].coord_names == (fc.FRT, fc.LEAD, 'pressure', 'latitude', 'longitude')
        assert ds['air_temperature_2m'].coord_names == (fc.FRT, fc.LEAD, 'height_2m', 'latitude', 'longitude')
        assert ds['u_wind_10m'].coord_names[2] == 'height_10m' and ds['u_wind_100m'].coord_names[2] == 'height_100m'
        assert ds['soil_moisture'].coord_names == (fc.FRT, fc.LEAD, 'depth', 'latitude', 'longitude')
        np.testing.assert_allclose(ds['depth'].data, IFS_SOIL_DEPTHS, atol=1e-3)  # packed depth coord
        assert ds['air_temperature'].chunk_shape == (1, 3, 1, 5, 7)
        # packed templates by default (0.01 K, finer than the GRIB's 0.03 K step); float32 only where the template would misrepresent
        assert ds['air_temperature'].dtype.precision == 2 and ds['air_temperature'].attrs['units'] == 'K'
        assert ds['air_temperature'].dtype.dtype_encoded == np.dtype('uint16')
        assert ds['cape'].dtype.precision is None and ds['land_sea_mask'].dtype.precision is None
        assert ds['soil_moisture'].dtype.precision == 3
        assert ds['relative_humidity'].attrs['units'] == '1'
        assert ds['wind_gust'].attrs['standard_name'] == 'wind_speed_of_gust'
        assert fc.complete_inits(ds) == ['2026-09-13T00:00']
        assert 'CC-BY-4.0' in ds.attrs['source']
        assert ds.crs.to_epsg() == 4326
        # one chunk-row per (variable, level): 6 level vars x 3 levels + 23 surface + 2 soil x 4 layers
        assert result['chunk_writes'] == 6 * 3 + 23 + 2 * 4


def test_values_match_closed_forms(converted):
    path, _ = converted
    with cfdb.open_dataset(str(path)) as ds:
        lon2, lat2 = _grids(ds)
        for i, step in enumerate((0, 3, 6)):
            # pressure levels are stored ascending: index 2 is 1000 hPa
            t1000 = _sq(ds['air_temperature'][0, i, 2, :, :].data)
            np.testing.assert_allclose(t1000, syn.t_pl(lat2, lon2, 1000, step), atol=0.01)
            gh500 = _sq(ds['geopotential_height'][0, i, 0, :, :].data)
            np.testing.assert_allclose(gh500, syn.gh_pl(lat2, lon2, 500, step), atol=0.2)  # gpm passthrough, no /g
            q850 = _sq(ds['specific_humidity'][0, i, 1, :, :].data)
            np.testing.assert_allclose(q850, syn.q_pl(lat2, lon2, 850, step), rtol=1e-3)
            rh850 = _sq(ds['relative_humidity'][0, i, 1, :, :].data)
            exp = thermo.rh_from_q_t_p(syn.q_pl(lat2, lon2, 850, step), syn.t_pl(lat2, lon2, 850, step), 85000.0)
            np.testing.assert_allclose(rh850, exp, atol=2e-3)
            assert rh850.min() >= 0.0 and rh850.max() <= 1.0
            t2 = _sq(ds['air_temperature_2m'][0, i, 0, :, :].data)
            np.testing.assert_allclose(t2, syn.t2(lat2, lon2, step), atol=0.01)
            rh2 = _sq(ds['relative_humidity_2m'][0, i, 0, :, :].data)
            np.testing.assert_allclose(
                rh2, thermo.rh_from_t_td(syn.t2(lat2, lon2, step), syn.td2(lat2, lon2, step)), atol=2e-3
            )
            u100 = _sq(ds['u_wind_100m'][0, i, 0, :, :].data)
            np.testing.assert_allclose(u100, syn.u100(lat2, lon2, step), atol=0.01)
            # SST: skt over water, missing over land
            sst = _sq(ds['sea_surface_temp'][0, i, 0, :, :].data)
            land = syn.is_land(lat2, lon2)
            assert land.any() and (~land).any()
            assert np.isnan(sst[land]).all()
            np.testing.assert_allclose(sst[~land], syn.skt(lat2, lon2, step)[~land], atol=0.01)
            np.testing.assert_array_equal(_sq(ds['land_sea_mask'][0, i, 0, :, :].data), land.astype('float32'))
            # sea ice: bitmap-masked land -> 0, no ice in this box -> all 0
            np.testing.assert_array_equal(_sq(ds['sea_ice'][0, i, 0, :, :].data), 0.0)
            # snow: none in the box (lat -50..-30) -> SNOW 0 and SNOWH 0
            np.testing.assert_array_equal(_sq(ds['snow_water_equiv'][0, i, 0, :, :].data), 0.0)
            np.testing.assert_array_equal(_sq(ds['snow_depth'][0, i, 0, :, :].data), 0.0)
            # orography from the 0 h message, broadcast to every lead, in metres
            terrain = _sq(ds['terrain_height'][0, i, 0, :, :].data)
            np.testing.assert_allclose(terrain, syn.z_sfc(lat2, lon2, 0) / syn.G, atol=0.05)
            assert (terrain[~land] < 0).all()  # negative orography survives
            # soil layer 2 (index 1)
            np.testing.assert_allclose(
                _sq(ds['soil_layer_temp'][0, i, 1, :, :].data), syn.sot(lat2, lon2, 2, step), atol=0.01
            )
            np.testing.assert_allclose(
                _sq(ds['soil_moisture'][0, i, 1, :, :].data), syn.vsw(lat2, lon2, 2, step), atol=1e-3
            )
        # accumulations -> increments along the lead axis, lead 0 missing
        tp = _sq(ds['precipitation'][0, :, 0, :, :].data)
        assert np.isnan(tp[0]).all()
        np.testing.assert_allclose(tp[1], (syn.tp(lat2, lon2, 3) - syn.tp(lat2, lon2, 0)) * 1000.0, atol=0.02)
        np.testing.assert_allclose(tp[2], (syn.tp(lat2, lon2, 6) - syn.tp(lat2, lon2, 3)) * 1000.0, atol=0.02)
        sw = _sq(ds['shortwave_radiation'][0, :, 0, :, :].data)
        assert np.isnan(sw[0]).all()
        np.testing.assert_allclose(sw[1], (syn.ssrd(lat2, lon2, 3) - syn.ssrd(lat2, lon2, 0)) / (3 * 3600.0), rtol=1e-3)


def test_seam_bbox_spellings_agree(tmp_path, ifs_cycle_dir):
    a = tmp_path / 'a.cfdb'
    b = tmp_path / 'b.cfdb'
    IfsIngest(ifs_cycle_dir).convert(a, bbox=BBOX, variables=['T2', 'T'])
    IfsIngest(ifs_cycle_dir).convert(b, bbox=(160.0, -50.0, -170.0, -30.0), variables=['T2', 'T'])
    with cfdb.open_dataset(str(a)) as da, cfdb.open_dataset(str(b)) as db:
        np.testing.assert_array_equal(da['longitude'].data, db['longitude'].data)
        np.testing.assert_array_equal(
            _sq(da['air_temperature_2m'][0, :, 0, :, :].data), _sq(db['air_temperature_2m'][0, :, 0, :, :].data)
        )


def test_global_without_bbox(tmp_path, ifs_cycle_dir):
    p = tmp_path / 'g.cfdb'
    IfsIngest(ifs_cycle_dir).convert(p, variables=['SP'])
    with cfdb.open_dataset(str(p)) as ds:
        np.testing.assert_array_equal(ds['longitude'].data, np.arange(0.0, 360.0, 5.0))
        np.testing.assert_array_equal(ds['latitude'].data, np.arange(-90.0, 91.0, 5.0))
        lon2, lat2 = _grids(ds)
        np.testing.assert_allclose(_sq(ds['surface_pressure'][0, 0, 0, :, :].data), syn.sp(lat2, lon2, 0), atol=2.0)


def test_target_levels_and_max_lead(tmp_path, ifs_cycle_dir):
    p = tmp_path / 'l.cfdb'
    r = IfsIngest(ifs_cycle_dir).convert(
        p, bbox=BBOX, variables=['T'], target_levels=[100000.0, 50000.0], max_lead_hours=3
    )
    assert r['n_leads'] == 2
    with cfdb.open_dataset(str(p)) as ds:
        np.testing.assert_array_equal(ds['pressure'].data, [50000.0, 100000.0])
        np.testing.assert_array_equal(ds[fc.LEAD].data, [0, 3])
        lon2, lat2 = _grids(ds)
        np.testing.assert_allclose(
            _sq(ds['air_temperature'][0, 1, 1, :, :].data), syn.t_pl(lat2, lon2, 1000, 3), atol=0.01
        )
    with pytest.raises(ValueError, match='not native IFS levels'):
        IfsIngest(ifs_cycle_dir).convert(tmp_path / 'x.cfdb', variables=['T'], target_levels=[92500.0])


def test_irregular_leads_refused(tmp_path, ifs_cycle_irregular):
    with pytest.raises(ValueError, match='not regularly spaced'):
        IfsIngest(ifs_cycle_irregular).convert(tmp_path / 'i.cfdb', bbox=BBOX, variables=['SP'])
    assert (
        IfsIngest(ifs_cycle_irregular).convert(tmp_path / 'i.cfdb', bbox=BBOX, variables=['SP'], max_lead_hours=6)[
            'n_leads'
        ]
        == 3
    )


def test_wps_preset_keys_are_available(ifs_cycle_dir):
    assert set(IFS_WPS_PRESET_KEYS) <= set(IfsIngest(ifs_cycle_dir).variables)


def test_required_messages_matches_the_ingest_needs(ifs_cycle_dir):
    """The download manifest and the ingest resolve the same sources; z is 0 h-only, nothing else is."""
    req = required_messages(IFS_WPS_PRESET_KEYS)
    assert {(c, s) for c, s, _ in req} == {
        *(('pl', s) for s in ('t', 'u', 'v', 'q', 'gh')),
        *(('sfc', s) for s in ('2t', '2d', '10u', '10v', 'msl', 'sp', 'skt', 'lsm', 'sithick', 'sd', 'rsn', 'z')),
        ('soil', 'sot'),
        ('soil', 'vsw'),
    }
    assert {s for c, s, inv in req if inv} == {'z'}
    # every source the ingest actually reads for the preset is in the set
    ing = IfsIngest(ifs_cycle_dir)
    read = {
        (ing.variables[k]['height'], sv)
        for k in ing.resolve_variables(IFS_WPS_PRESET_KEYS)
        for sv in ing.variables[k]['source_vars']
    }
    cat = {'levels': 'pl', 'soil': 'soil'}
    assert {(cat.get(h, 'sfc'), sv) for h, sv in read} == {(c, s) for c, s, _ in req}
    # extras add exactly the surface bundle; a cfdb short name fans out to every height
    assert {s for c, s, _ in required_messages(IFS_WPS_PRESET_KEYS + ['TP', 'u_wind'])} - {s for c, s, _ in req} == {
        'tp',
        '100u',
    }
    assert required_messages(None) >= req


# ---------------------------------------------------------------- lifecycle


def test_lifecycle_append_backfill_overwrite_truncate(tmp_path, ifs_cycle_dir, ifs_cycle_b, ifs_cycle_c):
    p = tmp_path / 'life.cfdb'
    variables = ['T2', 'T', 'SP']
    IfsIngest(ifs_cycle_dir).convert(p, bbox=BBOX, variables=variables)
    r = IfsIngest(ifs_cycle_c).convert(p, bbox=BBOX, variables=variables)  # +24 h: three auto-filled slots
    assert (r['status'], r['init_index'], r['autofilled']) == ('new', 4, 3)
    r = IfsIngest(ifs_cycle_b).convert(p, bbox=BBOX, variables=variables)  # back-fill the 12z slot
    assert (r['status'], r['init_index']) == ('backfill', 2)
    with pytest.raises(ValueError, match='immutable'):
        IfsIngest(ifs_cycle_dir).convert(p, bbox=BBOX, variables=variables)
    assert IfsIngest(ifs_cycle_dir).convert(p, bbox=BBOX, variables=variables, overwrite=True)['status'] == 'overwrite'

    with cfdb.open_dataset(str(p), 'w') as ds:
        frt = np.asarray(ds[fc.FRT].data)
        assert len(frt) == 5 and fc.complete_inits(ds) == ['2026-09-13T00:00', '2026-09-13T12:00', '2026-09-14T00:00']
        lon2, lat2 = _grids(ds)
        for idx, step_at_init in ((0, 0), (2, 0), (4, 0)):
            np.testing.assert_allclose(
                _sq(ds['air_temperature_2m'][idx, 0, 0, :, :].data), syn.t2(lat2, lon2, step_at_init), atol=0.01
            )
        assert np.isnan(_sq(ds['air_temperature_2m'][1, :, 0, :, :].data)).all()  # never filled
        # retention: drop everything before 12z, reclaim
        ds[fc.FRT].truncate(start='2026-09-13T12:00')
        assert ds.prune() > 0
        np.testing.assert_array_equal(ds[fc.FRT].data, frt[2:])
        np.testing.assert_allclose(_sq(ds['air_temperature_2m'][0, 0, 0, :, :].data), syn.t2(lat2, lon2, 0), atol=0.01)
        np.testing.assert_allclose(_sq(ds['air_temperature_2m'][2, 2, 0, :, :].data), syn.t2(lat2, lon2, 6), atol=0.01)
        assert ds[fc.FRT].step == 360


def test_open_handle_and_mismatches(tmp_path, ifs_cycle_dir, ifs_cycle_b):
    p = tmp_path / 'h.cfdb'
    IfsIngest(ifs_cycle_dir).convert(p, bbox=BBOX, variables=['SP'])
    ds = cfdb.open_dataset(str(p), 'w')
    r = IfsIngest(ifs_cycle_b).convert(ds, bbox=BBOX, variables=['SP'])
    assert r['status'] == 'new' and len(ds[fc.FRT].data) == 3
    ds.close()  # convert did not close the handle
    with pytest.raises(ValueError, match='does not match'):
        IfsIngest(ifs_cycle_b).convert(p, bbox=(150.0, -50.0, 190.0, -30.0), variables=['SP'])
    g = tmp_path / 'grid.cfdb'
    with cfdb.open_dataset(str(g), 'n') as ds:
        ds.create.coord.time(data=np.array(['2026-01-01'], dtype='datetime64[m]'))
    with pytest.raises(ValueError, match="expected 'grid_forecast'"):
        IfsIngest(ifs_cycle_b).convert(g, bbox=BBOX, variables=['SP'])


def test_packed_range_guard_refuses_a_value_past_the_template(tmp_path, ifs_cycle_dir, monkeypatch):
    """A value above a packed template's max would be stored as MISSING; the ingest must refuse instead."""
    from cfdb import dtypes

    from cfdb_ingest.ifs import check_packed_range, packed_range

    tiny = dtypes.dtype('float32', precision=1, min_value=0, max_value=100)  # 2 m temperature is ~290 K
    assert packed_range(tiny) == (-0.9, 6552.5)  # offset = min_value - 1, code 0 reserved, uint16 codes
    assert packed_range(dtypes.dtype('float32')) is None
    check_packed_range(np.array([np.nan, 1.0, 50.0], dtype='float32'), (0.1, 100.0), 'x')
    with pytest.raises(ValueError, match='exceed the packed template range'):
        check_packed_range(np.array([1.0, 100.1], dtype='float32'), (0.1, 100.0), 'x')
    with pytest.raises(ValueError, match='exceed'):
        check_packed_range(np.array([-0.5, 1.0], dtype='float32'), (0.1, 100.0), 'x')
    monkeypatch.setitem(
        IFS_VARIABLE_MAPPING['T2'], 'dtype', dtypes.dtype('float32', precision=1, min_value=0, max_value=10)
    )
    with pytest.raises(ValueError, match='air_temperature.*exceed'):  # no level T requested -> no _2m suffix
        IfsIngest(ifs_cycle_dir).convert(tmp_path / 'x.cfdb', variables=['T2'], bbox=(160, -50, 190, -30))


def test_soil_moisture_is_clipped_at_zero():
    from cfdb_ingest.ifs import _TRANSFORMS

    out = _TRANSFORMS['clip_nonneg']({'vsw': np.array([-3.6e-12, 0.0, 0.3], dtype='float32')}, None)
    assert out.dtype == np.dtype('float32') and out.tolist() == [0.0, 0.0, pytest.approx(0.3)]

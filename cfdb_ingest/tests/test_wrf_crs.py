"""
WRF CRS and x/y coordinates on the WPS sphere (cfdb-ingest 0.6.0).

The reference is the file's own XLAT/XLONG (computed by WPS geogrid on its 6 370 000 m sphere), never a
CRS built here: a check seeded from the code under test cannot see an error in it. Each test names the
regression it exists to catch.
"""
import shutil

import cfdb
import h5py
import numpy as np
import pyproj
import pytest

from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.wrf import WrfIngest, WPS_EARTH_RADIUS_M
from cfdb_ingest.wrf_synthetic import wps_ijll


def _xlat_xlong(path):
    with h5py.File(path, 'r') as h5:
        return h5['XLAT'][0].astype('float64'), h5['XLONG'][0].astype('float64')


def _max_distance_to_xlat(crs, x, y, xlat, xlong):
    """Largest great-circle distance (m) between the (x, y) lattice taken through ``crs`` and XLAT/XLONG."""
    xx, yy = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(xx, yy)
    return float(np.max(pyproj.Geod(ellps='WGS84').inv(lon, lat, xlong, xlat)[2]))


def _copy(src, tmp_path, name=None):
    dst = tmp_path / (name or src.name)
    shutil.copy(src, dst)
    return dst


class TestWpsSphereCrs:
    def test_lattice_matches_xlat_xlong(self, wrf_file_1):
        """Every cell centre within a few metres of WRF's own XLAT/XLONG.
        Catches: the WGS84 construction of < 0.6.0 (5.4 km on this d01 fixture)."""
        ing = WrfIngest(wrf_file_1)
        xlat, xlong = _xlat_xlong(wrf_file_1)
        err = _max_distance_to_xlat(ing.crs, ing.x, ing.y, xlat, xlong)
        assert err < 10.0, err

    def test_crs_parameters(self, wrf_file_1):
        """Sphere of the WPS radius; Lambert origin at MOAD_CEN_LAT.
        Catches: a CEN_LAT origin, which the XLAT fit alone cannot see (it only shifts y)."""
        ing = WrfIngest(wrf_file_1)
        assert ing.crs.ellipsoid.semi_major_metre == WPS_EARTH_RADIUS_M
        assert ing.crs.ellipsoid.semi_minor_metre == WPS_EARTH_RADIUS_M
        with h5py.File(wrf_file_1, 'r') as h5:
            moad = float(np.asarray(h5.attrs['MOAD_CEN_LAT']).item())
        assert ing.crs.to_cf()['latitude_of_projection_origin'] == pytest.approx(moad)

    def test_moad_origin_not_cen_lat(self, wrf_file_1, tmp_path):
        """With CEN_LAT moved away from MOAD_CEN_LAT, the origin follows MOAD_CEN_LAT.
        Catches: reading CEN_LAT (the fixture is a d01, where the two coincide)."""
        p = _copy(wrf_file_1, tmp_path)
        with h5py.File(p, 'r+') as h5:
            moad = float(np.asarray(h5.attrs['MOAD_CEN_LAT']).item())
            h5.attrs['CEN_LAT'] = np.float32(moad - 1.5)
        ing = WrfIngest(p)
        assert ing.crs.to_cf()['latitude_of_projection_origin'] == pytest.approx(moad)

    def test_wrong_dx_refused(self, wrf_file_1, tmp_path):
        """A DX that does not describe the grid is refused, not written.
        Catches: removing the lattice assertion."""
        p = _copy(wrf_file_1, tmp_path)
        with h5py.File(p, 'r+') as h5:
            h5.attrs['DX'] = np.float32(float(np.asarray(h5.attrs['DX']).item()) * 1.001)
        with pytest.raises(ValueError, match='grid check failed'):
            WrfIngest(p)

    def test_xlat_absent_falls_back(self, wrf_file_1, tmp_path):
        """No XLAT/XLONG: warn, lay the lattice around CEN_LAT/CEN_LON on the sphere."""
        xlat, xlong = _xlat_xlong(wrf_file_1)
        p = _copy(wrf_file_1, tmp_path)
        with h5py.File(p, 'r+') as h5:
            del h5['XLAT']
            del h5['XLONG']
        with pytest.warns(UserWarning, match='XLAT/XLONG absent'):
            ing = WrfIngest(p)
        assert _max_distance_to_xlat(ing.crs, ing.x, ing.y, xlat, xlong) < 25.0

    def test_moving_nest_refused(self, wrf_file_1, tmp_path):
        p = _copy(wrf_file_1, tmp_path)
        with h5py.File(p, 'r+') as h5:
            xlat = h5['XLAT'][:]
            xlat[-1] += 0.1
            h5['XLAT'][:] = xlat
        with pytest.raises(ValueError, match='moving'):
            WrfIngest(p)

    def test_cfdb_round_trip(self, wrf_file_1, cfdb_out):
        """The stored CRS equals the ingest's, and stored x/y through the stored CRS land on XLAT/XLONG.
        Catches: a CRS serialisation that loses the sphere, or a datum shift to EPSG:4326."""
        ing = WrfIngest(wrf_file_1)
        ing.convert(cfdb_out, variables=['T2'])
        xlat, xlong = _xlat_xlong(wrf_file_1)
        with cfdb.open_dataset(cfdb_out) as ds:
            assert ds.crs.equals(ing.crs)
            np.testing.assert_array_equal(ds['x'].data, ing.x)
            np.testing.assert_array_equal(ds['y'].data, ing.y)
            assert _max_distance_to_xlat(ds.crs, ds['x'].data, ds['y'].data, xlat, xlong) < 10.0


class TestHeaderConsistency:
    @pytest.mark.parametrize('attr,factor', [('DX', 1.001), ('BUCKET_MM', 2.0), ('TRUELAT1', 1.01)])
    def test_mismatched_attr_refused(self, wrf_file_1, wrf_file_2, tmp_path, attr, factor):
        """A later file whose grid/configuration differs from the first is refused.
        Catches: first-file-only reads (a d02 file in a d03 list, a different bucket size)."""
        p2 = _copy(wrf_file_2, tmp_path)
        with h5py.File(p2, 'r+') as h5:
            if attr not in h5.attrs:
                pytest.skip(f'{attr} not in the fixture')
            h5.attrs[attr] = np.float32(float(np.asarray(h5.attrs[attr]).item()) * factor)
        with pytest.raises(ValueError, match=attr):
            WrfIngest([wrf_file_1, p2])

    def test_attr_missing_from_later_file_refused(self, wrf_file_1, wrf_file_2, tmp_path):
        p2 = _copy(wrf_file_2, tmp_path)
        with h5py.File(p2, 'r+') as h5:
            del h5.attrs['BUCKET_MM']
        with pytest.raises(ValueError, match='BUCKET_MM'):
            WrfIngest([wrf_file_1, p2])

    def test_attr_only_in_later_file_refused(self, wrf_file_1, wrf_file_2, tmp_path):
        """Catches: comparing only the keys the FIRST file has (a later file's bucket would be ignored)."""
        # both copies in one directory, so the input sort order is the date order (the first file lacks it)
        p1 = _copy(wrf_file_1, tmp_path)
        p2 = _copy(wrf_file_2, tmp_path)
        with h5py.File(p1, 'r+') as h5:
            del h5.attrs['BUCKET_MM']
        ing_order = sorted([p1, p2])
        assert ing_order[0] == p1
        with pytest.raises(ValueError, match=r"\['BUCKET_MM'\] are absent from"):
            WrfIngest([p1, p2])

    def test_run_start_recorded_per_file(self, wrf_file_1, wrf_file_2, tmp_path):
        """Each file's SIMULATION_START_DATE is kept (a stitched hindcast spans several runs)."""
        p2 = _copy(wrf_file_2, tmp_path)
        with h5py.File(p2, 'r+') as h5:
            h5.attrs['SIMULATION_START_DATE'] = np.bytes_('2023-02-12_12:00:00')
        ing = WrfIngest([wrf_file_1, p2])
        assert ing._file_run_starts[1] == np.datetime64('2023-02-12T12:00', 'm')
        assert ing._file_run_starts[0] != ing._file_run_starts[1]


class _WrfIngestWgs84(WrfIngest):
    """Reproduces a cfdb-ingest < 0.6.0 output: the same Lambert parameters on the WGS84 ellipsoid."""

    xy_tolerance_m = 1e12

    def _parse_crs(self, h5):
        cf = super()._parse_crs(h5).to_cf()
        return pyproj.CRS.from_cf({k: v for k, v in cf.items()
                                   if k not in ('semi_major_axis', 'semi_minor_axis', 'inverse_flattening',
                                                'earth_radius', 'crs_wkt', 'reference_ellipsoid_name',
                                                'geographic_crs_name', 'horizontal_datum_name',
                                                'prime_meridian_name', 'longitude_of_prime_meridian',
                                                'projected_crs_name')})


class TestCrsMixRefused:
    def test_forecast_append_to_wgs84_target_refused(self, wrf_file_1, wrf_file_2, tmp_path):
        """Appending a 0.6.0 init to a target written with the old WGS84 CRS is refused, naming the remedy
        before any x/y comparison. Catches: dropping check_crs, or running it after validate_spatial."""
        p = tmp_path / 'fc.cfdb'
        old = _WrfIngestWgs84(wrf_file_1)
        assert old.crs.ellipsoid.semi_major_metre != WPS_EARTH_RADIUS_M
        old.convert(p, variables=['T2'], dataset_type='grid_forecast', forecast_reference_time='2023-02-12T00')
        with pytest.raises(ValueError, match='rebuild the target with cfdb-ingest >= 0.6.0'):
            WrfIngest(wrf_file_2).convert(p, variables=['T2'], dataset_type='grid_forecast',
                                          forecast_reference_time='2023-02-13T00')

    def test_forecast_append_same_crs_passes(self, wrf_file_1, wrf_file_2, tmp_path):
        p = tmp_path / 'fc.cfdb'
        WrfIngest(wrf_file_1).convert(p, variables=['T2'], dataset_type='grid_forecast',
                                      forecast_reference_time='2023-02-12T00')
        WrfIngest(wrf_file_2).convert(p, variables=['T2'], dataset_type='grid_forecast',
                                      forecast_reference_time='2023-02-13T00')
        with cfdb.open_dataset(p) as ds:
            assert len(ds['forecast_reference_time'].data) == 5  # 02-12T00 .. 02-13T00 at the 6-h step

    def test_grid_extend_to_wgs84_target_refused(self, wrf_file_1, wrf_file_2, tmp_path):
        """Grid extend mode: the same CRS check as forecast appends. Catches: grid.validate_target skipping check_crs."""
        p = tmp_path / 'g.cfdb'
        kw = dict(variables=['T2'], extend=True, squeeze_height=True, chunk_shape=(24, 37, 33))
        _WrfIngestWgs84(wrf_file_1).convert(p, **kw)
        with pytest.raises(ValueError, match='rebuild the target with cfdb-ingest >= 0.6.0'):
            WrfIngest(wrf_file_2).convert(p, **kw)


# Synthetic grids whose XLAT/XLONG come from a numpy port of WPS's own projection formulas
# (wrf_synthetic.wps_ijll, validated against the real d01 fixture's XLAT/XLONG to ~3 m).
WPS_GRIDS = {
    'ps_south': (2, dict(dx=10000.0, truelat1=-60.0, stdlon=170.0), -55.0, 160.0),
    'ps_north': (2, dict(dx=10000.0, truelat1=60.0, stdlon=-100.0), 55.0, -110.0),
    'merc_south': (3, dict(dx=10000.0, truelat1=-40.0, stdlon=170.0), -45.0, 165.0),
    'merc_north': (3, dict(dx=10000.0, truelat1=30.0, stdlon=0.0), 20.0, -10.0),
    'lc_north': (1, dict(dx=12000.0, truelat1=30.0, truelat2=60.0, stdlon=-98.0), 25.0, -110.0),
}


def test_wps_port_reproduces_real_xlat(wrf_file_1):
    """The port itself: Lambert through wps_ijll reproduces the real d01 XLAT/XLONG (else the tests below
    would check cfdb-ingest against a wrong reference)."""
    xlat, xlong = _xlat_xlong(wrf_file_1)
    with h5py.File(wrf_file_1, 'r') as h5:
        g = {k: float(np.asarray(h5.attrs[k]).item()) for k in ('DX', 'TRUELAT1', 'TRUELAT2', 'STAND_LON')}
    ny, nx = xlat.shape
    jj, ii = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing='ij')
    la, lo = wps_ijll(1, ii, jj, lat1=xlat[0, 0], lon1=xlong[0, 0], dx=g['DX'], truelat1=g['TRUELAT1'],
                      truelat2=g['TRUELAT2'], stdlon=g['STAND_LON'])
    d = pyproj.Geod(a=WPS_EARTH_RADIUS_M, b=WPS_EARTH_RADIUS_M).inv(lo, la, xlong, xlat)[2]
    assert d.max() < 5.0, d.max()


@pytest.mark.parametrize('name', sorted(WPS_GRIDS))
def test_projection_matches_wps_geometry(tmp_path, name):
    """Polar stereographic and Mercator (both hemispheres) and a northern Lambert: the CRS + fitted lattice
    reproduce WPS's cell centres. Catches: dropping the sphere, a wrong hemisphere or parameter mapping."""
    map_proj, proj, lat0, lon0 = WPS_GRIDS[name]
    p = syn.write_wrfout(tmp_path / f'wrfout_d01_{name}.nc', '2026-01-01T00', 2, 40, 50, map_proj=map_proj,
                         projection=proj, lat0=lat0, lon0=lon0, variables=('T2',))
    ing = WrfIngest(p)
    assert ing.crs.ellipsoid.semi_major_metre == WPS_EARTH_RADIUS_M
    xlat, xlong = _xlat_xlong(p)
    assert _max_distance_to_xlat(ing.crs, ing.x, ing.y, xlat, xlong) < 5.0

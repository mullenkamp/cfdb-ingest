"""
Tests for cfdb_ingest.wrf (WrfIngest) and cfdb_ingest.base (H5Ingest).

Uses subset wrfout files from cfdb_ingest/tests/data/.
"""
import pathlib
import tempfile

import h5py
import numpy as np
import pyproj
import pytest


# ======================================================================
# unstagger utility
# ======================================================================

class TestUnstagger:
    def test_unstagger_1d(self):
        from cfdb_ingest.wrf import unstagger
        data = np.array([1.0, 3.0, 5.0])
        result = unstagger(data, axis=0)
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_unstagger_2d_axis0(self):
        from cfdb_ingest.wrf import unstagger
        data = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])  # (3, 2)
        result = unstagger(data, axis=0)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[2.0, 4.0], [6.0, 8.0]])

    def test_unstagger_2d_axis1(self):
        from cfdb_ingest.wrf import unstagger
        data = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])  # (2, 3)
        result = unstagger(data, axis=1)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[2.0, 4.0], [3.0, 5.0]])

    def test_unstagger_3d(self):
        from cfdb_ingest.wrf import unstagger
        data = np.arange(24, dtype='float64').reshape(4, 3, 2)
        result = unstagger(data, axis=0)
        assert result.shape == (3, 3, 2)
        expected = (data[:3] + data[1:]) / 2.0
        np.testing.assert_array_almost_equal(result, expected)


# ======================================================================
# _wrf_attr helper
# ======================================================================

class TestWrfAttr:
    def test_scalar_array(self):
        from cfdb_ingest.wrf import _wrf_attr
        attrs = {'X': np.array([42.5])}
        assert _wrf_attr(attrs, 'X') == 42.5

    def test_scalar_int_array(self):
        from cfdb_ingest.wrf import _wrf_attr
        attrs = {'X': np.array([3])}
        assert _wrf_attr(attrs, 'X') == 3

    def test_bytes(self):
        from cfdb_ingest.wrf import _wrf_attr
        attrs = {'X': b'hello'}
        assert _wrf_attr(attrs, 'X') == 'hello'

    def test_plain_scalar(self):
        from cfdb_ingest.wrf import _wrf_attr
        attrs = {'X': 7}
        assert _wrf_attr(attrs, 'X') == 7


# ======================================================================
# WrfIngest — Initialization / Metadata
# ======================================================================

class TestWrfInit:
    def test_crs_is_lambert(self, wrf_single):
        crs = wrf_single.crs
        assert isinstance(crs, pyproj.CRS)
        cf = crs.to_cf()
        assert cf['grid_mapping_name'] == 'lambert_conformal_conic'

    def test_times_single_file(self, wrf_single):
        assert len(wrf_single.times) == 24
        assert wrf_single.times[0] == np.datetime64('2023-02-12T00:00', 'm')
        assert wrf_single.times[-1] == np.datetime64('2023-02-12T23:00', 'm')

    def test_times_multi_file(self, wrf_multi):
        assert len(wrf_multi.times) == 48
        assert wrf_multi.times[0] == np.datetime64('2023-02-12T00:00', 'm')
        assert wrf_multi.times[-1] == np.datetime64('2023-02-13T23:00', 'm')

    def test_spatial_coords_shape(self, wrf_single):
        assert wrf_single.x.shape == (99,)
        assert wrf_single.y.shape == (111,)

    def test_spatial_coords_spacing(self, wrf_single):
        dx = np.diff(wrf_single.x)
        dy = np.diff(wrf_single.y)
        np.testing.assert_allclose(dx, 27000.0, atol=0.1)
        np.testing.assert_allclose(dy, 27000.0, atol=0.1)

    def test_available_variables(self, wrf_single):
        var_keys = set(wrf_single.variables.keys())
        # These should all be present in the test files
        assert 'T2' in var_keys
        assert 'RAIN' in var_keys
        assert 'WIND10' in var_keys
        assert 'WIND_DIR10' in var_keys
        assert 'TSK' in var_keys
        assert 'SWDOWN' in var_keys
        assert 'GLW' in var_keys
        assert 'SNOWH' in var_keys
        assert 'T' in var_keys
        assert 'WIND' in var_keys
        assert 'WIND_DIR' in var_keys
        assert 'U10' in var_keys
        assert 'V10' in var_keys
        assert 'U' in var_keys
        assert 'V' in var_keys
        assert 'Q' in var_keys
        assert 'SLP' in var_keys
        # RH2 is not a standard WRF output; should be filtered
        # assert 'RH2' not in var_keys

    def test_cosalpha_sinalpha_loaded(self, wrf_single):
        assert wrf_single._cosalpha is not None
        assert wrf_single._sinalpha is not None
        assert wrf_single._cosalpha.shape == (111, 99)

    def test_bbox_geographic(self, wrf_single):
        bbox = wrf_single.bbox_geographic
        assert len(bbox) == 4
        min_lon, min_lat, max_lon, max_lat = bbox
        assert min_lat < max_lat
        # Domain is over New Zealand; latitudes should be in the southern hemisphere
        assert min_lat < -30
        assert max_lat < 0

    def test_input_paths_sorted(self, wrf_multi):
        paths = wrf_multi.input_paths
        assert all(isinstance(p, pathlib.Path) for p in paths)
        assert paths == sorted(paths)

    def test_file_not_found(self):
        from cfdb_ingest.wrf import WrfIngest
        with pytest.raises(FileNotFoundError):
            WrfIngest('/nonexistent/wrfout_d03_2015-01-01_00:00:00.nc')

    def test_single_path_as_string(self, wrf_file_1):
        from cfdb_ingest.wrf import WrfIngest
        wrf = WrfIngest(str(wrf_file_1))
        assert len(wrf.input_paths) == 1


# ======================================================================
# Variable Resolution
# ======================================================================

class TestVariableResolution:
    def test_resolve_by_mapping_key(self, wrf_single):
        assert wrf_single.resolve_variables(['T2']) == ['T2']

    def test_resolve_by_cfdb_name(self, wrf_single):
        # air_temp maps to both T2 (surface) and T (levels)
        result = wrf_single.resolve_variables(['air_temp'])
        assert set(result) == {'T2', 'T'}

    def test_resolve_by_source_var(self, wrf_single):
        # RAINC is a source var only for the RAIN mapping key
        assert wrf_single.resolve_variables(['RAINC']) == ['RAIN']

    def test_resolve_none_returns_all(self, wrf_single):
        result = wrf_single.resolve_variables(None)
        assert set(result) == set(wrf_single.variables.keys())

    def test_resolve_multiple(self, wrf_single):
        result = wrf_single.resolve_variables(['precip', 'wind_speed'])
        assert set(result) == {'RAIN', 'WIND10', 'WIND'}

    def test_resolve_deduplicates(self, wrf_single):
        # T2 is already included in air_temp's resolution (T2 + T)
        result = wrf_single.resolve_variables(['T2', 'air_temp'])
        assert set(result) == {'T2', 'T'}

    def test_resolve_unknown_raises(self, wrf_single):
        with pytest.raises(ValueError, match='Unknown variable'):
            wrf_single.resolve_variables(['nonexistent_var'])


# ======================================================================
# Conversion — 2D Variables
# ======================================================================

class TestConvert2D:
    def test_t2_values_match_raw(self, wrf_single, wrf_file_1, cfdb_out):
        """T2 (no transform) should match raw WRF data."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T06:00',
            end_date='2023-02-12T06:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            cfdb_data = np.squeeze(np.array(ds['air_temperature'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            raw_data = h5['T2'][6]

        # cfdb encodes with limited precision; allow small difference
        np.testing.assert_allclose(cfdb_data, raw_data, atol=0.02)

    def test_precip_increment_within_file(self, wrf_single, wrf_file_1, cfdb_out):
        """Precip increment = total[t] - total[t-1]."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['precip'],
            start_date='2023-02-12T06:00',
            end_date='2023-02-12T06:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            cfdb_data = np.squeeze(np.array(ds['precipitation'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            total_6 = h5['RAINNC'][6] + h5['RAINC'][6]
            total_5 = h5['RAINNC'][5] + h5['RAINC'][5]
            expected = (total_6 - total_5).astype('float32')

        np.testing.assert_allclose(cfdb_data, expected, atol=0.02)

    def test_precip_first_timestep_is_nan(self, wrf_single, cfdb_out):
        """First overall timestep of precip should be NaN."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['precip'],
            start_date='2023-02-12T00:00',
            end_date='2023-02-12T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            cfdb_data = np.squeeze(np.array(ds['precipitation'][0]))

        assert np.all(np.isnan(cfdb_data))

    def test_wind_speed_matches_manual(self, wrf_single, wrf_file_1, cfdb_out):
        """Wind speed after rotation should match manual calculation."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND10'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            ws_cfdb = np.squeeze(np.array(ds['wind_speed'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            u = h5['U10'][12].astype('float64')
            v = h5['V10'][12].astype('float64')
            cosa = h5['COSALPHA'][0]
            sina = h5['SINALPHA'][0]
            u_e = u * cosa + v * sina
            v_e = -u * sina + v * cosa
            expected = np.sqrt(u_e**2 + v_e**2).astype('float32')

        np.testing.assert_allclose(ws_cfdb, expected, atol=0.02)

    def test_wind_direction_range(self, wrf_single, cfdb_out):
        """Wind direction should be in [0, 360)."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND_DIR10'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            wd = np.squeeze(np.array(ds['wind_direction'][0]))

        assert np.nanmin(wd) >= 0.0
        assert np.nanmax(wd) <= 360.0

    def test_multiple_surface_vars(self, wrf_single, cfdb_out):
        """Convert several surface vars at once; height coord is built correctly."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2', 'TSK', 'SWDOWN', 'GLW', 'SNOWH'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) == 5

            # T2 at 2m, rest at 0m → height = [0.0, 2.0]
            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [0.0, 2.0])


# ======================================================================
# Conversion — Filtering
# ======================================================================

class TestFiltering:
    def test_date_filter(self, wrf_single, cfdb_out):
        """Date range selects correct number of timesteps."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T06:00',
            end_date='2023-02-12T10:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = np.array(ds['time'][:])
            assert len(times) == 5  # hours 6,7,8,9,10

    def test_bbox_clips_domain(self, wrf_single, cfdb_out):
        """Bbox should produce a smaller spatial domain."""
        import cfdb
        full_nx = len(wrf_single.x)
        full_ny = len(wrf_single.y)

        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            x = np.array(ds['x'][:])
            y = np.array(ds['y'][:])
            assert len(x) < full_nx
            assert len(y) < full_ny
            # The subset should still have reasonable size
            assert len(x) > 5
            assert len(y) > 5

    def test_bbox_no_overlap_raises(self, wrf_single, cfdb_out):
        """A bbox outside the domain should raise ValueError."""
        with pytest.raises(ValueError, match='does not overlap'):
            wrf_single.convert(
                cfdb_path=cfdb_out,
                variables=['T2'],
                bbox=(0.0, 0.0, 1.0, 1.0),  # Equator, nowhere near NZ
            )

    def test_date_and_bbox_combined(self, wrf_single, cfdb_out):
        """Both filters applied together."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T10:00',
            end_date='2023-02-12T14:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = np.array(ds['time'][:])
            x = np.array(ds['x'][:])
            y = np.array(ds['y'][:])
            assert len(times) == 5
            assert len(x) < len(wrf_single.x)
            assert len(y) < len(wrf_single.y)


# ======================================================================
# Conversion — Multi-file
# ======================================================================

class TestMultiFile:
    def test_all_timesteps_merged(self, wrf_multi, cfdb_out):
        """48 hours across 2 files."""
        import cfdb
        wrf_multi.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T22:00',
            end_date='2023-02-13T01:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = np.array(ds['time'][:])
            assert len(times) == 4
            assert times[0] == np.datetime64('2023-02-12T22:00')
            assert times[-1] == np.datetime64('2023-02-13T01:00')

    def test_precip_cross_file_no_nan(self, wrf_multi, cfdb_out):
        """Precip at the first timestep of file 2 should not be NaN."""
        import cfdb
        wrf_multi.convert(
            cfdb_path=cfdb_out,
            variables=['precip'],
            start_date='2023-02-13T00:00',
            end_date='2023-02-13T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            pr = np.squeeze(np.array(ds['precipitation'][0]))
            assert not np.all(np.isnan(pr))

    def test_precip_cross_file_values(self, wrf_multi, wrf_file_1, wrf_file_2, cfdb_out):
        """Precip increment at file boundary = file2[0] - file1[-1]."""
        import cfdb
        wrf_multi.convert(
            cfdb_path=cfdb_out,
            variables=['precip'],
            start_date='2023-02-13T00:00',
            end_date='2023-02-13T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            cfdb_pr = np.squeeze(np.array(ds['precipitation'][0]))

        i, j = 30, 25
        with h5py.File(wrf_file_1, 'r') as h5:
            prev = h5['RAINNC'][-1, i, j] + h5['RAINC'][-1, i, j]
        with h5py.File(wrf_file_2, 'r') as h5:
            curr = h5['RAINNC'][0, i, j] + h5['RAINC'][0, i, j]

        expected = float(curr - prev)
        np.testing.assert_allclose(float(cfdb_pr[i, j]), expected, atol=0.02)


# ======================================================================
# Conversion — 3D Variables
# ======================================================================

class TestConvert3D:
    def test_levels_requires_target_levels(self, wrf_single, cfdb_out):
        """Converting a level-interpolated var without target_levels should raise."""
        with pytest.raises(ValueError, match='target_levels'):
            wrf_single.convert(
                cfdb_path=cfdb_out,
                variables=['T'],
                start_date='2023-02-12T12:00',
                end_date='2023-02-12T12:00',
            )

    def test_levels_temp_shape_and_height(self, wrf_single, cfdb_out):
        """Level-interpolated temperature should have correct shape and height coord."""
        import cfdb
        levels = [100.0, 500.0, 1000.0, 2000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, levels)

            t_data = np.squeeze(np.array(ds['air_temperature'][0]))
            assert t_data.shape[0] == len(levels)

    def test_levels_temp_decreases_with_height(self, wrf_single, cfdb_out):
        """Temperature should generally decrease with height (standard lapse rate)."""
        import cfdb
        levels = [100.0, 1000.0, 3000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            t_data = np.squeeze(np.array(ds['air_temperature'][0]))
            mean_by_level = [np.nanmean(t_data[lev]) for lev in range(len(levels))]

            # Mean temp at 100m should be greater than at 3000m
            assert mean_by_level[0] > mean_by_level[-1]

    def test_surface_and_levels_merged(self, wrf_single, cfdb_out):
        """T2 (2m) and T (levels) merge into one air_temperature variable."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2', 'T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[100.0, 500.0],
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            # T2 and T both map to air_temp → one merged data var
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) == 1
            assert var_names[0] == 'air_temperature'

            # Height coordinate has surface (2m) + target levels
            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [2.0, 100.0, 500.0])

            # Data at all 3 height levels should be populated
            t_data = np.squeeze(np.array(ds['air_temperature'][0]))
            assert t_data.shape[0] == 3
            for lev in range(3):
                assert not np.all(np.isnan(t_data[lev]))
                # Temperature values should be physically reasonable (200-330 K)
                assert np.nanmean(t_data[lev]) > 200.0
                assert np.nanmean(t_data[lev]) < 330.0


# ======================================================================
# cfdb Output Structure
# ======================================================================

class TestCfdbOutput:
    def test_crs_set(self, wrf_single, cfdb_out):
        """CRS should be written to the cfdb dataset."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert ds.crs is not None

    def test_chunk_shape(self, wrf_single, cfdb_out):
        """Data var chunk shape should be (1, 1, ny, nx)."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            dv = ds['air_temperature']
            cs = dv.chunk_shape
            assert cs[0] == 1  # one timestep per chunk
            assert cs[1] == 1  # one height level per chunk
            assert cs[2] > 1   # spatial y
            assert cs[3] > 1   # spatial x

    def test_custom_chunk_shape(self, wrf_single, cfdb_out):
        """Custom chunk_shape should be applied to the output data variable."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            chunk_shape=(1, 1, 50, 50),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            cs = ds['air_temperature'].chunk_shape
            assert cs == (1, 1, 50, 50)

    def test_cf_attributes_template(self, wrf_single, cfdb_out):
        """Template variables should have CF standard_name and units."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            dv = ds['air_temperature']
            assert dv.attrs['standard_name'] == 'air_temperature'
            assert dv.attrs['units'] == 'K'

    def test_cf_attributes_non_template(self, wrf_single, cfdb_out):
        """Variables added as cfdb templates should have correct CF attrs."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['PSFC', 'SWDOWN', 'SNOWH'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            psfc = ds['surface_pressure']
            assert psfc.attrs['standard_name'] == 'surface_air_pressure'
            assert psfc.attrs['units'] == 'Pa'

            sw = ds['shortwave_radiation']
            assert sw.attrs['standard_name'] == 'surface_downwelling_shortwave_flux_in_air'
            assert sw.attrs['units'] == 'W m-2'

            snow = ds['snow_depth']
            assert snow.attrs['standard_name'] == 'surface_snow_thickness'
            assert snow.attrs['units'] == 'm'

    def test_cf_attributes_level_interp(self, wrf_single, cfdb_out):
        """Level-interpolated variable should get CF attrs via air_temp template."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[100.0, 500.0],
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            t_var = ds['air_temperature']
            assert t_var.attrs['standard_name'] == 'air_temperature'
            assert t_var.attrs['units'] == 'K'


# ======================================================================
# Conversion — 3D Wind
# ======================================================================

class TestConvert3DWind:
    def test_wind_speed_3d_shape(self, wrf_single, cfdb_out):
        """3D wind speed should have correct shape (n_levels, ny, nx)."""
        import cfdb
        levels = [100.0, 500.0, 1000.0, 2000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, levels)

            ws_data = np.squeeze(np.array(ds['wind_speed'][0]))
            assert ws_data.shape[0] == len(levels)

    def test_wind_speed_3d_reasonable_values(self, wrf_single, cfdb_out):
        """3D wind speed values should be >= 0 and typically < 100 m/s."""
        import cfdb
        levels = [100.0, 500.0, 1000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            ws = np.squeeze(np.array(ds['wind_speed'][0]))
            assert np.nanmin(ws) >= 0.0
            assert np.nanmax(ws) < 100.0

    def test_wind_speed_surface_and_levels_merged(self, wrf_single, cfdb_out):
        """WIND10 + WIND merge into one wind_speed variable with surface + level heights."""
        import cfdb
        levels = [100.0, 500.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND10', 'WIND'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) == 1
            assert var_names[0] == 'wind_speed'

            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [10.0, 100.0, 500.0])

            ws_data = np.squeeze(np.array(ds['wind_speed'][0]))
            assert ws_data.shape[0] == 3
            for lev in range(3):
                assert not np.all(np.isnan(ws_data[lev]))

    def test_wind_direction_3d_range(self, wrf_single, cfdb_out):
        """3D wind direction should be in [0, 360)."""
        import cfdb
        levels = [100.0, 500.0, 1000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['WIND_DIR'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            wd = np.squeeze(np.array(ds['wind_direction'][0]))
            assert np.nanmin(wd) >= 0.0
            assert np.nanmax(wd) <= 360.0


# ======================================================================
# Conversion — Wind Components
# ======================================================================

class TestConvertWindComponents:
    def test_u_wind_v_wind_10m(self, wrf_single, wrf_file_1, cfdb_out):
        """U10/V10 wind components should match manual rotation."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['U10', 'V10'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            u_cfdb = np.squeeze(np.array(ds['u_wind'][0]))
            v_cfdb = np.squeeze(np.array(ds['v_wind'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            u = h5['U10'][12].astype('float64')
            v = h5['V10'][12].astype('float64')
            cosa = h5['COSALPHA'][0]
            sina = h5['SINALPHA'][0]
            u_expected = (u * cosa + v * sina).astype('float32')
            v_expected = (-u * sina + v * cosa).astype('float32')

        np.testing.assert_allclose(u_cfdb, u_expected, atol=0.02)
        np.testing.assert_allclose(v_cfdb, v_expected, atol=0.02)

    def test_u_wind_3d_shape(self, wrf_single, cfdb_out):
        """3D U wind at levels should have correct shape."""
        import cfdb
        levels = [100.0, 500.0, 1000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['U'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            u_data = np.squeeze(np.array(ds['u_wind'][0]))
            assert u_data.shape[0] == len(levels)

    def test_wind_components_surface_and_levels_merged(self, wrf_single, cfdb_out):
        """U10 + U merge into one u_wind variable with surface + level heights."""
        import cfdb
        levels = [100.0, 500.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['U10', 'U'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) == 1
            assert var_names[0] == 'u_wind'

            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [10.0, 100.0, 500.0])

            u_data = np.squeeze(np.array(ds['u_wind'][0]))
            assert u_data.shape[0] == 3
            for lev in range(3):
                assert not np.all(np.isnan(u_data[lev]))


# ======================================================================
# Conversion — 3D Specific Humidity
# ======================================================================

class TestConvert3DQ:
    def test_specific_humidity_3d_shape(self, wrf_single, cfdb_out):
        """3D specific humidity should have correct shape."""
        import cfdb
        levels = [100.0, 500.0, 1000.0, 2000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['Q'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            q_data = np.squeeze(np.array(ds['specific_humidity'][0]))
            assert q_data.shape[0] == len(levels)

    def test_specific_humidity_3d_values(self, wrf_single, cfdb_out):
        """3D specific humidity should be physically reasonable (0 to ~0.04 kg/kg)."""
        import cfdb
        levels = [100.0, 500.0, 1000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['Q'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            q = np.squeeze(np.array(ds['specific_humidity'][0]))
            assert np.nanmin(q) >= 0.0
            assert np.nanmax(q) < 0.04

    def test_specific_humidity_surface_and_levels_merged(self, wrf_single, cfdb_out):
        """Q2 + Q merge into one specific_humidity variable."""
        import cfdb
        levels = [100.0, 500.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['Q2', 'Q'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) == 1
            assert var_names[0] == 'specific_humidity'

            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [2.0, 100.0, 500.0])

            q_data = np.squeeze(np.array(ds['specific_humidity'][0]))
            assert q_data.shape[0] == 3
            for lev in range(3):
                assert not np.all(np.isnan(q_data[lev]))


# ======================================================================
# Conversion — Sea Level Pressure
# ======================================================================

class TestConvertSLP:
    def test_slp_shape(self, wrf_single, cfdb_out):
        """SLP should have correct 2D shape."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['SLP'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            slp_data = np.squeeze(np.array(ds['mslp'][0]))
            x = np.array(ds['x'][:])
            y = np.array(ds['y'][:])
            assert slp_data.shape == (len(y), len(x))

    def test_slp_reasonable_values(self, wrf_single, cfdb_out):
        """SLP values should be around 95000-108000 Pa."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['SLP'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            slp = np.squeeze(np.array(ds['mslp'][0]))
            assert np.nanmin(slp) > 95000.0
            assert np.nanmax(slp) < 108000.0

    def test_slp_greater_than_surface_pressure(self, wrf_single, wrf_file_1, cfdb_out):
        """SLP should be >= PSFC for elevated terrain."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['SLP'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            slp = np.squeeze(np.array(ds['mslp'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            psfc = h5['PSFC'][12].astype('float32')

        # SLP should be >= PSFC everywhere (reduction to sea level increases pressure)
        # Allow small tolerance for floating point and near-sea-level points
        assert np.all(slp >= psfc - 1.0)

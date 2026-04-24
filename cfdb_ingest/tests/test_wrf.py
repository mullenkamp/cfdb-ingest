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
        assert 'Q_SH' in var_keys
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
        """Convert several surface vars at once; stored as (time, height_Xm, y, x)."""
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

            # Surface vars have named height coordinates
            assert 'height_0m' in ds.coord_names
            assert 'height_2m' in ds.coord_names

            # Each variable is (time, height_Xm, y, x)
            for dv in ds.data_vars:
                assert dv.ndims == 4


# ======================================================================
# Bucket-aware accumulation (bucket_mm > 0 reconstruction)
# ======================================================================


def _make_bucketed_wrfout(src_path, dst_path, bucket_mm, i_rainnc_field, i_rainc_field):
    """
    Copy an existing wrfout test file and inject synthetic I_RAINNC, I_RAINC
    bucket counters plus a BUCKET_MM global attribute, so we can test
    bucket-reconstruction logic without having to run WRF with bucket_mm>0.

    i_rainnc_field and i_rainc_field are 2D arrays (south_north, west_east)
    that broadcast to every time step (i.e. static bucket count).
    """
    import shutil
    shutil.copy(src_path, dst_path)
    with h5py.File(dst_path, 'r+') as h5:
        h5.attrs['BUCKET_MM'] = np.float32(bucket_mm)
        n_t = h5['RAINNC'].shape[0]
        ny, nx = i_rainnc_field.shape
        i_rainnc = np.broadcast_to(i_rainnc_field, (n_t, ny, nx)).astype('int32')
        i_rainc = np.broadcast_to(i_rainc_field, (n_t, ny, nx)).astype('int32')
        h5.create_dataset('I_RAINNC', data=i_rainnc, dtype='int32')
        h5.create_dataset('I_RAINC', data=i_rainc, dtype='int32')


class TestBucketAwareAccumulation:
    def test_plain_sum_when_bucket_disabled(self, wrf_single, wrf_file_1):
        """BUCKET_MM <= 0 on the file → plain source_vars sum, no reconstruction."""
        with h5py.File(wrf_file_1, 'r') as h5:
            got = wrf_single._accumulation_source_sum(
                h5, ['RAINNC', 'RAINC'], 5, (slice(None), slice(None))
            )
            expected = (h5['RAINNC'][5].astype('float64')
                        + h5['RAINC'][5].astype('float64'))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_bucket_reconstruction_adds_counter_term(self, wrf_file_1, tmp_path):
        """BUCKET_MM > 0 + I_<name> present → total = <name> + BUCKET_MM * I_<name>."""
        from cfdb_ingest.wrf import WrfIngest

        with h5py.File(wrf_file_1, 'r') as h5:
            ny, nx = h5['RAINNC'].shape[1:]
        # Distinct counts so we can verify the arithmetic cleanly.
        i_rainnc_field = np.full((ny, nx), 3, dtype='int32')
        i_rainc_field = np.full((ny, nx), 2, dtype='int32')
        bucket_mm = 100.0
        dst = tmp_path / 'wrfout_bucket.nc'
        _make_bucketed_wrfout(wrf_file_1, dst, bucket_mm, i_rainnc_field, i_rainc_field)

        ingest = WrfIngest(dst)
        with h5py.File(dst, 'r') as h5:
            got = ingest._accumulation_source_sum(
                h5, ['RAINNC', 'RAINC'], 5, (slice(None), slice(None))
            )
            expected = (h5['RAINNC'][5].astype('float64')
                        + bucket_mm * i_rainnc_field
                        + h5['RAINC'][5].astype('float64')
                        + bucket_mm * i_rainc_field)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_missing_companion_falls_back_to_plain(self, wrf_file_1, tmp_path):
        """BUCKET_MM > 0 but no I_<name> for a source var → that term is plain."""
        import shutil
        dst = tmp_path / 'wrfout_bucket_partial.nc'
        shutil.copy(wrf_file_1, dst)
        with h5py.File(dst, 'r+') as h5:
            h5.attrs['BUCKET_MM'] = np.float32(50.0)
            ny, nx = h5['RAINNC'].shape[1:]
            n_t = h5['RAINNC'].shape[0]
            # only I_RAINNC, no I_RAINC
            h5.create_dataset(
                'I_RAINNC',
                data=np.full((n_t, ny, nx), 4, dtype='int32'),
                dtype='int32',
            )

        from cfdb_ingest.wrf import WrfIngest
        ingest = WrfIngest(dst)
        with h5py.File(dst, 'r') as h5:
            got = ingest._accumulation_source_sum(
                h5, ['RAINNC', 'RAINC'], 3, (slice(None), slice(None))
            )
            expected = (h5['RAINNC'][3].astype('float64') + 50.0 * 4
                        + h5['RAINC'][3].astype('float64'))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_rain_tr_mapping_registered(self):
        """Ensure WRF_VARIABLE_MAPPING has a RAIN_TR entry for tracer precip."""
        from cfdb_ingest.wrf import WRF_VARIABLE_MAPPING
        assert 'RAIN_TR' in WRF_VARIABLE_MAPPING
        entry = WRF_VARIABLE_MAPPING['RAIN_TR']
        assert entry['cfdb_name'] == 'precip_tr'
        assert entry['source_vars'] == ['TR_RAINNC', 'TR_RAINC']
        assert entry['transform'] == 'accumulation_increment'

    def test_bucket_applies_to_tracer_source_vars(self, wrf_file_1, tmp_path):
        """The bucket-aware helper works for TR_* source_vars with I_TR_* companions.

        The base test file has no TR_* vars; synthesize a minimal scenario:
        add TR_RAINNC, TR_RAINC, I_TR_RAINNC, I_TR_RAINC to a copy, and
        confirm the helper reconstructs TR_<name> + bucket_mm * I_TR_<name>.
        """
        import shutil
        dst = tmp_path / 'wrfout_tr_bucket.nc'
        shutil.copy(wrf_file_1, dst)
        bucket_mm = 25.0
        with h5py.File(dst, 'r+') as h5:
            h5.attrs['BUCKET_MM'] = np.float32(bucket_mm)
            n_t, ny, nx = h5['RAINNC'].shape
            # Plausible TR_* values: a fraction of RAINNC, plus a bucket count.
            tr_rainnc_raw = (0.3 * h5['RAINNC'][:]).astype('float32')
            tr_rainc_raw = (0.3 * h5['RAINC'][:]).astype('float32')
            h5.create_dataset('TR_RAINNC', data=tr_rainnc_raw, dtype='float32')
            h5.create_dataset('TR_RAINC', data=tr_rainc_raw, dtype='float32')
            h5.create_dataset('I_TR_RAINNC',
                              data=np.full((n_t, ny, nx), 5, dtype='int32'),
                              dtype='int32')
            h5.create_dataset('I_TR_RAINC',
                              data=np.full((n_t, ny, nx), 2, dtype='int32'),
                              dtype='int32')

        from cfdb_ingest.wrf import WrfIngest
        ingest = WrfIngest(dst)
        with h5py.File(dst, 'r') as h5:
            got = ingest._accumulation_source_sum(
                h5, ['TR_RAINNC', 'TR_RAINC'], 4, (slice(None), slice(None))
            )
            expected = (h5['TR_RAINNC'][4].astype('float64') + bucket_mm * 5
                        + h5['TR_RAINC'][4].astype('float64') + bucket_mm * 2)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_increment_remains_correct_across_wrap(self, wrf_file_1, tmp_path):
        """
        Simulate a bucket wrap between two timesteps: at t=3 the counter is 2,
        at t=4 the counter is 3. The increment computation must use reconstructed
        totals, otherwise it produces a huge negative spike.
        """
        import shutil
        dst = tmp_path / 'wrfout_wrap.nc'
        shutil.copy(wrf_file_1, dst)
        bucket_mm = 10.0
        with h5py.File(dst, 'r+') as h5:
            h5.attrs['BUCKET_MM'] = np.float32(bucket_mm)
            n_t, ny, nx = h5['RAINNC'].shape
            # Counter is 2 for t <= 3, 3 for t >= 4 (simulated wrap at t=4).
            i_rainnc = np.zeros((n_t, ny, nx), dtype='int32')
            i_rainnc[:4] = 2
            i_rainnc[4:] = 3
            i_rainc = np.zeros((n_t, ny, nx), dtype='int32')
            h5.create_dataset('I_RAINNC', data=i_rainnc, dtype='int32')
            h5.create_dataset('I_RAINC', data=i_rainc, dtype='int32')

        from cfdb_ingest.wrf import WrfIngest
        ingest = WrfIngest(dst)
        with h5py.File(dst, 'r') as h5:
            # Manual reference using reconstruction:
            total_4 = (h5['RAINNC'][4].astype('float64') + bucket_mm * i_rainnc[4]
                       + h5['RAINC'][4].astype('float64'))
            total_3 = (h5['RAINNC'][3].astype('float64') + bucket_mm * i_rainnc[3]
                       + h5['RAINC'][3].astype('float64'))
            expected = total_4 - total_3

            ingest._prev_accum_total = None
            got = ingest._read_accumulation_increment(
                h5, 'RAIN', 4, (slice(None), slice(None))
            )
        np.testing.assert_allclose(got, expected, atol=1e-3)
        # Sanity: the plain (broken) increment would be wildly negative because
        # RAINNC at t=4 has wrapped — confirm our fix avoided that.
        assert np.all(got >= -1.0)


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


class TestOverlappingTimesteps:
    """Files with overlapping time ranges should deduplicate, first-file-wins."""

    def test_times_deduplicated(self, wrf_overlap):
        """Passing [file1, file2, file1] should still yield 48 unique times."""
        assert len(wrf_overlap.times) == 48
        assert wrf_overlap.times[0] == np.datetime64('2023-02-12T00:00', 'm')
        assert wrf_overlap.times[-1] == np.datetime64('2023-02-13T23:00', 'm')

    def test_raw_to_unique_marks_duplicates(self, wrf_overlap):
        """Sorted order is [file1, file1, file2]; second file1 should be all -1."""
        r2u = wrf_overlap._raw_to_unique
        # 24 + 24 + 24 = 72 raw timesteps
        assert len(r2u) == 72
        # First 24 (file1, first occurrence) are unique
        assert (r2u[:24] != -1).all()
        # Next 24 (file1, duplicate) are all -1
        assert (r2u[24:48] == -1).all()
        # Last 24 (file2) are unique
        assert (r2u[48:] != -1).all()

    def test_simple_var_rechunkit(self, wrf_overlap, wrf_file_1, cfdb_out):
        """Rechunkit path (no-transform) produces correct data with overlaps."""
        import cfdb
        wrf_overlap.convert(cfdb_path=cfdb_out, variables=['T2'])
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert np.array(ds['time'][:]).shape[0] == 48
            cfdb_data = np.squeeze(np.array(ds['air_temperature'][6]))

        with h5py.File(wrf_file_1, 'r') as h5:
            raw_data = h5['T2'][6]
        np.testing.assert_allclose(cfdb_data, raw_data, atol=0.02)

    def test_batch_var_with_overlap(self, wrf_overlap, wrf_file_1, cfdb_out):
        """Batch path (transform vars) produces correct data with overlaps."""
        import cfdb
        wrf_overlap.convert(
            cfdb_path=cfdb_out,
            variables=['WIND10'],
            start_date='2023-02-12T06:00',
            end_date='2023-02-12T06:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = np.array(ds['time'][:])
            assert len(times) == 1
            cfdb_ws = np.squeeze(np.array(ds['wind_speed'][0]))

        with h5py.File(wrf_file_1, 'r') as h5:
            u = h5['U10'][6].astype('float64')
            v = h5['V10'][6].astype('float64')
        expected = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(cfdb_ws, expected, atol=0.1)


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

    def test_surface_and_levels_separate(self, wrf_single, cfdb_out):
        """T2 (surface) and T (levels) stored as separate variables with different coords."""
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
            var_names = sorted(v.name for v in ds.data_vars)
            # T2 gets suffixed with height to avoid conflict: air_temperature_2m
            assert len(var_names) == 2
            assert 'air_temperature' in var_names
            assert 'air_temp_2m' in var_names

            # 3D variable on height coordinate
            t3d = ds['air_temperature']
            assert t3d.ndims == 4
            assert t3d.coord_names == ('time', 'height', 'y', 'x')

            height = np.array(ds['height'][:])
            np.testing.assert_array_equal(height, [100.0, 500.0])

            # Surface variable with named height coordinate
            t2 = ds['air_temp_2m']
            assert t2.ndims == 4
            assert t2.coord_names == ('time', 'height_2m', 'y', 'x')

            # Temperature values should be physically reasonable
            t3d_data = np.squeeze(np.array(t3d[0]))
            for lev in range(2):
                assert np.nanmean(t3d_data[lev]) > 200.0
                assert np.nanmean(t3d_data[lev]) < 330.0

            t2_data = np.squeeze(np.array(t2[0]))
            assert np.nanmean(t2_data) > 200.0
            assert np.nanmean(t2_data) < 330.0


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
        """Surface data var chunk shape should be (1, 1, ny, nx)."""
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
            assert cs[1] == 1  # one height value per chunk
            assert cs[2] > 1   # spatial y
            assert cs[3] > 1   # spatial x

    def test_custom_chunk_shape(self, wrf_single, cfdb_out):
        """Custom chunk_shape for 4D should be applied to level-interpolated vars."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            target_levels=[100.0, 500.0],
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

    def test_wind_speed_surface_and_levels_separate(self, wrf_single, cfdb_out):
        """WIND10 + WIND stored as separate variables with different coords."""
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
            var_names = sorted(v.name for v in ds.data_vars)
            assert len(var_names) == 2
            assert 'wind_speed' in var_names
            assert 'wind_speed_10m' in var_names

            # 3D on height
            ws3d = ds['wind_speed']
            assert ws3d.coord_names == ('time', 'height', 'y', 'x')

            # Surface with named height coordinate
            ws_sfc = ds['wind_speed_10m']
            assert ws_sfc.coord_names == ('time', 'height_10m', 'y', 'x')

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

    def test_wind_components_surface_and_levels_separate(self, wrf_single, cfdb_out):
        """U10 + U stored as separate variables with different coords."""
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
            var_names = sorted(v.name for v in ds.data_vars)
            assert len(var_names) == 2
            assert 'u_wind' in var_names
            assert 'u_wind_10m' in var_names

            assert ds['u_wind'].coord_names == ('time', 'height', 'y', 'x')
            assert ds['u_wind_10m'].coord_names == ('time', 'height_10m', 'y', 'x')


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
            variables=['Q_SH'],
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
            variables=['Q_SH'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            q = np.squeeze(np.array(ds['specific_humidity'][0]))
            assert np.nanmin(q) >= 0.0
            assert np.nanmax(q) < 0.04

    def test_specific_humidity_surface_and_levels_separate(self, wrf_single, cfdb_out):
        """Q2_SH + Q_SH stored as separate variables with different coords."""
        import cfdb
        levels = [100.0, 500.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['Q2_SH', 'Q_SH'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = sorted(v.name for v in ds.data_vars)
            assert len(var_names) == 2
            assert 'specific_humidity' in var_names
            assert 'specific_humidity_2m' in var_names

            assert ds['specific_humidity'].coord_names == ('time', 'height', 'y', 'x')
            assert ds['specific_humidity_2m'].coord_names == ('time', 'height_2m', 'y', 'x')


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


# ======================================================================
# Conversion — Pressure-level mode
# ======================================================================

class TestConvertPressureLevels:
    def test_pressure_coord_created(self, wrf_single, cfdb_out):
        """vertical_coord='pressure' creates a pressure coordinate, not height."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[90000.0, 70000.0, 50000.0],
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'pressure' in ds.coord_names
            assert 'height' not in ds.coord_names

    def test_pressure_levels_values(self, wrf_single, cfdb_out):
        """Pressure coordinate contains the requested levels."""
        import cfdb
        levels = [90000.0, 70000.0, 50000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            pressure = np.array(ds['pressure'][:])
            np.testing.assert_array_equal(pressure, sorted(levels))

    def test_pressure_temp_shape(self, wrf_single, cfdb_out):
        """3D temperature on pressure levels has correct shape."""
        import cfdb
        levels = [90000.0, 70000.0, 50000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            t = ds['air_temperature']
            assert t.ndims == 4
            assert t.coord_names == ('time', 'pressure', 'y', 'x')
            assert t.shape[1] == 3  # 3 pressure levels

    def test_pressure_temp_reasonable(self, wrf_single, cfdb_out):
        """Temperature values on pressure levels are physically reasonable."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[90000.0, 70000.0, 50000.0],
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            t_data = np.squeeze(np.array(ds['air_temperature'][0]))
            for lev in range(3):
                mean_t = np.nanmean(t_data[lev])
                assert mean_t > 200.0
                assert mean_t < 330.0

    def test_pressure_surface_and_3d_separate(self, wrf_single, cfdb_out):
        """Surface T2 and pressure-level T are separate variables."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['T2', 'T'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[90000.0, 70000.0],
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = sorted(v.name for v in ds.data_vars)
            assert 'air_temperature' in var_names
            assert 'air_temp_2m' in var_names

            assert ds['air_temperature'].coord_names == ('time', 'pressure', 'y', 'x')
            assert ds['air_temp_2m'].coord_names == ('time', 'height_2m', 'y', 'x')


# ======================================================================
# Conversion — Geopotential height 3D
# ======================================================================

class TestConvertGHT:
    def test_ght_shape(self, wrf_single, cfdb_out):
        """GHT variable has correct shape on height levels."""
        import cfdb
        levels = [100.0, 500.0, 1000.0]
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['GHT'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=levels,
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            ght = ds['geopotential_height']
            assert ght.ndims == 4
            assert ght.shape[1] == 3

    def test_ght_on_pressure_levels(self, wrf_single, cfdb_out):
        """GHT on pressure levels produces reasonable geopotential heights."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['GHT'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[90000.0, 50000.0],
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            # Pressure coord sorted ascending: [50000, 90000]
            pressure = np.array(ds['pressure'][:])
            ght = np.squeeze(np.array(ds['geopotential_height'][0]))

            idx_900 = np.searchsorted(pressure, 90000.0)
            idx_500 = np.searchsorted(pressure, 50000.0)

            # 900 hPa should be ~1000m, 500 hPa should be ~5500m
            assert 500.0 < np.nanmean(ght[idx_900]) < 2000.0
            assert 4000.0 < np.nanmean(ght[idx_500]) < 7000.0

    def test_ght_increases_with_lower_pressure(self, wrf_single, cfdb_out):
        """Geopotential height should increase as pressure decreases (ascending pressure coord)."""
        import cfdb
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['GHT'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
            bbox=(165.0, -47.0, 175.0, -40.0),
            target_levels=[90000.0, 70000.0, 50000.0],
            vertical_coord='pressure',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            # Pressure sorted ascending [50000, 70000, 90000]
            # GHT should decrease with increasing pressure (higher pressure = lower altitude)
            ght = np.squeeze(np.array(ds['geopotential_height'][0]))
            assert np.nanmean(ght[0]) > np.nanmean(ght[1]) > np.nanmean(ght[2])


# ======================================================================
# Conversion — Soil variables (skip if not in test data)
# ======================================================================

class TestConvertSoil:
    def test_soil_depth_coord_created(self, wrf_single, cfdb_out):
        """Soil variables create a depth coordinate."""
        import cfdb
        if 'SMOIS' not in wrf_single.variables:
            pytest.skip('SMOIS not available in test data')
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['SMOIS'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'depth' in ds.coord_names

    def test_soil_moisture_shape(self, wrf_single, cfdb_out):
        """Soil moisture has 4D shape (time, depth, y, x)."""
        import cfdb
        if 'SMOIS' not in wrf_single.variables:
            pytest.skip('SMOIS not available in test data')
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['SMOIS'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            sm = ds['soil_moisture']
            assert sm.ndims == 4
            assert sm.coord_names == ('time', 'depth', 'y', 'x')


# ======================================================================
# Conversion — New surface variables (skip if not in test data)
# ======================================================================

class TestNewSurfaceVars:
    def test_land_sea_mask_values(self, wrf_single, cfdb_out):
        """XLAND transform produces only 0.0 and 1.0 values."""
        import cfdb
        if 'XLAND' not in wrf_single.variables:
            pytest.skip('XLAND not available in test data')
        wrf_single.convert(
            cfdb_path=cfdb_out,
            variables=['XLAND'],
            start_date='2023-02-12T12:00',
            end_date='2023-02-12T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            mask = np.squeeze(np.array(ds['land_sea_mask'][0]))
            unique_vals = set(np.unique(mask))
            assert unique_vals <= {0.0, 1.0}

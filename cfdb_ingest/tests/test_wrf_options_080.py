"""
The 0.8.0 conversion options: ``names=`` (stored-name override), ``frame_step_minutes=`` (a strided time
axis, e.g. daily 00 UTC from hourly files), ``valid_time=`` (a one-frame static field), ``static_path=`` on
``WrfIngest`` (the wind rotation from a companion file), ``cell_methods='time: point'`` on instantaneous
fields, the strict encoding check on soil fields under ``extend``, and the off-grid ``end_date`` slot fix.

Every expected value comes from ``wrf_synthetic``'s closed forms or straight from the input file with h5py,
never from the code under test.
"""
import shutil

import cfdb
import h5py
import numpy as np
import pytest

import cfdb_ingest.base as base
from cfdb_ingest import wrf_synthetic as syn
from cfdb_ingest.wrf import WrfIngest
from cfdb_vars import short_name_map
from cfdb.utils import get_var_params

NY, NX = 6, 5
INIT = np.datetime64('2026-03-01T00', 'm')
H = np.timedelta64(1, 'h')
D = np.timedelta64(1, 'D')


def _run(tmp_path, days=4, name='run', variables=('T2', 'Q2', 'U10', 'V10', 'PREC_ACC_NC', 'PREC_ACC_C')):
    """One run from INIT, daily files of 24 hourly frames (plus WRF's single-frame end file)."""
    return syn.write_run(tmp_path / name, str(INIT), 24 * days, NY, NX, frames_per_file=24, variables=variables)


def _lead(t):
    return int((np.datetime64(t, 'm') - INIT) / H)


def _template_dtype(cfdb_name):
    """The cfdb-vars encoding of ``cfdb_name``, read from cfdb-vars itself (not through cfdb-ingest)."""
    short = cfdb_name if cfdb_name in short_name_map else None
    if short is None:
        short = next(k for k, v in short_name_map.items() if v == cfdb_name)
    return cfdb.dtypes.dtype(get_var_params(short, {'chunk_shape': None})[1]['dtype']).to_dict()


# ---------------------------------------------------------------- names=

def test_names_stores_under_override_with_template_encoding(tmp_path):
    """The variable is stored under the given name, with the KEY's cfdb-vars encoding and attrs.
    Catches: the override resolved as a template name (an unknown name falls back to plain float32)."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'n.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, names={'T2': 'temperature'})
    with cfdb.open_dataset(out) as ds:
        assert 'temperature' in ds.data_var_names and 'air_temperature' not in ds.data_var_names
        v = ds['temperature']
        assert v.dtype.to_dict() == _template_dtype('air_temperature')
        assert v.attrs['standard_name'] == 'air_temperature'
        np.testing.assert_allclose(v.data[5], syn.expected('T2', 5, NY, NX), atol=0.006)


def test_names_extend_append_matches_one_shot_and_keeps_strict_check(tmp_path, monkeypatch):
    """Extending a renamed variable appends to it (not a second variable) and still refuses an encoding change.
    Catches: the existence/strict checks looking up the template name instead of the override."""
    files = _run(tmp_path, days=2)
    kw = dict(variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(24, 3, 3), names={'T2': 'temperature'})
    split, one = tmp_path / 's.cfdb', tmp_path / 'o.cfdb'
    WrfIngest(files).convert(split, start_date=str(INIT), end_date=str(INIT + 23 * H), **kw)
    WrfIngest(files).convert(split, start_date=str(INIT + 24 * H), end_date=str(INIT + 47 * H), **kw)
    WrfIngest(files).convert(one, start_date=str(INIT), end_date=str(INIT + 47 * H), **kw)
    with cfdb.open_dataset(split) as a, cfdb.open_dataset(one) as b:
        assert a.data_var_names == ('temperature',)
        np.testing.assert_array_equal(a['temperature'].data, b['temperature'].data)

    orig = base._resolve_var_template

    def other_precision(cfdb_name, chunk_shape, dtype=None):
        name, params, attrs = orig(cfdb_name, chunk_shape, dtype)
        return name, {**params, 'dtype': cfdb.dtypes.dtype('float32', precision=1, min_value=0, max_value=5000)}, attrs

    monkeypatch.setattr(base, '_resolve_var_template', other_precision)
    with pytest.raises(ValueError, match='is encoded as'):
        WrfIngest(files).convert(split, start_date=str(INIT + 48 * H), end_date=str(INIT + 48 * H), **kw)
    with cfdb.open_dataset(split) as ds:
        assert len(ds['time'].data) == 48   # refused BEFORE the time axis was touched


def test_names_with_clip_nonneg(tmp_path):
    """clip_nonneg floors a renamed nonneg variable. Catches: the floor list built from template names."""
    files = _run(tmp_path, days=1)
    with h5py.File(files[0], 'r+') as h5:
        h5['Q2'][3, 2, 2] = -2e-6
    out = tmp_path / 'q.cfdb'
    WrfIngest(files).convert(out, variables=['Q2'], squeeze_height=True, clip_nonneg=True,
                             names={'Q2': 'water_vapour_mixing_ratio'})
    with cfdb.open_dataset(out) as ds:
        assert ds['water_vapour_mixing_ratio'].data[3, 2, 2] == 0.0


@pytest.mark.parametrize('names, match', [
    ({'PSFC': 'pressure'}, 'not among the converted'),             # key not requested
    ({'T2': 'same', 'Q2': 'same'}, 'same stored name'),             # two variables, one name
    ({'T2': 'time'}, 'coordinate'),                                 # clash with a coordinate
    ({'T2': ''}, 'non-empty string'),
])
def test_names_refusals(tmp_path, names, match):
    files = _run(tmp_path, days=1)
    with pytest.raises(ValueError, match=match):
        WrfIngest(files).convert(tmp_path / 'r.cfdb', variables=['T2', 'Q2'], squeeze_height=True, names=names)


# ---------------------------------------------------------------- frame_step_minutes=

def test_daily_stride_keeps_00utc_frames(tmp_path):
    """frame_step_minutes=1440 keeps only the frames on the epoch-aligned daily grid (00 UTC), values unchanged.
    Catches: the stride ignored, phase taken from the first frame, or values from neighbouring frames."""
    files = _run(tmp_path, days=3)
    out = tmp_path / 'd.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, frame_step_minutes=1440, extend=True,
                             chunk_shape=(4, 3, 3))
    with cfdb.open_dataset(out) as ds:
        times = ds['time'].data.astype('datetime64[m]')
        np.testing.assert_array_equal(times, INIT + np.arange(4) * D)
        for k, t in enumerate(times):
            np.testing.assert_allclose(ds['air_temperature'].data[k], syn.expected('T2', _lead(t), NY, NX), atol=0.006)


def test_daily_stride_phase_is_epoch_not_start_date(tmp_path):
    """An off-phase start_date cannot create an off-phase axis: the first slot is the next 00 UTC.
    Catches (review F4/S4b): a new dataset silently created at 06 or 23 UTC."""
    files = _run(tmp_path, days=3)
    out = tmp_path / 'p.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(4, 3, 3),
                             frame_step_minutes=1440, start_date=str(INIT + 6 * H), end_date=str(INIT + 71 * H))
    with cfdb.open_dataset(out) as ds:
        np.testing.assert_array_equal(ds['time'].data.astype('datetime64[m]'), INIT + np.arange(1, 3) * D)


def test_daily_stride_extend_split_equals_one_shot_with_gap(tmp_path):
    """create / prepend / append / overwrite on a daily axis equal a one-shot build; a skipped day is a
    placeholder. Catches: the stored step not 1440, slots misplaced across calls."""
    files = _run(tmp_path, days=6)
    kw = dict(variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(4, 3, 3), frame_step_minutes=1440)
    one, split = tmp_path / 'one.cfdb', tmp_path / 'split.cfdb'
    WrfIngest(files).convert(one, start_date=str(INIT), end_date=str(INIT + 6 * D), **kw)
    w = WrfIngest(files)
    w.convert(split, start_date=str(INIT + 2 * D), end_date=str(INIT + 3 * D), **kw)       # create
    w.convert(split, start_date=str(INIT), end_date=str(INIT + 1 * D), **kw)               # prepend
    res = w.convert(split, start_date=str(INIT + 5 * D), end_date=str(INIT + 6 * D), **kw)  # append across a gap
    assert res['gap_filled'] == 1
    w.convert(split, start_date=str(INIT + 4 * D), end_date=str(INIT + 4 * D), **kw)       # fill the gap
    w.convert(split, start_date=str(INIT + 1 * D), end_date=str(INIT + 2 * D), **kw)       # overwrite
    with cfdb.open_dataset(one) as a, cfdb.open_dataset(split) as b:
        ta, tb = a['time'].data.astype('datetime64[m]'), b['time'].data.astype('datetime64[m]')
        np.testing.assert_array_equal(ta, INIT + np.arange(7) * D)
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(a['air_temperature'].data, b['air_temperature'].data)
        assert not np.isnan(b['air_temperature'].data).any()


def test_daily_stride_off_grid_end_adds_no_slot(tmp_path):
    """end_date = 23:00 on the last day: the last slot is that day's 00 UTC, nothing missing, no extra slot.
    Catches (review S4a): np.arange(lo, hi + step) adding the next day as a placeholder."""
    files = _run(tmp_path, days=3)
    out = tmp_path / 'e.cfdb'
    res = WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(4, 3, 3),
                                   frame_step_minutes=1440, start_date=str(INIT), end_date=str(INIT + 2 * D + 23 * H))
    assert len(res['missing_frames']) == 0
    with cfdb.open_dataset(out) as ds:
        np.testing.assert_array_equal(ds['time'].data.astype('datetime64[m]'), INIT + np.arange(3) * D)


def test_hourly_off_grid_end_adds_no_slot(tmp_path):
    """The same off-by-one on the native step: end_date 22:30 must end the axis at 22:00.
    Catches: the _window_slots fix applied to the stride only."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'h.cfdb'
    res = WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(24, 3, 3),
                                   start_date=str(INIT), end_date=str(INIT + 22 * H + np.timedelta64(30, 'm')))
    assert len(res['missing_frames']) == 0
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].data.astype('datetime64[m]')[-1] == INIT + 22 * H


def test_daily_soil_stride_on_real_files(wrf_file_1, wrf_file_2, tmp_path):
    """SMOIS (4 layers) daily from two real d01 days: two 00 UTC slots, each the file's first frame, all layers.
    Catches: the stride on the soil path, a layer/depth mix-up."""
    out = tmp_path / 's.cfdb'
    WrfIngest([wrf_file_1, wrf_file_2]).convert(out, variables=['SMOIS'], extend=True, chunk_shape=(2, 4, 37, 33),
                                                frame_step_minutes=1440, names={'SMOIS': 'volumetric_water_content'})
    with cfdb.open_dataset(out) as ds:
        np.testing.assert_array_equal(ds['time'].data.astype('datetime64[m]'),
                                      np.array(['2023-02-12T00', '2023-02-13T00'], dtype='datetime64[m]'))
        v = ds['volumetric_water_content']
        for k, f in enumerate((wrf_file_1, wrf_file_2)):
            with h5py.File(f, 'r') as h5:
                np.testing.assert_allclose(v.data[k], h5['SMOIS'][0], atol=0.0006)


@pytest.mark.parametrize('kw, match', [
    (dict(variables=['T2'], frame_step_minutes=90), 'multiple of'),
    (dict(variables=['T2'], frame_step_minutes=0), 'positive'),
    (dict(variables=['PREC_ACC'], frame_step_minutes=1440), 'accumulat'),
    (dict(variables=['PREC_ACC'], frame_step_minutes=1440, time_label='start'), 'accumulat'),
])
def test_stride_refusals(tmp_path, kw, match):
    files = _run(tmp_path, days=2)
    with pytest.raises(ValueError, match=match):
        WrfIngest(files).convert(tmp_path / 'x.cfdb', squeeze_height=True, **kw)


def test_hourly_into_daily_target_refused(tmp_path):
    files = _run(tmp_path, days=3)
    out = tmp_path / 't.cfdb'
    kw = dict(variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(4, 3, 3))
    WrfIngest(files).convert(out, start_date=str(INIT), end_date=str(INIT + 1 * D), frame_step_minutes=1440, **kw)
    with pytest.raises(ValueError, match='step'):
        WrfIngest(files).convert(out, start_date=str(INIT + 2 * D), end_date=str(INIT + 2 * D + 3 * H), **kw)


# ---------------------------------------------------------------- valid_time=

def test_valid_time_relabels_one_frame(wrf_file_1, tmp_path):
    """A static field: one frame, stored at valid_time. Values straight from the file.
    Catches: the frame's own time kept, or a different frame picked."""
    out = tmp_path / 'hgt.cfdb'
    t0 = '2023-02-12T00:00'
    WrfIngest(wrf_file_1).convert(out, variables=['HGT'], squeeze_height=True, start_date=t0, end_date=t0,
                                  valid_time='1980-01-01T00:00', names={'HGT': 'altitude'})
    with cfdb.open_dataset(out) as ds:
        np.testing.assert_array_equal(ds['time'].data.astype('datetime64[m]'),
                                      np.array(['1980-01-01T00'], dtype='datetime64[m]'))
        with h5py.File(wrf_file_1, 'r') as h5:
            np.testing.assert_allclose(ds['altitude'].data[0], h5['HGT'][0], atol=0.06)


@pytest.mark.parametrize('kw, match', [
    (dict(), 'exactly one frame'),                                                   # 24 frames selected
    (dict(start_date='2023-02-12T00:00', end_date='2023-02-12T00:00', extend=True), 'extend'),
    (dict(start_date='2023-02-12T00:00', end_date='2023-02-12T00:00', frame_step_minutes=1440), 'frame_step'),
])
def test_valid_time_refusals(wrf_file_1, tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        WrfIngest(wrf_file_1).convert(tmp_path / 'v.cfdb', variables=['HGT'], squeeze_height=True,
                                      valid_time='1980-01-01T00:00', **kw)


# ---------------------------------------------------------------- static_path= on WrfIngest

def _strip_rotation(src, dst):
    shutil.copy(src, dst)
    with h5py.File(dst, 'r+') as h5:
        del h5['COSALPHA'], h5['SINALPHA']
    return dst


def test_static_path_restores_earth_winds(wrf_file_1, tmp_path):
    """A real Lambert wrfout pruned of COSALPHA/SINALPHA: without static_path the winds are withheld; with the
    original file as static_path, U10/V10/WIND_DIR10 equal a conversion of the original exactly.
    Catches: the companion's rotation not used, or used without the grid checks."""
    pruned = _strip_rotation(wrf_file_1, tmp_path / 'wrfout_d01_2023-02-12_00_00_00.nc')
    with pytest.warns(UserWarning, match='unavailable'):
        bare = WrfIngest(pruned)
    assert 'U10' not in bare.variables
    keys = ['U10', 'V10', 'WIND_DIR10']
    ref, got = tmp_path / 'ref.cfdb', tmp_path / 'got.cfdb'
    WrfIngest(wrf_file_1).convert(ref, variables=keys, squeeze_height=True)
    w = WrfIngest(pruned, static_path=wrf_file_1)
    w.convert(got, variables=keys, squeeze_height=True)
    with cfdb.open_dataset(ref) as a, cfdb.open_dataset(got) as b:
        for name in ('u_wind', 'v_wind', 'wind_direction'):
            np.testing.assert_array_equal(a[name].data, b[name].data)
        assert 'COSALPHA/SINALPHA of' in b.attrs['wind_rotation']


def test_static_path_other_grid_refused(wrf_file_1, tmp_path):
    pruned = _strip_rotation(wrf_file_1, tmp_path / 'wrfout_d01_2023-02-12_00_00_00.nc')
    other = syn.write_wrfout(tmp_path / 'other.nc', str(INIT), 1, 111, 99, map_proj=1,
                             projection=dict(dx=27000.0, truelat1=-45.0, truelat2=-45.0, stdlon=170.0))
    with pytest.raises(ValueError, match='not the same grid'):
        WrfIngest(pruned, static_path=other)


def test_static_path_without_rotation_refused(wrf_file_1, tmp_path):
    pruned = _strip_rotation(wrf_file_1, tmp_path / 'wrfout_d01_2023-02-12_00_00_00.nc')
    static = _strip_rotation(wrf_file_1, tmp_path / 'static.nc')
    with pytest.raises(ValueError, match='has no COSALPHA/SINALPHA'):
        WrfIngest(pruned, static_path=static)


# ---------------------------------------------------------------- cell_methods, nonneg, soil strict

def test_instantaneous_fields_carry_time_point(tmp_path):
    """Instantaneous fields say so; an end-labelled accumulation does not claim 'point'.
    Catches: the attribute missing, or written on accumulations."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'c.cfdb'
    WrfIngest(files).convert(out, variables=['T2', 'PREC_ACC'], squeeze_height=True)
    with cfdb.open_dataset(out) as ds:
        assert ds['air_temperature'].attrs['cell_methods'] == 'time: point'
        assert ds['precipitation'].attrs.get('cell_methods') != 'time: point'


def test_q2_sh_clip_nonneg(tmp_path):
    """A negative Q2 aborts Q2_SH unless clip_nonneg, which stores 0 (review F7)."""
    files = _run(tmp_path, days=1)
    with h5py.File(files[0], 'r+') as h5:
        h5['Q2'][3, 2, 2] = -2e-6
    with pytest.raises(ValueError, match='storable range'):
        WrfIngest(files).convert(tmp_path / 'a.cfdb', variables=['Q2_SH'], squeeze_height=True)
    out = tmp_path / 'b.cfdb'
    WrfIngest(files).convert(out, variables=['Q2_SH'], squeeze_height=True, clip_nonneg=True)
    with cfdb.open_dataset(out) as ds:
        assert ds['specific_humidity'].data[3, 2, 2] == 0.0


def test_soil_extend_encoding_enforced(wrf_file_1, wrf_file_2, tmp_path, monkeypatch):
    """Extend refuses a soil variable whose stored encoding differs from the incoming template.
    Catches (review F10): the soil path created without the strict check."""
    out = tmp_path / 's.cfdb'
    kw = dict(variables=['SMOIS'], extend=True, chunk_shape=(24, 4, 37, 33))
    WrfIngest([wrf_file_1, wrf_file_2]).convert(out, start_date='2023-02-12T00:00', end_date='2023-02-12T23:00', **kw)
    orig = base._resolve_var_template

    def other_precision(cfdb_name, chunk_shape, dtype=None):
        name, params, attrs = orig(cfdb_name, chunk_shape, dtype)
        return name, {**params, 'dtype': cfdb.dtypes.dtype('float32', precision=2, min_value=0, max_value=5)}, attrs

    monkeypatch.setattr(base, '_resolve_var_template', other_precision)
    with pytest.raises(ValueError, match='is encoded as'):
        WrfIngest([wrf_file_1, wrf_file_2]).convert(out, start_date='2023-02-13T00:00', end_date='2023-02-13T23:00',
                                                    **kw)


def test_wind_direction_is_the_from_direction(tmp_path):
    """WIND_DIR10 is the meteorological FROM direction and is labelled so (cfdb-vars >= 0.2.8). The value is checked
    with the textbook form atan2(-u, -v), a different expression from the code's (270 - atan2(v, u)).
    Catches: the 'wind_to_direction' label, or a TO-direction value (180 deg off)."""
    files = _run(tmp_path, days=1)
    out = tmp_path / 'w.cfdb'
    WrfIngest(files).convert(out, variables=['WIND_DIR10'], squeeze_height=True)
    u, v = syn.expected('U10', 5, NY, NX).astype('f8'), syn.expected('V10', 5, NY, NX).astype('f8')
    want = np.degrees(np.arctan2(-u, -v)) % 360.0
    with cfdb.open_dataset(out) as ds:
        assert ds['wind_direction'].attrs['standard_name'] == 'wind_from_direction'
        np.testing.assert_allclose(ds['wind_direction'].data[5], want, atol=0.06)


def test_clip_nonneg_reaches_a_height_suffixed_name(wrf_file_1, tmp_path):
    """Q2 next to QVAPOR is stored as 'mixing_ratio_2m'; clip_nonneg must still floor it. Before 0.8.0 the floor
    list held the bare template name 'mixing_ratio' and the suffixed variable was silently left unclipped."""
    f = tmp_path / 'wrfout_d01_2023-02-12_00_00_00.nc'
    shutil.copy(wrf_file_1, f)
    with h5py.File(f, 'r+') as h5:
        h5['Q2'][0, 3, 3] = -2e-6
    out = tmp_path / 'm.cfdb'
    WrfIngest(f).convert(out, variables=['Q2', 'QVAPOR'], target_levels=[100.0], start_date='2023-02-12T00:00',
                         end_date='2023-02-12T00:00', clip_nonneg=True)
    with cfdb.open_dataset(out) as ds:
        assert 'mixing_ratio_2m' in ds.data_var_names
        assert ds['mixing_ratio_2m'].data[0, 0, 3, 3] == 0.0


# ---------------------------------------------------------------- review round cfdb-ingest-080-code-1

def test_stride_without_extend_refused(tmp_path):
    """Without extend the axis would be compacted with an inferred step (a missing day closes up; review G2/F5)."""
    files = _run(tmp_path, days=2)
    with pytest.raises(ValueError, match='needs extend=True'):
        WrfIngest(files).convert(tmp_path / 'x.cfdb', variables=['T2'], squeeze_height=True, frame_step_minutes=1440)


def test_stride_phase_with_a_run_starting_at_06utc(tmp_path):
    """The first frame is 06 UTC: the daily axis is still 00 UTC (not 06), each slot the frame of its own time.
    Catches: the phase taken from the first input frame (review: survived every earlier test)."""
    files = syn.write_run(tmp_path / 'r6', str(INIT + 6 * H), 96, NY, NX, variables=('T2',))
    out = tmp_path / 'p6.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(4, 3, 3),
                             frame_step_minutes=1440)
    with cfdb.open_dataset(out) as ds:
        times = ds['time'].data.astype('datetime64[m]')
        np.testing.assert_array_equal(times, INIT + np.arange(1, 5) * D)
        for k, t in enumerate(times):
            lead = int((t - (INIT + 6 * H)) / H)
            np.testing.assert_allclose(ds['air_temperature'].data[k], syn.expected('T2', lead, NY, NX), atol=0.006)


def test_valid_time_takes_the_selected_frame(tmp_path):
    """Frame 30 of a 72 h run, a time-VARYING field: stored at valid_time with frame 30's values (review G5: the HGT
    test cannot tell frames apart, and selected frame 0)."""
    files = _run(tmp_path, days=3)
    t30 = str(INIT + 30 * H)
    out = tmp_path / 'v30.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], squeeze_height=True, start_date=t30, end_date=t30,
                             valid_time='1980-01-01T00:00')
    with cfdb.open_dataset(out) as ds:
        assert ds['time'].data.astype('datetime64[m]').tolist() == [np.datetime64('1980-01-01T00:00', 'm')]
        np.testing.assert_allclose(ds['air_temperature'].data[0], syn.expected('T2', 30, NY, NX), atol=0.006)
        assert not np.allclose(ds['air_temperature'].data[0], syn.expected('T2', 29, NY, NX), atol=0.006)


@pytest.mark.parametrize('kw, match', [
    (dict(variables=['PREC_ACC']), 'accumulations'),
    (dict(variables=['T2'], valid_time=0), 'date/time'),
    (dict(variables=['T2'], valid_time=''), 'date/time'),
])
def test_valid_time_more_refusals(tmp_path, kw, match):
    files = _run(tmp_path, days=1)
    kw = {'valid_time': '1980-01-01T00:00', **kw}
    with pytest.raises(ValueError, match=match):
        WrfIngest(files).convert(tmp_path / 'v.cfdb', squeeze_height=True, start_date=str(INIT + 3 * H),
                                 end_date=str(INIT + 3 * H), **kw)


@pytest.mark.parametrize('opt', [dict(frame_step_minutes=1440), dict(valid_time='1980-01-01T00:00')])
def test_stride_and_valid_time_refused_in_forecast_mode(tmp_path, opt):
    files = _run(tmp_path, days=1)
    with pytest.raises(ValueError, match='grid mode only'):
        WrfIngest(files).convert(tmp_path / 'f.cfdb', variables=['T2'], dataset_type='grid_forecast', **opt)


@pytest.mark.parametrize('name', ['x', 'y', 'height_2m', 'crs', 'latitude'])
def test_names_coordinate_like_refused(tmp_path, name):
    files = _run(tmp_path, days=1)
    with pytest.raises(ValueError, match='coordinate'):
        WrfIngest(files).convert(tmp_path / 'c.cfdb', variables=['T2'], squeeze_height=True, names={'T2': name})


def test_names_override_cannot_take_another_variables_default_name(tmp_path):
    files = _run(tmp_path, days=1)
    with pytest.raises(ValueError, match='same stored name'):
        WrfIngest(files).convert(tmp_path / 'm.cfdb', variables=['T2', 'Q2'], squeeze_height=True,
                                 names={'T2': 'mixing_ratio'})


def test_names_extend_refuses_another_key_into_a_named_variable(tmp_path):
    """TSK into T2's 'temperature': same encoding, so the strict check passed it (review F1: silent wrong field)."""
    files = _run(tmp_path, days=1, variables=('T2', 'TSK'))
    kw = dict(squeeze_height=True, extend=True, chunk_shape=(24, 3, 3), start_date=str(INIT),
              end_date=str(INIT + 23 * H))
    out = tmp_path / 'k.cfdb'
    WrfIngest(files).convert(out, variables=['T2'], names={'T2': 'temperature'}, **kw)
    with pytest.raises(ValueError, match='holds T2'):
        WrfIngest(files).convert(out, variables=['TSK'], names={'TSK': 'temperature'}, **kw)
    with cfdb.open_dataset(out) as ds:
        np.testing.assert_allclose(ds['temperature'].data[5], syn.expected('T2', 5, NY, NX), atol=0.006)


@pytest.mark.parametrize('names', [None, {'T2': 'temperatrue'}])
def test_names_extend_refuses_a_forgotten_or_misspelt_name(tmp_path, names):
    """A band without names= (or a misspelt one) would start a second variable and report a normal append
    (review F2: silent holes under both names)."""
    files = _run(tmp_path, days=2)
    kw = dict(variables=['T2'], squeeze_height=True, extend=True, chunk_shape=(24, 3, 3))
    out = tmp_path / 'f.cfdb'
    WrfIngest(files).convert(out, start_date=str(INIT), end_date=str(INIT + 23 * H), names={'T2': 'temperature'}, **kw)
    with pytest.raises(ValueError, match="already stored as \\['temperature'\\]"):
        WrfIngest(files).convert(out, start_date=str(INIT + 24 * H), end_date=str(INIT + 47 * H), names=names, **kw)
    with cfdb.open_dataset(out) as ds:
        assert ds.data_var_names == ('temperature',) and len(ds['time'].data) == 24


def test_forecast_append_refuses_a_forgotten_name(tmp_path):
    a = syn.write_run(tmp_path / 'a', str(INIT), 6, NY, NX, variables=('T2',))
    b = syn.write_run(tmp_path / 'b', str(INIT + 6 * H), 6, NY, NX, variables=('T2',))
    out = tmp_path / 'fc.cfdb'
    WrfIngest(a).convert(out, variables=['T2'], dataset_type='grid_forecast', names={'T2': 'temperature'})
    with pytest.raises(ValueError, match='already stored as'):
        WrfIngest(b).convert(out, variables=['T2'], dataset_type='grid_forecast')


def test_soil_extend_chunk_mismatch_refused_before_axis_touched(wrf_file_1, wrf_file_2, tmp_path):
    out = tmp_path / 's.cfdb'
    kw = dict(variables=['SMOIS'], extend=True, names={'SMOIS': 'volumetric_water_content'})
    WrfIngest([wrf_file_1, wrf_file_2]).convert(out, start_date='2023-02-12T00:00', end_date='2023-02-12T23:00',
                                                chunk_shape=(24, 4, 37, 33), **kw)
    with pytest.raises(ValueError, match='chunk_shape'):
        WrfIngest([wrf_file_1, wrf_file_2]).convert(out, start_date='2023-02-13T00:00', end_date='2023-02-13T23:00',
                                                    chunk_shape=(12, 4, 37, 33), **kw)
    with cfdb.open_dataset(out) as ds:
        assert len(ds['time'].data) == 24


def test_cell_methods_on_soil_and_level_but_not_forecast(wrf_file_1, tmp_path):
    a, b = tmp_path / 'a.cfdb', tmp_path / 'b.cfdb'
    w = WrfIngest(wrf_file_1)
    w.convert(a, variables=['SMOIS', 'T'], target_levels=[500.0], start_date='2023-02-12T00:00',
              end_date='2023-02-12T01:00')
    with cfdb.open_dataset(a) as ds:
        assert ds['soil_moisture'].attrs['cell_methods'] == 'time: point'
        assert ds['air_temperature'].attrs['cell_methods'] == 'time: point'
    files = _run(tmp_path, days=1)
    WrfIngest(files).convert(b, variables=['T2'], dataset_type='grid_forecast')
    with cfdb.open_dataset(b) as ds:
        assert 'cell_methods' not in ds['air_temperature'].attrs.data


def test_soil_chunks_written_once(wrf_file_1, wrf_file_2, tmp_path, monkeypatch):
    """SMOIS goes through the cross-file rechunker: every (time, depth, y, x) chunk is stored once, not once per
    input file and layer (review F4: 48 writes per chunk for 12 files x 4 layers)."""
    import collections
    import booklet
    counts = collections.Counter()
    orig = booklet.main.VariableLengthValue.set

    def spy(self, key, value, *a, **k):
        if isinstance(key, str) and key.startswith('soil_moisture!'):
            counts[key] += 1
        return orig(self, key, value, *a, **k)

    monkeypatch.setattr(booklet.main.VariableLengthValue, 'set', spy)
    WrfIngest([wrf_file_1, wrf_file_2]).convert(tmp_path / 'w.cfdb', variables=['SMOIS'], extend=True,
                                                chunk_shape=(48, 4, 37, 33))
    assert counts and set(counts.values()) == {1}, counts


def test_static_path_of_another_grid_refused_even_when_inputs_carry_cosalpha(wrf_file_1, tmp_path):
    """The inputs' own rotation is used, but a companion of another grid is still a caller error (review F7)."""
    other = syn.write_wrfout(tmp_path / 'other.nc', str(INIT), 1, 111, 99, map_proj=1,
                             projection=dict(dx=27000.0, truelat1=-45.0, truelat2=-45.0, stdlon=170.0))
    with pytest.raises(ValueError, match='not the same grid'):
        WrfIngest(wrf_file_1, static_path=other)


def test_wind_direction_on_a_rotated_grid_against_the_geography(wrf_file_1, tmp_path):
    """Real Lambert d01 (rotation > 20 deg), rotation borrowed via static_path: the stored FROM direction agrees
    with one built from the grid's own geography (finite differences of XLAT/XLONG), not from any rotation formula.
    Interior cells with |V| > 1 m/s; tolerance measured by review F (max 0.051 deg).
    Catches: rotation skipped (~25-53 deg), sign flipped, or a TO direction (180 deg)."""
    pruned = _strip_rotation(wrf_file_1, tmp_path / 'wrfout_d01_2023-02-12_00_00_00.nc')
    out = tmp_path / 'dir.cfdb'
    t = '2023-02-12T06:00'
    WrfIngest(pruned, static_path=wrf_file_1).convert(out, variables=['WIND_DIR10'], squeeze_height=True,
                                                      start_date=t, end_date=t)
    with h5py.File(wrf_file_1, 'r') as h5:
        k = [b''.join(r).decode() for r in h5['Times'][:]].index('2023-02-12_06:00:00')
        u, v = h5['U10'][k].astype('f8'), h5['V10'][k].astype('f8')
        ex, ey = syn._fd_basis(h5['XLAT'][0], h5['XLONG'][0])
    ue, ve = u * ex[0] + v * ey[0], u * ex[1] + v * ey[1]
    want = np.degrees(np.arctan2(-ue, -ve)) % 360.0
    with cfdb.open_dataset(out) as ds:
        got = ds['wind_direction'].data[0].astype('f8')
    inner = (slice(1, -1), slice(1, -1))
    mask = np.hypot(u, v)[inner] > 1.0
    d = np.abs((got[inner] - want[inner] + 180.0) % 360.0 - 180.0)[mask]
    assert mask.sum() > 1000 and d.max() < 0.2, (mask.sum(), d.max())

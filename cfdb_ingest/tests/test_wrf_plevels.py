"""
WRF pressure-level diagnostics (``wrfplevels``, stream auxhist23) -> cfdb, and the frame guard on WrfIngest.

Each test names the defect it exists to catch; the rotation tests use routes independent of the code under
test (a real wrfout's COSALPHA/SINALPHA, and the finite-difference grid basis of XLAT/XLONG).
"""
import pathlib
import warnings

import cfdb
import h5py
import numpy as np
import pytest

import cfdb_ingest.wrf_synthetic as syn
from cfdb_ingest.base import _check_level_count
from cfdb_ingest.wrf import WRF_PLEV_VARIABLE_MAPPING, WrfIngest, WrfPlevIngest, grid_rotation

NY, NX, NT = 20, 30, 8
LEVELS_ASC = sorted(syn.PLEV_LEVELS)


def _plev(tmp_path, name='wrfplevels_d01_2026-01-01_00_00_00.nc', **kw):
    kw.setdefault('n_times', NT)
    return syn.write_wrfplevels(tmp_path / name, '2026-01-01T00', kw.pop('n_times'), NY, NX, **kw)


def _convert(ing, path, **kw):
    kw.setdefault('chunk_shape', (NT, 1, NY, NX))
    result = ing.convert(path, **kw)
    with cfdb.open_dataset(path) as ds:
        data = {name: np.asarray(ds[name].data) for name in ds.data_var_names}
        data['pressure'] = np.asarray(ds['pressure'].data)
    return result, data


def _leads(n=NT, step=3):
    return [k * step for k in range(n)]


# ------------------------------------------------------------------ T0: the rotation formula vs real WRF output

def test_rotation_formula_matches_real_cosalpha(wrf_file_1):
    """The analytic rotation reproduces WRF's own COSALPHA/SINALPHA on a real SH tangent-cone Lambert d01 -- an
    anchor outside this package, so a hemisphere or wrap error cannot pass by agreeing with a synthetic built by
    the same formula. Catches: a flipped hemisphere sign, a missing antimeridian wrap, the secant cone on a
    tangent grid."""
    with h5py.File(wrf_file_1, 'r') as h5:
        a = {k: float(np.asarray(h5.attrs[k]).item()) for k in ('MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON')}
        xlong = h5['XLONG'][0]
        alpha_wrf = np.arctan2(h5['SINALPHA'][0].astype('f8'), h5['COSALPHA'][0].astype('f8'))
    cosa, sina, validated = grid_rotation(int(a['MAP_PROJ']), a['TRUELAT1'], a['TRUELAT2'], a['STAND_LON'], xlong)
    assert validated
    assert np.degrees(np.abs(alpha_wrf)).max() > 20.0   # a sign error would be ~2*alpha here, not ~0
    assert np.max(np.abs(np.arctan2(sina, cosa) - alpha_wrf)) < 1e-5


def test_rotation_formula_branches():
    lon = np.array([[170.0, 180.0, -170.0]])
    c, s, ok = grid_rotation(6, 0.0, 0.0, 0.0, lon)
    assert ok and np.all(c == 1) and np.all(s == 0)
    c, s, ok = grid_rotation(3, -40.0, -40.0, 170.0, lon)
    assert not ok and np.all(s == 0)                     # Mercator: formula, but unvalidated
    _, _, ok = grid_rotation(1, -30.0, -60.0, 170.0, lon)
    assert not ok                                         # secant cone: unvalidated
    _, _, ok = grid_rotation(2, -60.0, -60.0, 170.0, lon)
    assert not ok                                         # polar stereographic: unvalidated
    with pytest.raises(ValueError, match='rotated-pole'):
        grid_rotation(6, 0.0, 0.0, 0.0, lon, pole_lat=40.0)
    # continuity across the antimeridian: 180 and -170 are 10 deg apart, not 350
    c, s, _ = grid_rotation(1, -45.0, -45.0, 175.0, lon)
    a = np.degrees(np.arctan2(s, c))[0]
    assert np.allclose(np.diff(a), 10.0 * np.sin(np.radians(45.0)) * -1.0, atol=1e-6)


# ------------------------------------------------------------------ T1: rotation recovers the earth wind

def _alpha(path):
    with h5py.File(path, 'r') as h5:
        c, s, _ = grid_rotation(1, -45.0, -45.0, 175.0, h5['XLONG'][0])
    return np.arctan2(s, c)


@pytest.mark.parametrize('with_static', [False, True])
def test_rotation_recovers_earth_wind(tmp_path, cfdb_out, with_static):
    """U_PL/V_PL were written grid-relative from a known earth-relative wind through the finite-difference basis of
    XLAT/XLONG (wrf_synthetic._fd_basis) -- never through a rotation formula -- so recovering that wind tests the
    ingest's rotation independently. Tolerance measured: interior FD angle error <= 6e-6 rad (x 21 m/s = 1e-4 m/s)
    plus the 0.01 m/s packing; the one-sided edge ring reaches 5.3e-3 rad (0.11 m/s) and is bounded separately.
    Catches: a flipped sin sign (~7 m/s), cos/sin swapped, rotation skipped, a companion from another grid."""
    f = _plev(tmp_path, mountain=False)
    static = None
    if with_static:
        static = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX,
                                  map_proj=1, projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0,
                                  variables=('T2',))
    ing = WrfPlevIngest(f, static_path=static)
    assert ('wrfout' in ing._rotation_source) is with_static
    alpha = _alpha(f)
    interior = np.zeros((NY, NX), bool)
    interior[1:-1, 1:-1] = True
    assert np.degrees(np.abs(alpha[interior])).max() >= 5.0   # the test is blind where alpha ~ 0

    _, d = _convert(ing, cfdb_out, variables=['U_PL', 'V_PL'])
    for t, lead in enumerate(_leads()):
        for k, p in enumerate(LEVELS_ASC):
            ue, ve = syn.plev_earth_wind(p, lead)
            err = np.maximum(np.abs(d['u_wind'][t, k] - ue), np.abs(d['v_wind'][t, k] - ve))
            assert err[interior].max() < 0.01, (p, lead, err[interior].max())
            assert err[~interior].max() < 0.2, (p, lead, err[~interior].max())


def test_rotated_sentinel_cannot_escape(tmp_path, cfdb_out):
    """-999 in U_PL/V_PL becomes NaN BEFORE rotation: a rotated sentinel would be a plausible-looking number
    (-999*cos + -999*sin), not -999, and no later check could recognise it."""
    f = _plev(tmp_path)
    _, d = _convert(WrfPlevIngest(f), cfdb_out, variables=['U_PL', 'V_PL'])
    masks = syn.plev_masks(NY, NX)
    for k, p in enumerate(LEVELS_ASC):
        m = masks[('other', p)]
        assert np.isnan(d['u_wind'][:, k][:, m]).all() and np.isnan(d['v_wind'][:, k][:, m]).all()
        assert np.isfinite(d['u_wind'][:, k][:, ~m]).all()


# ------------------------------------------------------------------ T2: levels land on their own pressure

def _check_levels(d, levels, var='geopotential_height', name='GHT_PL'):
    assert d['pressure'].tolist() == sorted(levels)
    for k, p in enumerate(sorted(levels)):
        exp = syn.plev_expected(name, p, 0, NY, NX)
        got = d[var][0, k]
        ok = np.isfinite(got)
        assert np.allclose(got[ok], exp[ok], atol=0.06), (var, p)


@pytest.mark.parametrize('order', ['descending', 'ascending'])
@pytest.mark.parametrize('subset', [None, [50000.0, 85000.0]])
def test_levels_land_on_their_pressure(tmp_path, cfdb_out, order, subset):
    """Every level's values sit on its own pressure label, for a single-source field (GHT) and the rotated winds
    (multi-source path), on files storing the levels descending (WRF) or ascending, full set or subset.
    Catches: the order-blind native-level shortcut (a descending file mirror-swapped), and the multi-source path
    ignoring the level selection (an ascending subset put 700 hPa in the 850 slot)."""
    levels = syn.PLEV_LEVELS if order == 'descending' else LEVELS_ASC
    f = _plev(tmp_path, levels=levels, mountain=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # the ascending file is (rightly) warned about
        ing = WrfPlevIngest(f)
    _, d = _convert(ing, cfdb_out, variables=['GHT_PL', 'U_PL', 'V_PL'], target_levels=subset)
    want = subset if subset is not None else levels
    _check_levels(d, want)
    for k, p in enumerate(sorted(want)):
        ue, ve = syn.plev_earth_wind(p, 0)
        assert abs(np.nanmedian(d['u_wind'][0, k]) - ue) < 0.01 and abs(np.nanmedian(d['v_wind'][0, k]) - ve) < 0.01


def test_level_count_invariant():
    """A block whose level axis does not match its output indices is refused, not truncated or mislabelled."""
    class _V:
        name = 'x'
    _check_level_count(_V(), 5, [0, 1, 2, 3, 4])
    _check_level_count(_V(), None, [0])
    with pytest.raises(ValueError, match='silent mislabel'):
        _check_level_count(_V(), 5, [0, 1])


# ------------------------------------------------------------------ T3: missing values

@pytest.mark.parametrize('chunk_t', [NT, 3])
def test_missing_values_become_nan_per_variable(tmp_path, cfdb_out, chunk_t):
    """-999 -> NaN in every variable, GHT explicitly: GHT is the one field whose packed range (offset -1001 m)
    would store -999 as a real height, so check_encodable cannot catch it. The masks differ per variable (GHT is
    interpolated on full levels and stays valid above the surface where T is missing), and masked_cells counts them
    -- for the single-source fields AND the rotated winds (multi-source path), over one time block and over several
    (chunk_t=3: 8 frames in 3 blocks). Catches: counts overwritten per block instead of summed; counts taken before the
    level pick in the multi-source path (mirrored onto the wrong pressures of a descending file)."""
    f = _plev(tmp_path)
    result, d = _convert(WrfPlevIngest(f), cfdb_out, chunk_shape=(chunk_t, 1, NY, NX))
    masks = syn.plev_masks(NY, NX)
    for k, p in enumerate(LEVELS_ASC):
        mg, mo = masks[('GHT', p)], masks[('other', p)]
        g, t = d['geopotential_height'][:, k], d['air_temperature'][:, k]
        assert np.isnan(g[:, mg]).all() and np.isfinite(g[:, ~mg]).all()
        assert np.isnan(t[:, mo]).all() and np.isfinite(t[:, ~mo]).all()
        assert result['masked_cells']['GHT_PL'][p] == mg.sum() * NT
        for src in ('T_PL', 'U_PL', 'V_PL'):
            assert result['masked_cells'][src][p] == mo.sum() * NT, (src, p, result['masked_cells'][src][p])
        assert np.nanmin(g) > 0.0          # no -999 survived anywhere
    p_hi = max(LEVELS_ASC)
    band = masks[('other', p_hi)] & ~masks[('GHT', p_hi)]
    assert band.any() and np.isfinite(d['geopotential_height'][:, -1][:, band]).all()


def test_whole_level_missing_warns(tmp_path, cfdb_out):
    """A level missing everywhere (above the model top, or a non-descending press_levels) is reported."""
    f = _plev(tmp_path, missing_levels=(30000.0,), mountain=False)
    with pytest.warns(UserWarning, match='missing everywhere'):
        result, d = _convert(WrfPlevIngest(f), cfdb_out, variables=['GHT_PL'])
    assert np.isnan(d['geopotential_height'][:, 0]).all()
    assert result['masked_cells']['GHT_PL'][30000.0] == NY * NX * NT


def test_relative_humidity_is_a_clipped_fraction(tmp_path, cfdb_out):
    f = _plev(tmp_path, mountain=False)
    _, d = _convert(WrfPlevIngest(f), cfdb_out, variables=['RH_PL'])
    rh = d['relative_humidity']
    assert np.nanmax(rh) <= 1.0 and np.all(rh[:, :, 0, 0] == 1.0)   # WRF's 104 % cell
    assert np.allclose(rh[0, 2, 1:, 1:], syn.plev_expected('RH_PL', LEVELS_ASC[2], 0, NY, NX)[1:, 1:], atol=0.002)


def test_native_level_variable_on_batch_path_refused(tmp_path, cfdb_out, monkeypatch):
    """The per-timestep batch path skips the missing-value hook and the level selection; a native-level variable
    whose transform is not a registered block transform (a typo) must be refused, not written unmasked."""
    f = _plev(tmp_path)
    mapping = dict(WRF_PLEV_VARIABLE_MAPPING)
    mapping['T_PL'] = dict(mapping['T_PL'], transform='not_registered')
    monkeypatch.setattr('cfdb_ingest.wrf.WRF_PLEV_VARIABLE_MAPPING', mapping)
    with pytest.raises(RuntimeError, match='per-timestep path'):
        WrfPlevIngest(f).convert(cfdb_out, variables=['T_PL'])


# ------------------------------------------------------------------ T4: refusals, each with an ok_ control

POLAR = dict(dx=20000.0, truelat1=-60.0, truelat2=-60.0, stdlon=170.0)


def test_unvalidated_projection_needs_a_companion(tmp_path, cfdb_out):
    """Polar stereographic: the formula is not validated against real WRF output, so without a companion wrfout
    the winds are withheld (with the reason) while scalars convert; with one whose COSALPHA agrees they convert."""
    f = _plev(tmp_path, map_proj=2, projection=POLAR, lat0=-55.0, lon0=160.0, mountain=False)
    with pytest.warns(UserWarning, match='unavailable'):
        ing = WrfPlevIngest(f)
    assert 'U_PL' not in ing.variables and 'GHT_PL' in ing.variables
    with pytest.raises(ValueError, match='static_path'):
        ing.convert(cfdb_out, variables=['U_PL'])
    ing.convert(cfdb_out, variables=['GHT_PL'])                     # ok_: scalars need no rotation

    static = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX,
                              map_proj=2, projection=POLAR, lat0=-55.0, lon0=160.0, variables=('T2',))
    ing = WrfPlevIngest(f, static_path=static)
    assert 'U_PL' in ing.variables
    _, d = _convert(ing, tmp_path / 'b.cfdb', variables=['U_PL'])
    assert np.isfinite(d['u_wind']).all()


def test_companion_from_another_grid_refused(tmp_path):
    f = _plev(tmp_path, mountain=False)
    other = syn.write_wrfout(tmp_path / 'wrfout_other.nc', '2026-01-01T00', 1, NY, NX, map_proj=1,
                             projection=dict(syn.PLEV_PROJECTION, stdlon=170.0), lat0=-50.0, lon0=160.0,
                             variables=('T2',))
    with pytest.raises(ValueError, match='not the same grid'):
        WrfPlevIngest(f, static_path=other)


def test_companion_with_shifted_cells_refused(tmp_path):
    """Same header, cell centres 0.01 deg (~1 km) off: refused by the XLAT/XLONG check alone (the header check
    cannot see it)."""
    f = _plev(tmp_path, mountain=False)
    static = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX,
                              map_proj=1, projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0,
                              variables=('T2',))
    with h5py.File(static, 'r+') as h5:
        h5['XLAT'][...] = h5['XLAT'][...] + np.float32(0.01)
    with pytest.raises(ValueError, match='XLAT/XLONG differ'):
        WrfPlevIngest(f, static_path=static)


def test_companion_with_different_header_refused(tmp_path):
    """Same cell centres, different grid header (here DX): refused by the attribute check alone."""
    f = _plev(tmp_path, mountain=False)
    static = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX,
                              map_proj=1, projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0,
                              variables=('T2',))
    with h5py.File(static, 'r+') as h5:
        h5.attrs['DX'] = np.float32(12000.0)
    with pytest.raises(ValueError, match='DX'):
        WrfPlevIngest(f, static_path=static)


def test_companion_with_wrong_rotation_refused(tmp_path):
    """Same grid, but COSALPHA/SINALPHA that disagree with the formula (sign flipped): refused, not trusted."""
    f = _plev(tmp_path, mountain=False)
    static = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX,
                              map_proj=1, projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0,
                              variables=('T2',))
    with h5py.File(static, 'r+') as h5:
        h5['SINALPHA'][...] = -h5['SINALPHA'][...]
    with pytest.raises(ValueError, match='disagrees'):
        WrfPlevIngest(f, static_path=static)


def test_pressure_levels_must_agree_across_files(tmp_path):
    a = _plev(tmp_path, name='wrfplevels_d01_2026-01-01_00_00_00.nc')
    b = syn.write_wrfplevels(tmp_path / 'wrfplevels_d01_2026-01-02_00_00_00.nc', '2026-01-02T00', NT, NY, NX,
                             levels=(90000.0, 85000.0, 50000.0))
    with pytest.raises(ValueError, match='one set of pressure levels'):
        WrfPlevIngest([a, b])
    WrfPlevIngest([a, syn.write_wrfplevels(tmp_path / 'wrfplevels_d01_2026-01-03_00_00_00.nc', '2026-01-02T00',
                                           NT, NY, NX)])          # ok_: same levels


def test_field_off_the_pressure_axis_refused(tmp_path):
    """A *_PL field whose level length differs from P_PL (a file split or pruned per level) is refused: P_PL
    would not describe its axis."""
    f = _plev(tmp_path, mountain=False)
    with h5py.File(f, 'r+') as h5:
        three = h5['GHT_PL'][:, :3]
        del h5['GHT_PL']
        h5.create_dataset('GHT_PL', data=three)
    with pytest.raises(ValueError, match='P_PL has 5 levels'):
        WrfPlevIngest(f)


def test_convert_options_refused(tmp_path, cfdb_out):
    ing = WrfPlevIngest(_plev(tmp_path, mountain=False))
    for kw in (dict(vertical_coord='height'), dict(extend=True), dict(dataset_type='grid_forecast'),
               dict(squeeze_height=True), dict(time_label='start')):
        with pytest.raises(ValueError):
            ing.convert(cfdb_out, variables=['GHT_PL'], **kw)


def test_directory_input_globs_plevels(tmp_path):
    _plev(tmp_path, mountain=False)
    syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 1, NY, NX, map_proj=1,
                     projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0, variables=('T2',))
    assert [p.name for p in WrfPlevIngest(tmp_path).input_paths] == ['wrfplevels_d01_2026-01-01_00_00_00.nc']


# ------------------------------------------------------------------ T5: WrfIngest withholds winds it cannot rotate

def test_wrfingest_withholds_unrotatable_winds(tmp_path):
    """An auxiliary-stream file carries U10/V10 but no COSALPHA. Before 0.7.0 WrfIngest stored them unrotated as
    eastward/northward wind (13 m/s wrong on a real file). Now the rotated keys are withheld with a warning, a
    request for them is refused with the reason -- including 'U10', which would otherwise resolve by source name to
    WIND10, the speed -- and the frame-invariant speed stays."""
    f = _plev(tmp_path, mountain=False)
    with pytest.warns(UserWarning, match='unavailable'):
        ing = WrfIngest(f)
    for key in ('U10', 'V10', 'WIND_DIR10'):
        assert key not in ing.variables
    assert 'WIND10' in ing.variables
    for request in (['U10'], ['u_wind'], ['WIND_DIR10']):
        with pytest.raises(ValueError, match='rotation'):
            ing.resolve_variables(request)
    assert ing.resolve_variables(['WIND10']) == ['WIND10']


def test_wrfingest_keeps_native_vimf_without_cosalpha(tmp_path):
    """Keyed on the ACTIVE transform: WRF writes native VIMF_U/V earth-relative, so it stays available."""
    p = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 2, NY, NX, map_proj=1,
                         projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0, variables=('T2', 'U10', 'V10'))
    with h5py.File(p, 'r+') as h5:
        del h5['COSALPHA'], h5['SINALPHA']
        h5.create_dataset('VIMF_U', data=np.ones((2, NY, NX), 'f4'))
        h5.create_dataset('VIMF_V', data=np.ones((2, NY, NX), 'f4'))
    with pytest.warns(UserWarning):
        ing = WrfIngest(p)
    assert 'VIMF_U' in ing.variables and 'U10' not in ing.variables


def test_wrfingest_rotation_known_ok_controls(wrf_file_1, tmp_path):
    """ok_: a real projected wrfout with COSALPHA, and a lat-lon grid without it (no rotation needed), keep every
    wind key and raise no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        ing = WrfIngest(wrf_file_1)
        assert {'U10', 'V10', 'WIND_DIR10'} <= set(ing.variables)
        p = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 2, NY, NX,
                             variables=('T2', 'U10', 'V10'))
        with h5py.File(p, 'r+') as h5:
            del h5['COSALPHA'], h5['SINALPHA']
        assert 'U10' in WrfIngest(p).variables


def test_synthetic_projected_wrfout_carries_true_rotation(tmp_path):
    """wrf_synthetic wrote an identity COSALPHA for every projection before 0.7.0 -- a grid no real wrfout has."""
    p = syn.write_wrfout(tmp_path / 'wrfout_d01.nc', '2026-01-01T00', 1, NY, NX, map_proj=1,
                         projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0, variables=('T2',))
    with h5py.File(p, 'r') as h5:
        assert np.degrees(np.abs(np.arcsin(h5['SINALPHA'][0]))).max() > 5.0


# ------------------------------------------------------------------ chunking: every output chunk written once, every
# source frame read once (the chunked-I/O contract of 0.6.0, on the new level-selecting paths)

@pytest.fixture
def io_counts(monkeypatch):
    """
    (chunk writes per booklet key, source HDF5 chunks touched per (variable, file, chunk index)).

    Reads are counted per HDF5 CHUNK a selection overlaps, not per frame: WRF tiles each frame 2 x 2 (the
    real layout, see write_wrfplevels), so a reader may touch a frame several times while decompressing each
    tile once. A chunk (> h5py's 1 MB default chunk cache at production size) touched twice is decompressed
    twice.
    """
    import collections
    import itertools

    import booklet

    import cfdb_ingest.base as base
    writes = collections.Counter()
    reads = collections.Counter()
    orig_set = booklet.main.VariableLengthValue.set
    orig_call = base._ConcatTimeSource.__call__

    def spy_set(self, key, value, *a, **k):
        if isinstance(key, str) and '!' in key:
            writes[key] += 1
        return orig_set(self, key, value, *a, **k)

    def spy_call(self, slices):
        t = slices[0]
        for i, ds in enumerate(self._datasets):
            f0, f1 = int(self._cum[i]), int(self._cum[i + 1])
            lo, hi = max(t.start, f0) - f0, min(t.stop, f1) - f0
            if lo >= hi:
                continue
            sel = (slice(lo, hi),) + tuple(slices[1:])
            chunks = ds.chunks or ds.shape
            spans = [range(s.start // c, (s.stop - 1) // c + 1) for s, c in zip(sel, chunks)]
            for idx in itertools.product(*spans):
                reads[(ds.name.lstrip('/'), i) + idx] += 1
        return orig_call(self, slices)

    monkeypatch.setattr(booklet.main.VariableLengthValue, 'set', spy_set)
    monkeypatch.setattr(base._ConcatTimeSource, '__call__', spy_call)
    return writes, reads


@pytest.mark.parametrize('subset', [None, [50000.0, 85000.0]])
def test_each_chunk_written_once_each_frame_read_once(tmp_path, io_counts, subset):
    """Three daily files of 8 frames, a time chunk of 12 that straddles them, the first frame dropped (so blocks
    must align to the OUTPUT chunk grid, not the input), for the single-source path (GHT) and the rotating
    multi-source path (U/V), all levels and a descending-file subset. Every output chunk is written exactly once,
    and every HDF5 chunk holding a kept frame is decompressed exactly once, on WRF's real chunk layout (one
    frame x all levels x a 2 x 2 spatial tile, measured on a real wrfzlevels file)."""
    files = [syn.write_wrfplevels(tmp_path / f'wrfplevels_d01_2026-01-0{d}_00_00_00.nc', f'2026-01-0{d}T00', 8,
                                  NY, NX, simulation_start='2026-01-01T00', mountain=False) for d in (1, 2, 3)]
    ing = WrfPlevIngest(files)
    writes, reads = io_counts
    out = tmp_path / 'o.cfdb'
    result = ing.convert(out, variables=['GHT_PL', 'U_PL', 'V_PL'], target_levels=subset,
                         chunk_shape=(12, 1, NY, NX), start_date=str(ing.times[1]))
    n_lev = len(subset) if subset else len(LEVELS_ASC)
    n_time_chunks = int(np.ceil(result['n_times'] / 12))   # a new dataset's time axis starts at a chunk boundary
    for var in ('geopotential_height', 'u_wind', 'v_wind'):
        keys = {k: n for k, n in writes.items() if k.startswith(var + '!')}
        assert all(n == 1 for n in keys.values()), keys
        assert len(keys) == n_time_chunks * n_lev, (var, len(keys))
    for src in ('GHT_PL', 'U_PL', 'V_PL'):
        per_chunk = [n for key, n in reads.items() if key[0] == src]
        assert per_chunk and max(per_chunk) == 1, (src, max(per_chunk))
        assert len(per_chunk) == result['n_times'] * 4        # each kept frame's 2 x 2 tiles, all levels in one


# ------------------------------------------------------------------ added after code review cfdb-ingest-plev-code-1
# (each kills a mutant the review showed surviving the whole suite)

def test_masked_cells_keyed_by_pressure_for_unsorted_subset(tmp_path, cfdb_out):
    """target_levels given out of order: masked_cells must still be keyed by the right pressure."""
    f = _plev(tmp_path)
    result, _ = _convert(WrfPlevIngest(f), cfdb_out, variables=['T_PL'], target_levels=[90000.0, 50000.0])
    masks = syn.plev_masks(NY, NX)
    assert result['masked_cells']['T_PL'] == {50000.0: 0, 90000.0: int(masks[('other', 90000.0)].sum() * NT)}


def test_whole_level_warning_within_bbox(tmp_path, cfdb_out):
    """Under a bbox the 'missing everywhere' test must count the bbox's cells, not the full grid's."""
    f = _plev(tmp_path, missing_levels=(30000.0,), mountain=False)
    ing = WrfPlevIngest(f)
    lon, lat = ing.bbox_geographic[0], ing.bbox_geographic[1]
    with pytest.warns(UserWarning, match='missing everywhere'):
        result, d = _convert(ing, cfdb_out, variables=['GHT_PL'], bbox=(lon - 1, lat - 1, lon + 6, lat + 4))
    n_bbox = d['geopotential_height'].shape[2] * d['geopotential_height'].shape[3]
    assert 0 < n_bbox < NY * NX
    assert result['masked_cells']['GHT_PL'][30000.0] == n_bbox * NT


def test_rotated_pole_without_cosalpha_withholds(tmp_path):
    """MAP_PROJ 6 is unrotated only on a true lat-lon grid; a rotated pole without COSALPHA must withhold winds."""
    p = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 2, NY, NX,
                         variables=('T2', 'U10', 'V10'))
    with h5py.File(p, 'r+') as h5:
        del h5['COSALPHA'], h5['SINALPHA']
        h5.attrs['POLE_LAT'] = np.float32(45.0)
    with pytest.warns(UserWarning, match='unavailable'):
        ing = WrfIngest(p)
    assert 'U10' not in ing.variables


def test_three_d_winds_withheld_without_cosalpha(tmp_path):
    """A wrfout pruned of COSALPHA but keeping 3-D U/V: the 3-D wind components, direction and the VIMF FALLBACK
    (which rotates) are withheld; speed stays."""
    nz = 3
    p = syn.write_wrfout(tmp_path / 'wrfout_d01_2026-01-01_00_00_00.nc', '2026-01-01T00', 2, NY, NX, map_proj=1,
                         projection=syn.PLEV_PROJECTION, lat0=-50.0, lon0=160.0, variables=('T2', 'U10', 'V10'))
    with h5py.File(p, 'r+') as h5:
        del h5['COSALPHA'], h5['SINALPHA']
        for name, shape in (('U', (2, nz, NY, NX + 1)), ('V', (2, nz, NY + 1, NX)), ('PH', (2, nz + 1, NY, NX)),
                            ('PHB', (2, nz + 1, NY, NX)), ('QVAPOR', (2, nz, NY, NX)), ('P', (2, nz, NY, NX)),
                            ('PB', (2, nz, NY, NX))):
            h5.create_dataset(name, data=np.ones(shape, 'f4'))
    with pytest.warns(UserWarning):
        ing = WrfIngest(p)
    for key in ('U', 'V', 'WIND_DIR', 'VIMF_U', 'VIMF_V'):
        assert key not in ing.variables, key
    assert 'WIND' in ing.variables


@pytest.mark.parametrize('n_times,chunk_t', [(NT, NT), (1, 1)])
def test_level_count_invariant_is_wired_into_both_writers(tmp_path, cfdb_out, monkeypatch, n_times, chunk_t):
    """A block transform returning one level too many is refused at the write, through the block writer
    (several frames) and the single-frame writer (one frame) -- not truncated onto the requested levels."""
    f = _plev(tmp_path, n_times=n_times, mountain=False)
    orig = WrfPlevIngest._block_percent_to_fraction

    def one_too_many(self, sources, y_sl, x_sl, block_cache):
        out = orig(self, sources, y_sl, x_sl, block_cache)
        return np.concatenate([out, out[:, :1]], axis=1)

    monkeypatch.setattr(WrfPlevIngest, '_block_percent_to_fraction', one_too_many)
    with pytest.raises(ValueError, match='silent mislabel'):
        WrfPlevIngest(f).convert(cfdb_out, variables=['RH_PL'], chunk_shape=(chunk_t, 1, NY, NX))


def test_column_integral_ignores_level_subset(tmp_path):
    """ERA5 VIMF integrates over EVERY native level; requesting a level subset for another variable in the same call
    must not reach it (it did silently in review: vimf_u mean -3.90 -> -14.08)."""
    import pathlib

    from cfdb_ingest import Era5Ingest
    pl = sorted((pathlib.Path(__file__).parent / 'data' / 'era5' / 'pl').iterdir())
    means = []
    for lv, name in ((None, 'all.cfdb'), ([70000.0, 92500.0], 'subset.cfdb')):
        Era5Ingest(pl).convert(tmp_path / name, variables=['VIMF_U', 'T'], target_levels=lv)
        with cfdb.open_dataset(tmp_path / name) as ds:
            means.append(np.asarray(ds['vimf_u'].data))
    np.testing.assert_allclose(means[0], means[1])


ROTATION_GRIDS = {
    'lc_tangent_s': (1, dict(dx=20000.0, truelat1=-45.0, truelat2=-45.0, stdlon=175.0), -50.0, 160.0),
    'lc_secant_s': (1, dict(dx=20000.0, truelat1=-30.0, truelat2=-60.0, stdlon=175.0), -50.0, 160.0),
    'lc_secant_n': (1, dict(dx=20000.0, truelat1=30.0, truelat2=60.0, stdlon=-98.0), 25.0, -120.0),
    'ps_south': (2, dict(dx=20000.0, truelat1=-60.0, truelat2=-60.0, stdlon=170.0), -60.0, 140.0),
    'ps_north': (2, dict(dx=20000.0, truelat1=60.0, truelat2=60.0, stdlon=-100.0), 55.0, -130.0),
    'merc_south': (3, dict(dx=20000.0, truelat1=-40.0, truelat2=-40.0, stdlon=170.0), -45.0, 165.0),
}


@pytest.mark.parametrize('name', sorted(ROTATION_GRIDS))
def test_rotation_formula_matches_grid_geometry(name):
    """Every branch of grid_rotation against the finite-difference basis of the cell centres WPS's own formulas place
    (wps_ijll) -- a route that shares no code with it (the companion test's COSALPHA comes from grid_rotation itself,
    so it cannot see a wrong cone constant). The finite-difference error grows with grid spacing near the pole (60 km,
    75 N: 2.3e-4 rad; 10 km: 4e-7), so the grids are 20 km. Tolerance calibrated from these grids: interior
    disagreement measured <= 8.6e-6 rad (polar north, |alpha| to 35 deg); 5e-5 is ~6x that, and a cone error is
    ~0.1 rad. Catches: a wrong cone constant, hemisphere sign or wrap in any branch."""
    map_proj, proj, lat0, lon0 = ROTATION_GRIDS[name]
    ny, nx = 30, 40
    jj, ii = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing='ij')
    lat, lon = syn.wps_ijll(map_proj, ii, jj, lat1=lat0, lon1=lon0, dx=proj['dx'], truelat1=proj['truelat1'],
                            truelat2=proj['truelat2'], stdlon=proj['stdlon'])
    ex, _ = syn._fd_basis(lat, lon)
    alpha_geom = np.arctan2(-ex[1], ex[0])[1:-1, 1:-1]
    c, s, _ = grid_rotation(map_proj, proj['truelat1'], proj['truelat2'], proj['stdlon'], lon)
    diff = np.abs(np.angle(np.exp(1j * (np.arctan2(s, c)[1:-1, 1:-1] - alpha_geom))))
    if map_proj != 3:
        assert np.degrees(np.abs(alpha_geom)).max() > 5.0      # a cone error is invisible where alpha ~ 0
    assert diff.max() < 5e-5, (name, diff.max())


def test_cli_wps_preset_refuses_withheld_winds(tmp_path):
    """--preset wps must not produce an export without winds when they are withheld."""
    from typer.testing import CliRunner

    from cfdb_ingest.cli import app
    f = _plev(tmp_path, mountain=False)          # carries U10/V10/T2/Q2, no COSALPHA -- as a real aux stream
    result = CliRunner().invoke(app, ['wrf', str(f), str(tmp_path / 'o.cfdb'), '--preset', 'wps'])
    assert result.exit_code != 0
    assert 'unavailable' in result.output


def test_convert_all_warns_about_withheld_winds(tmp_path, cfdb_out):
    f = _plev(tmp_path, mountain=False)
    with pytest.warns(UserWarning):
        ing = WrfIngest(f)
    with pytest.warns(UserWarning, match='EXCEPT'):
        ing.convert(cfdb_out)


def test_zero_missing_value_refused(tmp_path):
    with pytest.raises(ValueError, match='non-zero'):
        WrfPlevIngest(_plev(tmp_path, mountain=False), missing_value=0.0)


def test_forecast_only_kwargs_refused(tmp_path, cfdb_out):
    ing = WrfPlevIngest(_plev(tmp_path, mountain=False))
    for kw in (dict(overwrite=True), dict(forecast_reference_time='2026-01-01T00'), dict(leads=[0, 3])):
        with pytest.raises(ValueError, match='not implemented'):
            ing.convert(cfdb_out, variables=['GHT_PL'], **kw)


# ------------------------------------------------------------------ T6: REAL WRF output (Hetzner plev_test_2023-02,
# run f_on, extrap_below_grnd = 1, as the pipeline uploaded it), cropped to the Southern Alps by
# create_test_data.create_plevel_subset. Every expectation is derived from the raw file itself.

REAL_DIR = pathlib.Path(__file__).parent / 'data' / 'wrf_plevels'
REAL_PLEV = REAL_DIR / 'wrfplevels_d01_2023-02-12_00_00_00.nc'
REAL_WRFOUT = REAL_DIR / 'wrfout_d01_2023-02-12_00_00_00.nc'
REAL_FIELDS = {'GHT_PL': 'geopotential_height', 'T_PL': 'air_temperature', 'Q_PL': 'mixing_ratio',
               'U_PL': 'u_wind', 'V_PL': 'v_wind'}


@pytest.fixture(scope='module')
def real_raw():
    with h5py.File(REAL_PLEV, 'r') as h5:
        raw = {n: h5[n][:] for n in REAL_FIELDS}
        levels = h5['P_PL'][0].astype('float64')
        stamped = np.asarray(h5.attrs['p_lev_press_levels_pa'], dtype='float64')
        extrap = int(np.asarray(h5.attrs['p_lev_extrap_below_grnd']).item())
    with h5py.File(REAL_WRFOUT, 'r') as h5:
        cosa, sina = h5['COSALPHA'][0].astype('float64'), h5['SINALPHA'][0].astype('float64')
        hgt = h5['HGT'][0]
    return dict(raw=raw, levels=levels, stamped=stamped, extrap=extrap, cosa=cosa, sina=sina, hgt=hgt)


def _real_convert(path, **kw):
    ing = WrfPlevIngest(REAL_PLEV, static_path=REAL_WRFOUT)
    result = ing.convert(path, chunk_shape=(8, 1, len(ing.y), len(ing.x)), **kw)
    with cfdb.open_dataset(path) as ds:
        data = {n: np.asarray(ds[n].data) for n in ds.data_var_names}
        data['pressure'] = np.asarray(ds['pressure'].data)
    return ing, result, data


def test_real_file_layout_and_levels(real_raw):
    """The real file is what the ingest expects: descending P_PL equal to the pipeline's stamp, the fields on it,
    missing mode 1 (the mode this fixture exists for)."""
    assert real_raw['extrap'] == 1
    assert np.array_equal(real_raw['levels'], real_raw['stamped'])
    assert (np.diff(real_raw['levels']) < 0).all()
    assert all(a.shape[1] == len(real_raw['levels']) for a in real_raw['raw'].values())


def test_real_file_southern_alps_missing(tmp_path, real_raw):
    """A real Southern Alps cell is NaN at 900 AND 850 hPa in every frame, in every field; masked_cells equals the
    file's own -999 count per (field, level); GHT keeps the band above the surface where T is missing; nothing
    else changes beyond the storage precision; no -999 survives."""
    ing, result, d = _real_convert(tmp_path / 'real.cfdb')
    press = d['pressure'].tolist()
    order = [real_raw['levels'].tolist().index(p) for p in press]       # stored level k <- file level order[k]
    raw = real_raw['raw']
    alps = ((raw['T_PL'][:, real_raw['levels'].tolist().index(90000.0)] == -999)
            & (raw['T_PL'][:, real_raw['levels'].tolist().index(85000.0)] == -999)).all(axis=0)
    assert alps.sum() >= 10 and real_raw['hgt'][alps].max() > 1500.0     # the fixture holds the thing tested
    tol = {'geopotential_height': 0.06, 'air_temperature': 0.006, 'mixing_ratio': 1e-6}
    for src, name in REAL_FIELDS.items():
        stored, file_ordered = d[name], raw[src][:, order]
        miss = file_ordered == -999.0
        for p in (90000.0, 85000.0):
            if src != 'GHT_PL':
                assert np.isnan(stored[:, press.index(p)][:, alps]).all(), (src, p)
        assert np.array_equal(np.isnan(stored), miss), src
        for k, p in enumerate(press):
            assert result['masked_cells'][src][p] == int(miss[:, k].sum()), (src, p)
        if name in tol:
            assert np.abs(stored[~miss] - file_ordered[~miss]).max() <= tol[name], name
    k850 = press.index(85000.0)
    band = np.isnan(d['air_temperature'][:, k850]) & np.isfinite(d['geopotential_height'][:, k850])
    assert band.any()
    assert not any((np.asarray(a) == -999.0).any() for a in d.values())


def test_real_file_rotation(tmp_path, real_raw):
    """The real grid-relative U/V are rotated with the wrfout's own COSALPHA/SINALPHA (the formula agrees on this
    grid, else the companion is refused): stored = u cos + v sin, -u sin + v cos to the 0.01 m/s storage step. This
    box sits near the central meridian (|alpha| 3-8 deg), so the test also proves in place that a sign-flipped
    rotation would miss by far more than the tolerance."""
    ing, _, d = _real_convert(tmp_path / 'real.cfdb', variables=['U_PL', 'V_PL'])
    assert 'wrfout' in ing._rotation_source
    order = [real_raw['levels'].tolist().index(p) for p in d['pressure'].tolist()]
    u, v = real_raw['raw']['U_PL'][:, order].astype('f8'), real_raw['raw']['V_PL'][:, order].astype('f8')
    ok = u != -999.0
    cosa, sina = real_raw['cosa'], real_raw['sina']
    assert np.abs((d['u_wind'] - (u * cosa + v * sina))[ok]).max() <= 0.006
    assert np.abs((d['v_wind'] - (-u * sina + v * cosa))[ok]).max() <= 0.006
    flipped = np.hypot(d['u_wind'] - (u * cosa - v * sina), d['v_wind'] - (u * sina + v * cosa))[ok]
    assert flipped.max() > 100 * 0.006                    # the tolerance cannot hide a sign error here


def test_real_file_each_chunk_read_once(tmp_path, io_counts):
    """On WRF's real tiled layout: every HDF5 chunk decompressed once, every output chunk written once."""
    writes, reads = io_counts
    ing = WrfPlevIngest(REAL_PLEV, static_path=REAL_WRFOUT)
    result = ing.convert(tmp_path / 'real.cfdb', chunk_shape=(8, 1, len(ing.y), len(ing.x)))
    data_keys = {k: n for k, n in writes.items() if k.split('!')[0] in REAL_FIELDS.values()}
    assert data_keys and max(data_keys.values()) == 1 and len(data_keys) == len(REAL_FIELDS) * 5
    for src in REAL_FIELDS:
        per_chunk = [n for key, n in reads.items() if key[0] == src]
        assert max(per_chunk) == 1 and len(per_chunk) == result['n_times'] * 4, src

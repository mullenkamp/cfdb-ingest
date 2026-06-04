"""
Regression tests for cross-cadence cfdb chunk-aligned writes.

Each cfdb chunk should be written exactly once during conversion, regardless
of how many source files contribute to it. The sentinel for write
amplification is ``ds.prune()`` — if it removes any items, some chunk got
rewritten partially.
"""
import pathlib
import uuid

import cfdb
import h5py
import numpy as np
import pytest

from cfdb_ingest.wrf import WrfIngest


# ---------------------------------------------------------------------------
# Synthetic minimal-WRF fixture builder
# ---------------------------------------------------------------------------

def _write_synthetic_wrfout(path: pathlib.Path, start_time: np.datetime64,
                            file_t: int, ny: int, nx: int, hour_step: int = 3,
                            seed: int = 0):
    """
    Write a minimal lat-lon (MAP_PROJ=6) wrfout file containing exactly the
    datasets/attributes WrfIngest needs to convert ``T2``.

    ``hour_step`` is the spacing between successive timesteps in hours.
    """
    rng = np.random.default_rng(seed)
    # Times array: (file_t, 19) of 'YYYY-MM-DD_HH:MM:SS' bytes — exactly what
    # WRF writes in its Times variable.
    times = []
    base = np.datetime64(start_time, 's')
    for i in range(file_t):
        t = base + np.timedelta64(i * hour_step * 3600, 's')
        # numpy renders 's' resolution as 'YYYY-MM-DDTHH:MM:SS' (19 chars).
        s = str(t).replace('T', '_')
        assert len(s) == 19, f'unexpected timestamp width: {s!r}'
        times.append([c.encode() for c in s])
    times_arr = np.array(times, dtype='S1')

    # Lat/lon coordinate arrays — 1° spacing, centred near (0, 0).
    lats = np.linspace(-1.0, 1.0, ny, dtype='float32')
    lons = np.linspace(-1.0, 1.0, nx, dtype='float32')
    xlat = np.broadcast_to(lats[:, None], (ny, nx)).astype('float32')
    xlong = np.broadcast_to(lons[None, :], (ny, nx)).astype('float32')
    xlat_t = np.broadcast_to(xlat, (file_t, ny, nx))
    xlong_t = np.broadcast_to(xlong, (file_t, ny, nx))

    t2 = (rng.standard_normal((file_t, ny, nx)) + 273.15).astype('float32')

    with h5py.File(path, 'w') as h5:
        h5.attrs['MAP_PROJ'] = np.int32(6)
        h5.attrs['TITLE'] = ' OUTPUT FROM WRF SYNTHETIC TEST'.ljust(74)
        h5.attrs['CEN_LAT'] = np.float32(0.0)
        h5.attrs['CEN_LON'] = np.float32(0.0)
        h5.attrs['DX'] = np.float32(111000.0)
        h5.attrs['DY'] = np.float32(111000.0)

        h5.create_dataset('Times', data=times_arr)
        h5.create_dataset('XLAT', data=xlat_t)
        h5.create_dataset('XLONG', data=xlong_t)
        h5.create_dataset('T2', data=t2,
                          chunks=(1, ny, nx), compression='gzip', compression_opts=1)


@pytest.fixture
def make_wrfout_files(tmp_path):
    """
    Build a sequence of synthetic wrfout files with controllable cadence.

    Returns a callable ``make(file_t, n_files, ny=20, nx=20, hour_step=3)``
    that writes ``n_files`` files into a fresh subdir and returns their paths
    in time order.
    """
    counter = {'i': 0}

    def make(file_t: int, n_files: int, ny: int = 20, nx: int = 20, hour_step: int = 3):
        counter['i'] += 1
        d = tmp_path / f'wrfout_{counter["i"]:03d}'
        d.mkdir()
        files = []
        for f_idx in range(n_files):
            # Each file's first timestep is the next contiguous time after the
            # previous file's last (no overlap).
            start = np.datetime64('2023-01-01T00:00', 'h') + np.timedelta64(f_idx * file_t * hour_step, 'h')
            p = d / f'wrfout_d01_{str(start).replace(":", "_")}.nc'
            _write_synthetic_wrfout(p, start, file_t, ny, nx, hour_step=hour_step,
                                    seed=f_idx)
            files.append(p)
        return files

    return make


# ---------------------------------------------------------------------------
# The sentinel test: prune() == 0 across cadence/chunk-shape combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('file_t,n_files,chunk_t', [
    (1,  24, 24),    # 1 ts/file, 24-ts chunks
    (8,  12, 24),    # 8 ts/file (matches the user's wvt_nudging_sst d01 cadence)
    (8,  11, 24),    # last file partial, 88 ts / 24-ts chunks
    (24,  4, 24),    # perfectly aligned
    (24,  4, 168),   # chunk bigger than total timesteps
    (3,  20,  7),    # chunk_t and file_t coprime
    (8,  12,  1),    # smallest chunks (default-equivalent)
])
def test_no_write_amplification_single_source(make_wrfout_files, tmp_path, file_t, n_files, chunk_t):
    """A chunk-aligned converter must never leave obsolete chunk versions."""
    files = make_wrfout_files(file_t=file_t, n_files=n_files)
    out = tmp_path / f'{uuid.uuid4().hex}.cfdb'
    w = WrfIngest(files)
    ny, nx = len(w.y), len(w.x)
    w.convert(out, variables=['T2'], chunk_shape=(chunk_t, 1, ny, nx))

    with cfdb.open_dataset(out, 'w') as ds:
        pruned = ds.prune()
    # cfdb's booklet store leaves up to ~1 baseline obsolete entry per dataset
    # creation. Real write amplification scales with the number of cfdb
    # chunks (e.g. user's 1021 prunes for ~510 chunks); a small constant
    # threshold distinguishes the two.
    assert pruned <= 1, (
        f'{pruned} obsolete chunks for file_t={file_t} n_files={n_files} '
        f'chunk_t={chunk_t}: chunk-aligned writes failed'
    )


@pytest.mark.parametrize('file_t,n_files,chunk_t', [
    (8,  12, 24),    # 8 ts/file (matches user's wvt_nudging_sst d01 cadence)
    (8,  11, 24),    # last file partial
    (24,  4, 24),    # perfectly aligned baseline
    (3,  20,  7),    # chunk_t and file_t coprime
])
def test_no_write_amplification_multi_source(make_wrfout_files, tmp_path, file_t, n_files, chunk_t):
    """
    Same sentinel for the multi-source rechunker path. Uses RH2 (T2 + Q2 + PSFC)
    to exercise the path. The synthetic fixture only has T2; we extend it to
    write Q2 and PSFC alongside.
    """
    files = make_wrfout_files(file_t=file_t, n_files=n_files)
    # Augment each synthetic file with Q2 and PSFC (same shape as T2).
    rng = np.random.default_rng(0)
    for p in files:
        with h5py.File(p, 'a') as h5:
            shape = h5['T2'].shape
            if 'Q2' not in h5:
                h5.create_dataset('Q2', data=rng.uniform(0, 0.02, shape).astype('float32'),
                                  chunks=(1, shape[1], shape[2]))
            if 'PSFC' not in h5:
                h5.create_dataset('PSFC', data=rng.uniform(99000, 102000, shape).astype('float32'),
                                  chunks=(1, shape[1], shape[2]))
    out = tmp_path / f'{uuid.uuid4().hex}.cfdb'
    w = WrfIngest(files)
    ny, nx = len(w.y), len(w.x)
    w.convert(out, variables=['RH2'], chunk_shape=(chunk_t, 1, ny, nx))

    with cfdb.open_dataset(out, 'w') as ds:
        pruned = ds.prune()
    assert pruned <= 1, (
        f'{pruned} obsolete chunks (multi-source) for file_t={file_t} '
        f'n_files={n_files} chunk_t={chunk_t}'
    )


@pytest.mark.parametrize('file_t,n_files,chunk_t', [
    (8,  12, 24),
    (8,  11, 24),
    (24,  4, 24),
])
def test_no_write_amplification_accumulation(make_wrfout_files, tmp_path, file_t, n_files, chunk_t):
    """
    accumulation_increment (RAIN with RAINNC+RAINC) must also write each cfdb
    chunk exactly once. Augments synthetic WRF files with cumulative RAINNC and
    RAINC.
    """
    files = make_wrfout_files(file_t=file_t, n_files=n_files)
    rng = np.random.default_rng(0)
    # Build a globally-monotonic cumulative across all files.
    total_t = file_t * n_files
    cum_template = np.cumsum(rng.uniform(0, 0.5, (total_t, 20, 20)).astype('float32'), axis=0)
    for fi, p in enumerate(files):
        with h5py.File(p, 'a') as h5:
            f0 = fi * file_t
            f1 = f0 + file_t
            shape = (file_t, 20, 20)
            if 'RAINNC' not in h5:
                h5.create_dataset('RAINNC', data=cum_template[f0:f1].astype('float32'),
                                  chunks=(1, 20, 20))
            if 'RAINC' not in h5:
                h5.create_dataset('RAINC', data=np.zeros(shape, dtype='float32'),
                                  chunks=(1, 20, 20))
    out = tmp_path / f'{uuid.uuid4().hex}.cfdb'
    w = WrfIngest(files)
    ny, nx = len(w.y), len(w.x)
    w.convert(out, variables=['RAIN'], chunk_shape=(chunk_t, 1, ny, nx))

    with cfdb.open_dataset(out, 'w') as ds:
        pruned = ds.prune()
    assert pruned <= 1, (
        f'{pruned} obsolete chunks (accumulation) for file_t={file_t} '
        f'n_files={n_files} chunk_t={chunk_t}'
    )


def test_chunk_shape_independence(make_wrfout_files, tmp_path):
    """Conversion output values must not depend on chunk_shape."""
    files = make_wrfout_files(file_t=8, n_files=12)
    w = WrfIngest(files)
    ny, nx = len(w.y), len(w.x)

    out_a = tmp_path / 'a.cfdb'
    out_b = tmp_path / 'b.cfdb'
    WrfIngest(files).convert(out_a, variables=['T2'], chunk_shape=(1, 1, ny, nx))
    WrfIngest(files).convert(out_b, variables=['T2'], chunk_shape=(24, 1, ny, nx))

    with cfdb.open_dataset(out_a) as a, cfdb.open_dataset(out_b) as b:
        np.testing.assert_array_equal(a['air_temperature'].data, b['air_temperature'].data)


# ---------------------------------------------------------------------------
# ERA5 — same sentinel against the real ERA5 test fixtures
# ERA5 has one var per file, 24 ts each, 2 files per var -> 48 ts total.
# chunk_t=48 forces rechunker to span both files per variable.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('chunk_t', [1, 24, 48])
def test_no_write_amplification_era5(tmp_path, chunk_t):
    from cfdb_ingest.era5 import Era5Ingest
    sfc_dir = pathlib.Path(__file__).parent / 'data' / 'era5' / 'sfc'
    e = Era5Ingest(sfc_dir)
    ny, nx = len(e.y), len(e.x)
    out = tmp_path / f'{uuid.uuid4().hex}.cfdb'
    # Convert just one simple single-source variable to exercise the
    # _populate_with_rechunkit path.
    e.convert(out, variables=['SP'], chunk_shape=(chunk_t, 1, ny, nx))

    with cfdb.open_dataset(out, 'w') as ds:
        pruned = ds.prune()
    assert pruned <= 1, f'{pruned} obsolete chunks at chunk_t={chunk_t} (era5)'


# ---------------------------------------------------------------------------
# Same sentinel against the real (24 ts/file) test data, varied chunk_t
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('chunk_t', [1, 12, 24, 48])
def test_no_write_amplification_real_wrfout(wrf_file_1, wrf_file_2, tmp_path, chunk_t):
    """
    Two real wrfout files (24 ts each, 48 ts total). chunk_t=48 specifically
    exercises 'one cfdb chunk fed by two files' which is the exact pattern
    that bites the user's 8-ts/file dataset at chunk_t=24.
    """
    files = [wrf_file_1, wrf_file_2]
    w = WrfIngest(files)
    ny, nx = len(w.y), len(w.x)
    out = tmp_path / f'{uuid.uuid4().hex}.cfdb'
    w.convert(out, variables=['T2'], chunk_shape=(chunk_t, 1, ny, nx))

    with cfdb.open_dataset(out, 'w') as ds:
        pruned = ds.prune()
    assert pruned <= 1, f'{pruned} obsolete chunks at chunk_t={chunk_t} (baseline allowed: 1)'

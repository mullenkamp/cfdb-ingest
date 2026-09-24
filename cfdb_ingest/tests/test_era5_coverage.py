"""
ERA5 per-variable file sets through the cross-file rechunk paths (cfdb-ingest 0.6.0).

ERA5 stores one variable per file, so different variables can cover different times, and their file lists
need not sort in the same order. Built from the committed fixtures by copying files and shifting their
``time`` coordinate.
"""
import shutil

import cfdb
import h5py
import numpy as np
import pytest

from cfdb_ingest.era5 import Era5Ingest
from cfdb_ingest.tests.test_era5 import PL_DIR, SFC_DIR

Q_FILE = 'e5.oper.an.pl.128_133_q.ll025sc.2020010100_2020010123.nc'
U_FILE = 'e5.oper.an.pl.128_131_u.ll025uv.2020010100_2020010123.nc'


def _shifted_copy(src, dst, hours, var=None, factor=1.0):
    """A copy of ``src`` ``hours`` later; ``var`` scaled by ``factor`` so a mispaired frame shows."""
    shutil.copy(src, dst)
    with h5py.File(dst, 'r+') as h5:
        h5['time'][:] = h5['time'][:] + hours
        if var is not None:
            h5[var][:] = h5[var][:] * factor
    return dst


def _vimf(files, out):
    Era5Ingest(files).convert(out, variables=['VIMF_U'])
    with cfdb.open_dataset(out) as ds:
        return ds['time'].data.astype('datetime64[m]'), ds['vimf_u'].data


def test_time_absent_from_one_variable_is_missing_not_zero(tmp_path):
    """Surface pressure covers two days, mean sea level pressure one: MSL's second day is missing (NaN).
    Catches: writing the time-mapped source's zero pad rows for a time the variable does not have."""
    out = tmp_path / 'msl.cfdb'
    Era5Ingest(SFC_DIR).convert(out, variables=['MSL', 'SP'])
    with cfdb.open_dataset(out) as ds:
        assert len(ds['time'].data) == 48
        msl = [n for n in ds.data_var_names if 'sea_level' in n or n in ('mslp', 'msl')]
        assert msl, ds.data_var_names
        v = ds[msl[0]].data
    assert not np.isnan(v[:24]).any()
    assert np.isnan(v[24:]).all()


def test_multi_source_pairs_frames_by_time(tmp_path):
    """VIMF_U = f(Q, U): each source's frames are matched by time, even when one variable's files sort in a
    different order from its times. Catches: pairing by raw row (Q day 1 with U day 2)."""
    d = tmp_path / 'pl'
    d.mkdir()
    shutil.copy(PL_DIR / Q_FILE, d / Q_FILE)
    shutil.copy(PL_DIR / U_FILE, d / U_FILE)
    _shifted_copy(PL_DIR / Q_FILE, d / Q_FILE.replace('2020010100_2020010123', '2020010200_2020010223'), 24)
    # U's second day (values doubled, so a mispairing shows) under a name that sorts BEFORE its first day
    u2 = _shifted_copy(PL_DIR / U_FILE, d / U_FILE.replace('2020010100_2020010123', '2019123100_2019123123'),
                       24, var='U', factor=2.0)
    t, v = _vimf(sorted(d.iterdir()), tmp_path / 'both.cfdb')
    assert len(t) == 48

    d1 = tmp_path / 'day1'
    d1.mkdir()
    for f in (Q_FILE, U_FILE):
        shutil.copy(PL_DIR / f, d1 / f)
    _, v1 = _vimf(sorted(d1.iterdir()), tmp_path / 'day1.cfdb')
    d2 = tmp_path / 'day2'
    d2.mkdir()
    shutil.copy(d / Q_FILE.replace('2020010100_2020010123', '2020010200_2020010223'), d2 / 'q2.nc')
    shutil.copy(u2, d2 / 'u2.nc')
    _, v2 = _vimf(sorted(d2.iterdir()), tmp_path / 'day2.cfdb')
    assert not np.allclose(v1, v2)  # the two days differ, so a swapped pairing cannot pass
    np.testing.assert_array_equal(v[:24], v1)
    np.testing.assert_array_equal(v[24:], v2)


def test_multi_source_coverage_mismatch_refused(tmp_path):
    """Q covers two days, U one: a frame-by-frame combination is undefined, so it is refused clearly."""
    d = tmp_path / 'pl'
    d.mkdir()
    shutil.copy(PL_DIR / Q_FILE, d / Q_FILE)
    shutil.copy(PL_DIR / U_FILE, d / U_FILE)
    _shifted_copy(PL_DIR / Q_FILE, d / Q_FILE.replace('2020010100_2020010123', '2020010200_2020010223'), 24)
    with pytest.raises(ValueError, match='cover different times'):
        Era5Ingest(sorted(d.iterdir())).convert(tmp_path / 'x.cfdb', variables=['VIMF_U'])

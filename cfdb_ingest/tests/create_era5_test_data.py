"""
Generate synthetic ERA5 test files for CI.

Creates small NetCDF files matching the NCAR ERA5 structure:
- Surface files: (time, latitude, longitude) with one variable each
- Pressure level files: (time, level, latitude, longitude) with one variable each
- Invariant files: (time=1, latitude, longitude) with one variable each

Usage:
    uv run python -m cfdb_ingest.tests.create_era5_test_data
"""
import pathlib

import h5py
import numpy as np

OUTPUT_DIR = pathlib.Path(__file__).parent / 'data' / 'era5'

# Small grid: 10 lat x 12 lon, covering a portion of the real clipped region
LAT = np.arange(-40.0, -37.5, 0.25)   # 10 points, descending stored as ascending here then reversed
LON = np.arange(170.0, 173.0, 0.25)   # 12 points

# Descending latitude (as ERA5 stores it)
LAT_DESC = LAT[::-1]

# Time: 24 hourly timesteps for one day (2020-01-01)
# ERA5 time is hours since 1900-01-01
BASE_HOURS = int((np.datetime64('2020-01-01') - np.datetime64('1900-01-01')) / np.timedelta64(1, 'h'))
TIME_SFC = np.arange(BASE_HOURS, BASE_HOURS + 24, dtype='int32')  # 24 hours

# Second day for multi-file tests
TIME_SFC_2 = np.arange(BASE_HOURS + 24, BASE_HOURS + 48, dtype='int32')

# Pressure level: same 24h but daily files in real ERA5
TIME_PL = TIME_SFC.copy()

# Pressure levels in hPa (subset of the full 37)
LEVELS = np.array([500.0, 700.0, 850.0, 925.0, 1000.0], dtype='float64')

NY = len(LAT)
NX = len(LON)
NT_SFC = len(TIME_SFC)
NT_PL = len(TIME_PL)
NZ = len(LEVELS)

np.random.seed(42)


def _create_file(path, var_name, data, lat, lon, time, level=None):
    """Create a single ERA5-like NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        f.create_dataset('latitude', data=lat, dtype='float64')
        f.create_dataset('longitude', data=lon, dtype='float64')
        f.create_dataset('time', data=time, dtype='int32')
        f.create_dataset('utc_date', data=time, dtype='int32')  # simplified

        if level is not None:
            f.create_dataset('level', data=level, dtype='float64')

        f.create_dataset(var_name, data=data, dtype='float32', chunks=True)


def create_surface_files():
    """Create surface variable files."""
    sfc_dir = OUTPUT_DIR / 'sfc'

    # SP (surface pressure) — realistic range ~95000-103000 Pa
    sp_data = np.random.uniform(95000, 103000, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_134_sp.ll025sc.2020010100_2020010123.nc',
        'SP', sp_data, LAT_DESC, LON, TIME_SFC,
    )

    # SP day 2 (for multi-file tests)
    sp_data_2 = np.random.uniform(95000, 103000, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_134_sp.ll025sc.2020010200_2020010223.nc',
        'SP', sp_data_2, LAT_DESC, LON, TIME_SFC_2,
    )

    # VAR_2T (2m temperature) — realistic range ~250-310 K
    t2_data = np.random.uniform(260, 300, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_167_2t.ll025sc.2020010100_2020010123.nc',
        'VAR_2T', t2_data, LAT_DESC, LON, TIME_SFC,
    )

    # VAR_2D (2m dewpoint) — realistic range ~240-295 K
    d2_data = np.random.uniform(250, 290, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_168_2d.ll025sc.2020010100_2020010123.nc',
        'VAR_2D', d2_data, LAT_DESC, LON, TIME_SFC,
    )

    # VAR_10U (10m U wind) — realistic range ~-20 to 20 m/s
    u10_data = np.random.uniform(-15, 15, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_165_10u.ll025sc.2020010100_2020010123.nc',
        'VAR_10U', u10_data, LAT_DESC, LON, TIME_SFC,
    )

    # VAR_10V (10m V wind)
    v10_data = np.random.uniform(-15, 15, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_166_10v.ll025sc.2020010100_2020010123.nc',
        'VAR_10V', v10_data, LAT_DESC, LON, TIME_SFC,
    )

    # MSL (mean sea level pressure)
    msl_data = np.random.uniform(98000, 104000, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_151_msl.ll025sc.2020010100_2020010123.nc',
        'MSL', msl_data, LAT_DESC, LON, TIME_SFC,
    )

    # CI (sea ice cover) — range 0-1
    ci_data = np.random.uniform(0, 0.3, (NT_SFC, NY, NX)).astype('float32')
    _create_file(
        sfc_dir / 'e5.oper.an.sfc.128_031_ci.ll025sc.2020010100_2020010123.nc',
        'CI', ci_data, LAT_DESC, LON, TIME_SFC,
    )


def create_pressure_level_files():
    """Create pressure level variable files."""
    pl_dir = OUTPUT_DIR / 'pl'

    # T (temperature on pressure levels) — range ~200-300 K
    t_data = np.random.uniform(210, 300, (NT_PL, NZ, NY, NX)).astype('float32')
    _create_file(
        pl_dir / 'e5.oper.an.pl.128_130_t.ll025sc.2020010100_2020010123.nc',
        'T', t_data, LAT_DESC, LON, TIME_PL, level=LEVELS,
    )

    # U (U wind on pressure levels)
    u_data = np.random.uniform(-30, 30, (NT_PL, NZ, NY, NX)).astype('float32')
    _create_file(
        pl_dir / 'e5.oper.an.pl.128_131_u.ll025uv.2020010100_2020010123.nc',
        'U', u_data, LAT_DESC, LON, TIME_PL, level=LEVELS,
    )

    # V (V wind on pressure levels)
    v_data = np.random.uniform(-30, 30, (NT_PL, NZ, NY, NX)).astype('float32')
    _create_file(
        pl_dir / 'e5.oper.an.pl.128_132_v.ll025uv.2020010100_2020010123.nc',
        'V', v_data, LAT_DESC, LON, TIME_PL, level=LEVELS,
    )

    # Z (geopotential on pressure levels) — m2/s2
    z_data = np.random.uniform(0, 60000, (NT_PL, NZ, NY, NX)).astype('float32')
    _create_file(
        pl_dir / 'e5.oper.an.pl.128_129_z.ll025sc.2020010100_2020010123.nc',
        'Z', z_data, LAT_DESC, LON, TIME_PL, level=LEVELS,
    )

    # Q (specific humidity on pressure levels) — kg/kg
    q_data = np.random.uniform(0, 0.02, (NT_PL, NZ, NY, NX)).astype('float32')
    _create_file(
        pl_dir / 'e5.oper.an.pl.128_133_q.ll025sc.2020010100_2020010123.nc',
        'Q', q_data, LAT_DESC, LON, TIME_PL, level=LEVELS,
    )


def create_invariant_files():
    """Create invariant (time-independent) files."""
    inv_dir = OUTPUT_DIR / 'inv'
    time_inv = np.array([BASE_HOURS], dtype='int32')

    # Z (surface geopotential) — m2/s2
    z_data = np.random.uniform(0, 5000, (1, NY, NX)).astype('float32')
    _create_file(
        inv_dir / 'e5.oper.invariant.128_129_z.ll025sc.2020010100_2020010100.nc',
        'Z', z_data, LAT_DESC, LON, time_inv,
    )

    # LSM (land-sea mask) — 0 or 1
    lsm_data = np.random.choice([0.0, 1.0], (1, NY, NX)).astype('float32')
    _create_file(
        inv_dir / 'e5.oper.invariant.128_172_lsm.ll025sc.2020010100_2020010100.nc',
        'LSM', lsm_data, LAT_DESC, LON, time_inv,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Creating ERA5 test data in {OUTPUT_DIR}')

    create_surface_files()
    print('  Surface files created')

    create_pressure_level_files()
    print('  Pressure level files created')

    create_invariant_files()
    print('  Invariant files created')

    print('Done.')


if __name__ == '__main__':
    main()

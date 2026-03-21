"""
Benchmark cfdb-ingest convert() for various variable configurations.

Uses the small test data (99x111, 16 levels) in cfdb_ingest/tests/data/.
Run with: uv run python benchmarks/bench_convert.py
"""
import csv
import pathlib
import shutil
import sys
import tempfile
import time

from cfdb_ingest.wrf import WrfIngest

TEST_DIR = pathlib.Path(__file__).resolve().parent.parent / 'cfdb_ingest' / 'tests' / 'data'
WRF_FILE_1 = TEST_DIR / 'wrfout_d01_2023-02-12_00:00:00.nc'
WRF_FILE_2 = TEST_DIR / 'wrfout_d01_2023-02-13_00:00:00.nc'

CONFIGS = [
    {
        'name': '1x_2D_T2',
        'variables': ['T2'],
        'target_levels': None,
        'bbox': None,
    },
    {
        'name': '1x_3D_T',
        'variables': ['T'],
        'target_levels': [100, 500, 1000],
        'bbox': None,
    },
    {
        'name': '3x_3D_shared_geo',
        'variables': ['T', 'THETA', 'QVAPOR'],
        'target_levels': [100, 500, 1000],
        'bbox': None,
    },
    {
        'name': '4x_3D_shared_wind_geo',
        'variables': ['U', 'V', 'WIND', 'WIND_DIR'],
        'target_levels': [100, 500, 1000],
        'bbox': None,
    },
    {
        'name': 'all_3D',
        'variables': ['T', 'THETA', 'THETA_E', 'U', 'V', 'WIND', 'WIND_DIR', 'QVAPOR', 'Q_SH', 'RH'],
        'target_levels': [100, 500, 1000],
        'bbox': None,
    },
    {
        'name': 'all_3D_bbox',
        'variables': ['T', 'THETA', 'THETA_E', 'U', 'V', 'WIND', 'WIND_DIR', 'QVAPOR', 'Q_SH', 'RH'],
        'target_levels': [100, 500, 1000],
        'bbox': (165.0, -47.0, 175.0, -40.0),
    },
]

N_TIMESTEPS = 6


def run_benchmark(config, wrf, tmp_dir):
    out_path = tmp_dir / f'{config["name"]}.cfdb'
    kwargs = {
        'cfdb_path': out_path,
        'variables': config['variables'],
        'start_date': '2023-02-12T00:00',
        'end_date': '2023-02-12T05:00',
    }
    if config['target_levels'] is not None:
        kwargs['target_levels'] = config['target_levels']
    if config['bbox'] is not None:
        kwargs['bbox'] = config['bbox']

    t0 = time.perf_counter()
    wrf.convert(**kwargs)
    elapsed = time.perf_counter() - t0

    # Clean up
    if out_path.exists():
        shutil.rmtree(out_path, ignore_errors=True)

    return elapsed


def main():
    if not WRF_FILE_1.exists():
        print(f"Test data not found: {WRF_FILE_1}")
        sys.exit(1)

    print("Initializing WrfIngest...")
    wrf = WrfIngest([WRF_FILE_1, WRF_FILE_2])

    # Warmup run
    print("Warmup...")
    with tempfile.TemporaryDirectory() as tmp:
        run_benchmark(CONFIGS[0], wrf, pathlib.Path(tmp))

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        for config in CONFIGS:
            print(f"  {config['name']}...", end=' ', flush=True)
            elapsed = run_benchmark(config, wrf, tmp_dir)
            results.append({'config': config['name'], 'time_s': elapsed})
            print(f"{elapsed:.3f}s")

    # Print summary
    print("\n--- Results ---")
    print(f"{'Config':<30} {'Time (s)':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['config']:<30} {r['time_s']:>10.3f}")

    # Write CSV
    csv_path = pathlib.Path(__file__).parent / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'time_s'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()

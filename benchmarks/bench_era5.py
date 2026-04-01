"""
Benchmark cfdb-ingest convert() for various ERA5 variable configurations.

Uses the small test data in cfdb_ingest/tests/data/era5/.
Run with: uv run python benchmarks/bench_era5.py
"""
import csv
import pathlib
import shutil
import sys
import tempfile
import time

from cfdb_ingest.era5 import Era5Ingest

TEST_DIR = pathlib.Path(__file__).resolve().parent.parent / 'cfdb_ingest' / 'tests' / 'data' / 'era5'

CONFIGS = [
    {
        'name': '1x_sfc',
        'variables': ['VAR_2T'],
        'target_levels': None,
        'bbox': None,
        'split': False,
    },
    {
        'name': '3x_sfc',
        'variables': ['VAR_2T', 'VAR_10U', 'VAR_10V'],
        'target_levels': None,
        'bbox': None,
        'split': False,
    },
    {
        'name': '1x_pl',
        'variables': ['T'],
        'target_levels': [50000.0, 85000.0],
        'bbox': None,
        'split': False,
    },
    {
        'name': '3x_pl',
        'variables': ['T', 'U', 'V'],
        'target_levels': [50000.0, 85000.0],
        'bbox': None,
        'split': False,
    },
    {
        'name': 'all_mixed',
        'variables': ['VAR_2T', 'VAR_10U', 'VAR_10V', 'T', 'U', 'V', 'Z_PL', 'Z_INV'],
        'target_levels': [50000.0, 85000.0],
        'bbox': None,
        'split': False,
    },
    {
        'name': 'all_mixed_bbox',
        'variables': ['VAR_2T', 'VAR_10U', 'VAR_10V', 'T', 'U', 'V', 'Z_PL', 'Z_INV'],
        'target_levels': [50000.0, 85000.0],
        'bbox': (165.0, -47.0, 175.0, -40.0),
        'split': False,
    },
    {
        'name': 'split_mixed',
        'variables': ['VAR_2T', 'VAR_10U', 'VAR_10V', 'T', 'U', 'V', 'Z_PL', 'Z_INV'],
        'target_levels': [50000.0, 85000.0],
        'bbox': None,
        'split': True,
    },
]

def run_benchmark(config, era5, tmp_dir):
    kwargs = {
        'variables': config['variables'],
        'start_date': '2020-01-01T00:00',
        'end_date': '2020-01-01T23:00',
        'split': config['split'],
    }
    
    if config['split']:
        out_path = tmp_dir / config['name']
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = tmp_dir / f'{config["name"]}.cfdb'
        
    kwargs['cfdb_path'] = out_path

    if config['target_levels'] is not None:
        kwargs['target_levels'] = config['target_levels']
    if config['bbox'] is not None:
        kwargs['bbox'] = config['bbox']

    t0 = time.perf_counter()
    era5.convert(**kwargs)
    elapsed = time.perf_counter() - t0

    return elapsed


def main():
    if not TEST_DIR.exists():
        print(f"Test data not found: {TEST_DIR}")
        sys.exit(1)

    print("Initializing Era5Ingest...")
    era5 = Era5Ingest(list(TEST_DIR.rglob('*.nc')))

    # Warmup run
    print("Warmup...")
    with tempfile.TemporaryDirectory() as tmp:
        run_benchmark(CONFIGS[0], era5, pathlib.Path(tmp))

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        for config in CONFIGS:
            print(f"  {config['name']}...", end=' ', flush=True)
            elapsed = run_benchmark(config, era5, tmp_dir)
            results.append({'config': config['name'], 'time_s': elapsed})
            print(f"{elapsed:.3f}s")

    # Print summary
    print("\n--- Results ---")
    print(f"{'Config':<30} {'Time (s)':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['config']:<30} {r['time_s']:>10.3f}")

    # Write CSV
    csv_path = pathlib.Path(__file__).parent / 'era5_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'time_s'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()

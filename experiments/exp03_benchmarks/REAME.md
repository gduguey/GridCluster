
# Benchmark Experiments

This folder has three scripts used to generate benchmark data and run GTEP experiments.

## 1. `create_benchmarks_data.py`
Creates the benchmark datasets (clusters + time series) under `./benchmark_results/<method>__<cfg>/`.

Run first:
```bash
python create_benchmarks_data.py
```

## 2. `benchmarks.py`

Runs the full three-stage GTEP pipeline (aggregated → spatial → temporal) for **all** benchmark folders created above.

Run second:

```bash
python benchmarks.py
```

## 3. `single_run.py`

Runs the three-stage GTEP pipeline for **one chosen case** (edit `method` and `cfg` at the top of the file).

Use this if you want to test/debug just one configuration:

```bash
python single_run.py
```

### Typical workflow

1. Generate data with `create_benchmarks_data.py`
2. Run everything in parallel with `benchmarks.py`
3. Use `single_run.py` for focused checks

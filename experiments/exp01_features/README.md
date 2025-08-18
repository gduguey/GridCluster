# Weight-Grid Runs (N5 / N10 / N15)

These three scripts sweep a **grid of weighting vectors** for the aggregation features and evaluate each setting at a few (N, K) combinations, with N the number of spatial clusters and K the number of representative days.

- `run_grid_N5.py`   → runs `(N=5,  K ∈ {7,14,21})`, writes `manifest_N5_grid.csv`
- `run_grid_N10.py`  → runs `(N=10, K ∈ {7,14,21})`, writes `manifest_N10_grid.csv`
- `run_grid_N15.py`  → runs `(N=15, K ∈ {7,14,21})`, writes `manifest_N15_grid.csv`

Each script:
1. Loads **`enriched_weight_grid.csv`** from the same folder.
2. Builds a list of hyper-parameter entries where `weights = {position, time_series, duration_curves, ramp_duration_curves, intra_correlation, inter_correlation}`.
3. For each (N, K) in the script, attaches:
   - `n_representative_nodes = N`
   - `k_representative_days = K`
4. Runs all weight entries **in parallel** via `ProcessPoolExecutor`:
   - Calls `run_experiment(...)`,
   - Streams results into a **manifest CSV** (overwrites as it progresses).

## Input file format

`enriched_weight_grid.csv` must have at least these columns:

```

id, desc, position, time_series, duration_curves,
ramp_duration_curves, intra_correlation, inter_correlation

```

- `id` can be any unique string.
- Weights don’t have to sum to 1; the pipeline normalizes if needed.

## How to run

```bash
# They are independent
python run_grid_N5.py
python run_grid_N10.py
python run_grid_N15.py
````

## Outputs

* A manifest CSV per script (e.g., `manifest_N10_grid.csv`) containing one row per experiment stage and weight vector, including objectives.
* Solver logs and artifacts are written wherever your pipeline config points (e.g., under `results/`), unchanged by these scripts.

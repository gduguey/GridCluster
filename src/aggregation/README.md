# Aggregation Module

This package implements the **spatial and temporal aggregation pipeline** used to prepare input data for capacity expansion planning. It supports two network resolutions:

* **Fine network**: \~385 nodes, built directly from population, demand, and capacity factor data.
* **Coarse network**: 17 zones, aggregated from the fine network.

The coarse network is always derived from the fine network using nearest-zone mapping. Both network options can be used in subsequent temporal aggregation and capacity expansion modeling (`src/gtep`).

## File Overview

* **`settings.py`** — Central configuration system

  * Immutable preprocessing parameters (year, granularity, active features)
  * Mutable hyperparameters (weights, number of representative nodes/days)
  * Derived file paths

* **`utils.py`** — Data utilities and network building

  * Load wind, solar, demand, and population data
  * Build **fine** and **coarse** networks
  * Feature computation (position, time series, duration curves, correlations)
  * Results export

* **`models.py`** — Aggregation models

  * Distance metric computation (position, time series, correlations)
  * Clustering methods: **k-medoids** and **optimization (MIP with Gurobi)**
  * Temporal aggregation (representative days)

* **`pipeline.py`** — Pipeline orchestration

  * **Static preprocessing**: build fine or coarse network and compute features
  * **Dynamic processing**: run aggregation with given hyperparameters, evaluate results, and save outputs

## Pipeline Flow

1. **Choose network granularity**:

   * `granularity="fine"` → build 385-node fine network.
   * `granularity="coarse"` → build fine network, then aggregate into 17 zones.

2. **Build features** for each node:

   * Position, raw time series, duration curves, ramp curves, correlations.

3. **Spatial aggregation**:

   * Compute feature distances
   * Aggregate nodes into representative clusters using either:

     * **k-medoids** (multiple initializations, inertia-based), or
     * **optimization** (formulated as MILP in Gurobi).

4. **Temporal aggregation**:

   * Sample representative days from clustered nodes
   * Build a reduced set of daily time slices.

5. **Results export**:

   * Nodes, branches, and time series written to `results/joint_aggregation_results/`.

## Configuration Settings

The pipeline is configured via `Config` (see `settings.py`). Key options:

| Category            | Parameter                | Type        | Description                                                                                                     |
| ------------------- | ------------------------ | ----------- | --------------------------------------------------------------------------------------------------------------- |
| **Preprocessing**   | `year`                   | `int`       | Data year (2007–2013)                                                                                           |
|                     | `granularity`            | `str`       | `"fine"` (385 nodes) or `"coarse"` (17 zones)                                                                   |
|                     | `active_features`        | `list[str]` | Features to include (`position`, `time_series`, `duration_curves`, `ramp_duration_curves`, `intra_correlation`) |
| **Hyperparameters** | `n_representative_nodes` | `int`       | Number of representative spatial clusters                                                                       |
|                     | `k_representative_days`  | `int`       | Number of representative days (1–365)                                                                           |
|                     | `inter_correlation`      | `bool`      | Whether to include correlations between series at different nodes (inter)                                                                     |
|                     | `weights`                | `dict`      | Feature weights (auto-generated or manual override)                                                             |
| **Paths**           | Derived automatically    | `Path`      | File system paths for demand, wind/solar CF, population, etc.                                                   |

## Example Usage

```python
from aggregation.pipeline import StaticPreprocessor, DynamicProcessor

# --- Step 1. Static preprocessing ---
# Choose "fine" (385 nodes) or "coarse" (17 zones)
preproc = StaticPreprocessor(granularity="coarse", year=2013).preprocess()

# --- Step 2. Dynamic run with hyperparameters ---
dyn = DynamicProcessor(preprocessor=preproc)

weights={
        'position': 1.0,
        'time_series': 0.8,
        'duration_curves': 1.2,
        'ramp_duration_curves': 1.0,
        'intra_correlation': 1.0,
        'inter_correlation': 1.0
    }

results, version_hash, day_weights, metadata = dyn.run_with_hyperparameters(
    weights=weights,
    n_representative_nodes=10,
    k_representative_days=12
)

print("Results saved under version:", version_hash)
print("Representative day weights:", day_weights)
```

This produces a reduced dataset (nodes, branches, representative time series) saved under
`results/joint_aggregation_results/v<hash>/`.

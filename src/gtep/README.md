# GTEP (Generation & Transmission Expansion Planning)

This package solves a expansion problem over representative days and nodes derived by the aggregation module, and deaggregates it 2 stages:

1. **Aggregated GTEP** on the reduced network (from the aggregation pipeline)
2. **Spatial de-aggregation** (distributes investments across original spatial members, spatially deaggregated)
3. **Temporal de-aggregation** (fixes investments, operates over full horizon, fully deaggregated)

It consumes the artifacts produced by `src/aggregation/` and writes results + logs under a versioned folder.

## Files

* **`models.py`** — core optimization models

  * `AggregatedGTEP`: single-stage model on aggregated nodes & representative days
  * `SpatialGTEP`: ties aggregated investments to member buses via linking constraints
  * `TemporalGTEP`: fixes investments, operates the original system over the full time span
  * `GTEPBase`: shared scaffolding (variables, constraints, objective, solve wrapper)

* **`pipeline.py`** — experiment runner

  * `run_aggregated`, `run_spatial`, `run_temporal`: build inputs, solve, save artifacts
  * `run_experiment(grid, processor, …)`: iterate over hyperparameter cases (N, K, weights)
  * `consume(gen, root)`: collect run metadata into a manifest `DataFrame`
  * `save_artifacts` / `load_artifacts`: serialize/deserialize complete results

* **`types.py`** — typed dataclasses

  * `Params`, `RepPeriod`, `InputData`: model inputs
  * `SolveStats`, `InvestmentResults`, `OperationResults`, `GTEPResults`: outputs
  * `RunArtifacts`: bundle of (results, input data, hashes, paths, clusters)

* **`utils.py`** — GTEP helpers

  * `SolveLogger`: CSV logger for Gurobi bound/objective over time
  * `build_input_data`: convert the aggregation pipeline outputs into `InputData` (scaling, rounding) for the GTEP
  * `uniform_day_weights`, `ensure_unique`, and small utilities

## Inputs & Outputs

### Inputs

From `src/aggregation/` you pass one of the three result blocks:

* **`spatiotemporal`** — aggregated nodes & representative days
* **`temporal_only`** — original nodes with representative days
* **`original`** — original nodes with full time horizon

`build_input_data(...)` turns a block into:

* `InputData.buses` (DataFrame with `bus_id`, `Lat`, `Lon`)
* `InputData.branches` (complete-graph branches built from nodes)
* `InputData.rep_period` (`Load`, `PV`, `Wind` with hourly rows)
* `InputData.params` (`Params` dataclass, units scaled to chosen power/cost scales)

### Outputs

Each stage returns a `RunArtifacts` object and writes to disk:

* **Pickle**: `results/gtep_output/<stage>/<stage>_<aggHash>_<gtepHash>.pkl` (full `RunArtifacts`)
* **CSV Log**: `results/gtep_logs/<stage>/*.csv` (time, best obj, best bound)

`RunArtifacts.results` contains:

* `InvestmentResults`: `PV`, `Wind`, `CCGT` (integer units), `Storage` (power), `Tran` (per line)
* `OperationResults`: hourly dispatches per bus/branch (`y_*`, `y_flow`)
* `SolveStats`: build time, run time, objective value

## Model sketch

### Decision variables (per bus unless noted)

* Investments: `PV`, `Wind`, `CCGT_units` (integer), `StoragePower`, `TransCap` (per line)
* Operations (hourly): `y_ccgt`, `y_shed`, `y_curtail`, `y_charge`, `y_discharge`, `y_soc`, `y_flow` (per line, ±)

### Core constraints

* Power balance at each bus/hour (with flows in/out)
* CCGT capacity & symmetric ramp constraints
* Storage power/energy limits and SOC dynamics
* Transmission thermal limits per line
* Spatial link (in `SpatialGTEP`): group sums equal aggregated plan

### Objective

Capex (PV, Wind, CCGT units, Storage power, Transmission by line length) + weighted operating costs over representative days (`d_ccgt * gen + d_shed * shed`).

## Typical workflows

### A) Run all three stages for each (C, D, weights)

```python
from pathlib import Path
import pandas as pd
from aggregation.pipeline import StaticPreprocessor, DynamicProcessor
from gtep.pipeline import run_experiment, consume

# 1) Build aggregated data (choose "coarse" 17 zones OR "fine" 385 nodes)
pre = StaticPreprocessor(granularity="coarse", year=2013).preprocess()
dyn = DynamicProcessor(preprocessor=pre)

# 2) Define a tiny grid of hyperparameters
grid = [
    {"id":"case_01","desc":"C10-D12","n_representative_nodes":10,"k_representative_days":12,
     "weights":{"position":0.4,"time_series":0.4,"duration_curves":0.2,"ramp_duration_curves":0.2,"intra_correlation":0.1,"inter_correlation":0.4}},
    {"id":"case_02","desc":"C12-D16","n_representative_nodes":12,"k_representative_days":16,
     "weights":{"position":0.4,"time_series":0.4,"duration_curves":0.2,"ramp_duration_curves":0.2,"intra_correlation":0.1,"inter_correlation":0.4}},
]

# 3) Run GTEP pipeline
exp_root = Path(".")
gen = run_experiment(grid, dyn, exp_root=exp_root, threads=8)

# 4) Collect results into a manifest
manifest = consume(gen, root=exp_root)
print(manifest)
```

### B) Run a single stage directly

```python
from pathlib import Path
from gtep.pipeline import run_aggregated, run_spatial, run_temporal

# Assume you already have "agg_res, agg_hash, day_weights, meta" from aggregation:
# results, version_hash, day_weights, metadata = dyn.run_with_hyperparameters(**hyper)
art_agg = run_aggregated(agg_res, agg_hash, exp_root=Path("."), day_weights=list(day_weights.values()), threads=8)

if art_agg.results:
    art_sp = run_spatial(agg_res, agg_hash, Path("."), list(day_weights.values()), art_agg.results.inv, threads=8)
    if art_sp.results:
        art_tm = run_temporal(agg_res, agg_hash, Path("."), art_sp.results.inv, threads=8)
```

## Configuration & scaling

`build_input_data(...)` accepts:

* `power_scale`: `"MW" | "GW" | "TW"` (default `"GW"`)
* `cost_scale`:  `"$" | "k$" | "M$"` (default `"M$"`)
* It rounds numeric inputs to a small number of significant digits for compact models/logs.

Key parameters (`Params`):

* `PV_resource`, `Wind_resource` (installed MW cap)
* `CCGT_max_cap` (MW per unit), `CCGT_ramp` (fraction/h)
* `c_*` investment costs, `d_*` variable/penalty costs
* `rate_duration` (storage hours), `eta_charge`, `eta_discharge`, `Initial_SOC`
* Transmission: `Length[line]`, `TranMax[line]`, `MW_per_mile`, `c_tran`

> Note: `types.Params` uses snake\_case, the builder maps from human-friendly keys to dataclass fields.

## Requirements

* **Gurobi** (license + Python bindings)
* Python ≥ 3.10, packages: `numpy`, `pandas`, `flask` (for `json`), plus the aggregation deps if you run end-to-end.

## Notes & tips

* `run_experiment` **yields** stage artifacts; use `consume` to build a manifest `DataFrame`.
* Time limits, thread counts, and logging are set per stage in `pipeline.py`.
* Representative-day weights:

  * Aggregated/spatial: use weights returned by aggregation
  * Temporal: uses `uniform_day_weights` across the full horizon by default
* Transmission network in these utilities is a **complete graph** derived from bus coordinates; replace with your real topology if available.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

# GridCluster

**Composite-feature spatial clustering and representative-day selection for power system planning**

📍 By [Gabriel Duguey](mailto:gduguey@mit.edu)  
Master of Engineering in Data Science for Engineering Systems  
Massachusetts Institute of Technology, 2025

---

## Overview

**GridCluster** is the codebase for my MIT master's thesis, which proposes a task-aware framework for spatial and temporal aggregation in long-term power system planning models. The goal is to reduce model complexity, while preserving fidelity in investment outcomes, by carefully clustering nodes and selecting representative time periods based on composite planning-relevant signals.

This repository enables:
- Feature-driven **spatial clustering** of grid nodes,
- Representative-day **temporal reduction**,
- A tractable **CEP optimization model** for generation and transmission planning, and
- Full-resolution **deaggregation** and evaluation of investment decisions

A detailed case study on New England demonstrates the method's performance across various aggregation levels and feature weightings.

---

## Motivation

Modern CEP models require high-resolution inputs: hourly time series at hundreds or thousands of grid nodes. Solving such models at scale is computationally prohibitive. Naive aggregation (e.g., by political zones or raw capacity factors) can lead to suboptimal siting and distorted results.

**GridCluster** reframes aggregation as a design problem, proposing:
- A modular, **feature-aware pipeline**, with tunable similarity metrics,
- Combined spatial and temporal aggregation tailored to system structure,
- Evaluation via **full-resolution deaggregation**, not just reconstruction error.

---

## Installation

It is recommended to use a virtual environment.

1. **Clone the repository:**
```bash
git clone https://github.com/gduguey/GridCluster
cd GridCluster
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

### Pipeline Steps

1. **Configuration Setup**

   * Centralized in `src/settings.py` using a `Config` class composed of:

     * `PreprocessingConfig`: year, granularity, features
     * `PathConfig`: file paths by year and granularity
     * `HyperParameters`: clustering parameters and weights

2. **Data Import and Interpolation**

   * `HighResDataProcessor` or `LowResDataProcessor` loads raw demand, solar, and wind
   * Applies k-NN or mass-conserving kernel interpolation (Numba-accelerated)

3. **Feature Computation**

   * Computes planning-relevant features: duration curves, ramp rates, correlation, etc.
   * Fully vectorized with Numba for speed

4. **Spatial Aggregation**

   * `SpatialAggregation` supports:

     * MILP optimization using Gurobi
     * K-medoids clustering with seed control
   * Uses composite distance metrics (weighted sum of normalized distances)

5. **Temporal Aggregation**

   * Selects representative days using k-medoids on daily system-wide profiles
   * Scales and reshapes time series to reflect cluster means

6. **Export and Evaluation**

   * Aggregated `nodes`, `branches`, and time series are exported to disk
   * A full-resolution **deaggregation module** evaluates investment feasibility and cost

---

## Key Configuration Options

| Setting                              | Affects                     | Description                                                                                            |
| ------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------ |
| `data_preproc.granularity`           | Input resolution            | Choose `high` (k-bus) or `low` (county-level)                                                          |
| `data_preproc.active_features`       | Feature set                 | Select from: `position`, `time_series`, `duration_curves`, `ramp_duration_curves`, `intra_correlation` |
| `model_hyper.n_representative_nodes` | Spatial aggregation size    | Number of zones (P)                                                                                    |
| `model_hyper.k_representative_days`  | Temporal aggregation size   | Number of representative days (Q)                                                                      |
| `model_hyper.weights`                | Feature weighting           | Dict of relative feature importance                                                                    |
| `spatial_method`                     | Aggregation method          | Choose `'optimization'` or `'kmedoids'`                                                                |
| `model_hyper.kmed_seed`              | Reproducibility             | Seed for K-medoids clustering                                                                          |
| `model_hyper.inter_correlation`      | Optional global correlation | Adds a system-wide distance term if `True`                                                             |

---

## Example Results

In the New England test case, GridCluster outperforms common baselines like:

* Geographic (Voronoi) clustering
* Political zones
* Capacity factor–based aggregation

Best-performing configurations reduce **total system cost by up to 13%** compared to heuristic methods. Correlation-based features drive most of this performance.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{duguey2025gridcluster,
  title        = {Task-Aware Spatio-Temporal Aggregation for Capacity Expansion Planning Models},
  author       = {Gabriel Duguey},
  school       = {Massachusetts Institute of Technology},
  year         = {2025},
  note         = {\url{https://github.com/gduguey/GridCluster}}
}
```

---

## Acknowledgments

This work was developed as part of my thesis under the supervision of [Prof. Saurabh Amin](https://cee.mit.edu/people_individual/saurabh-amin/). I thank Aron Brenner for his crucial help with design, CEP implementation, and experimentation, and Rahman Khorramfar for his ongoing guidance.

---

## Contact

For questions, ideas, or collaborations:
**Gabriel Duguey** – [gduguey@mit.edu](mailto:gduguey@mit.edu)

```

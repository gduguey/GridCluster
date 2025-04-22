![License](https://img.shields.io/badge/license-MIT-blue.svg)
# MIT Master Thesis: Uncertainty-Aware Spatio-Temporal Aggregation for Power System Capacity Expansion Problems

## Author
**Gabriel Duguey: gduguey@mit.edu**  
Master of Engineering in Data Science for Engineering Systems, MIT 2025


## Description  
This project is part of my Master's thesis at MIT, exploring the benefits of jointly aggregating spatially and temporally, as well as applying random sampling to account for uncertainty in highly granular networks. The objective of these aggregation techniques is to improve the tractability and computational efficiency of solving Power System Capacity Expansion problems, while preserving the precision and quality of the optimization results. The research uses New England's power grid as a case study to evaluate the effectiveness of this approach.  

The repository is made available for reference and educational purposes. While you are welcome to download and explore the code and data, the repository is protected to prevent any unauthorized modifications. If you have questions, suggestions, or feedback, please feel free to reach out or fork the repository to propose changes.

## Installation

It's recommended to use a virtual environment to manage project dependencies.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/DUGU630/SpatioTemporal-Aggregation-CEP
    ```

2. **Install Dependencies**:
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

# Data Aggregation Pipeline Overview

The data aggregation pipeline involves several steps, each with specific choices and methods for aggregating spatial data. Below is a straightforward explanation of the pipeline, highlighting the key intersections where aggregation choices are made.

A Jupyter notebook is provided in `demo/demo.ipynb` to demonstrate this pipeline.

### Pipeline Steps

1. **Configuration Setup**  
The **src/settings.py** module defines a top‐level `Config` class that ties together three components:  
   - **`PreprocessingConfig`** (immutable): data‐year, interpolation neighbors, decay alpha, granularity, and the list of active node‐level features.  
   - **`PathConfig`** (immutable): dynamically builds all input/output file paths (e.g. demand, capacity‐factor, raw/processed data) based on year and granularity.  
   - **`HyperParameters`** (mutable, via Pydantic): model settings such as number of representative nodes and days, seed and initializations for KMedoids, inter‐correlation flag, and feature‐weight overrides.  

   All fields are validated on assignment, and feature‐weights are auto‐generated (or overridden) to ensure consistency between preprocessing and aggregation steps.

2. **Data Import and Interpolation**  
   The `DataProcessor` in **src/utils.py** delegates to either `HighResDataProcessor` or `LowResDataProcessor` based on your `config.granularity`. Each loader reads node and branch tables plus raw demand, wind, and solar files, then applies either k‑NN or mass‑conserving kernel interpolation (via Numba) to project all time series onto the network’s node coordinates. The result is a unified dict of DataFrames ready for feature computation.

3. **Feature Computation**  
   The `Network` class in **src/utils.py** takes the interpolated nodes and time series and computes only the user‑enabled features—geographic position, raw series, duration and ramp‑duration curves, and intra‑node correlations—optionally filtering to a specific date range. It vectorizes time‑series operations for speed, wraps them in Numba‑compatible loops, and returns a per‑node feature dictionary.

4. **Aggregation Methods**  
   The `SpatialAggregation` class in **src/models.py** drives full spatial clustering. It first builds per‑feature distance matrices (`DistanceCalculator`), normalizes and caches them (`IOManager`), then offers two aggregation strategies:  
   – **Optimization** (Gurobi): solves a MILP to pick exactly _n_ representatives that minimize total assignment distance.  
   – **K‑Medoids Clustering** (scikit‑learn‑extra): runs PAM with multiple initializations and seed control to find medoids.  
   Both return cluster‑to‑members maps and representative indices.

5. **Visualization**  
   The `Visualizer` in **src/visualization.py** consumes the raw node locations and cluster assignments to produce geographic plots. It can display original versus aggregated networks on interactive maps.

6. **Temporal Aggregation**  
   The `TemporalAggregation` class in **src/models.py** takes each spatial cluster, samples one node’s rescaled time series to match the cluster mean, then reshapes the full‑year data into daily vectors. It computes a day‑to‑day distance matrix and runs the same K‑Medoids procedure to select _k_ representative days, which are then passed back for final CEP input assembly.

7. **Results Assembly and Export**  
   The `Results` class in **src/utils.py** takes your original data (nodes, branches, time series), together with the spatial and temporal aggregation outputs, and produces the final tables ready for your CEP model. It  
     - precomputes a mapping from each original bus to its representative node,  
     - builds aggregated `nodes` and `branches` DataFrames (summing susceptances and capacities across all lines between reps),  
     - filters and concatenates the representative nodes’ time series over the chosen days, and  
     - writes out CSVs plus a JSON metadata file into a versioned directory keyed by your configuration hash.  

### Key Intersections for Aggregation Choices

The following configuration settings determine how and where aggregation is performed, and how the various methods interact:

| Configuration Option              | Affects                                | Description                                                                                                      |
|-----------------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **`data_preproc.granularity`**    | Data import & interpolation resolution | • `"high"` → uses `HighResDataProcessor` (3 k‑bus network) <br> • `"low"`  → uses `LowResDataProcessor` (67 counties) |
| **`data_preproc.cf_k_neighbors`** | k‑NN interpolation                     | Number of nearest neighbors for capacity factor interpolation in `HighResDataProcessor._process_capacity_factors`. |
| **`data_preproc.demand_decay_alpha`** | Kernel interpolation of demand        | Decay parameter α controlling mass‑conserving kernel in `HighResDataProcessor._process_demand`.                   |
| **`data_preproc.active_features`**| Feature computation & distance metrics | Select any subset of `['position','time_series','duration_curves','ramp_duration_curves','intra_correlation']`.   |
| **`model_hyper.inter_correlation`** | Network‑wide correlation metric       | If true (and `time_series` is active), adds an extra distance metric over all node time‐series pairs.             |
| **`model_hyper.weights`**         | Feature weighting in spatial agg.      | Dict of feature→weight shares; by default auto‐generated equally over `active_features` (plus `inter_correlation` if enabled), or manually overridden. |
| **`model_hyper.n_representative_nodes`** | Spatial aggregation size           | Number of clusters (P) in both the MILP optimizer and K‑Medoids.                                                   |
| **`spatial_method`** (method arg) | Choice of spatial algorithm            | Pass `'optimization'` to solve the Gurobi MILP or `'kmedoids'` to run PAM via `SpatialAggregation.Clusterer`.      |
| **`model_hyper.kmed_seed`**, **`kmed_n_init`** | K‑Medoids reproducibility         | Seed and number of restarts for the PAM algorithm to guard against local minima.                                 |
| **`model_hyper.k_representative_days`** | Temporal aggregation size           | Number of representative days (Q) that `TemporalAggregation` extracts via K‑Medoids over daily profiles.         |

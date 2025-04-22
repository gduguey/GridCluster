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

1. **Configuration Setup**:
   - The `Config` class in `utils_mistral.py` sets up the configuration parameters for the aggregation process. These parameters include the year of data, method for demand calculation, number of neighbors for interpolation, time scale, file paths, weights for different features, and the number of representative nodes.

2. **Data Import and Interpolation**:
   - The `DataProcessor` class in `utils_mistral.py` handles the import and interpolation of data. It reads node data, demand data, and capacity factor data (wind and solar). It then interpolates the capacity factor data to match the node locations using a custom interpolation method.

3. **Feature Computation**:
   - The `Network` class in `utils_mistral.py` computes various features for each node, including position, time series, duration curves, ramp duration curves, correlations, and supply-demand mismatch. These features are used for aggregation.

4. **Aggregation Methods**:
   - The `SpatialAggregation` class in `models.py` provides different methods for aggregating the nodes (spatially). The primary methods are optimization and k-medoids clustering.
     - **Optimization**: Formulates and solves an optimization model to minimize the total distance between nodes and their representatives.
     - **K-Medoids Clustering**: Performs k-medoids clustering using a precomputed distance matrix to find representative nodes.

5. **Visualization**:
   - The `Visualization` class in `utils_mistral.py` handles the visualization of the aggregation results. It can plot maps showing the original nodes and their representative nodes.

6. **Temporal Aggregation**:
   - The `TemporalAggregation` class in `models.py` selects representative days by sampling a node from each spatial cluster, forming day arrays by concatenating the sampled nodes' time series data, and applying k-medoids clustering to identify the representative days.

### Key Intersections for Aggregation Choices

1. **Demand Calculation Method**:
   - The choice between `total_demand` and `k-interpolation` for demand calculation affects how demand is aggregated: `total_demand` computes the supply-demand mismatch by comparing each node's capacity factors to the network's total demand, while `k-interpolation` uses `k_weight_demand` to interpolate different demands at each node, then compares each node's capacity factors to its interpolated demand.

2. **Time Scale for Supply-Demand Mismatch**:
   - The `time_scale` (`yearly`, `monthly`, `weekly`) determines the granularity of the supply-demand mismatch calculation.

3. **Interpolation Method**:
   - The `custom_interpolate` method in `DataProcessor` uses k-nearest neighbors for interpolating capacity factor data (and demand data if `k-interpolation` is selected in `Config`), affecting the accuracy of the interpolated values.

4. **Distance Metrics**:
   - The `compute_distance_metrics` method in the `SpatialAggregation` class calculates various distance metrics (position, time series, duration curves, ramping duration curves, inter correlation, intra correlation, and supply demand mismatch), which are used in both optimization and clustering methods.
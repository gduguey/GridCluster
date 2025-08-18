#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
from sklearn_extra.cluster import KMedoids
from collections import defaultdict
import geopandas as gpd
import pandas as pd
from sklearn.metrics import pairwise_distances
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────────
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXP_DIR      = THIS_FILE.parent
BENCHMARK_ROOT = EXP_DIR / "benchmark_results"
shp_path = (
    PROJECT_ROOT
    / "DATA" / "raw" / "ne_population"
    / "cb_2022_us_state_500k"
    / "cb_2022_us_state_500k.shp"
)
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Imports from your codebase ─────────────────────────────────────────────────
from src.aggregation.utils import Results
from src.aggregation.models import TemporalAggregation
from src.aggregation.pipeline import StaticPreprocessor, DynamicProcessor

### ---------- Benchmark configurations ----------
experiment_setups = [
    {"name": "high_spatial_low_temporal", "N": 10, "K": 3},
    {"name": "low_spatial_high_temporal", "N": 3, "K": 30},
    {"name": "political_regions", "N": 6, "K": 12}
]

benchmark_methods = {
    "T4": {
        "weights": [0.083, 0.25, 0.083, 0.25, 0.083, 0.25],
        "type": "standard"
    },
    "R25": {
        "weights": [0.05, 0.016, 0.066, 0.034, 0.514, 0.32],
        "type": "standard"
    },
    "position_only": {
        "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "type": "standard"
    },
    "time_series_only": {
        "weights": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "type": "standard"
    },
    "avg_cf_kmedoids": {
        "type": "custom"
    },
    "random": {
        "type": "custom"
    },
    "political": {
        "type": "custom",
        "only_config": "political_regions"
    }
}

FEATURES = ["position", "time_series", "duration_curves", "ramp_duration_curves", "intra_correlation", "inter_correlation"]


### ---------- Helper: generate clusters ----------
def generate_random_clusters(N, n_points, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, N, size=n_points)

    label_to_nodes = defaultdict(list)
    for i, lbl in enumerate(labels):
        label_to_nodes[lbl].append(i)

    clusters = {}
    for node_list in label_to_nodes.values():
        rep = min(node_list)
        clusters[rep] = node_list

    return {
        "clusters": clusters,
        "representatives": sorted(clusters.keys())
    }

def generate_avg_cf_kmedoids_clusters(node_df, N, seed=42, n_init=10):
    """
    Cluster nodes using K-Medoids over average wind and solar capacity factors.
    Uses Euclidean distance, PAM algorithm, and multiple initializations.
    """
    rng = np.random.default_rng(seed)
    cf = node_df[["avg_cf_wind", "avg_cf_solar"]].values
    distance_matrix = pairwise_distances(cf, metric="euclidean")

    best_inertia = float("inf")
    best_result = None

    for _ in range(n_init):
        random_state = rng.integers(0, 1e9)
        kmed = KMedoids(
            n_clusters=N,
            metric="precomputed",
            method="pam",
            init="heuristic",
            max_iter=300,
            random_state=random_state,
        )
        kmed.fit(distance_matrix)

        if kmed.inertia_ < best_inertia:
            best_inertia = kmed.inertia_
            labels = kmed.labels_
            medoids = kmed.medoid_indices_.tolist()

            # First map cluster index → node list
            label_to_nodes = defaultdict(list)
            for i, lbl in enumerate(labels):
                label_to_nodes[lbl].append(i)

            # Then remap to rep_id (medoid) → node list
            clusters = {}
            for cluster_idx, node_list in label_to_nodes.items():
                rep = medoids[cluster_idx]
                clusters[rep] = node_list

            best_result = {
                "clusters": clusters,
                "representatives": sorted(clusters.keys()),
                "metadata": {
                    "inertia": kmed.inertia_,
                    "n_iter": kmed.n_iter_,
                    "labels": labels,
                },
            }


    return best_result

def generate_political_clusters(node_df):
    """
    Cluster nodes by U.S. state using their latitude and longitude.
    Requires a valid US state shapefile under /DATA/raw/ne_population/.
    """
    # Load US state shapes
    state_gdf = gpd.read_file(shp_path)
    state_gdf = state_gdf.to_crs("EPSG:4326")  # Ensure same CRS as lat/lon

    # Convert node_df to GeoDataFrame
    node_gdf = gpd.GeoDataFrame(
        node_df.copy(),
        geometry=gpd.points_from_xy(node_df["Lon"], node_df["Lat"]),
        crs="EPSG:4326"
    )

    # Spatial join: assign each node to a state polygon
    joined = gpd.sjoin(node_gdf, state_gdf, how="left", predicate="within")

    # `STUSPS` contains the state abbreviation (e.g., 'MA')
    if "STUSPS" not in joined.columns:
        raise ValueError("State abbreviation column 'STUSPS' not found after join")

    raw_clusters = defaultdict(list)
    for idx, row in joined.iterrows():
        state = row["STUSPS"]
        if pd.isna(state):
            print(f"Warning: Node {idx} has no state assigned, skipping.")
            continue
        raw_clusters[state].append(idx)

    # Now convert to clusters with rep_id as key
    clusters = {}
    for node_list in raw_clusters.values():
        rep = min(node_list)
        clusters[rep] = node_list

    return {
        "clusters": clusters,
        "representatives": sorted(clusters.keys())
    }

def save_results_benchmark_style(results: dict[str, dict], method: str, cfg_name: str, day_weights: list[int]):
    """
    Save results under BENCHMARK_ROOT / f"{method}_{cfg_name}", keeping the same format and logic as _save_results.
    """

    # Compute target folder
    save_path = BENCHMARK_ROOT / f"{method}__{cfg_name}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save results
    for results_name, subdict in results.items():
        if results_name == "clusters":
            # Handle clusters separately in JSON format
            cluster_path = save_path / "clustering"
            cluster_path.mkdir(parents=True, exist_ok=True)
            for cluster_type, cluster_dict in subdict.items():
                print(cluster_dict)
                cluster_dict = {int(k): v for k, v in cluster_dict.items()}
                cluster_file = cluster_path / f"{cluster_type}_clusters.json"
                with open(cluster_file, "w") as f:
                    json.dump(cluster_dict, f, indent=2)

            # Save day_weights if provided
            weights_file = cluster_path / "day_weights.json"
            with open(weights_file, "w") as f:
                json.dump(day_weights, f, indent=2)
            continue

        # Save all DataFrames and sub-dictionaries
        results_path = save_path / results_name
        results_path.mkdir(parents=True, exist_ok=True)
        for name, df in subdict.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(results_path / f"{name}.csv", index=False)
            else:
                for sub_name, sub_df in df.items():
                    sub_df.to_csv(results_path / f"{name}_{sub_name}.csv", index=False)

    print(f"Results saved to {save_path}")

### ---------- Main Loop ----------

static_prep = StaticPreprocessor(granularity="coarse").preprocess()
processor = DynamicProcessor(static_prep)

def run_all_benchmarks():
    for cfg in experiment_setups:
        N = cfg["N"]
        K = cfg["K"]
        cfg_name = cfg["name"]

        print(f"\n=== Running experiments for config: {cfg_name} (N={N}, K={K}) ===")

        for method, method_info in benchmark_methods.items():
            if method_info.get("only_config") and method_info["only_config"] != cfg_name:
                continue

            print(f"\n→ Method: {method}")

            if method_info["type"] == "standard":
                call_kwargs = {
                    "weights":                dict(zip(FEATURES, method_info["weights"])),
                    "n_representative_nodes": N,
                    "k_representative_days":  K,
                }
                agg_res, agg_hash, day_weights, meta = processor.run_with_hyperparameters(**call_kwargs)
                day_weights = list(day_weights.values())
            else:
                node_df = processor.preprocessor.network_data['nodes'].reset_index(drop=True)
                ts = processor.preprocessor.network_data['time_series']
                ntw = processor.ntw

                if method == "random":
                    spatial_results = generate_random_clusters(N, len(node_df))
                elif method == "avg_cf_kmedoids":
                    node_df["avg_cf_wind"] = ts["wind"].mean()
                    node_df["avg_cf_solar"] = ts["solar"].mean()
                    spatial_results = generate_avg_cf_kmedoids_clusters(node_df, N)
                elif method == "political":
                    spatial_results = generate_political_clusters(node_df)
                else:
                    raise ValueError(f"Unknown custom method: {method}")

                current_config = processor.base_config
                current_config.model_hyper.n_representative_nodes = N
                current_config.model_hyper.k_representative_days = K
                temporal_agg = TemporalAggregation(current_config, ntw.features, spatial_results)
                temporal_results = temporal_agg.aggregate()
                day_weights = {rep_day: len(days) for rep_day, days in temporal_results["clusters"].items()}
                day_weights = list(day_weights.values())
                results = Results(current_config, processor.preprocessor.network_data, spatial_results, temporal_results, auto_save=False)
                agg_res = results.results
            save_results_benchmark_style(agg_res, method=method, cfg_name=cfg_name, day_weights=day_weights)


run_all_benchmarks()

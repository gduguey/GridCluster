#!/usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import sys

# ─── Paths ─────────────────────────────────────────────────────────────
THIS_FILE       = Path(__file__).resolve()
PROJECT_ROOT    = THIS_FILE.parents[2]
EXP_DIR         = THIS_FILE.parent
BENCHMARK_ROOT  = EXP_DIR / "benchmark_results"
RESULTS_ROOT    = EXP_DIR / "results" / "gtep_output"

sys.path.insert(0, str(PROJECT_ROOT))

# ─── Your pipeline logic ────────────────────────────────────────────────
from src.gtep.pipeline import run_aggregated, run_spatial, run_temporal

# ─── Settings for parallelism ───────────────────────────────────────────
threads_per_solve = 8
method = "random"
cfg    = "high_spatial_low_temporal"
folder = BENCHMARK_ROOT / f"{method}__{cfg}"

# ─── Utilities ──────────────────────────────────────────────────────────

def load_agg_res(method: str, cfg_name: str) -> dict[str, dict]:
    """
    Load results from BENCHMARK_ROOT / f"{method}__{cfg_name}" in the same structure
    as produced by `save_results_benchmark_style(...)`.
    Returns:
        - results: dict[str, dict], where keys are sections like 'clusters', 'spatiotemporal', ...
        - day_weights: list[int]
    """

    save_path = BENCHMARK_ROOT / f"{method}__{cfg_name}"
    results = {}

    for item in save_path.iterdir():
        if item.name == "clustering":
            clustering_dict = {}
            for cluster_file in item.glob("*_clusters.json"):
                cluster_type = cluster_file.stem.replace("_clusters", "")
                with open(cluster_file) as f:
                    cluster_data = json.load(f)
                    cluster_data = {int(k): v for k, v in cluster_data.items()}
                    clustering_dict[cluster_type] = cluster_data
            results["clusters"] = clustering_dict

        else:
            results[item.name] = {}
            for file in item.glob("*.csv"):
                stem = file.stem
                df = pd.read_csv(file)

                if stem.startswith("time_series_"):
                    # Parse nested time series: wind, solar, demand
                    _, subkey = stem.split("time_series_", 1)
                    if "time_series" not in results[item.name]:
                        results[item.name]["time_series"] = {}
                    results[item.name]["time_series"][subkey] = df
                else:
                    results[item.name][stem] = df

    return results

# ─── Loop over benchmark configs ────────────────────────────────────────

print(f"\n=== Running GTEP for {method} / {cfg} ===")

try:
    agg_res = load_agg_res(method, cfg)
    temporal_results = agg_res["clusters"]["temporal"]
    day_weights = {rep_day: len(days) for rep_day, days in temporal_results.items()}
    day_weights = list(day_weights.values())

    art_agg = run_aggregated(agg_res, agg_hash=folder.name, exp_root=EXP_DIR, day_weights=day_weights, threads=threads_per_solve)
    if art_agg.results is None:
        raise RuntimeError("Aggregated stage failed.")

    art_spatial = run_spatial(agg_res, folder.name, EXP_DIR, day_weights, art_agg.results.inv, threads=threads_per_solve)
    if art_spatial.results is None:
        raise RuntimeError("Spatial stage failed.")

    art_temp = run_temporal(agg_res, folder.name, EXP_DIR, art_spatial.results.inv, threads=threads_per_solve)
    if art_temp.results is None:
        raise RuntimeError("Temporal stage failed.")

    print(f"\n✓ Completed all stages for {method} / {cfg}")
    print(f"→ Objective (aggregated): {art_agg.results.stats.objective:.3f}")
    print(f"→ Objective (spatial):    {art_spatial.results.stats.objective:.3f}")
    print(f"→ Objective (temporal):   {art_temp.results.stats.objective:.3f}")

except Exception as e:
    print(f"Failed for {method} / {cfg}: {e}")
    


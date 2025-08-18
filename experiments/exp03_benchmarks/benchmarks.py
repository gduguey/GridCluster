#!/usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ─────────────────────────────────────────────────────────────
THIS_FILE       = Path(__file__).resolve()
PROJECT_ROOT    = THIS_FILE.parents[2]
EXP_DIR         = THIS_FILE.parent
BENCHMARK_ROOT  = EXP_DIR / "benchmark_results"
RESULTS_ROOT    = EXP_DIR / "results" / "gtep_output"

sys.path.insert(0, str(PROJECT_ROOT))

# ─── Your pipeline logic ────────────────────────────────────────────────
from src.gtep.pipeline import run_aggregated, run_spatial, run_temporal
from src.gtep.utils import ensure_unique
from src.gtep.types import RunArtifacts

# ─── Settings for parallelism ───────────────────────────────────────────
threads_per_solve = 8
cpu_count = multiprocessing.cpu_count()
max_workers = max(1, cpu_count // threads_per_solve)
print(f"→ Using {max_workers} processes x {threads_per_solve} threads per solve")

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

def run_gtep_for_folder(folder: Path) -> list[dict]:
    method, cfg = folder.name.split("__")

    print(f"\n=== Running GTEP stages for {method} / {cfg} ===")

    try:
        agg_res = load_agg_res(method, cfg)
        temporal_results = agg_res["clusters"]["temporal"]
        day_weights = {rep_day: len(days) for rep_day, days in temporal_results.items()}
        day_weights = list(day_weights.values())
    except Exception as e:
        print(f"Failed to load input for {method} / {cfg}: {e}")
        return []
    
    manifest_rows = []

    try:
        # --- Stage 1: Aggregated ---
        art_agg = run_aggregated(agg_res, agg_hash=folder.name, exp_root=EXP_DIR, day_weights=day_weights, threads=threads_per_solve)
        manifest_rows.append(make_row(art_agg, method, cfg, "aggregated"))

        if art_agg.results is None:
            print(f"Failed to run aggregated stage for {method} / {cfg}")
            return manifest_rows

        # --- Stage 2: Spatial ---
        art_spatial = run_spatial(agg_res, folder.name, EXP_DIR, day_weights, art_agg.results.inv, threads=threads_per_solve)
        manifest_rows.append(make_row(art_spatial, method, cfg, "spatial"))

        if art_spatial.results is None:
            print(f"Failed to run spatial stage for {method} / {cfg}")
            return manifest_rows

        # --- Stage 3: Temporal ---
        art_temp = run_temporal(agg_res, folder.name, EXP_DIR, art_spatial.results.inv, threads=threads_per_solve)
        manifest_rows.append(make_row(art_temp, method, cfg, "temporal"))

    except Exception as e:
        print(f"Error during GTEP runs for {method} / {cfg}: {e}")

    return manifest_rows  

def run_all_parallel():
    benchmark_dirs = sorted(BENCHMARK_ROOT.glob("*__*"))
    manifest_path = ensure_unique(EXP_DIR / "manifest_benchmark_gtep.csv")
    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_gtep_for_folder, folder): folder.name for folder in benchmark_dirs}

        for fut in as_completed(futures):
            try:
                rows = fut.result()
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(manifest_path, index=False)
            except Exception as e:
                print(f"Error for {futures[fut]}: {e}")

    print(f"\nManifest saved to {manifest_path}")

def make_row(art: RunArtifacts, method: str, cfg: str, stage: str) -> dict:
    return {
        "method":     method,
        "config":     cfg,
        "stage":      stage,
        "agg_hash":   art.agg_hash,
        "gtep_hash":  art.gtep_hash,
        "artifact":   str(art.save_path),
        "log_path":   str(art.log_path),
        "objective":  getattr(art.results.stats, "objective", float("nan")) if art.results else float("nan"),
        "error":      None if art.results else "infeasible",
    }

run_all_parallel()

#!/usr/bin/env python
from pathlib import Path
import sys
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXP_DIR = THIS_FILE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregation.pipeline import StaticPreprocessor, DynamicProcessor
from src.gtep.pipeline import consume, run_experiment
from src.gtep.utils import ensure_unique

# Fixed weights
fixed_weights = {
    "position":             0.45,
    "time_series":          0.025,
    "duration_curves":      0.025,
    "ramp_duration_curves": 0.025,
    "intra_correlation":    0.025,
    "inter_correlation":    0.45,
}

# Build grid over (N, K)
N_vals = [3, 5, 8, 10, 15, 17]   # can extend to 30 later
K_vals = [3, 7, 12, 15, 20, 25, 30]  # can extend to 30 later
grid = []
for N in N_vals:
    for K in K_vals:
        grid.append({
            "id": f"P4_n{N}_k{K}",
            "desc": f"P4 weights with N={N}, K={K}",
            "weights": fixed_weights,
            "n_representative_nodes": N,
            "k_representative_days": K,
        })

# Prepare processor
static_prep = StaticPreprocessor(granularity="coarse").preprocess()
processor = DynamicProcessor(static_prep)

# Parallel settings
threads_per_solve = 8
cpu_count = multiprocessing.cpu_count()
grid_workers = max(1, cpu_count // threads_per_solve)

def eval_hp(hp):
    df = consume(run_experiment([hp], processor, EXP_DIR, threads=threads_per_solve), PROJECT_ROOT)
    df["id"] = hp["id"]
    return df

out_path = ensure_unique(EXP_DIR / "manifest_pareto_coarse_P4.csv")
results = []
with ProcessPoolExecutor(max_workers=grid_workers) as exe:
    futures = {exe.submit(eval_hp, hp): hp for hp in grid}
    for fut in as_completed(futures):
        try:
            results.append(fut.result())
            pd.concat(results, ignore_index=True).to_csv(out_path, index=False)
        except Exception as e:
            print(f"Error processing {futures[fut]['id']}: {e}", file=sys.stderr)

print(f"Results written to {out_path}")

#!/usr/bin/env python
import sys
from pathlib import Path
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ───────────────────────────────────────────────────────────────────────
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXP_DIR      = THIS_FILE.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Imports from your codebase ─────────────────────────────────────────────────
from src.aggregation.pipeline import StaticPreprocessor, DynamicProcessor
from src.gtep.pipeline     import consume, run_experiment
from src.gtep.utils        import ensure_unique

# ─── 0) Load enriched weight grid from CSV ──────────────────────────────────────
csv_path = EXP_DIR / "enriched_weight_grid.csv"
weight_df = pd.read_csv(csv_path)

# Build base_grid of hyper-parameter dicts
base_grid = []
for _, row in weight_df.iterrows():
    base_grid.append({
        "id":   row["id"],
        "desc": row.get("desc", ""),
        "weights": {
            "position":             row["position"],
            "time_series":          row["time_series"],
            "duration_curves":      row["duration_curves"],
            "ramp_duration_curves": row["ramp_duration_curves"],
            "intra_correlation":    row["intra_correlation"],
            "inter_correlation":    row["inter_correlation"],
        }
    })

# ─── 1) Build processors ────────────────────────────────────────────────────────
static_prep = StaticPreprocessor(granularity="coarse").preprocess()
processor   = DynamicProcessor(static_prep)

# ─── 2) Parallelism settings ────────────────────────────────────────────────────
threads_per_solve = 8
cpu_count         = multiprocessing.cpu_count()
grid_workers      = max(1, cpu_count // threads_per_solve)
print(f"→ Using {grid_workers} processes x {threads_per_solve} threads per solve")

def eval_hp(hp, config_label):
    """
    Evaluate a single hyper-parameter dict hp under the fixed "processor".
    Returns a DataFrame with your stages + objective, plus the id/config.
    """
    df = consume(
        run_experiment([hp],       # one-item grid
                       processor,
                       EXP_DIR,
                       threads=threads_per_solve), PROJECT_ROOT
    )
    df["id"]     = hp["id"]
    df["config"] = config_label
    return df

# ─── 3) Define 10 (N, K) configurations ─────────────────────────────────────────
configs = [
    {"label":"N5_K7",  "N":5,  "K":7},
    {"label":"N5_K14", "N":5,  "K":14},
    {"label":"N5_K21", "N":5,  "K":21},
]

# ─── 4) Run full grid for each (N, K) ───────────────────────────────────────────
manifest_path = ensure_unique(EXP_DIR / "manifest_N5_grid.csv")
all_results = []
for cfg in configs:
    # attach N, K to every hp
    grid = []
    for hp in base_grid:
        entry = hp.copy()
        entry.update({
            "n_representative_nodes": cfg["N"],
            "k_representative_days":  cfg["K"],
        })
        grid.append(entry)

    # parallel evaluation
    results = []
    with ProcessPoolExecutor(max_workers=grid_workers) as exe:
        futs = {exe.submit(eval_hp, hp, cfg["label"]): hp for hp in grid}
        for fut in as_completed(futs):
            try:
                all_results.append(fut.result())
                # overwrite manifest with everything so far
                pd.concat(all_results, ignore_index=True).to_csv(manifest_path, index=False)
            except Exception as e:
                print(f"Error on {futs[fut]['id']}: {e}", file=sys.stderr)

# ─── 5) Save combined results ──────────────────────────────────────────────────
print(f"Full manifest written to {manifest_path}")

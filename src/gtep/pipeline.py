# pipeline.py — experiment runner and multi-stage GTEP pipeline

import hashlib
import pickle
from pathlib import Path
from dataclasses import asdict
from typing import Iterable, List
from flask import json
import gurobipy as gp
import pandas as pd

from .utils import SolveLogger, uniform_day_weights, ensure_unique, build_input_data
from .types import InvestmentResults, RunArtifacts
from .models import AggregatedGTEP, SpatialGTEP, TemporalGTEP

# =========================================================
# ===================  PIPELINE HELPERS  ==================
# =========================================================

def run_experiment(grid, processor, exp_root: Path, threads: int):
    """
    grid: iterable of dicts with spatial/temporal hyperparams
    processor: your DynamicProcessor
    Returns a manifest DataFrame with paths + KPIs.
    """
    for hp in grid:
        case_id = hp["id"]
        n, k = hp["n_representative_nodes"], hp["k_representative_days"]
        print(f"\n=== Starting case {case_id} (N={n}, K={k}) ===")

        call_kwargs = {
            "weights":                hp["weights"],
            "n_representative_nodes": n,
            "k_representative_days":  k,
        }

        # --- A) aggregation ---
        print(f"[{case_id}] Running aggregation...")
        agg_res, agg_hash, day_weights, meta = processor.run_with_hyperparameters(**call_kwargs)
        day_weights = list(day_weights.values())
        print(f"[{case_id}] Aggregation done. agg_hash={agg_hash}")

        # --- B) GTEP stages ---
        print(f"[{case_id}] Solving aggregated GTEP...")
        agg_art = run_aggregated(agg_res, agg_hash, exp_root, day_weights, threads=threads)
        yield ("agg", hp, agg_art)
        if agg_art.results is None:
            print(f"[{case_id}] Aggregated GTEP infeasible, skipping spatial and temporal stages")
            continue
        print(f"[{case_id}] Aggregated GTEP solved with objective {agg_art.results.stats.objective}")

        print(f"[{case_id}] Solving spatial de-aggregation GTEP...")
        spat_art = run_spatial(
            agg_res, agg_hash, exp_root, day_weights,
            agg_art.results.inv,
            threads=threads
        )
        yield ("spatial", hp, spat_art)
        if spat_art.results is None:
            print(f"[{case_id}] Spatial GTEP infeasible, skipping temporal stage")
            continue
        print(f"[{case_id}] Spatial results saved to {spat_art.save_path} with objective {spat_art.results.stats.objective}")

        print(f"[{case_id}] Solving temporal de‑aggregation GTEP...")
        temp_art = run_temporal(
            agg_res, agg_hash, exp_root,
            spat_art.results.inv,
            threads=threads
        )
        if temp_art.results is None:
            print(f"[{case_id}] Temporal GTEP infeasible, skipping")
        else:
            obj = temp_art.results.stats.objective
            print(f"[{case_id}] Temporal results saved to {temp_art.save_path} with objective {obj}")
        yield ("temporal", hp, temp_art)

    print("\n=== All cases completed ===")

def consume(gen: Iterable, root: Path) -> pd.DataFrame:
    rows = []
    for stage, hp, art in gen:
        ok = art.results is not None
        abs_path = art.save_path
        if abs_path is not None:
            try:
                rel_path = str(Path(abs_path).relative_to(root))
            except ValueError:
                # if somehow it's outside root, fall back to absolute
                rel_path = str(abs_path)
        else:
            rel_path = None
        rows.append({
            "id":                hp["id"],
            "desc":              hp["desc"],
            "stage":             stage,
            "error":             None if ok else "infeasible",
            "agg_hash":          art.agg_hash,
            "gtep_hash":         art.gtep_hash,
            "artifact_path":     rel_path,
            "objective":         getattr(art.results.stats, "objective", float("nan")) if ok else float("nan"),
        })
    return pd.DataFrame(rows)

def run_aggregated(agg_results, agg_hash: str, exp_root: Path, day_weights: List[float], threads: int = 1) -> RunArtifacts:
    data = build_input_data(agg_results["spatiotemporal"], power_scale='GW', cost_scale='M$', sig_round=3, verbose=True)
    gtep_hash = hashlib.md5(json.dumps(asdict(data.params), sort_keys=True).encode()).hexdigest()[:8]
    log_path = exp_root / "results" / "gtep_logs" / "aggregated" / f"aggregated_{agg_hash}_{gtep_hash}.csv"
    clusters = agg_results["clusters"]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1)
        env.setParam("Threads",   threads)
        env.setParam("TimeLimit", 2 * 60 * 60) # in seconds
        env.start()
        logger = SolveLogger(log_path)
        model = AggregatedGTEP(env, data.buses, data.branches, data.params)
        try:
            res = model.optimize(data.rep_period, day_weights, logger)
        except RuntimeError as e:
            if "No feasible solution" in str(e):
                print(f"[{agg_hash}] Infeasible aggregated GTEP → skipping")
                res = None
            else:
                raise
    art = RunArtifacts(
        results=res,
        data=data,
        agg_hash=agg_hash,
        gtep_hash=gtep_hash,
        save_path=None,
        log_path=log_path,
        clusters=clusters
    )
    if res is not None:
        art.save_path = save_artifacts(art, "aggregated", exp_root)
    return art

def run_spatial(agg_results, agg_hash: str, exp_root: Path, day_weights: List[float], agg_inv: InvestmentResults, threads: int = 1) -> RunArtifacts:
    data = build_input_data(agg_results["temporal_only"], power_scale='GW', cost_scale='M$', sig_round=3, verbose=True)
    gtep_hash = hashlib.md5(json.dumps(asdict(data.params), sort_keys=True).encode()).hexdigest()[:8]
    log_path = exp_root / "results" / "gtep_logs" / "spatial" / f"spatial_{agg_hash}_{gtep_hash}.csv"
    clusters = agg_results["clusters"]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1)
        env.setParam("Threads",   threads)
        env.setParam("TimeLimit", 4 * 60 * 60) # in seconds
        env.start()
        logger = SolveLogger(log_path)
        model = SpatialGTEP(env, data.buses, data.branches, data.params, agg_inv, clusters["spatial"], agg_results["spatiotemporal"]["branches"])
        try:
            res = model.optimize(data.rep_period, day_weights, logger)
        except RuntimeError as e:
            if "No feasible solution" in str(e):
                print(f"[{agg_hash}] Infeasible spatial GTEP → skipping")
                res = None
            else:
                raise
    art = RunArtifacts(
        results=res,
        data=data,
        agg_hash=agg_hash,
        gtep_hash=gtep_hash,
        save_path=None,
        log_path=log_path,
        clusters=clusters
    )
    if res is not None:
        art.save_path = save_artifacts(art, "spatial", exp_root)
    return art

def run_temporal(agg_results, agg_hash: str, exp_root: Path, fixed_inv: InvestmentResults, threads: int = 1) -> RunArtifacts:
    data = build_input_data(agg_results["original"], power_scale='GW', cost_scale='M$', sig_round=3, verbose=True)
    gtep_hash = hashlib.md5(json.dumps(asdict(data.params), sort_keys=True).encode()).hexdigest()[:8]
    log_path = exp_root / "results" / "gtep_logs" / "temporal" / f"temporal_{agg_hash}_{gtep_hash}.csv"
    clusters = agg_results["clusters"]
    day_weights = uniform_day_weights(data.rep_period.T)  # uniform weights over full horizon
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1)
        env.setParam("Threads",   threads)
        env.setParam("TimeLimit", 2 * 60 * 60) # in seconds
        env.start()
        logger = SolveLogger(log_path)
        model = TemporalGTEP(env, data.buses, data.branches, data.params, fixed_inv)
        try:
            res = model.optimize(data.rep_period, day_weights, logger)
        except RuntimeError as e:
            if "No feasible solution" in str(e):
                print(f"[{agg_hash}] Infeasible temporal GTEP → skipping")
                res = None
            else:
                raise
    art = RunArtifacts(
        results=res,
        data=data,
        agg_hash=agg_hash,
        gtep_hash=gtep_hash,
        save_path=None,
        log_path=log_path,
        clusters=clusters
    )
    if res is not None:
        art.save_path = save_artifacts(art, "temporal", exp_root)
    return art

def save_artifacts(art: RunArtifacts, stage: str, exp_root: Path, overwrite: bool = False) -> Path:
    """
    Pickle the entire RunArtifacts object to disk.
    Returns the path where it's saved.
    """
    fname = f"{stage}_{art.agg_hash}_{art.gtep_hash}.pkl"
    out = exp_root / "results" / "gtep_output" / stage / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        out = ensure_unique(out)

    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f"Saved full RunArtifacts to {out}")
    return out

def load_artifacts(path: Path) -> RunArtifacts:
    with open(path, "rb") as f:
        return pickle.load(f)
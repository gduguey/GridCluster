# utils.py — logging, data loading, scaling, and helper functions for GTEP

import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import gurobipy as gp

from .types import Params, RepPeriod, InputData

# =========================================================
# ==================  LOGGING / CALLBACK  =================
# =========================================================

class SolveLogger:
    """Minimal CSV logger for MIP gap evolution."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.data = []
        self.best_obj = None
        self.best_bd = None
        self.start = time.time()

    def callback(self, model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd  = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            if self.best_obj != cur_obj or self.best_bd != cur_bd:
                self.best_obj = cur_obj
                self.best_bd  = cur_bd
                self.data.append([time.time() - self.start, cur_obj, cur_bd])
                pd.DataFrame(self.data, columns=["t", "obj", "bd"]).to_csv(self.path, index=False)

# =========================================================
# ===================  DATA UTILITIES  ====================
# =========================================================

def haversine(lat1, lon1, lat2, lon2, unit='km'):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    if unit == 'km':
        r = 6371.0
    elif unit == 'miles':
        r = 3958.8
    return c * r

def build_input_data(results_block: dict,
                     power_scale='GW', cost_scale='M$', sig_round=3, verbose=True) -> InputData:
    buses, branches, load, pv, wind, p = scale_inputs(
        *read_inputs(results_block),
        power_scale=power_scale, cost_scale=cost_scale,
        sig_round=sig_round, verbose=verbose
    )
    return InputData(buses, branches, RepPeriod(load, pv, wind), p)

def round_scalar(x, sig):
    if isinstance(x, (int, float)):
        mag = np.floor(np.log10(abs(x))) if x != 0 else 0 # magnitude, x = a * 10^mag so log10(x) = log10(a) + mag with log10(a) in [0, 1) with clean handling of log10(0) = -inf
        dec = int(np.clip(sig - 1 - mag, 0, 10)) # number of decimals to keep (−1 means rounding to tens), clipped to 0 and 10 to round to integers or 10 decimals max
        return round(x, dec)
    print(f"Warning: Unsupported type {type(x)} for rounding. Returning as is.")
    return x

def smart_round(obj, sig):
    if isinstance(obj, pd.DataFrame):
        return obj.map(lambda x: round_scalar(x, sig))
    if isinstance(obj, dict):
        return {k: smart_round(v, sig) for k, v in obj.items()}
    return round_scalar(obj, sig)

def read_inputs(results) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    pv = results['time_series']['solar']
    wind = results['time_series']['wind']
    load = results['time_series']['demand']
    buses = results['nodes']
    branches = results['branches']

    load.columns = load.columns.astype(int)
    pv.columns   = pv.columns.astype(int)
    wind.columns = wind.columns.astype(int)

    # Apply 60% increase to all load values to project 2050 demand
    load = load * 1.6

    params = {}
    params['Initial SOC'] = 0.5 # initial state of charge
    params['eta_discharge'] = 0.922 # efficiency of discharge
    params['eta_charge'] = 0.922 # efficiency of charge
    params['PV Resource Availability'] = 2360325 
    params['Wind Resource Availability'] = 57900
    params['CCGT Ramp Rate'] = 0.05
    params['CCGT Max Cap'] = 640.0
    params['c_ccgt'] = 142087
    params['c_pv'] = 47620
    params['c_wind'] = 82412
    params['c_storage'] = 75247
    crpyears_tran = 40
    discount_rate_tran = 0.08
    crf = discount_rate_tran / (1 - (1 + discount_rate_tran) ** -crpyears_tran)
    params['c_tran'] = 3780 * crf
    params['d_ccgt'] = 4.46 * 6.85
    params['d_shed'] = 10000
    params['Rate Duration'] = 4
    params['Length'] = {}

    for ix, row in branches.iterrows():
        f, t = row['from_bus_id'], row['to_bus_id']
        flon, flat = buses.loc[buses.bus_id == f, ['Lon','Lat']].iloc[0]
        tlon, tlat = buses.loc[buses.bus_id == t, ['Lon','Lat']].iloc[0]
        length_miles = haversine(flat, flon, tlat, tlon, unit='miles')
        params['Length'][ix] = length_miles

    median_len = np.median(list(params['Length'].values()))
    rating_typical = 715.0  # MW (from WECC-weighted HVAC number)
    MW_per_mile = rating_typical / median_len
    params['MW_per_mile'] = MW_per_mile
    params['TranMax'] = {ix: MW_per_mile * params['Length'][ix] for ix in params['Length']}

    print("=== DATA LOADING ===")
    print("PV shape ", pv.shape)
    print("Wind shape ", wind.shape)
    print("Load shape ", load.shape)
    print("\n")

    return buses, branches, load, pv, wind, params

def scale_inputs(buses, branches, load, pv, wind, params,
                 power_scale='GW', cost_scale='M$', sig_round=4, verbose=True):
    
    p_scale = {'MW':1,'GW':1e-3,'TW':1e-6}[power_scale]
    c_scale = {'$':1,'k$':1e-3,'M$':1e-6}[cost_scale]

    load_s = smart_round(load * p_scale, sig_round)
    pv_s   = pv.round(3)
    wind_s = wind.round(3)

    params_s = dict(params)  # shallow copy

    def mul_key(klist, fac):
        for k in klist:
            if k in params_s:
                params_s[k] = params_s[k] * fac

    mul_key(['PV Resource Availability','Wind Resource Availability','CCGT Max Cap','MW_per_mile'], p_scale)
    mul_key(['c_ccgt','c_pv','c_wind','c_storage','c_tran','d_ccgt','d_shed'], c_scale / p_scale)

    if 'TranMax' in params_s:
        params_s['TranMax'] = {k: v * p_scale for k, v in params_s['TranMax'].items()}

    params_s = smart_round(params_s, sig_round)

    if verbose:
        print("=== SCALING CHECK ===")
        print(f"Power: MW → {power_scale} (×{p_scale}), Cost: $ → {cost_scale} (×{c_scale})")
        print(f"    PV CF range: [{pv_s.values.min():.3g}, {pv_s.values.max():.3g}]")
        print(f"    Wind CF range: [{wind_s.values.min():.3g}, {wind_s.values.max():.3g}]")
        lmin, lmax = load.values.min(), load.values.max()
        lsmin, lsmax = load_s.values.min(), load_s.values.max()
        print(f"    Load range MW → {power_scale}:  [{lmin:.3g}, {lmax:.3g}] → [{lsmin:.3g}, {lsmax:.3g}]")
        # rough coefficient ranges
        keys = ['c_ccgt','c_pv','c_wind','c_storage','c_tran','d_ccgt','d_shed',
                'PV Resource Availability','Wind Resource Availability','CCGT Max Cap']
        ov, sv = [], []
        for k in keys:
            if k in params:
                ov.append(params[k])
                sv.append(params_s[k])
        if 'TranMax' in params:
            ov.extend(params['TranMax'].values())
            sv.extend(params_s['TranMax'].values())
        ov = [v for v in ov if v > 1e-10]
        sv = [v for v in sv if v > 1e-10]
        if ov:
            print(f"    Param range: [{min(ov):.3g}, {max(ov):.3g}] → [{min(sv):.3g}, {max(sv):.3g}]")
        print()

    # Cast to Params dataclass
    params_dc = Params(
        Initial_SOC=params_s['Initial SOC'],
        eta_discharge=params_s['eta_discharge'],
        eta_charge=params_s['eta_charge'],
        PV_resource=params_s['PV Resource Availability'],
        Wind_resource=params_s['Wind Resource Availability'],
        CCGT_ramp=params_s['CCGT Ramp Rate'],
        CCGT_max_cap=params_s['CCGT Max Cap'],
        c_ccgt=params_s['c_ccgt'],
        c_pv=params_s['c_pv'],
        c_wind=params_s['c_wind'],
        c_storage=params_s['c_storage'],
        c_tran=params_s['c_tran'],
        d_ccgt=params_s['d_ccgt'],
        d_shed=params_s['d_shed'],
        rate_duration=params_s['Rate Duration'],
        MW_per_mile=params_s['MW_per_mile'],
        TranMax=params_s['TranMax'],
        Length=params_s['Length'],
    )

    return buses, branches, load_s, pv_s, wind_s, params_dc

def uniform_day_weights(T: int, hours_per_day: int = 24) -> List[float]:
    n_days = T // hours_per_day
    return [1] * n_days

def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        alt = path.with_stem(f"{path.stem}__{i}")
        if not alt.exists():
            return alt
        i += 1
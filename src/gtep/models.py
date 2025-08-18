# models.py — core GTEP optimization models (aggregated, spatial, temporal)

import time
from typing import Dict, List
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from .utils import SolveLogger
from .types import SolveStats, InvestmentResults, OperationResults, GTEPResults, Params, RepPeriod

# =========================================================
# ====================  MODEL CLASSES  ====================
# =========================================================

class GTEPBase:
    """Common scaffolding. Subclasses must implement build_model()."""
    def __init__(self, env: gp.Env, buses: pd.DataFrame, branches: pd.DataFrame, params: Params):
        self.env = env
        self.model = gp.Model(env=self.env)
        self.buses = buses['bus_id'].tolist()
        self.branches = branches
        self.p = params

        self.ref_bus = self.buses[0]
        self.non_ref_buses = self.buses[1:]
        self._topology()

    def _topology(self):
        self.lines_in  = {bus: [] for bus in self.buses}
        self.lines_out = {bus: [] for bus in self.buses}
        for ix, row in self.branches.iterrows():
            self.lines_in[row['to_bus_id']].append(ix)
            self.lines_out[row['from_bus_id']].append(ix)

    # ---------- Vars ----------
    def _add_investment_vars(self):
        self.PV     = self.model.addVars(self.buses, name="PV")
        self.Wind   = self.model.addVars(self.buses, name="Wind")
        self.CCGT   = self.model.addVars(self.buses, vtype=GRB.INTEGER, name="CCGT_units")
        self.Storage= self.model.addVars(self.buses, name="StoragePower")
        self.Tran   = self.model.addVars(self.branches.index, name="TransCap")

    def _add_operation_vars(self, T):
        self.T = T
        self.y_ccgt      = self.model.addVars(self.buses, T, name="y_ccgt")
        self.y_shed      = self.model.addVars(self.buses, T, name="y_shed")
        self.y_curtail   = self.model.addVars(self.buses, T, name="y_curtail")
        self.y_charge    = self.model.addVars(self.buses, T, name="y_charge")
        self.y_discharge = self.model.addVars(self.buses, T, name="y_discharge")
        self.y_soc       = self.model.addVars(self.buses, T, name="y_soc")
        self.y_flow      = self.model.addVars(self.branches.index, T, lb=-GRB.INFINITY, name="y_flow")

    # ---------- Constraints ----------
    def _investment_constraints(self):
        self.model.addConstr(gp.quicksum(self.PV)   <= self.p.PV_resource)
        self.model.addConstr(gp.quicksum(self.Wind) <= self.p.Wind_resource)
        self.model.addConstrs(self.Tran[i] <= self.p.TranMax[i] for i in self.branches.index)

    def _operation_constraints(self, rp: RepPeriod):
        loads   = rp.Load.reset_index(drop=True).astype(float)
        pv_cfs  = rp.PV.reset_index(drop=True).astype(float)
        wind_cfs= rp.Wind.reset_index(drop=True).astype(float)

        # Power balance
        self.model.addConstrs(
            (
                pv_cfs[bus].values[t] * self.PV[bus]
                + wind_cfs[bus].values[t] * self.Wind[bus]
                + self.y_ccgt[bus, t]
                + self.y_discharge[bus, t]
                + self.y_shed[bus, t]
                + gp.quicksum(self.y_flow[l, t] for l in self.lines_in[bus])
                - gp.quicksum(self.y_flow[l, t] for l in self.lines_out[bus])
                - loads[bus][t]
                - self.y_charge[bus, t]
                - self.y_curtail[bus, t]
                == 0
            )
            for bus in self.buses for t in range(self.T)
        )

        # CCGT
        self.model.addConstrs(self.y_ccgt[b,t] <= self.p.CCGT_max_cap * self.CCGT[b] for b in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_ccgt[b,t] - self.y_ccgt[b,(t-1)%self.T] <=  self.p.CCGT_ramp * self.p.CCGT_max_cap * self.CCGT[b] for b in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_ccgt[b,t] - self.y_ccgt[b,(t-1)%self.T] >= -self.p.CCGT_ramp * self.p.CCGT_max_cap * self.CCGT[b] for b in self.buses for t in range(self.T))

        # Storage
        self.model.addConstrs(self.y_soc[b,t] <= self.p.rate_duration * self.Storage[b]               for b in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_charge[b,t]    <= self.Storage[b]                                for b in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_discharge[b,t] <= self.Storage[b]                                for b in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_soc[b, t] == self.y_soc[b, t-1] + self.p.eta_charge * self.y_charge[b, t] - (1/self.p.eta_discharge) * self.y_discharge[b, t] for b in self.buses for t in range(1, self.T))
        self.model.addConstrs(self.y_soc[b, 0] == self.p.Initial_SOC * self.p.rate_duration * self.Storage[b] for b in self.buses)

        # Transmission
        self.model.addConstrs(self.y_flow[i, t] <= self.Tran[i] for i in self.branches.index for t in range(self.T))
        self.model.addConstrs(self.y_flow[i, t] >= -self.Tran[i] for i in self.branches.index for t in range(self.T))

    # ---------- Objective ----------
    def _objective_common(self, day_weights: List[float]):
        obj = 0
        obj += self.p.c_pv   * gp.quicksum(self.PV[b]    for b in self.buses)
        obj += self.p.c_wind * gp.quicksum(self.Wind[b]  for b in self.buses)
        obj += self.p.c_ccgt * self.p.CCGT_max_cap * gp.quicksum(self.CCGT[b] for b in self.buses)
        obj += self.p.c_storage * gp.quicksum(self.Storage[b] for b in self.buses)
        for i in self.branches.index:
            obj += self.p.c_tran * self.p.Length[i] * self.Tran[i]

        H = 24
        for i, w in enumerate(day_weights):
            h0, h1 = H*i, H*(i+1)
            obj += w * ( self.p.d_ccgt * gp.quicksum(self.y_ccgt[b,t] for b in self.buses for t in range(h0,h1))
                       + self.p.d_shed * gp.quicksum(self.y_shed[b,t]  for b in self.buses for t in range(h0,h1)) )
        return obj

    # ---------- Solve ----------
    def optimize(self, rp: RepPeriod, day_weights: List[float], logger: SolveLogger | None = None) -> GTEPResults:
        start_build = time.time()
        self.build_model(rp, day_weights)  # delegate to subclass
        self.model.setObjective(self.objective, GRB.MINIMIZE)
        build_time = time.time() - start_build

        start_run = time.time()
        cb_method = getattr(logger, "callback", None)
        if callable(cb_method):
            # wrap the bound method into a true function
            def _gurobi_callback(model, where):
                return cb_method(model, where)
            self.model.optimize(callback=_gurobi_callback)
        else:
            self.model.optimize()
        run_time = time.time() - start_run

        if self.model.SolCount == 0:
            raise RuntimeError("No feasible solution.")

        res = GTEPResults(
            stats=SolveStats(build_time, run_time, self.model.objVal),
            inv=InvestmentResults(
                PV=self._pull(self.PV),
                Wind=self._pull(self.Wind),
                CCGT=self._pull(self.CCGT),
                Storage=self._pull(self.Storage),
                Tran=self._pull(self.Tran),
            ),
            op=OperationResults(
                y_ccgt=self._pull(self.y_ccgt, two_index=True),
                y_curtail=self._pull(self.y_curtail, two_index=True),
                y_shed=self._pull(self.y_shed, two_index=True),
                y_charge=self._pull(self.y_charge, two_index=True),
                y_discharge=self._pull(self.y_discharge, two_index=True),
                y_soc=self._pull(self.y_soc, two_index=True),
                y_flow=self._pull(self.y_flow, two_index=True),
            )
        )
        return res

    @staticmethod
    def _pull(var, two_index=False):
        if two_index:
            return {(i,j): round(getattr(var[i,j], "X", var[i,j]),3) for i,j in var.keys()}
        return {i: round(getattr(var[i], "X", var[i]),3) for i in var.keys()}

# ---------- Concrete models ----------

class AggregatedGTEP(GTEPBase):
    def build_model(self, rp: RepPeriod, day_weights: List[float]):
        self._add_investment_vars()
        self._add_operation_vars(T=len(rp.Load))
        self._investment_constraints()
        self._operation_constraints(rp)
        self.objective = self._objective_common(day_weights)

class SpatialGTEP(GTEPBase):
    def __init__(self, env, buses, branches, params,
                 agg_inv: InvestmentResults,
                 cluster_map: Dict[int, List[int]],
                 agg_lines: pd.DataFrame):
        super().__init__(env, buses, branches, params)
        self.agg_inv = agg_inv
        self.cluster_map = cluster_map # {rep_id: [members]}
        self.agg_lines = agg_lines     # DataFrame with branch_id, from_bus_id, to_bus_id

    def build_model(self, rp: RepPeriod, day_weights: List[float]):
        self._add_investment_vars()
        self._add_operation_vars(T=len(rp.Load))
        self._investment_constraints()
        self._operation_constraints(rp)
        self._spatial_link_constraints()
        self.objective = self._objective_common(day_weights)

    def _spatial_link_constraints(self):
        for rep_id, members in self.cluster_map.items():
            self.model.addConstr(gp.quicksum(self.PV[b]   for b in members) == self.agg_inv.PV[rep_id])
            self.model.addConstr(gp.quicksum(self.Wind[b] for b in members) == self.agg_inv.Wind[rep_id])
            self.model.addConstr(gp.quicksum(self.CCGT[b] for b in members) == self.agg_inv.CCGT[rep_id])
            self.model.addConstr(gp.quicksum(self.Storage[b] for b in members) == self.agg_inv.Storage[rep_id])

        for _, row in self.agg_lines.iterrows():
            f_rep, t_rep, br_id = row["from_bus_id"], row["to_bus_id"], row["branch_id"]
            from_buses = self.cluster_map[f_rep]
            to_buses   = self.cluster_map[t_rep]
            inter_lines = self.branches[
                (self.branches['from_bus_id'].isin(from_buses)) &
                (self.branches['to_bus_id'].isin(to_buses))
            ].index
            self.model.addConstr(gp.quicksum(self.Tran[i] for i in inter_lines) == self.agg_inv.Tran[br_id])

class TemporalGTEP(GTEPBase):
    def __init__(self, env, buses, branches, params, fixed_inv: InvestmentResults):
        super().__init__(env, buses, branches, params)
        self.fixed_inv = fixed_inv

    def build_model(self, rp: RepPeriod, day_weights: List[float]):
        # Fix first-stage vars
        self.PV      = self.fixed_inv.PV
        self.Wind    = self.fixed_inv.Wind
        self.CCGT    = self.fixed_inv.CCGT
        self.Storage = self.fixed_inv.Storage
        self.Tran    = self.fixed_inv.Tran

        self._add_operation_vars(T=len(rp.Load))
        self._operation_constraints(rp)
        self.objective = self._objective_common(day_weights)
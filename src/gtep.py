import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
from gurobipy import GRB
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix
import multiprocessing
    
# GTEP ---------------------------------------------------------------------------------

class GTEPBase:
    """Base class for GTEP models with common functionality"""
    def __init__(self, env, data, params):
        # Common initialization
        self.env = env
        self.model = gp.Model(env=self.env)
        self.buses = data[0]['bus_id'].tolist()
        self.branches = data[1]
        self.params = params
        self.objective = 0

        # Network setup
        self.ref_bus = self.buses[0]
        self.non_ref_buses = self.buses[1:]
        self._setup_network_topology()
        
    def _setup_network_topology(self):
        """Setup line connections"""
        self.lines = {'in': {bus: [] for bus in self.buses}, 
                      'out': {bus: [] for bus in self.buses}}
        for ix, row in self.branches.iterrows():
            self.lines['in'][row['to_bus_id']].append(ix)
            self.lines['out'][row['from_bus_id']].append(ix)

    def solve(self, rep_period, weights, callback=None):
        """Common solve method for all models"""
        start = time.time()
        self.build_model(rep_period, weights)
        self.model.setObjective(self.objective, GRB.MINIMIZE)
        buildtime = time.time() - start
        
        start = time.time()
        self.model.optimize(callback)
        runtime = time.time() - start
        
        if self.model.SolCount > 0:
            return {
                'buildtime': buildtime,
                'runtime': runtime,
                'objective': self.model.objVal,
                'inv': {
                    'PV': self._get_solutions(self.PV),
                    'Wind': self._get_solutions(self.Wind),
                    'CCGT': self._get_solutions(self.CCGT),
                    'Storage Energy': self._get_solutions(self.StorageEnergy),
                    'Storage Power': self._get_solutions(self.StoragePower),
                    'Tran': self._get_solutions(self.Tran)
                }
            }
        return None

    def _add_investment_variables(self):
        """Add first-stage investment variables (common for all models)"""
        self.PV = self.model.addVars(self.buses)
        self.Wind = self.model.addVars(self.buses)
        self.CCGT = self.model.addVars(self.buses, vtype=GRB.INTEGER)
        self.StorageEnergy = self.model.addVars(self.buses)
        self.StoragePower = self.model.addVars(self.buses)
        self.Tran = self.model.addVars(self.branches.shape[0])

    def _add_operation_variables(self, rep_period):
        """Add second-stage operation variables (common for all models)"""
        self.T = len(rep_period['Load'])
        self.y_ccgt = self.model.addVars(self.buses, self.T) # dispatchable generation fuel
        self.y_shed = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # load shedding (potentially negative)
        self.y_charge = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # battery charging
        self.y_soc = self.model.addVars(self.buses, self.T) # battery state-of-charge
        self.y_pos_shed = self.model.addVars(self.buses, self.T) # max(0, load shedding)
        self.y_flow = self.model.addVars(self.branches.shape[0], self.T, lb=-GRB.INFINITY) # flows
        self.y_theta = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY)  # phase angles

    def _add_investment_constraints(self):
        """Add investment constraints"""
        # Resource availability constraints
        self.model.addConstr(gp.quicksum(self.PV) <= self.params['PV Resource Availability'])
        self.model.addConstr(gp.quicksum(self.Wind) <= self.params['Wind Resource Availability'])
        
        # Storage energy-power coupling
        self.model.addConstrs(self.StorageEnergy[bus] <= self.params['Rate Duration'] * self.StoragePower[bus] for bus in self.buses)
        
        # Transmission capacity constraints
        for ix, row in self.branches.iterrows():
            self.model.addConstr(self.Tran[ix] <= self.params['Transmission Max Cont Rating'])
 
    def _add_operation_constraints(self, rep_period):
        """Add operational constraints common to all models"""
        loads = rep_period['Load'].reset_index(drop=True).astype(float)
        pv_cfs = rep_period['PV'].reset_index(drop=True).astype(float)
        wind_cfs = rep_period['Wind'].reset_index(drop=True).astype(float)

        injection = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        outflow = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        for bus in self.buses:
            in_lines = self.lines['in'][bus]
            out_lines = self.lines['out'][bus]
            for t in range(self.T):
                injection[(bus,t)] = gp.LinExpr((1,self.y_flow[line, t]) for line in in_lines)
                injection[(bus,t)] += self.y_shed[bus,t]
                outflow[(bus,t)] = gp.LinExpr((1,self.y_flow[line, t]) for line in out_lines)
                outflow[(bus,t)] += loads[bus][t]
                injection[(bus,t)] += pv_cfs[bus].values[t] * self.PV[bus]
                injection[(bus,t)] += wind_cfs[bus].values[t] * self.Wind[bus]
                injection[(bus,t)] += self.y_ccgt[bus,t]
                outflow[(bus,t)] += self.y_charge[bus,t]

        # power balance constraint
        self.model.addConstrs(injection[(bus,t)] - outflow[(bus,t)] == 0 for bus in self.buses for t in range(self.T))

        # CCGT generation constraint
        self.model.addConstrs(self.y_ccgt[bus,t] <= self.params['CCGT Max Cap'] * self.CCGT[bus] for bus in self.buses for t in range(self.T))

        # ramp up constraint
        self.model.addConstrs(self.y_ccgt[bus,t] - self.y_ccgt[bus,(t-1)%self.T] <= self.params['CCGT Ramp Rate'] * self.CCGT[bus] for bus in self.buses for t in range(self.T))

        # ramp down constraint
        self.model.addConstrs(self.y_ccgt[bus,t] - self.y_ccgt[bus,(t-1)%self.T] >= -self.params['CCGT Ramp Rate'] * self.CCGT[bus] for bus in self.buses for t in range(self.T))

        # storage energy capacity constraint
        self.model.addConstrs(self.y_soc[bus,t] <= self.StorageEnergy[bus] for bus in self.buses for t in range(self.T))

        # storage power capacity constraint
        self.model.addConstrs(self.y_charge[bus,t] <= self.StoragePower[bus] for bus in self.buses for t in range(self.T))
        self.model.addConstrs(self.y_charge[bus,t] >= -self.StoragePower[bus] for bus in self.buses for t in range(self.T))

        # charging dynamics constraint
        self.model.addConstrs(self.y_soc[bus,(t-1)%self.T] + self.y_charge[bus,(t-1)%self.T] == self.y_soc[bus,t] for bus in self.buses for t in range(self.T))

        # positive load shedding constraint
        self.model.addConstrs(self.y_pos_shed[bus,t] >= self.y_shed[bus,t] for bus in self.buses for t in range(self.T))

        # transmission capacity constraints
        for ix, row in self.branches.iterrows():
            line = ix
            i, j = row['from_bus_id'], row['to_bus_id']
            self.model.addConstrs(self.y_flow[line,t] == (self.params['Susceptance'][line]) * (self.y_theta[i,t] - self.y_theta[j,t]) for t in range(self.T))
            self.model.addConstrs(self.y_flow[line,t] <= self.Tran[line] for t in range(self.T)) # flow capacity constraint
            self.model.addConstrs(self.y_flow[line,t] >= -self.Tran[line] for t in range(self.T)) # flow capacity constraint

        # phase angle constraints
        self.model.addConstrs(self.y_theta[bus,t] <= 180 for bus in self.non_ref_buses for t in range(self.T))
        self.model.addConstrs(self.y_theta[bus,t] >= -180 for bus in self.non_ref_buses for t in range(self.T))
        self.model.addConstrs(self.y_theta[self.ref_bus,t] == 0 for t in range(self.T))

    def _add_objective_terms(self, weights):
        """Add objective terms common to all models"""
        # Investment costs
        self.objective += self.params['c_pv'] * gp.quicksum(self.PV)        
        self.objective += self.params['c_wind'] * gp.quicksum(self.Wind)
        self.objective += self.params['c_ccgt'] * self.params['CCGT Max Cap'] * gp.quicksum(self.CCGT)
        self.objective += self.params['c_stor_energy'] * gp.quicksum(self.StorageEnergy)
        self.objective += self.params['c_stor_power'] * gp.quicksum(self.StoragePower)
        for ix, row in self.branches.iterrows():
            self.objective += self.params['c_tran'] * self.params['Length'][ix] * self.Tran[ix]
        
        # Operational costs
        for i in range(len(weights)):
            self.objective += weights[i] * (self.params['d_ccgt'] * gp.quicksum(self.y_ccgt[bus,t] for bus in self.buses for t in range(24*i,24*(i+1))) + self.params['d_shed'] * gp.quicksum(self.y_pos_shed[bus,t] for bus in self.buses for t in range(24*i,24*(i+1))))
       
    def _get_solutions(self, var):
        if isinstance(var, dict):
            return {k: self._get_solutions(v) for k, v in var.items()}
        else:
            return np.round(var.X, 3)
        
# Fully aggregated GTEP -------------------------------------------------------------------------------------

class AggregatedGTEP(GTEPBase):
    """Fully aggregated model (spatial and temporal aggregation)"""
    def build_model(self, rep_period, weights):
        self._add_investment_variables()
        self._add_operation_variables(rep_period)
        self._add_investment_constraints()
        self._add_operation_constraints(rep_period)
        self._add_objective_terms(weights)

# Spatially deaggregated GTEP ----------------------------------------------------------------------------------

class SpatialGTEP(GTEPBase):
    """Spatially deaggregated model with temporal aggregation"""
    def __init__(self, env, data, params, AggregatedGTEP_results, spatial_agg_results, agg_lines):
        super().__init__(env, data, params)
        self.agg_results = AggregatedGTEP_results  # Results from AggregatedGTEP
        self.spatial_agg_results = spatial_agg_results  # Spatial aggregation mapping
        self.agg_lines = agg_lines  # Aggregated lines
    
    def build_model(self, rep_period, weights):
        self._add_investment_variables()
        self._add_operation_variables(rep_period)
        self._add_investment_constraints()
        self._add_operation_constraints(rep_period)
        self._add_objective_terms(weights)
        self._add_spatial_constraints()
    
    def _add_spatial_constraints(self):
        """Add constraints tying investments to aggregated results"""
        # # Node investment constraints
        for rep_id, members in self.spatial_agg_results['clusters'].items():
            self.model.addConstr(gp.quicksum(self.PV[bus_id] for bus_id in members) == self.agg_results['PV'][rep_id])
            self.model.addConstr(gp.quicksum(self.Wind[bus_id] for bus_id in members) == self.agg_results['Wind'][rep_id])
            self.model.addConstr(gp.quicksum(self.CCGT[bus_id] for bus_id in members) == self.agg_results['CCGT'][rep_id])
            self.model.addConstr(gp.quicksum(self.StorageEnergy[bus_id] for bus_id in members) == self.agg_results['Storage Energy'][rep_id])
        #     self.model.addConstr(gp.quicksum(self.StoragePower[bus_id] for bus_id in members) == self.agg_results['Storage Power'][rep_id])
        #     print(self.agg_results['PV'][rep_id], self.agg_results['Wind'][rep_id], self.agg_results['CCGT'][rep_id],
        #           self.agg_results['Storage Energy'][rep_id], self.agg_results['Storage Power'][rep_id])
        # print(self.branches)
        # Transmission constraints
        for _, agg_line in self.agg_lines.iterrows():
            from_rep = agg_line["from_bus_id"]
            to_rep = agg_line["to_bus_id"]
            
            # Find lines between these clusters
            from_buses = self.spatial_agg_results['clusters'][from_rep]
            to_buses = self.spatial_agg_results['clusters'][to_rep]
            
            inter_lines = self.branches[
                (self.branches['from_bus_id'].isin(from_buses) & 
                 self.branches['to_bus_id'].isin(to_buses))
            ].index
            
            self.model.addConstr(
                gp.quicksum(self.Tran[line] for line in inter_lines) == 
                self.agg_results['Tran'][agg_line['branch_id']]
            )
        
# Temporally deaggregated GTEP ---------------------------------------------------------------------------------

class TemporalGTEP(GTEPBase):
    """Fully deaggregated model with fixed investments"""
    def __init__(self, env, data, params, fixed_inv):
        super().__init__(env, data, params)
        self.fixed_inv = fixed_inv  # Fixed investment decisions
    
    def build_model(self, rep_period, weights):
        self._add_investment_variables()
        self._add_operation_variables(rep_period)
        self._add_investment_constraints()
        self._add_operation_constraints(rep_period)
        self._add_objective_terms(weights)
        self._fix_investments()
    
    def _fix_investments(self):
        """Fix investment variables to predetermined values"""
        for bus in self.buses:
            self.PV[bus].lb = self.fixed_inv['PV'][bus]
            self.PV[bus].ub = self.fixed_inv['PV'][bus]
            self.Wind[bus].lb = self.fixed_inv['Wind'][bus]
            self.Wind[bus].ub = self.fixed_inv['Wind'][bus]
            self.CCGT[bus].lb = self.fixed_inv['CCGT'][bus]
            self.CCGT[bus].ub = self.fixed_inv['CCGT'][bus]
            self.StorageEnergy[bus].lb = self.fixed_inv['Storage Energy'][bus]
            self.StorageEnergy[bus].ub = self.fixed_inv['Storage Energy'][bus]
            # self.StoragePower[bus].lb = self.fixed_inv['Storage Power'][bus]
            # self.StoragePower[bus].ub = self.fixed_inv['Storage Power'][bus]
        
        for branch in self.branches.index:
            self.Tran[branch].lb = self.fixed_inv['Tran'][branch]
            self.Tran[branch].ub = self.fixed_inv['Tran'][branch]    

# Utils ---------------------------------------------------------------------------------

def harversine(lat1, lon1, lat2, lon2, unit = 'km'):
    """
    Calculate the great circle distance in kilometers between two points on the earth (specified in decimal degrees)
    """
    from math import radians, cos, sin, asin, sqrt
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    if unit == 'km':
        r = 6371.0
    elif unit == 'miles':
        r = 3958.8
    return c * r 
    
def read_data(results):
    pv = results['time_series']['solar']
    wind = results['time_series']['wind']
    load = results['time_series']['demand']
    buses = results['nodes']
    branches = results['branches']
    
    load.columns = load.columns.astype(int)
    pv.columns = pv.columns.astype(int)
    wind.columns = wind.columns.astype(int)

    print("PV shape ",pv.shape)
    print("Wind shape ",wind.shape)
    print("Load shape ",load.shape)
    
    params = {}
    params['PV Resource Availability'] = 500000 # MW
    params['Wind Resource Availability'] = 110000 # MW
    params['Transmission Max Cont Rating'] = 3000 # MW
    params['CCGT Ramp Rate'] = 248.4
    params['CCGT Max Cap'] = 355.0
    params['c_ccgt'] = 31167 # $/MW
    params['c_pv'] = 22400 # $/MW
    params['c_wind'] = 26933 # $/MW
    params['c_stor_energy'] = 4300 # $/MWh
    params['c_stor_power'] = 5200 # $/MW
    params['c_tran'] = 246.6 # $/MW/mile
    params['d_ccgt'] = 20 # $/MWh
    params['d_shed'] = 10000 # $/MWh
    params['Rate Duration'] = 4
    params['Susceptance'] = {}
    params['Length'] = {}
    for ix, row in branches.iterrows():
        from_bus_id, to_bus_id = row['from_bus_id'], row['to_bus_id'] 
        from_lat = buses[buses["bus_id"] == from_bus_id]["Lat"].iloc[0]
        from_lon = buses[buses["bus_id"] == from_bus_id]["Lon"].iloc[0]
        to_lat = buses[buses["bus_id"] == to_bus_id]["Lat"].iloc[0]
        to_lon = buses[buses["bus_id"] == to_bus_id]["Lon"].iloc[0]
        length_miles = harversine(from_lat, from_lon, to_lat, to_lon, unit='miles') # Length in miles
        params['Length'][ix] = length_miles
        X_actual = 0.486 * length_miles * 1.609344 # 0.486 is the base reactance in ohms per to km
        V_base = 138 # kV, base voltage
        B_mw_per_rad = (V_base ** 2) / X_actual # Susceptance in MW per radian
        params['Susceptance'][ix] = B_mw_per_rad if length_miles != 0 else 0 

    print("Data loaded successfully.")

    return buses, branches, load.round(4), pv.round(4), wind.round(4), params
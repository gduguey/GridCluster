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

class ED:
    def __init__(self, rep_period, data, params):

        start = time.time()
        self.rep_period = rep_period
        self.buses, self.branches, self.gen = data
        self.buses = self.buses['Bus ID'].unique()
        self.params = params
        self.model = gp.Model()

        self.T = len(self.rep_period['Load'])
        self.loads = self.rep_period['Load'].reset_index(drop=True).astype(float)
        self.pv_cfs = self.rep_period['PV'].reset_index(drop=True).astype(float)
        self.wind_cfs = self.rep_period['Wind'].reset_index(drop=True).astype(float)
        self.ccgt_buses = np.sort(self.gen.loc[self.gen['Category'].isin(['Oil CT', 'Coal', 'Gas CC', 'Gas CT', 'Oil ST'])]['Bus ID'].unique())
        self.pv_buses = self.gen.loc[self.gen['Unit Group'] == 'PV']['Bus ID'].unique()
        self.wind_buses = self.gen.loc[self.gen['Unit Group'] == 'WIND']['Bus ID'].unique()
        self.vre_buses = self.pv_buses.tolist() + self.wind_buses.tolist()
        self.ref_bus = self.buses[0]
        self.non_ref_buses = self.buses[1:]
        self.lines = {}
        self.lines['in'], self.lines['out'] = {bus: [] for bus in self.buses}, {bus: [] for bus in self.buses}
        for ix, row in self.branches.iterrows():
            self.lines['in'][row['To Bus']].append(row['UID'])
            self.lines['out'][row['From Bus']].append(row['UID'])
        self.cons = {'flow balance': {}, 'CCGT max cap': {}, 'CCGT ramp up': {}, 'CCGT ramp down': {},
                     'storage balance': {}, 'storage energy cap': {}, 'storage charge cap': {}, 'storage discharge cap': {},
                     'pos shed': {}, 'DC flow': {}, 'flow UB': {}, 'flow LB': {}, 'ref bus': {}, 'angle UB': {}, 'angle LB': {}}

        self.y_ccgt = self.model.addVars(self.ccgt_buses, self.T) # dispatchable generation fuel
        self.y_flow = self.model.addVars(self.branches['UID'].unique(), self.T, lb=-GRB.INFINITY) # flows
        self.y_charge = self.model.addVars(self.vre_buses, self.T, lb=-GRB.INFINITY) # battery charging
        self.y_shed = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # load shedding (potentially negative)
        self.y_soc = self.model.addVars(self.vre_buses, self.T) # battery state-of-charge
        self.y_pos_shed = self.model.addVars(self.buses, self.T) # max(0, load shedding)
        self.y_theta = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # phase angles

        self.cons['flow balance'] = {}
        injection = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        for bus in self.buses:
            in_lines = self.lines['in'][bus]
            out_lines = self.lines['out'][bus]
            for t in range(self.T):
                injection[(bus,t)] = gp.LinExpr((1,self.y_flow[line, t]) for line in in_lines) + self.y_shed[bus,t]
                injection[(bus,t)] -= gp.LinExpr((1,self.y_flow[line, t]) for line in out_lines)
            if bus in self.ccgt_buses:
                for t in range(self.T):
                    injection[(bus,t)] += self.y_ccgt[bus,t]
            if bus in self.vre_buses:
                for t in range(self.T):
                    injection[(bus,t)] -= self.y_charge[bus,t]

            for t in range(self.T):
                self.cons['flow balance'][(bus,t)] = self.model.addConstr(injection[(bus,t)] == 0)
                if bus in self.ccgt_buses:
                    self.cons['CCGT max cap'][(bus,t)] = self.model.addConstr(self.y_ccgt[bus,t] <= 0)
                    self.cons['CCGT ramp up'][(bus,t)] = self.model.addConstr(self.y_ccgt[bus,t] - self.y_ccgt[bus,(t-1)%self.T] <= 0)
                    self.cons['CCGT ramp down'][(bus,t)] = self.model.addConstr(-self.y_ccgt[bus,t] + self.y_ccgt[bus,(t-1)%self.T] <= 0)
                if bus in self.vre_buses:
                    self.cons['storage balance'][(bus,t)] = self.model.addConstr(self.y_soc[bus,t] - self.y_soc[bus,(t-1)%self.T] - self.y_charge[bus,(t-1)%self.T] == 0)
                    self.cons['storage energy cap'][(bus,t)] = self.model.addConstr(self.y_soc[bus,t] <= 0)
                    self.cons['storage charge cap'][(bus,t)] = self.model.addConstr(self.y_charge[bus,t] <= 0)
                    self.cons['storage discharge cap'][(bus,t)] = self.model.addConstr(-self.y_charge[bus,t] <= 0)
                self.cons['pos shed'][(bus,t)] = self.model.addConstr(-self.y_pos_shed[bus,t] + self.y_shed[bus,t] <= 0)
                if bus in self.non_ref_buses:
                    self.cons['angle UB'][(bus,t)] = self.model.addConstr(self.y_theta[bus,t] <= 180)
                    self.cons['angle LB'][(bus,t)] = self.model.addConstr(-self.y_theta[bus,t] <= 180)

        for ix, row in self.branches.iterrows():
            ij = row['UID']
            i, j = row['From Bus'], row['To Bus']
            for t in range(self.T):
                self.cons['DC flow'][(ij,t)] = self.model.addConstr(self.y_flow[ij,t] - self.params['Susceptance'][ij] * (self.y_theta[i,t] - self.y_theta[j,t]) == 0)
                self.cons['flow UB'][(ij,t)] = self.model.addConstr(self.y_flow[ij,t] <= 0) # flow capacity constraint
                self.cons['flow LB'][(ij,t)] = self.model.addConstr(-self.y_flow[ij,t] <= 0) # flow capacity constraint
        for t in range(self.T):
            self.cons['ref bus'][t] = self.model.addConstr(self.y_theta[self.ref_bus,t] == 0)

        objective = self.params['d_ccgt'] * self.y_ccgt.sum() + self.params['d_shed'] * self.y_pos_shed.sum()
        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.update()
        self.init_time = time.time() - start

    def solve(self, inv):
        start = time.time()
        net_load = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        for bus in self.buses:
            bus_loads = self.loads[bus].values
            for t in range(self.T):
                net_load[(bus,t)] += bus_loads[t]
                if bus in self.pv_buses:
                    net_load[(bus,t)] -= self.pv_cfs[bus].values[t] * inv['PV'][bus]
                elif bus in self.wind_buses:
                    net_load[(bus,t)] -= self.wind_cfs[bus].values[t] * inv['Wind'][bus]

            for t in range(self.T):
                self.model.setAttr('RHS', self.cons['flow balance'][(bus,t)], net_load[(bus,t)])
                if bus in self.ccgt_buses:
                    self.model.setAttr('RHS', self.cons['CCGT max cap'][(bus,t)], self.params['CCGT Max Cap'] * inv['CCGT'][bus])
                    self.model.setAttr('RHS', self.cons['CCGT ramp up'][(bus,t)], self.params['CCGT Ramp Rate'] * inv['CCGT'][bus])
                    self.model.setAttr('RHS', self.cons['CCGT ramp down'][(bus,t)], self.params['CCGT Ramp Rate'] * inv['CCGT'][bus])
                if bus in self.vre_buses:
                    self.model.setAttr('RHS', self.cons['storage energy cap'][(bus,t)], inv['Storage Energy'][bus])
                    self.model.setAttr('RHS', self.cons['storage charge cap'][(bus,t)], inv['Storage Power'][bus])
                    self.model.setAttr('RHS', self.cons['storage discharge cap'][(bus,t)], inv['Storage Power'][bus])

        for ix, row in self.branches.iterrows():
            ij = row['UID']
            i, j = row['From Bus'], row['To Bus']
            for t in range(self.T):
                self.model.setAttr('RHS', self.cons['flow UB'][(ij,t)], inv['Tran'][ij])
                self.model.setAttr('RHS', self.cons['flow LB'][(ij,t)], inv['Tran'][ij])

        update_time = time.time() - start
        start = time.time()
        self.model.optimize()
        runtime = time.time() - start

        results = {'init time': self.init_time, 'buildtime': update_time, 'runtime': runtime, 'objective': self.model.objVal}

        return results

class GTEP:
    def __init__(self, env, data, params):
        self.env = env
        self.model = gp.Model(env=self.env)
        self.buses, self.branches = data
        self.buses = self.buses['bus_id']
        self.params = params
        self.objective = 0

        self.ref_bus = self.buses[0]
        self.non_ref_buses = self.buses[1:]
        self.lines = {}
        self.lines['in'], self.lines['out'] = {bus: [] for bus in self.buses}, {bus: [] for bus in self.buses}
        for ix, row in self.branches.iterrows():
            self.lines['in'][row['to_bus_id']].append(ix)
            self.lines['out'][row['from_bus_id']].append(ix)

    # def get_capex(self):
    #     capex = 0
    #     capex += self.params['c_ccgt'] * self.params['CCGT Max Cap'] * sum([x.X for x in self.CCGT.values()])
    #     capex += self.params['c_pv'] * sum([x.X for x in self.PV.values()])
    #     capex += self.params['c_wind'] * sum([x.X for x in self.Wind.values()])
    #     capex += self.params['c_stor_energy'] * sum([x.X for x in self.StorageEnergy.values()])
    #     capex += self.params['c_stor_power'] * sum([x.X for x in self.StoragePower.values()])
    #     for ix, row in self.branches.iterrows():
    #         capex += self.params['c_tran'] * row['Length'] * self.Tran[row['UID']].X
    #     return capex

    def solve(self, rep_period, weights, callback=None):
        start = time.time()

        # PROBLEM VARIABLES
        self.PV = self.model.addVars(self.buses)
        self.model.addConstr(gp.quicksum(self.PV) <= self.params['PV Resource Availability'])
        self.objective += self.params['c_pv'] * gp.quicksum(self.PV)

        self.Wind = self.model.addVars(self.buses)
        self.model.addConstr(gp.quicksum(self.Wind) <= self.params['Wind Resource Availability'])
        self.objective += self.params['c_wind'] * gp.quicksum(self.Wind)

        self.CCGT = self.model.addVars(self.buses, vtype=GRB.INTEGER)
        self.objective += self.params['c_ccgt'] * self.params['CCGT Max Cap'] * gp.quicksum(self.CCGT)

        self.StorageEnergy = self.model.addVars(self.buses)
        self.StoragePower = self.model.addVars(self.buses)
        self.model.addConstrs(self.StorageEnergy[bus] <= self.params['Rate Duration'] * self.StoragePower[bus] for bus in self.buses)
        self.objective += self.params['c_stor_energy'] * gp.quicksum(self.StorageEnergy)
        self.objective += self.params['c_stor_power'] * gp.quicksum(self.StoragePower)

        self.Tran = self.model.addVars(self.branches.shape[0])
        for ix, row in self.branches.iterrows():
            self.model.addConstr(self.Tran[ix] <= self.params['Transmission Max Cont Rating']) ##############################################
            self.objective += self.params['c_tran'] * self.params['Length'][ix] * self.Tran[ix] ################################################

        self.T = len(rep_period['Load'])

        self.y_ccgt = self.model.addVars(self.buses, self.T) # dispatchable generation fuel
        self.y_shed = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # load shedding (potentially negative)
        self.y_charge = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY) # battery charging
        self.y_soc = self.model.addVars(self.buses, self.T) # battery state-of-charge
        self.y_pos_shed = self.model.addVars(self.buses, self.T) # max(0, load shedding)
        self.y_flow = self.model.addVars(self.branches.shape[0], self.T, lb=-GRB.INFINITY) # flows
        # self.y_theta = self.model.addVars(self.buses, self.T, lb=-GRB.INFINITY)  # phase angles

        loads = rep_period['Load'].reset_index(drop=True).astype(float)
        pv_cfs = rep_period['PV'].reset_index(drop=True).astype(float)
        wind_cfs = rep_period['Wind'].reset_index(drop=True).astype(float)

        injection = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        outflow = {(bus,t): 0 for bus in self.buses for t in range(self.T)}
        # net_load = {(bus,t): 0 for bus in self.buses for t in range(self.T)}

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
        ################################################### is * self.CCGT[bus] necessary here ?  and why make it cyclic with (t-1)%self.T (when t=0 (t-1)%self.T = self.T) ?
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
        self.model.addConstrs(self.y_pos_shed[bus,t] >= self.y_shed[bus,t] for bus in self.buses for t in range(self.T)) ############################################################

        # transmission capacity constraints
        for ix, row in self.branches.iterrows():
            line = ix
            i, j = row['from_bus_id'], row['to_bus_id']
            # self.model.addConstrs(self.y_flow[line,t] == (self.params['Susceptance'][line]) * (self.y_theta[i,t] - self.y_theta[j,t]) for t in range(self.T))
            self.model.addConstrs(self.y_flow[line,t] <= self.Tran[line] for t in range(self.T)) # flow capacity constraint ####################################################################
            self.model.addConstrs(self.y_flow[line,t] >= -self.Tran[line] for t in range(self.T)) # flow capacity constraint

        # phase angle constraints
        # self.model.addConstrs(self.y_theta[bus,t] <= 180 for bus in self.non_ref_buses for t in range(self.T))
        # self.model.addConstrs(self.y_theta[bus,t] >= -180 for bus in self.non_ref_buses for t in range(self.T))
        # self.model.addConstrs(self.y_theta[self.ref_bus,t] == 0 for t in range(self.T))

        for i in range(len(weights)):
            self.objective += weights[i] * (self.params['d_ccgt'] * gp.quicksum(self.y_ccgt[bus,t] for bus in self.buses for t in range(24*i,24*(i+1))) + self.params['d_shed'] * gp.quicksum(self.y_pos_shed[bus,t] for bus in self.buses for t in range(24*i,24*(i+1))))

        self.model.setObjective(self.objective, GRB.MINIMIZE)
        buildtime = time.time() - start

        start = time.time()
        self.model.optimize(callback)
        runtime = time.time() - start

        if self.model.SolCount > 0:
            results = {'buildtime': buildtime, 'runtime': runtime, 'objective': self.model.objVal,
                       'inv': {'PV': get_solutions(self.PV), 'Wind': get_solutions(self.Wind), 'CCGT': get_solutions(self.CCGT), 
                               'Storage Energy': get_solutions(self.StorageEnergy), 'Storage Power': get_solutions(self.StoragePower), 
                               'Tran': get_solutions(self.Tran)}}
            return results
        else:
            return None

# def round_dict_values(data, decimals):
#     return {key: round(value, decimals) if isinstance(value, (int, float)) else value for key, value in data.items()}

# def aggregate(df, average=True):
#     df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']]) + pd.to_timedelta((df['Period'] - 1) * 5, unit='m')
#     df.set_index('Datetime', inplace=True)
#     if average:
#         df = df.resample('h').mean().reset_index()
#     else:
#         df = df.resample('h').sum().reset_index()
#     df = df[df.columns[5:]]
#     start_date = '2024-01-01 00:00:00'
#     end_date = '2024-12-31 23:00:00'
#     datetime_series = pd.date_range(start=start_date, end=end_date, freq='h')
#     df.index = datetime_series
#     df['Week'] = df.index.isocalendar().week
#     df['Day'] = df.index.dayofyear
#     df['Hour'] = df.index.hour
#     df = df.set_index(['Week', 'Day', 'Hour'])
#     return df

# def normalize_distance_matrix(mat):
#     return (mat - mat.min()) / (mat.max() - mat.min())

def harversine(lat1, lon1, lat2, lon2):
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
    r = 3958.8  # Radius of earth in miles
    return c * r 

def get_solutions(var):
    if isinstance(var, dict):
        return {k: get_solutions(v) for k, v in var.items()}
    else:
        return np.round(var.X, 3)

def read_data(results):
    # pv = pd.read_csv('RTS-GMLC/RTS_Data/timeseries_data_files/PV/REAL_TIME_pv.csv')
    # wind = pd.read_csv('RTS-GMLC/RTS_Data/timeseries_data_files/WIND/REAL_TIME_wind.csv')
    # gen = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/gen.csv')
    # load = pd.read_csv('RTS-GMLC/RTS_Data/timeseries_data_files/Load/REAL_TIME_regional_Load.csv')
    # buses = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/bus.csv')
    # branches = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/branch.csv')

    # pv = pd.read_csv('./agg_results/time_series_solar.csv')
    # wind = pd.read_csv('./agg_results/time_series_wind.csv')
    # load = pd.read_csv('./agg_results/time_series_demand.csv')
    # buses = pd.read_csv('./agg_results/nodes.csv')
    # branches = pd.read_csv('./agg_results/branches.csv')

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
        # params['Susceptance'][ix] = row['b']
        from_lat = buses[buses["bus_id"] == from_bus_id]["Lat"].iloc[0]
        from_lon = buses[buses["bus_id"] == from_bus_id]["Lon"].iloc[0]
        to_lat = buses[buses["bus_id"] == to_bus_id]["Lat"].iloc[0]
        to_lon = buses[buses["bus_id"] == to_bus_id]["Lon"].iloc[0]
        dist = harversine(from_lat, from_lon, to_lat, to_lon)
        params['Length'][ix] = dist

    return buses, branches, load.round(4), pv.round(4), wind.round(4), params
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:56:36 2023
Read problem data for the wind-solar siting optimization problem
@author: Rahman Khorramfar
"""
import numpy as np
import pandas as pd
import xarray as xr
import math

class Storage:
    def __init__(self):  # no constructor
        self.Type = str()
        self.capex = float()
        self.eff_round = float()
        self.eff_single = float()
        self.FOM = float()
        self.lifetime = int()
        self.est_coef = float()
        self.num_storages = []
        self.duration = int()
        self.decay_eff = float()
    def populate_storage_data(self, Setting):
        self.Storages = list()
        df = pd.read_csv(f'../Storage_params.csv')
        storage_types = Setting.storage_types
        for s in range(len(storage_types)):
            strg = Storage()
            strg.Type = storage_types[s]
            idx = np.min(np.where(df['Storage technology'] == strg.Type))
            strg.capex = (df['energy capex($/kWh)'][idx]+df['power capex($/kW)']
                          [idx]/df['duration (hr)'][idx])*1000  # per MWh
            strg.eff_round = df['round-trip efficiency'][idx]
            strg.eff_single = df['sing-eff'][idx]
            strg.FOM = strg.capex*0.025
            strg.lifetime = int(df['lifetime'][idx])  # yr
            strg.est_coef = Setting.WACC * \
                ((1+Setting.WACC)**strg.lifetime) / \
                ((1+Setting.WACC)**strg.lifetime-1)
            strg.duration = int(df['duration (hr)'][idx])  # hr
            strg.decay_eff = (1-1/df['duration (hr)'][idx])
            self.Storages.append(strg)
        self.num_storages = len(self.Storages)

class Plant:
    def __init__(self):  # no constructor
        self.Type = str()  # name of the plant type, e.g., solar, wind-offshore
        self.capex = float()
        self.FOM = float()
        self.VOM = float()
        self.heat_rate = float()
        self.lifetime = int()
        self.est_coef = float()
        self.density= float()

    def populate_plant_data(self,Setting):
        self.Plants = list()
        df = pd.read_csv(f'../Plant_params.csv')
        plant_types = Setting.plant_types
        for p in range(len(plant_types)):
            plt = Plant()
            plt.Type = plant_types[p]
            idx = np.min(np.where(df['plant_type'] == plt.Type))
            # plt.nameplate_cap = df['Nameplate capacity (MW)'][idx];
            plt.capex = df['CAPEX($/kw)'][idx]*1000  # to MW
            plt.FOM = df['FOM ($/kW-yr)'][idx]*1000  # to MW
            plt.VOM = df['VOM ($/MWh)'][idx]
            plt.density=df['Density(MW/km2)'][idx] # MW/km2           
            plt.heat_rate = df['Heat Rate  (MMBtu/MWh)'][idx]
            plt.lifetime = int(df['Lifetime (year)'][idx])
            plt.est_coef = Setting.WACC * \
                ((1+Setting.WACC)**plt.lifetime) / \
                ((1+Setting.WACC)**plt.lifetime-1)
            self.Plants.append(plt)
        self.num_plants = len(self.Plants)

class REDat:
    def __init__(self):  # no constructor
        self.Type = str()
        self.CF = list()
        self.lat = list()
        self.lon = list()
        self.area = list()
        self.num_loc = int()
    def ReadREData(self, Setting):
        self.REPlants = list()
        plant_types = Setting.RE_plant_types
        for p in range(len(plant_types)):
            red = REDat()
            red.Type = plant_types[p]
            CF = []
            for iy in Setting.test_year:
                CF_orig = xr.open_dataset(Setting.REfile[red.Type]+str(iy)+'.nc')['cf']
                if Setting.RE_cell_size[red.Type] == 5:
                    area = CF_orig[0, :]*0
                    area[:] = 1000000000
                elif Setting.RE_cell_size[red.Type] == 3:
                    area = CF_orig[0, :].squeeze()*0
                    area[:] = 1000000000
                else:
                    if iy == Setting.test_year[0]:
                        lats = CF_orig.lat.data
                        # create an empty xarray with same size as CF_orig
                        area = CF_orig[0, :, :].squeeze()*0
                        rr = -111198.77
                        # standard area of a grid cell in km2
                        standard_area = (
                            (rr*Setting.RE_cell_size[red.Type])**2)/1000/1000
                        for ilat in range(len(lats)):
                            area[ilat, :] = standard_area * \
                                math.cos(-0.0174533*lats[ilat])
                        area = xr.where(
                            np.isnan(CF_orig[0, :, :].squeeze()), np.nan, area)
                        if Setting.landr == 1:
                            #apply land use mask if landr=1
                            landusedata = xr.open_dataset(Setting.landusefile[red.Type])['mask']
                            available_area_r = area*landusedata
                            available_area_r = xr.where(available_area_r >= 1, available_area_r, np.nan)
                if Setting.landr == 1:
                    data = xr.where(available_area_r >= 1, CF_orig, np.nan)
                    available_area = available_area_r
                else:
                    available_area = area
                    data = CF_orig
                # red.area = available_area.stack(z=("lat", "lon")).dropna('z', how='all')
                if Setting.RE_cell_size[red.Type] >= 3:
                    data=data.rename({'ncells':'z'})
                else:
                    data = data.stack(z=("lat", "lon")).dropna('z', how='all')
                # remove Feb 29
                CF_yr = data.sel(
                    time=~((data.time.dt.month == 2) & (data.time.dt.day == 29))).data
                CF.append(CF_yr)
            red.CF = np.concatenate(CF, axis=0)
            red.CF=np.where(red.CF<Setting.minCF,0,red.CF)
            if Setting.RE_cell_size[red.Type] >=3:
                red.area = available_area.rename({'ncells':'z'})
            else:
                red.area = available_area.stack(z=("lat", "lon")).dropna('z', how='all')
            red.lat = data.lat.data
            red.lon = data.lon.data
            red.num_loc = red.CF.shape[1]
            self.REPlants.append(red)
        self.num_re_plants = len(self.REPlants)

class Data(REDat, Plant, Storage):
    def __init__(self, Setting):  # no constructor
        super().__init__()
        self.ReadREData(Setting)
        self.populate_plant_data(Setting)
        self.populate_storage_data(Setting)
        df = pd.read_csv(Setting.demandfile)
        # REMOVE FEB 29
        dft = df.set_index(pd.to_datetime(df['Date'], format='%m/%d/%y'))
        dft = dft.loc[dft.index.year.isin(Setting.test_year)]
        dft = dft.loc[~((dft.index.month == 2) & (dft.index.day == 29))]
        self.demand = dft['ISONE_grs_ld'].to_numpy()
        self.num_plan_periods = len(self.demand)

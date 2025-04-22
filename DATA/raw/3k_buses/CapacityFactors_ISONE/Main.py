# -*- coding: utf-8 -*-
import sys
import pandas as pd
import os
sys.path.append('../Models_mincost/')
import NCDataCost_mincost as NCDataCost
import Modules_mincost as Modules
import print_results_mincost as pr

import gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
import time
import itertools

class Setting:
    demandfile = str()
    RE_cell_size = dict()  # degree
    RE_plant_types = list()  # set of RE plant types considered
    REfile = dict()
    landusefile = dict()
    solver_gap = 0.001  # x100 percent
    wall_clock_time_lim = 100000  # seconds
    weather_model = str()
    print_results_header = 1
    print_detailed_results = 1
    test_name = str()
    datadir = str()
    UB_dispatchable_cap = dict()
    lost_load_thres = float()
    gas_price = float()
    storage_types = list()
    plant_types = list()
    wake = int()
    gas_plant_types = list()
    val_lost_load = float()
    val_curtail = float()
    num_y = int()
    test_year = list()
    ens_id = int()
    year_list=list()
    minCF = 0.005 # remove small numbers
    
def runmodel(Setting):
    if Setting.wake == 1:
        Setting.REfile['wind-onshore'] = '%s/WindData/%s/wakecf/cf_Wind_%.2fm_' % (Setting.datadir, Setting.weather_model, Setting.RE_cell_size['wind-onshore'])
    else:
        Setting.REfile['wind-onshore'] = '%s/WindData/%s/simplecf/cf_Wind_%.2fm_' % (Setting.datadir, Setting.weather_model, Setting.RE_cell_size['wind-onshore'])

    if Setting.weather_model == "WTK":
        Setting.REfile['solar-UPV'] = '%s/SolarData/NSRDB/output/PVwatts/netcdf/cf_Solar_%.2fm_' % (Setting.datadir, Setting.RE_cell_size['solar-UPV'])
    else:
        Setting.REfile['solar-UPV'] = '%s/SolarData/%s/output/PVwatts/netcdf/cf_Solar_%.2fm_' % (Setting.datadir, Setting.weather_model, Setting.weather_model, Setting.RE_cell_size['solar-UPV'])

    Setting.landusefile['solar-UPV'] = '/nobackup1/lyqiu/ISONE/landuse_data/NREL_Sitingmap/wind_coe_composite_50_ISNE_%.2fmean.nc' % (Setting.RE_cell_size['solar-UPV'])
    Setting.landusefile['wind-onshore'] = '/nobackup1/lyqiu/ISONE/landuse_data/NREL_Sitingmap/wind_coe_composite_50_ISNE_%.2fmean.nc' % (Setting.RE_cell_size['wind-onshore'])
    stime = time.time()
    dat = NCDataCost.Data(Setting)
    Model = gp.Model()
    Modules.define_DVs(dat, Setting, Model)
    Modules.add_constraints(dat, Setting, Model)
    obj = Modules.add_obj_func(dat, Setting, Model)
    Model.modelSense = GRB.MINIMIZE
    Model.setObjective(obj)
    Model.setParam('OutputFlag', 1)
    Model.setParam('MIPGap', Setting.solver_gap)
    Model.setParam('Timelimit', Setting.wall_clock_time_lim)
    Model.setParam('Presolve', 2)  # -1 to 2
    Model.setParam('Method', 1)  # -1 to 2
    Model.optimize()
    Modules.get_DV_vals(Model, Setting)
    pr.print_results(dat, Setting, stime, Model)
    Model.reset()
    del (Model)

###############Input data and parameters######################################
Setting.demandfile = '../demand/ISONE_grossload_metdata_spliced_22yr_UTC0.csv'
Setting.RE_plant_types = ['solar-UPV', 'wind-onshore']
Setting.gas_plant_types = ['ng', 'CCGT']
Setting.plant_types = Setting.RE_plant_types + Setting.gas_plant_types
######################
paras=pd.read_csv('../Other_params.csv',index_col=0)
Setting.WACC = paras.loc['WACC'].Value
Setting.lost_load_thres = paras.loc['lost_load_thres'].Value
Setting.gas_price = paras.loc['gas_price'].Value
Setting.storage_types = ['Li-ion']
Setting.test_name = "onshoresolar"
Setting.datadir = '/nobackup1/lyqiu/ISONE/'
Setting.year_list = list(range(2007, 2014))
#######################################################################################
####input from command line
Setting.weather_model = str(sys.argv[1]) #WR
Setting.wake = int(sys.argv[2]) # if wake=1, then wake effect is considered
Setting.landr = int(sys.argv[3]) # if landr=1, then land restriction is considered
Setting.RE_cell_size['wind-onshore'] = float(sys.argv[4]) # OR wind
Setting.RE_cell_size['solar-UPV'] = float(sys.argv[5]) # OR solar
Setting.UB_dispatchable_cap['ng'] = float(sys.argv[6]) #maximum capacity of natural gas (0-1)
Setting.UB_dispatchable_cap['CCGT'] = float(sys.argv[7])  # maximum capacity of CCGT (0-1)
Setting.test_num_y = int(sys.argv[8])
id =int(sys.argv[9]) # percentile
  

# mdl = 'WTK'
# wake = 0
# landr = 0
# onwindsize = 0.06
# solarsize = 0.14
# cap_ng =0
# cap_cc = 0.05
# num_y = 7
# ensid=1

year_lists = list(itertools.combinations(Setting.year_list, Setting.test_num_y))
suffix='ng_%d_cc_%d_wake_%d_landr_%d_wind-onshore%.2f_solar-UPV%.2f_%s_Load.csv' % (Setting.UB_dispatchable_cap['ng']*100,
Setting.UB_dispatchable_cap['CCGT']*100, Setting.wake, Setting.landr, 
Setting.RE_cell_size['wind-onshore'], Setting.RE_cell_size['solar-UPV'], Setting.weather_model)
Setting.outputdir = '../Result_minCost/'

if id != 0:
    Setting.ens_id = id
    csvfilename = '%s/%s_sub%dyrs_ens%d_%s' % (Setting.outputdir, Setting.test_name, Setting.test_num_y, Setting.ens_id, suffix)
    print(csvfilename)
    if not os.path.exists(csvfilename):
        print('generating:'+csvfilename)
        Setting.test_year = year_lists[Setting.ens_id-1]
        runmodel(Setting)
else:
    for en in range(len(year_lists)):
        Setting.test_year = year_lists[en]
        Setting.ens_id = en+1
        csvfilename = '%s/%s_sub%dyrs_ens%d_%s' % (Setting.outputdir, Setting.test_name, Setting.test_num_y, Setting.ens_id, suffix)
        print(csvfilename)
        if not os.path.exists(csvfilename):
            print('generating:'+csvfilename)
            runmodel(Setting)

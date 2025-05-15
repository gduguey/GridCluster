from modules import *
import pickle

buses, branches, load, pv, wind, params = read_data()
eval_period = {'Load': load, 'PV': pv, 'Wind': wind}
for key in ['c_ccgt', 'c_pv', 'c_wind', 'c_stor_energy', 'c_stor_power', 'c_tran', 'd_ccgt', 'd_shed']:
    params[key] = params[key] * 1e-5

def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._data.append([time.time() - model._start, cur_obj, cur_bd])
            pd.DataFrame(model._data).to_csv('full_solve_log.csv')


with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 1)
    env.start()

    data = [buses, branches]

    gtep = GTEP(env, data, params)
    gtep.model._obj = None
    gtep.model._bd = None
    gtep.model._data = []
    gtep.model._start = time.time()
    cep_results = gtep.solve(eval_period, [365/6]*6, callback=data_cb)

    # Save results to a pickle file
    with open('cep_results_v2.pkl', 'wb') as f:
        pickle.dump(cep_results, f)

    # # Load results from the pickle file
    # with open('cep_results.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)
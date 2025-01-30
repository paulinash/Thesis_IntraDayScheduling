import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
import numpy as np
import os
import pandas as pd


def get_gt(timeframe):
    # but it should take day as a value    
    gt = pd.read_csv('data/ground_truth/residential4_prosumption.csv', index_col=0)
    gt.index = pd.to_datetime(gt.index)
    if timeframe is not None:
        gt = gt.loc[timeframe[0]:timeframe[1]]
    return gt

def adjust_time_horizon(x, start, end=24):
    # gets dictionary and slices it to corect time horizon via a list
    items = list(x.items())
    sliced_items = items[start:]
    return dict(sliced_items)

def get_gt_battery_evolution(model, pb_nom_gt):
    # takes ground truth battery pb and calculates the battery state with the evolution equation
    battery_evolution = np.empty(len(model.model.time_e))
    battery_evolution[0] = model.e0

    for t in range(len(model.model.time_e)-1):
        battery_evolution[t+1] = battery_evolution[t] - model.t_inc * pb_nom_gt[t] - model.t_inc * model.mu * np.abs(pb_nom_gt[t])
    return battery_evolution

def get_ground_truth_pg_pb(model, pl_gt):
    # takes a model and the whole 24 hour ground truth and returns the true pg and pb by using ground truth and model allocation
    low_x = list(model.model.x_low.get_values().values())
    high_x = list(model.model.x_high.get_values().values())
    pg_nom = list(model.model.pg_nom.get_values().values())
    pb_nom = list(model.model.pb_nom.get_values().values())
    pl = [model.model.pl[t] for t in model.model.time]

    # adjust pl_gt to correct length for intra day problems
    length = len(low_x)
    pl_gt = list(pl_gt.values.flatten())[-length:]

    delta_pl = [x-y for x,y in zip(pl_gt, pl)]

    pb_nom_gt = np.empty(len(pl))
    pg_nom_gt = np.empty(len(pl))

    for i in range(len(pl)):
        if delta_pl[i] > low_x[i] and delta_pl[i] < high_x[i]:
            # if delta_pl is in interval [x,x] then battery takes the uncertainty, thus pg_nom stays the same
            pb_nom_gt[i] = pb_nom[i] + delta_pl[i]
            pg_nom_gt[i] = pg_nom[i] + 0
        elif delta_pl[i] <= low_x[i]:
            # if pdelta_pl < low_x then the pb_nom takes x_low uncertainty and pg_nom takes the rest
            pb_nom_gt[i] = pb_nom[i] + low_x[i]
            pg_nom_gt[i] = pg_nom[i] + delta_pl[i] - low_x[i]
        else:
            pb_nom_gt[i] = pb_nom[i] + high_x[i]
            pg_nom_gt[i] = pg_nom[i] + delta_pl[i] - high_x[i]

    return pg_nom_gt, pb_nom_gt
    
def compute_quantiles(model, quantiles=[0.05, 0.95]):
    pg_nom = list(model.model.pg_nom.get_values().values())

    quant_low, quant_high  = quantiles
    # Compute the quantiles of the prosumption uncertainty
    prosumption_low = []
    prosumption_high = []
    for t in model.model.time:
        # Function with form cdf(x) - quantile = 0
        func_temp_low = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_low
        func_temp_high = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_high
        prosumption_low_temp = fsolve(func_temp_low, x0=-0.5)[0]
        prosumption_high_temp = fsolve(func_temp_high, x0=0.5)[0]
        prosumption_low.append(prosumption_low_temp)
        prosumption_high.append(prosumption_high_temp)
    # Compute the quantile values of the truncated distribution (i.e. the grid power uncertainty)
    pg_truncated_quantile_low = []
    pg_truncated_quantile_high = []
    for i in range(len(prosumption_low)):
        pg_trunc_low_temp = prosumption_low[i] - model.model.x_low[model.model.time.at(i+1)].value
        pg_trunc_high_temp = prosumption_high[i] - model.model.x_high[model.model.time.at(i+1)].value
        pg_truncated_quantile_low.append(pg_trunc_low_temp)
        pg_truncated_quantile_high.append(pg_trunc_high_temp)
    # Shift the uncertaintites to represent quantiles of the deviations from the nominal grid power
    pg_quantile_low = [pg_nomi + deviation if deviation < 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_low)]
    pg_quantile_high = [pg_nomi + deviation if deviation > 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_high)]
    return pg_quantile_low, pg_quantile_high



    



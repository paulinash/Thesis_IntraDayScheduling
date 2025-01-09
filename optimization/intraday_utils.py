import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
import numpy as np
import os
import pandas as pd

# TODO we have this timeframe here and in main. change that when that code works!
timeframe = ['2017-04-01 06:00:00', '2017-04-02 05:00:00']  


def get_gt(model):
    # TODO is model here irrelevant?
    # but it should take day as a value
    gt = pd.read_csv('data/ground_truth/residential4_prosumption.csv', index_col=0)
    gt.index = pd.to_datetime(gt.index)
    if timeframe is not None:
        gt = gt.loc[timeframe[0]:timeframe[1]]
    return gt

def adjust_time_horizon(x, start, end):
    # gets dictionary and slices it to corect time horizon via a list
    # TODO could be wrong
    items = list(x.items())
    sliced_items = items[start-1:]
    return dict(sliced_items)

def get_gt_battery_evolution(model, pb_nom_gt):
    # takes ground truth battery pb and calculates the battery state with the evolution equation
    battery_evolution = np.empty(len(model.model.time_e))
    battery_evolution[0] = model.e0

    for t in range(len(model.model.time_e)-1):
        battery_evolution[t+1] = battery_evolution[t] - model.t_inc * pb_nom_gt[t] - model.t_inc * model.mu * np.abs(pb_nom_gt[t])
    return battery_evolution

def get_ground_truth_pg_pb(model):
    # takes a model and returns the true pg and pb by using ground truth and model allocation
    low_x = list(model.model.x_low.get_values().values())
    high_x = list(model.model.x_high.get_values().values())
    pg_nom = list(model.model.pg_nom.get_values().values())
    pb_nom = list(model.model.pb_nom.get_values().values())
    pl = [model.model.pl[t] for t in model.model.time]
    pl_gt = get_gt(model)

    # TODO get correct timeframe for pl_gt (otherise always 24 long)
    length = len(low_x)


    delta_pl = [x-y for x,y in zip(pl_gt.values.flatten().tolist(), pl)]

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
    




    



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
from results_processing import custom_x_axis_formatter, get_file_path
from intraday_utils import get_ground_truth_pg_pb, get_gt_battery_evolution, compute_quantiles, get_gt
from utils import get_24_hour_timeframe
import numpy as np
import pandas as pd
import os

colors = ['#43AA8B', '#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff']

def postprocess_results_intra_rolling_horizon(models, timeframe, time_slots):
    ''' Postprocess the results of the intraday optimizations. '''

    plot_battery_evolution_intra_rolling_horizon(models, timeframe, time_slots)
    plot_probabilistic_power_schedule_intra_rolling_horizon(models, timeframe, time_slots)

def plot_battery_evolution_intra_rolling_horizon(models, timeframe, time_slots):
    ''' Plots the optimal battery evolution over time. '''

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    whole_e_gt = np.empty(25)
    all_timestamps_set = set()
    for model in models:
        e_nominal = list(model.model.e_nom.get_values().values())
        e_prob_max = list(model.model.e_max.get_values().values())
        e_prob_min = list(model.model.e_min.get_values().values())

        # get correct timeframe (24 hours but with different starting point)
        if color_counter == 0: 
            # day ahead model
            point_in_time = 0
        else:
            point_in_time = time_slots[color_counter-1]
            hour = point_in_time + 6
            if hour >= 24:
                hour = hour-24
        new_start_time = get_24_hour_timeframe(timeframe[0], time_delta = point_in_time)[1] # this is '2017-07-13 08:00:00' for time_delta = 2
        intra_day_timeframe = get_24_hour_timeframe(new_start_time)

        # get ground truth battery evolution
        pl_gt = get_gt(intra_day_timeframe)
        gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
        e_gt = get_gt_battery_evolution(model, gt_pb)

        # TODO not sure if this is correct
        # TODO the complete time horizon needs to be considered for plt.plot
        #print('################################')
        #print(point_in_time)
        #print(e_gt)
        #print(len(e_gt))
        whole_e_gt = np.concatenate((whole_e_gt[:point_in_time], e_gt))
        #print(whole_e_gt)
        #print(len(whole_e_gt))

        e_max = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_max)]
        e_min = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_min)]

        time_e = [str(t) for t in model.model.time_e]
        ordered_time_e = model.model.time_e.ordered_data()
        all_timestamps_set.update(ordered_time_e)

        if color_counter == 0:
            ax.plot(time_e, e_nominal, color=colors[color_counter], linewidth=2, label='Nominal Battery State')
            ax.plot(time_e, e_gt, color = colors[color_counter], linewidth=2, linestyle='dashed', label='Ground truth battery state')
            first_time_e = time_e
            first_ordered_time_e = ordered_time_e
        else:
            ax.plot(time_e, e_gt, color = colors[color_counter], linewidth=2, linestyle='dashed')
            ax.plot(time_e, e_nominal, color=colors[color_counter], linewidth=2)
        
        ax.fill_between(time_e, e_min, e_max, color=colors[color_counter], alpha=0.2)
        ax.plot(time_e, e_min, linewidth=1, color=colors[color_counter])
        ax.plot(time_e, e_max, linewidth=1, color=colors[color_counter])
        color_counter = color_counter + 1 

    ax.axhline(y=model.e_limit_max, color='k', linestyle='--', linewidth='2', label='Battery Limits')
    ax.axhline(y=model.e_limit_min, color='k', linestyle='--', linewidth='2')
    #ax.plot(first_time_e, whole_e_gt, color='black', linewidth='1', label='Whole Ground Truth')

    # get correct legend labeling
    timestamps = tuple(sorted(all_timestamps_set))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, timestamps)))
    plt.xticks(np.arange(0, len(timestamps), 2), rotation=45)
    plt.ylim([-0.5, model.e_limit_max + 0.6])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.94), bbox_transform=ax.transAxes)
    plt.ylabel('Battery Storage [kWh]')
    file_path = get_file_path('battery_evolution_intra.png')
    plt.savefig(file_path, dpi=200)
    #plt.show()

def plot_probabilistic_power_schedule_intra_rolling_horizon(models, timeframe, time_slots, quantiles=[0.05, 0.95]):
    ''' Plot the probabilistic power schedule. '''

    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    whole_pg_gt = np.empty(24)
    all_timestamps_set = set()

    for model in models:
        time = [str(t) for t in model.model.time]
        ordered_time = model.model.time.ordered_data()
        all_timestamps_set.update(ordered_time)
        pg_nom = list(model.model.pg_nom.get_values().values())
       
        # Get quantiles
        quant_low, quant_high  = quantiles
        pg_quantile_low, pg_quantile_high = compute_quantiles(model, quantiles)

        # Unconditional Expected Deviations
        #pg_exp_low = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value for t in model.model.time]
        #pg_exp_high = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value for t in model.model.time]

        # Conditional ExpectedDeviations
        pg_exp_low_cond = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value / model.model.prob_low[t].value for t in model.model.time]
        pg_exp_high_cond = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value / model.model.prob_high[t].value for t in model.model.time]


        # get correct timeframe
        if color_counter == 0: 
            # day ahead model
            point_in_time = 0
        else:
            point_in_time = time_slots[color_counter-1]
            hour = point_in_time + 6
            if hour >= 24:
                hour = hour-24
        new_start_time = get_24_hour_timeframe(timeframe[0], time_delta = point_in_time)[1] # this is be '2017-07-13 08:00:00' for time_delta = 2
        intra_day_timeframe = get_24_hour_timeframe(new_start_time)

        # Ground truth
        pl_gt = get_gt(intra_day_timeframe)
        gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
        gt_pg = list(gt_pg)

        
        # Whole ground truth
        # TODO still needs to be looked at, if it is correct and included in plot
        whole_pg_gt = np.concatenate((whole_pg_gt[:point_in_time], gt_pg))   

        # Ensure that the conditional expected deviations are within the quantiles for visualization purposes.
        # With specifying the quantiles, we limit the range that interests us. Thus, the conditional expected deviations
        # should be within the quantiles. For highly asymmetrical distributions, this could not hold. Thus, for asymmetrical
        # PDFs, either the quantiles should be large (5%-95% or 1%-99%) or the following 5 lines should be commented out.
        # Otherwise, a plot that is easily misinterpreted could be generated.
        for i in range(len(pg_exp_low_cond)):
            if pg_exp_low_cond[i] < pg_quantile_low[i]:
                pg_exp_low_cond[i] = pg_quantile_low[i]
            if pg_exp_high_cond[i] > pg_quantile_high[i]:
                pg_exp_high_cond[i] = pg_quantile_high[i]
        
        if color_counter == len(time_slots): #  if we are in the last round
            # Add a last time point for the step plot.            
            time_pandas = [pd.Timestamp(t) for t in time]
            last_hour = time_pandas[-1] + pd.Timedelta(hours=1)  
            time_pandas.append(last_hour)
            time = [t.strftime('%Y-%m-%d %H:%M:%S') for t in time_pandas]
            pg_nom.append(pg_nom[-1])  # Repeat the last y-value (nominal grid power) for the step plot.
            pg_exp_low_cond.append(pg_exp_low_cond[-1])  
            pg_exp_high_cond.append(pg_exp_high_cond[-1])
            pg_quantile_low.append(pg_quantile_low[-1])
            pg_quantile_high.append(pg_quantile_high[-1])
            gt_pg.append(gt_pg[-1])

        if color_counter == 0:
            ax.step(time, pg_nom, label='DiS', color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, np.ravel(pg_quantile_low), '--', label=f'{int(100*quant_low)} - {int(100*quant_high)}% Quantile', color=colors[color_counter], linewidth=1.5, where='post')
            ax.step(time, gt_pg, label='Ground truth', color=colors[color_counter], linestyle='dotted', linewidth=2, where='post')
            first_ordered_time = ordered_time
            first_time = time
        else:
            ax.step(time, pg_nom, color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, np.ravel(pg_quantile_low), '--', color=colors[color_counter], linewidth=1.5, where='post')
            ax.step(time, gt_pg, color=colors[color_counter], linestyle='dotted', linewidth=2, where='post')
            
        ax.step(time, np.ravel(pg_quantile_high), '--', color=colors[color_counter], linewidth=1.5, where='post')
        color_counter = color_counter + 1

    whole_pg_gt = list(whole_pg_gt)
    whole_pg_gt.append(whole_pg_gt[-1])

    # get correct legend labeling
    timestamps = tuple(sorted(all_timestamps_set))
    #ax.step(timestamps, whole_pg_gt, color='black', linewidth='1', label='Whole Ground Truth', where='post')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, timestamps)))
    plt.xticks(np.arange(0, len(timestamps), 2), rotation=45)

    plt.legend(loc='lower right')
    plt.ylabel('Grid Power [kW]')
    file_path = get_file_path('dispatch_schedule_intra.png')
    plt.savefig(file_path, dpi=200)
    plt.show()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
from results_processing import custom_x_axis_formatter, get_file_path
from intraday_utils import get_ground_truth_pg_pb, get_gt_battery_evolution
import numpy as np
import os

colors = ['#ddcc77', '#88ccee', '#44aa99', '#117733', '#332288', '#cc6677', '#cc6677']

def postprocess_results_intra(models):
    ''' Postprocess the results of the optimization. '''

    plot_battery_evolution_intra(models)

    plot_probabilistic_power_schedule_intra(models)

def plot_battery_evolution_intra(models):
    ''' Plots the optimal battery evolution over time. '''

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    whole_gt = []
    # TODO plot the whole ground truth in black
    for model in models:
        e_nominal = list(model.model.e_nom.get_values().values())
        e_prob_max = list(model.model.e_max.get_values().values())
        e_prob_min = list(model.model.e_min.get_values().values())
        # get ground truth battery evolution
        gt_pg, gt_pb = get_ground_truth_pg_pb(model)
        e_gt = get_gt_battery_evolution(model, gt_pb)

        e_max = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_max)]
        e_min = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_min)]

        time_e = [str(t) for t in model.model.time_e]
        ordered_time_e = model.model.time_e.ordered_data()

        if color_counter == 0:
            ax.plot(time_e, e_nominal, color=colors[color_counter], linewidth=2, label='Nominal Battery State')
            ax.plot(time_e, e_gt, color = colors[color_counter], linewidth=2, linestyle='dashed', label='Ground truth battery state')
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

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, first_ordered_time_e)))
    plt.xticks(np.arange(0, len(first_ordered_time_e), 2), rotation=45)
    plt.ylim([-0.5, model.e_limit_max + 0.6])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.94), bbox_transform=ax.transAxes)
    plt.ylabel('Battery Storage [kWh]')
    file_path = get_file_path('battery_evolution_intra.png')
    plt.savefig(file_path, dpi=200)
    #plt.show()

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

def plot_probabilistic_power_schedule_intra(models, quantiles=[0.05, 0.95]):
    ''' Plot the probabilistic power schedule. '''

    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    
    for model in models:
        time = [str(t.hour) for t in model.model.time]
        ordered_time = model.model.time.ordered_data()
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

        # Ground truth
        gt_pg, gt_pb = get_ground_truth_pg_pb(model)
        gt_pg = list(gt_pg)
        
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

        time.append(str(int(time[-1])+0.95))  # Add a last time point for the step plot.
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
        else:
            ax.step(time, pg_nom, color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, np.ravel(pg_quantile_low), '--', color=colors[color_counter], linewidth=1.5, where='post')
            ax.step(time, gt_pg, color=colors[color_counter], linestyle='dotted', linewidth=2, where='post')

        ax.step(time, np.ravel(pg_quantile_high), '--', color=colors[color_counter], linewidth=1.5, where='post')
        color_counter = color_counter + 1
    # TODO use later for prob plot
    #probs_low = list(model.model.prob_low.get_values().values())
    #probs_high = list(model.model.prob_high.get_values().values())

    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)

    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, first_ordered_time)))
    plt.xticks(np.arange(0, len(first_ordered_time), 2), rotation=45)

    plt.legend(loc='lower right')
    plt.ylabel('Grid Power [kW]')
    file_path = get_file_path('dispatch_schedule_intra.png')
    plt.savefig(file_path, dpi=200)
    plt.show()
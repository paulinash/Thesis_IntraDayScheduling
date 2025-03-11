import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
from results_processing import custom_x_axis_formatter, get_file_path
from intraday_utils import get_ground_truth_pg_pb, get_gt_battery_evolution, compute_quantiles, get_gt
import numpy as np
import os

colors = ['#43AA8B', '#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff']

def postprocess_results_intra(models, timeframe):
    ''' Postprocess the results of the intraday optimizations. '''

    #print_objective_values(models)
    plot_battery_evolution_intra(models, timeframe)
    #plot_probabilities_of_deviations_intra(models)
    #plot_costs_intra(models, timeframe)
    plot_probabilistic_power_schedule_intra(models, timeframe)

def print_objective_values(models):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    model_counter = 0
    for model in models:
        # Grid Schedule
        
        if model_counter != 0:
            time = [str(t) for t in model.model.time]
            pg_nom = list(model.model.pg_nom.get_values().values())
            DiS_Schedule = list(model.day_ahead_schedule.values())
            # TODO because Dis-Schedule is now shorter (20 in first run), this affects the other quantities
            grid_list = [(x-y)**2 for x,y in zip(pg_nom, DiS_Schedule)]
            prob_low = list(model.model.prob_low.get_values().values())
            prob_high = list(model.model.prob_high.get_values().values())
            exp_pg_low = list(model.model.exp_pg_low.get_values().values())
            exp_pg_high = list(model.model.exp_pg_high.get_values().values())
            prob_list = [-a*b + c*d for a,b,c,d in zip(prob_low, exp_pg_low, prob_high, exp_pg_high)]
            new_list = [a+b for a,b in zip(grid_list, prob_list)]

            pg_nom_plus = list(model.model.pg_nom_plus.get_values().values())
            pg_nom_minus = list(model.model.pg_nom_minus.get_values().values())
            pg_nom_list = [x**2+y**2 for x,y in zip(pg_nom_plus, pg_nom_minus)]

    
            plt.plot(time, new_list, color='red')# TODO thus we land at an error
            plt.plot(time, pg_nom_list, color='blue')
        model_counter += 1
    plt.show()

def plot_battery_evolution_intra(models, timeframe):
    ''' Plots the optimal battery evolution over time. '''

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    whole_e_gt = np.empty(25)
    for model in models:
        e_nominal = list(model.model.e_nom.get_values().values())
        e_prob_max = list(model.model.e_max.get_values().values())
        e_prob_min = list(model.model.e_min.get_values().values())
        
        # get ground truth battery evolution
        pl_gt = get_gt(timeframe)
        gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
        e_gt = get_gt_battery_evolution(model, gt_pb)
        start = len(e_gt)
        whole_e_gt[-start:] = e_gt

        e_max = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_max)]
        e_min = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_min)]

        time_e = [str(t) for t in model.model.time_e]
        ordered_time_e = model.model.time_e.ordered_data()

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
    ax.plot(first_time_e, whole_e_gt, color='black', linewidth='1', label='Whole Ground Truth')

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

def plot_costs_intra(models, timeframe):

    plt.rcParams.update({'font.size':15})
    fig,ax = plt.subplots(figsize=(10,6))

    color_counter = 0
    for model in models:
        
        # TODO es werden sowohl nominal als auch gt costs geplottet. aber ist essentiell einfach prob dis plot
        time = [str(t.hour) for t in model.model.time]
        ordered_time = model.model.time.ordered_data()

        # Nominal costs
        # TODO das hier ist einfach pg_nom aufgeteilt in + und - teil. pg_nom ist gleiches wie DiS 0
        p_plus_nom = list(model.model.pg_nom_plus.get_values().values())
        p_minus_nom = list(model.model.pg_nom_minus.get_values().values())

        # weighted Nominal costs with, p+ is purchase price (~40ct), p- is selling price (~8ct)
        purchase_price = 0.40
        selling_price = 0.08
        p_plus_nom_weighted = [purchase_price*p for p in p_plus_nom]
        p_minus_nom_weighted = [selling_price*p for p in p_minus_nom]

        # Ground Truth costs
        pl_gt = get_gt(timeframe)
        gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
        gt_pg = list(gt_pg)
        costs_gt = [purchase_price*x if x > 0 else selling_price*x for x in gt_pg]

        if color_counter == 0:
            ax.step(time, p_plus_nom_weighted, label='Nominal Purchase price', linestyle='dotted', color=colors[color_counter], linewidth=1, where='post')
            ax.step(time, p_minus_nom_weighted, label='Nominal Selling price', linestyle='dashed', color=colors[color_counter], linewidth=1, where='post')
            ax.step(time, costs_gt, label='Ground Truth Purchase price', color=colors[color_counter], linewidth=1, where='post')
            first_ordered_time = ordered_time
        else:
            ax.step(time, costs_gt, color=colors[color_counter], linewidth=1, where='post')
            ax.step(time, p_plus_nom_weighted, linestyle='dotted', color=colors[color_counter], linewidth=1, where='post')
            ax.step(time, p_minus_nom_weighted, linestyle='dashed', color=colors[color_counter], linewidth=1, where='post')
        color_counter = color_counter + 1

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, first_ordered_time)))
    plt.xticks(np.arange(0, len(first_ordered_time), 2), rotation=45)
    plt.legend(loc='lower right')
    plt.ylabel('Costs')
    file_path = get_file_path('cost_plot_intra.png')
    plt.savefig(file_path, dpi=200)



def plot_probabilities_of_deviations_intra(models):
    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    for model in models:
        time = [str(t.hour) for t in model.model.time]
        ordered_time = model.model.time.ordered_data()
        probs_low = list(model.model.prob_low.get_values().values())
        probs_high = list(model.model.prob_high.get_values().values())
        neg_probs_low = [-1*x for x in probs_low]

        if color_counter == 0:
            ax.step(time, probs_high, label='Probability of upward deviations', linestyle='dotted', color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, neg_probs_low, label='Neg Probability of downward deviations', linestyle='dashed', color=colors[color_counter], linewidth=2, where='post')
            first_ordered_time = ordered_time
        else:
            ax.step(time, probs_high, linestyle='dotted', color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, neg_probs_low, linestyle='dashed', color=colors[color_counter], linewidth=2, where='post')
        color_counter = color_counter + 1

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, first_ordered_time)))
    plt.xticks(np.arange(0, len(first_ordered_time), 2), rotation=45)
    plt.legend(loc='lower right')
    plt.ylabel('Probability of deviations')
    file_path = get_file_path('prob_of_deviations_intra.png')
    plt.savefig(file_path, dpi=200)


def plot_probabilistic_power_schedule_intra(models, timeframe, quantiles=[0.05, 0.95]):
    ''' Plot the probabilistic power schedule. '''

    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    color_counter = 0
    whole_pg_gt = np.empty(24)

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
        pl_gt = get_gt(timeframe)
        gt_pg, gt_pb = get_ground_truth_pg_pb(model, pl_gt)
        gt_pg = list(gt_pg)
        
        # Whole ground truth
        start = len(pg_nom)
        whole_pg_gt[-start:] = gt_pg
        
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
            first_time = time
        else:
            ax.step(time, pg_nom, color=colors[color_counter], linewidth=2, where='post')
            ax.step(time, np.ravel(pg_quantile_low), '--', color=colors[color_counter], linewidth=1.5, where='post')
            ax.step(time, gt_pg, color=colors[color_counter], linestyle='dotted', linewidth=2, where='post')
            
        ax.step(time, np.ravel(pg_quantile_high), '--', color=colors[color_counter], linewidth=1.5, where='post')
        color_counter = color_counter + 1
    
    whole_pg_gt = list(whole_pg_gt)
    whole_pg_gt.append(whole_pg_gt[-1])
    ax.step(first_time, whole_pg_gt, color='black', linewidth='1', label='Whole Ground Truth', where='post')

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
''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data, load_costs
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values, show_costs
from experiment_tracking import start_experiment, log_data, end_experiment, log_results
from intraday_solve import solve_intra_day_problems_rolling_horizon, solve_intra_day_problems_rolling_horizon_optimal_eps_policy
from intraday_results_processing_rolling_horizon import postprocess_results_intra_rolling_horizon, get_costs_intra
from pareto_front import calculate_multiple_pareto_fronts
from utils import get_24_hour_timeframe
import numpy as np

show_base_results = False
intra_day_approach = True #!plot_costs only works for hourly resolution!
multiple_pareto_fronts = False # !resulting cost values are only sensible for hourly resolution!

# choose resolution
#time_slots = [4,10]
#time_slots = [4,8,12,16,20] # 4 hourly resolution
#time_slots = [2,4,6,8,10,12,14,16,18,20,22] # 2 hourly resolution
time_slots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] # hourly resolution

number_scalarisations=10 # TODO back to 10
self_suff = True
timeframe1 = ['2017-05-02 06:00:00', '2017-05-04 05:00:00'] # GT much more negative than forecasts
timeframe2 = ['2017-07-02 06:00:00', '2017-07-04 05:00:00'] # GT closer to zero than forecasts
timeframe3 = ['2017-06-09 06:00:00', '2017-06-11 05:00:00'] # GT in mean close to forecast, works for every hour

def main():

    fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_6/' 
    params_path = 'data/parameters/params_case4.json'
    timeframe = timeframe1

    # create 24 hour timeframe from timeframe for day-ahead problem
    day_ahead_timeframe = get_24_hour_timeframe(timeframe[0])

    # Load data
    forecasts = load_forecasts(fc_folder, timeframe=day_ahead_timeframe)
    params = load_params(params_path)
    input_data = preprocess_data(forecasts, params)

    # Run the optimization model
    model = BaseOptimizationModel(input_data)
    model.solve()

    ######## Process and visualize the results
    if show_base_results:
        validate_expected_values(model)
        show_costs(model, day_ahead_timeframe)
        postprocess_results(model, day_ahead_timeframe)

    #### Intra Day Approach
    # To plot Intra Schedules
    if intra_day_approach:
        weights = [0.5,0.5] # grid, ss
        scalarisation_approach = 'weighted sum'
        models = solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation_approach, params_path, weight_1=weights[0], weight_2=weights[1])
        postprocess_results_intra_rolling_horizon(models, timeframe, time_slots, day_ahead_timeframe)

    #### Pareto fronts for MULTIPLE time_slots with rolling horizon
    # And resulting Schedules from optimal epsilons
    if multiple_pareto_fronts:
        # get costs of intra day models
        weights = [0.3,0.7] # grid and ss weights
        intra_day_models = solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation='weighted sum', params_path=params_path, weight_1=weights[0], weight_2=weights[1])
        # grid costs and ss costs array. one entry in a cost array are the summed costs for that model througout the prediction horizon
        grid_costs_list_intra, ss_costs_list_intra = get_costs_intra(intra_day_models) # ATTENTION costs only sensible for hourly resolution
        
        # get Pareto fronts
        scalarisation_approach = 'epsilon constraint'
        chosen_epsilons = calculate_multiple_pareto_fronts(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path, grid_costs_list_intra,ss_costs_list_intra)
        
        # get schedules for optimal epsilon policy
        models = solve_intra_day_problems_rolling_horizon_optimal_eps_policy(model, forecasts, params, time_slots, timeframe, params_path, chosen_epsilons)
        postprocess_results_intra_rolling_horizon(models, timeframe, time_slots, day_ahead_timeframe)
        
    
if __name__ == '__main__':
    main()

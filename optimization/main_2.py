''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data, load_costs
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values, show_costs
from experiment_tracking import start_experiment, log_data, end_experiment, log_results
from intraday_solve import solve_intra_day_problems, solve_intra_day_problems_rolling_horizon, solve_intra_day_problems_rolling_horizon_optimal_eps_policy
from intraday_results_processing import postprocess_results_intra
from intraday_results_processing_rolling_horizon import postprocess_results_intra_rolling_horizon, get_costs_intra
from pareto_front import calculate_pareto_front_by_scalarisation, calculate_pareto_front_by_scalarisation_rolling_horizon, calculate_multiple_pareto_fronts
from utils import get_24_hour_timeframe
import numpy as np

show_base_results = False
intra_day_approach = False # # ATTENTION: use weighted sum here
scalarisation_bool = False
scalarisation_approach_list = ['weighted sum', 'epsilon constraint'] # weighted sum not really in use anymore, except for basic intra day problem
scalarisation_approach = scalarisation_approach_list[1]
multiple_pareto_fronts = True # ATTENTION: use epsilon constraint here
dynamic_costs = False

#time_slots = [4,12,20]
#time_slots = [4,8,12,16,20] # corresponds to 10am, 2pm, 6pm, 10pm, 2am 
#time_slots = [2,4,6,8,10,12,14,16,18,20,22]
time_slots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
number_scalarisations=10 # for now at least choose 6 please
self_suff = True

def main_2():

    fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_6/' 
    params_path = 'data/parameters/params_case4.json'
    costs_folder = 'data/electricity_costs/'
    
    timeframe1 = ['2017-05-02 06:00:00', '2017-05-04 05:00:00'] # GT much more negative than forecasts
    #timeframe1 = ['2017-05-05 06:00:00', '2017-05-07 05:00:00'] # similar to 2.-4.5. but at time 12 it didnt solve
    timeframe2 = ['2017-07-02 06:00:00', '2017-07-04 05:00:00'] # GT closer to zero than forecasts
    timeframe3 = ['2017-06-09 06:00:00', '2017-06-11 05:00:00'] # GT in mean close to forecast, works for every hour
    timeframe = timeframe1
    
    day_ahead_timeframe = get_24_hour_timeframe(timeframe[0])

    # Load data
    forecasts = load_forecasts(fc_folder, timeframe=day_ahead_timeframe)
    params = load_params(params_path)
    input_data = preprocess_data(forecasts, params)
    
    if dynamic_costs:
        ### Loading costs
        costs = load_costs(costs_folder, timeframe)
        print(costs)
    

    # Run the optimization model
    model = BaseOptimizationModel(input_data)
    model.solve()

    ######## Process and visualize the results
    if show_base_results:
        validate_expected_values(model)
        show_costs(model)
        postprocess_results(model, day_ahead_timeframe)

    #### Intra Day Approach
    # To plot Intra Schedules
    # TODO: it is not clear which policy gets chosen here
    # TODO epsilon needs to be adjusted here
    # TODO include costs here
    if intra_day_approach:
        models = solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation_approach, params_path)
        postprocess_results_intra_rolling_horizon(models, timeframe, time_slots)

    ###### Scalarisation Approaches (only usable for one time step)
    # Not in use anymore
    if scalarisation_bool:
        calculate_pareto_front_by_scalarisation_rolling_horizon(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path)

    #### Pareto fronts for MULTIPLE time_slots with rolling horizon
    # TODO include costs here
    if multiple_pareto_fronts:
        # get costs of intra day models
        weights = [0.3,0.7] # grid and ss weights
        intra_day_models = solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation='weighted sum', params_path=params_path, weight_1=weights[0], weight_2=weights[1])
        # grid costs and ss costs array. one entry in a cost array are the summed costs for that model througout the prediction horizon
        grid_costs_list_intra, ss_costs_list_intra = get_costs_intra(intra_day_models) # ATTENTION costs only sensible for hourly resolution
        
        chosen_epsilons = calculate_multiple_pareto_fronts(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path, grid_costs_list_intra,ss_costs_list_intra)
        # get schedules for optimal epsilon policy
        
        #models = solve_intra_day_problems_rolling_horizon_optimal_eps_policy(model, forecasts, params, time_slots, timeframe, params_path, chosen_epsilons)
        #postprocess_results_intra_rolling_horizon(models, timeframe, time_slots)
        
    
if __name__ == '__main__':
    main_2()

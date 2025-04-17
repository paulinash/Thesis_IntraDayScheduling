''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data, load_costs
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values
from experiment_tracking import start_experiment, log_data, end_experiment, log_results
from intraday_solve import solve_intra_day_problems, solve_intra_day_problems_rolling_horizon
from intraday_results_processing import postprocess_results_intra
from intraday_results_processing_rolling_horizon import postprocess_results_intra_rolling_horizon
from pareto_front import calculate_pareto_front_by_scalarisation, calculate_pareto_front_by_scalarisation_rolling_horizon, calculate_multiple_pareto_fronts
from utils import get_24_hour_timeframe
import numpy as np

show_base_results = False
intra_day_approach = False
scalarisation_bool = False
scalarisation_approach_list = ['weighted sum', 'epsilon constraint'] # weighted sum not really in use anymore
scalarisation_approach = scalarisation_approach_list[1]
multiple_pareto_fronts = True
dynamic_costs = False


time_slots = [4,8,12,16,18] # corresponds to 10am, 2pm, 6pm, 10pm, 2am 
number_scalarisations=15 # for now at least choose 6 please
self_suff = True

def main_2():

    fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_6/' 
    params_path = 'data/parameters/params_case4.json'
    costs_folder = 'data/electricity_costs/'
    timeframe = ['2017-06-09 06:00:00', '2017-06-11 05:00:00'] # always choose 48 hour timeframe please

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
        calculate_multiple_pareto_fronts(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path)

    
if __name__ == '__main__':
    main_2()

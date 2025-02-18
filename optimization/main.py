''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values
from experiment_tracking import start_experiment, log_data, end_experiment, log_results
from intraday_solve import solve_intra_day_problems
from intraday_results_processing import postprocess_results_intra
from pareto_front import calculate_pareto_front_by_scalarisation
import numpy as np

show_base_results = False
intra_day_approach = False
scalarisation_bool = True
scalarisation_approach_list = ['weighted sum', 'epsilon constraint']
scalarisation_approach = scalarisation_approach_list[0] # alternatively 'epsilon constraint'


# TODO for scalarisation only one model is possible right now, for intra day you can choose multiple models in time_slots
time_slots = [10]
number_scalarisations=11
self_suff = True

def main():
    # Example 1: Normal Distribution
    # fc_folder = 'data/parametric_forecasts/normal_dist_forecast_2024-10-31/'
    # params_path = 'data/parameters/params_normal_dist.json'
    # timeframe = ['2017-05-18 06:00:00', '2017-05-19 05:00:00']

    # Example 2: Sum of two logistic functions (Recreate paper results)
    fc_folder = 'data/parametric_forecasts/s2l_dist_forecast_2024-10-02/' 
    params_path = 'data/parameters/params_case2.json'
    #timeframe = ['2017-06-09 06:00:00', '2017-06-10 05:00:00']  # Forecasted nighttime PDFs can be too tight for plotting or solving the optimization model (See limitations section in the paper).
                                                                # Simple solution: Either choose a different date, adjust the timeframe to exclude nighttime (e.g. 06:00-23:00), or use gaussian distribution.
                                                                # Better solution: Redo the forecasts with additional constraints, or approximate nighttime PDFs with a different distribution.
    timeframe = ['2017-04-01 06:00:00', '2017-04-02 05:00:00']
    # Load data
    forecasts = load_forecasts(fc_folder, timeframe=timeframe)
    params = load_params(params_path)
    input_data = preprocess_data(forecasts, params)

    # Run the optimization model
    model = BaseOptimizationModel(input_data)
    model.solve()

    ######## Process and visualize the results
    if show_base_results:
        validate_expected_values(model)
        postprocess_results(model)


    #### Intra Day Approach
    if intra_day_approach:
        models = solve_intra_day_problems(model, forecasts, params, time_slots, timeframe)
        postprocess_results_intra(models, timeframe)

    ###### Scalarisation Approaches
    if scalarisation_bool:
        calculate_pareto_front_by_scalarisation(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach)

    
if __name__ == '__main__':
    main()

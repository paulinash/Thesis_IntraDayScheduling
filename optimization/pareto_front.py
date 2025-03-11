import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from input_data import load_forecasts, load_params, preprocess_data
from intraday_optimization_model import IntraDayOptimizationModel
from epsilon_const_optimization_model import EpsilonConstraintOptimizationModel
from intraday_solve import solve_intra_day_problems, solve_intra_day_problems_rolling_horizon
from results_processing import get_file_path
import numpy as np
from utils import get_24_hour_timeframe
from intraday_utils import adjust_time_horizon, get_ground_truth_pg_pb, get_gt_battery_evolution, get_gt

# TODO this whole class only works for 1 model
colors = ['yellow', 'gold', 'goldenrod', 'darkgoldenrod', 'peru', 'chocolate', 'saddlebrown', 'olive', 'darkolivegreen', 'dimgray', 'black']

def get_objective_values_1m(model, self_suff=True):
    # obtains a model and returns the sum of values related to grid uncertainty in obj function and related to self sufficiency
    pg_nom = np.array(list(model.model.pg_nom.get_values().values()))
    DiS_Schedule = np.array(list(model.day_ahead_schedule.values()))
    # Dis schedule is now shorter than pg_nom, so only calculate the difference for values in dis Schedule
    min_length = min(len(pg_nom), len(DiS_Schedule))
    pg_nom_truncated = pg_nom[:min_length]
    DiS_Schedule_truncated = DiS_Schedule[:min_length]
    schedule_list = (pg_nom_truncated - DiS_Schedule_truncated)**2

    prob_low = np.array(list(model.model.prob_low.get_values().values()))
    prob_high = np.array(list(model.model.prob_high.get_values().values()))
    exp_pg_low = np.array(list(model.model.exp_pg_low.get_values().values()))
    exp_pg_high = np.array(list(model.model.exp_pg_high.get_values().values()))
    prob_list = -prob_low*exp_pg_low + prob_high*exp_pg_high

    # List that contains all values of objective function that consider grid uncertainty
    sum_grid = sum(schedule_list) + sum(prob_list)

    pg_nom_plus = np.array(list(model.model.pg_nom_plus.get_values().values()))
    pg_nom_minus = np.array(list(model.model.pg_nom_minus.get_values().values()))

    # List that containts all values of objective function that consider self sufficiency
    if self_suff:
        price_list = pg_nom_plus**2 + pg_nom_minus**2
    else: # Promoting cost efficiency
        price_list = pg_nom_plus**2 - pg_nom_minus**2
    sum_price = sum(price_list)
    return sum_grid, sum_price


def plot_pareto_front(x,y, self_suff, color='#00876C'):
    # TODO this needs to be changed so that in multiple pareto fronts the dots all land into one plot (delete plt.subplots()) and just call it once before calling plot_pareto_front for the first time
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6)) 
    plt.scatter(x,y, c=color)
    plt.xlabel('Uncertainty in Grid')
    if self_suff:
        plt.ylabel('Self sufficiency costs')
    else:
        plt.ylabel('Cost efficiency costs')
    plt.title('Pareto front of grid vs. price')
    file_path = get_file_path('pareto_front.png')
    plt.savefig(file_path, dpi=200)
    #plt.show()

    
def calculate_pareto_front_by_scalarisation(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation):
    weights_1 = np.linspace(0,1,number_scalarisations)
    weights_2 = [1-w for w in weights_1]
    # TODO in 'epsilon constraint' approach the epsilons list need to be found manually
    epsilons = np.linspace(21,33,number_scalarisations)
     

    grid_values = []
    price_values = []
    for i in range(number_scalarisations):
        models = solve_intra_day_problems(model, forecasts, params, time_slots, timeframe, scalarisation, weight_1=weights_1[i], weight_2=weights_2[i], epsilon=epsilons[i], self_suff=self_suff)
        
        grid_value, price_value = get_objective_values_1m(models[1], self_suff)
        grid_values.append(grid_value)
        price_values.append(price_value)
        print(grid_values)
        print(price_values)
    plot_pareto_front(grid_values, price_values, self_suff)

def calculate_pareto_front_by_scalarisation_rolling_horizon(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation, params_path):
    weights_1 = np.linspace(0,1,number_scalarisations)
    weights_2 = [1-w for w in weights_1]
    # TODO in 'epsilon constraint' approach the epsilons list need to be found manually
    epsilons = np.linspace(10,33,number_scalarisations)
     
    grid_values = []
    price_values = []
    for i in range(number_scalarisations):
        models = solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation, params_path, weight_1=weights_1[i], weight_2=weights_2[i], epsilon=epsilons[i], self_suff=self_suff)
        
        grid_value, price_value = get_objective_values_1m(models[1], self_suff)
        grid_values.append(grid_value)
        price_values.append(price_value)
        print(grid_values)
        print(price_values)
    plot_pareto_front(grid_values, price_values, self_suff)

    
def calculate_multiple_pareto_fronts(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path):
    models = [model]
    model_t = model

    # These are arrays filled with arrays of grid values, price values
    grid_values_array = []
    price_values_array = []
    old_time = 0
    counter = 0
    for point_in_time in time_slots:
        new_time = point_in_time
        start_time = new_time - old_time
        
        # convert to list to slice and then convert back to dictionary
        day_ahead_schedule = adjust_time_horizon(model.model.pg_nom.get_values(), point_in_time) # here model.model because we always take schedule from model 1
        # this gets us values from k until end of e_nom from last problem
        e_nom = adjust_time_horizon(model_t.model.e_nom.get_values(), start_time) 
        e_prob_max = adjust_time_horizon(model_t.model.e_max.get_values(), start_time)
        e_prob_min = adjust_time_horizon(model_t.model.e_min.get_values(), start_time)
        #### get input data, consider data for hour k till k+24
        hour = point_in_time + 6
        if hour >= 24:
            hour = hour-24
        fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-03-06_hour_' + str(hour) + '/' 

        # get timeframe for new optimization problem
        new_start_time = get_24_hour_timeframe(timeframe[0], time_delta = time_slots[counter])[1] # this should be '2017-07-13 08:00:00' for time_delta = 2
        intra_day_timeframe = get_24_hour_timeframe(new_start_time)
        forecasts = load_forecasts(fc_folder, timeframe=intra_day_timeframe)
        params = load_params(params_path)

        ### Start calculation ###
        # get initial battery state e_nominal_gt by running ground truth prosumption through allocation
        # get timeframe for start calculation
        if counter == 0: #  first round, then take day-ahead-timeframe
            start_timeframe = timeframe
            start_timeframe = get_24_hour_timeframe(start_timeframe[0])
        else:
            new_start_time = get_24_hour_timeframe(timeframe[0], time_delta = time_slots[counter-1])[1] # this should be '2017-07-13 08:00:00' for time_delta = 2
            start_timeframe = get_24_hour_timeframe(new_start_time)
        pl_gt = get_gt(start_timeframe)
        pg_nom_gt, pb_nom_gt = get_ground_truth_pg_pb(model_t, pl_gt)
        e_nominal_gt = list(get_gt_battery_evolution(model_t, pb_nom_gt))

        params['e0'] = e_nominal_gt[start_time] 
        input_data = preprocess_data(forecasts, params)

        #### Now iterate through different weights to obtain a pareto front ###
        weights_1 = np.linspace(0,1,number_scalarisations)
        weights_2 = [1-w for w in weights_1]
        epsilons = np.linspace(10,33,number_scalarisations)
        ## TODO in 'epsilon constraint' approach the epsilons list need to be found manually

        grid_values = []
        price_values = []
        weighted_models = []
        for i in range(number_scalarisations):
            if scalarisation_approach == 'weighted sum':
                weighted_model = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, weights_1[i], weights_2[i], self_suff)
            elif scalarisation_approach == 'epsilon constraint':
                weighted_model = EpsilonConstraintOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilons[i], self_suff)
            weighted_model.solve()
            grid_value, price_value = get_objective_values_1m(weighted_model, self_suff)
            grid_values.append(grid_value)
            price_values.append(price_value)
            weighted_models.append(weighted_model)
        plot_pareto_front(grid_values, price_values, self_suff, color=colors[counter])

            
        grid_values_array.append(grid_values)
        price_values_array.append(price_values)

        # TODO somehow choose a specific model of the ones in weighted_models (list of models for specific timestep with different weights)
        # right now, just take 10th
        chosen_policy_model = weighted_models[10]
        models.append(chosen_policy_model)

        # model_t is the model in the timestep before so we get the variables from this model for the next round
        model_t = chosen_policy_model
        old_time = new_time
        counter = counter+1
    plt.show()


        
    
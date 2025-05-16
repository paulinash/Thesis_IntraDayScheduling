from input_data import load_forecasts, load_params, preprocess_data
from intraday_optimization_model import IntraDayOptimizationModel
from epsilon_const_optimization_model import EpsilonConstraintOptimizationModel
from intraday_utils import adjust_time_horizon, get_ground_truth_pg_pb, get_gt_battery_evolution, get_gt
from utils import get_24_hour_timeframe
import matplotlib.pyplot as plt
import numpy as np

def solve_intra_day_problems_rolling_horizon(model, forecasts, params, time_slots, timeframe, scalarisation, params_path, weight_1=0.5, weight_2=0.5, epsilon=10, self_suff=True):
    models = [model]
    model_t = model

    old_time = 0
    counter = 0
    for point_in_time in time_slots:
        new_time = point_in_time
        start_time = new_time - old_time
        
        # convert to list to slice and then convert back to dictionary
        day_ahead_schedule = adjust_time_horizon(model_t.model.pg_nom.get_values(), start_time) # here model.model because we always take schedule from model 1
        
        # this gets us values from k until end of e_nom from last problem
        e_nom = adjust_time_horizon(model_t.model.e_nom.get_values(), start_time) 
        e_prob_max = adjust_time_horizon(model_t.model.e_max.get_values(), start_time)
        e_prob_min = adjust_time_horizon(model_t.model.e_min.get_values(), start_time)

        #### get input data, consider data for hour k till k+24
        hour = point_in_time + 6
        if hour >= 24:
            hour = hour-24
        fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_' + str(hour) + '/' 

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
        if scalarisation == 'weighted sum':
            intra_day_model = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, weight_1, weight_2, self_suff)
        elif scalarisation == 'epsilon constraint':
            intra_day_model = EpsilonConstraintOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilon, self_suff)
            # TODO cost function and e_limit constraint was not adapted here yet
        result = intra_day_model.solve()

        models.append(intra_day_model)

        # model_t is the model in the timestep before so we get the variables from this model for the next round
        model_t = intra_day_model
        old_time = new_time
        counter = counter+1

    return models
    
def solve_intra_day_problems(model, forecasts, params, time_slots, timeframe, scalarisation, weight_1=0.5, weight_2=0.5, epsilon=10, self_suff=True):
    time_horizon = 24
    models = [model]
    model_t = model

    old_time = 0
    for point_in_time in time_slots:
        new_time = point_in_time
        start_time = new_time - old_time

        # convert to list to slice and then convert back to dictionary
        #x_low_input = adjust_time_horizon(model_t.model.x_low.get_values(), start_time, time_horizon)
        #x_high_input = adjust_time_horizon(model_t.model.x_high.get_values(), start_time, time_horizon)
        day_ahead_schedule = adjust_time_horizon(model.model.pg_nom.get_values(), point_in_time, time_horizon) # here model.model because we always take schedule from model 1

        e_nom = adjust_time_horizon(model_t.model.e_nom.get_values(), start_time, time_horizon+1)
        e_prob_max = adjust_time_horizon(model_t.model.e_max.get_values(), start_time, time_horizon+1)
        e_prob_min = adjust_time_horizon(model_t.model.e_min.get_values(), start_time, time_horizon+1)
    
        # get initial battery state e_nominal_gt by running ground truth prosumption through allocation
        pl_gt = get_gt(timeframe)
        pg_nom_gt, pb_nom_gt = get_ground_truth_pg_pb(model_t, pl_gt)
        e_nominal_gt = list(get_gt_battery_evolution(model_t, pb_nom_gt))
        params['e0'] = e_nominal_gt[start_time] 

        #### shorten input data to appropiate time horizon k:24
        input_data = preprocess_data(forecasts, params)
        input_data['fc_exp'] = input_data['fc_exp'][point_in_time:24]
        input_data['fc_weights'] = input_data['fc_weights'][point_in_time:24]

        if scalarisation == 'weighted sum':
            intra_day_model = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, weight_1, weight_2, self_suff)
        elif scalarisation == 'epsilon constraint':
            intra_day_model = EpsilonConstraintOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilon, self_suff)
        result = intra_day_model.solve()


        models.append(intra_day_model)

        # model_t is the model in the timestep before so we get the variables from this model for the next round
        model_t = intra_day_model
        old_time = new_time

    return models


def solve_intra_day_problems_rolling_horizon_optimal_eps_policy(model, forecasts, params, time_slots, timeframe, params_path, epsilons, self_suff=True):
    models = [model]
    model_t = model

    old_time = 0
    counter = 0
    for point_in_time in time_slots:
        
        new_time = point_in_time
        start_time = new_time - old_time
        
        # convert to list to slice and then convert back to dictionary
        day_ahead_schedule = adjust_time_horizon(model_t.model.pg_nom.get_values(), start_time) # here model.model because we always take schedule from model 1
        
        # this gets us values from k until end of e_nom from last problem
        e_nom = adjust_time_horizon(model_t.model.e_nom.get_values(), start_time) 
        e_prob_max = adjust_time_horizon(model_t.model.e_max.get_values(), start_time)
        e_prob_min = adjust_time_horizon(model_t.model.e_min.get_values(), start_time)

        #### get input data, consider data for hour k till k+24
        hour = point_in_time + 6
        if hour >= 24:
            hour = hour-24
        fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_' + str(hour) + '/' 

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

        intra_day_model = EpsilonConstraintOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilons[counter], self_suff)
        result = intra_day_model.solve()

        models.append(intra_day_model)

        # model_t is the model in the timestep before so we get the variables from this model for the next round
        model_t = intra_day_model
        old_time = new_time
        counter = counter+1

    return models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from input_data import load_forecasts, load_params, preprocess_data
from intraday_optimization_model import IntraDayOptimizationModel
from epsilon_const_optimization_model import EpsilonConstraintOptimizationModel
from results_processing import get_file_path
import numpy as np
from utils import get_24_hour_timeframe
from intraday_utils import adjust_time_horizon, get_ground_truth_pg_pb, get_gt_battery_evolution, get_gt

colors = ['yellow', 'gold', 'goldenrod', 'darkgoldenrod', 'peru', 'chocolate', 'saddlebrown', 'olive', 'darkolivegreen', 'dimgray', 'black']

def calculate_eucl_weighted_distance(x,y,w):
     x1,x2 = x
     y1,y2 = y
     w1,w2 = w
     return np.sqrt(w1*(x1-y1)**2 + w2*(x2-y2)**2)

        
    
def get_objective_values_1m(model, self_suff=True):
    # obtains a model and returns the sum of values related to grid uncertainty in obj function and related to self sufficiency
    pg_nom = np.array(list(model.model.pg_nom.get_values().values()))
    DiS_Schedule = np.array(list(model.day_ahead_schedule.values()))
    # Dis schedule is now shorter than pg_nom, so only calculate the difference for values in dis Schedule
    min_length = min(len(pg_nom), len(DiS_Schedule))
    pg_nom_truncated = pg_nom[:min_length]
    DiS_Schedule_truncated = DiS_Schedule[:min_length]
    schedule_list = model.c31*(pg_nom_truncated - DiS_Schedule_truncated)**2

    prob_low = np.array(list(model.model.prob_low.get_values().values()))
    prob_high = np.array(list(model.model.prob_high.get_values().values()))
    exp_pg_low = np.array(list(model.model.exp_pg_low.get_values().values()))
    exp_pg_high = np.array(list(model.model.exp_pg_high.get_values().values()))
    prob_list = -model.c31*prob_low*exp_pg_low + model.c32*prob_high*exp_pg_high

    # List that contains all values of objective function that consider grid uncertainty
    sum_grid = sum(schedule_list) + sum(prob_list)

    pg_nom_plus = np.array(list(model.model.pg_nom_plus.get_values().values()))
    pg_nom_minus = np.array(list(model.model.pg_nom_minus.get_values().values()))
    # List that containts all values of objective function that consider self sufficiency
    if self_suff:
        price_list = model.c11*pg_nom_plus**2 + model.c21*pg_nom_minus**2
    else: # Promoting cost efficiency
        price_list = model.c11*pg_nom_plus**2 - model.c21*pg_nom_minus**2
    sum_price = sum(price_list)
    return sum_grid, sum_price



def plot_3d_pareto_fronts(grid_values_array, price_values_array, chosen_policy_grid, chosen_policy_price, hours_list, grid_costs_list_intra, ss_costs_list_intra):
        hours_list = np.array(hours_list)
        time_steps = np.arange(len(hours_list))
        grid_values_array = np.array(grid_values_array)
        price_values_array = np.array(price_values_array)
        chosen_policy_grid = np.array(chosen_policy_grid)
        chosen_policy_price = np.array(chosen_policy_price)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use a colormap to assign different colors to each time step
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))
        # Loop through each time step and plot points
        for i, t in enumerate(time_steps):
            grid_vals = grid_values_array[i]
            price_vals = price_values_array[i]
            ax.plot(grid_vals, [t]*len(grid_vals), price_vals, color = colors[i], linewidth=1)
            ax.scatter(
                grid_vals,              # X = grid
                [t]*len(grid_vals),     # Y = time
                price_vals,             # Z = price
                color=colors[i],
                label=f'Time {hours_list[i]}' if t==0 or t==time_steps[-1] else None,
                s=20  # point size
            )
            # Plot the chosen policy point in red
            ax.scatter(
                chosen_policy_grid[i],     # X
                t,                         # Y
                chosen_policy_price[i],   # Z
                facecolors='none',
                edgecolors='red',
                s=60,                      # slightly bigger for emphasis
                marker='o',
                label=f'Chosen MOO Policy' if i == (len(time_steps)-1) else None  # Avoid duplicate legend labels
            )
            ax.scatter(
                grid_costs_list_intra[i+1],
                t,
                ss_costs_list_intra[i+1],
                facecolors='none',
                edgecolors='blue',
                s=60,
                marker='o',
                label=f'Basic Intra-Day Policy' if i == (len(time_steps)-1) else None
            )   
            
        # Connect red dots with a line
        ax.plot(
            chosen_policy_grid,
            time_steps,
            chosen_policy_price,
            color='red',
            linewidth=2.5,
            )
        ax.plot( 
            grid_costs_list_intra[1:],
            time_steps,
            ss_costs_list_intra[1:],
            color='blue',
            linewidth=2.5,
            )
        
        ax.set_yticks(time_steps[::4])
        ax.set_yticklabels(hours_list[::4])
        ax.set_xlabel('Grid Uncertainty and Deviations')
        ax.set_ylabel('Time')
        ax.set_zlabel('Self-sufficiency')
        ax.legend()
        ax.view_init(elev=20, azim=300)

        file_path = get_file_path('pareto_front_3d.png')
        plt.savefig(file_path, dpi=200)
        #plt.show()

    
def plot_2d_slices(chosen_policy_grid, chosen_policy_price, hours_list, grid_costs_list_intra, ss_costs_list_intra, price_values_low, price_values_high, grid_values_low, grid_values_high):
    hours_list = np.array(hours_list)
    time_steps = np.arange(len(hours_list))
    grid_costs_list_intra = np.array(grid_costs_list_intra)
    ss_costs_list_intra = np.array(ss_costs_list_intra)
    chosen_policy_grid = np.array(chosen_policy_grid)
    chosen_policy_price = np.array(chosen_policy_price)
    price_values_high = np.array(price_values_high)
    price_values_low = np.array(price_values_low)
    grid_values_low = np.array(grid_values_low)
    grid_values_high = np.array(grid_values_high)

    # get error terms
    error_grid_low = np.array(chosen_policy_grid-grid_values_low)
    error_grid_high = np.array(grid_values_high-chosen_policy_grid)
    error_price_low = np.array(chosen_policy_price-price_values_low)
    error_price_high = np.array(price_values_high-chosen_policy_price)


    # Time vs. Grid Plot
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(hours_list, chosen_policy_grid, yerr=[error_grid_low, error_grid_high], fmt='o', color='red', ecolor='red', capsize=5, label='Chosen MOO Policy')
    ax.scatter(hours_list, grid_costs_list_intra[1:], label='Basic Intra-Day Policy')
    ax.set_xticks(time_steps[::2])
    ax.set_xticklabels(hours_list[::2])
    for label in ax.get_xticklabels():
        label.set_rotation(45)    
    ax.set_ylabel('Grid Uncertainty and Deviations')
    ax.set_xlabel('Time')
    ax.legend()
    file_path = get_file_path('pareto_front_2d_slice_time_vs_grid.png')
    plt.savefig(file_path, dpi=200)


    # Time vs. SS Plot
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(hours_list, chosen_policy_price, yerr=[error_price_low, error_price_high], fmt='o', color='red', ecolor='red', capsize=5, label='Chosen MOO Policy')
    ax.scatter(hours_list, ss_costs_list_intra[1:], label='Basic Intra-Day Policy')
    ax.set_xticks(time_steps[::2])
    ax.set_xticklabels(hours_list[::2])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_ylabel('Self-sufficiency')
    ax.set_xlabel('Time')
    ax.legend()
    # Show plot
    file_path = get_file_path('pareto_front_2d_slice_time_vs_ss.png')
    plt.savefig(file_path, dpi=200)


    
def calculate_multiple_pareto_fronts(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations, scalarisation_approach, params_path, grid_costs_list_intra,ss_costs_list_intra):
    models = [model]
    model_t = model

    # These are arrays filled with arrays of grid values, price values
    grid_values_array = []
    price_values_array = []
    chosen_policy_grid = []
    chosen_policy_price = []
    old_time = 0
    counter = 0
    hours_list = []
    epsilon_boundaries_low = []
    epsilon_boundaries_high = []
    grid_boundaries_low = []
    grid_boundaries_high = []
    chosen_epsilons_list = []

    for point_in_time in time_slots:
        new_time = point_in_time
        start_time = new_time - old_time
        
        # convert to list to slice and then convert back to dictionary
        day_ahead_schedule = adjust_time_horizon(model_t.model.pg_nom.get_values(), start_time) 
        
        # this gets us values from k until end of e_nom from last problem
        e_nom = adjust_time_horizon(model_t.model.e_nom.get_values(), start_time) 
        e_prob_max = adjust_time_horizon(model_t.model.e_max.get_values(), start_time)
        e_prob_min = adjust_time_horizon(model_t.model.e_min.get_values(), start_time)

        #### get input data, consider data for hour k till k+24
        hour = point_in_time + 6
        if hour >= 24:
            hour = hour-24
        fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-04-03_hour_' + str(hour) + '/' 
        hour_in_format = f"{hour:02d}:00"
        hours_list.append(hour_in_format)

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
        epsilons = np.linspace(40,100,number_scalarisations)# just so there is some initialization, doesnt actually get chosen in this manual way

        # Find manual epsilon range dependent on weighted sum edge cases
        if scalarisation_approach == 'epsilon constraint':
            model_eps_bound_1 = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, 0,1, self_suff) # complete weight on ss
            model_eps_bound_2 = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, 1,0, self_suff) # complete weight on grid 
            model_eps_bound_1.solve()
            model_eps_bound_2.solve()
            grid_value_1, price_value_1 = get_objective_values_1m(model_eps_bound_1, self_suff) # here the price value is really low
            grid_value_2, price_value_2 = get_objective_values_1m(model_eps_bound_2, self_suff) #  here the grid value is really low
            ideal_point = [grid_value_2, price_value_1]
            epsilons = np.linspace(price_value_1, price_value_2, number_scalarisations)

            epsilon_boundaries_low.append(price_value_1) # low price value
            grid_boundaries_high.append(grid_value_1) # high grid value
            epsilon_boundaries_high.append(price_value_2) # high price value
            grid_boundaries_low.append(grid_value_2) # low grid value
            
        grid_values = []
        price_values = []
        weighted_models = []
        euclid_dist_to_ideal_point = []
        weights = [0.3,0.7] # weights for grid value, self sufficiency value

        for i in range(number_scalarisations):
            if scalarisation_approach == 'weighted sum':
                weighted_model = IntraDayOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, weights_1[i], weights_2[i], self_suff)
                
            elif scalarisation_approach == 'epsilon constraint':
                weighted_model = EpsilonConstraintOptimizationModel(input_data, day_ahead_schedule, e_nom, e_prob_min, e_prob_max, epsilons[i], self_suff)
            weighted_model.solve()
                
            # get cost values from weighted_model
            grid_value, price_value = get_objective_values_1m(weighted_model, self_suff)
            
            grid_values.append(grid_value)
            price_values.append(price_value)
            value = [grid_value, price_value]
            distance = calculate_eucl_weighted_distance(value, ideal_point, weights)
            euclid_dist_to_ideal_point.append(distance)
            weighted_models.append(weighted_model)

        grid_values_array.append(grid_values)
        price_values_array.append(price_values)

        # get chosen policy
        policy_index = np.argmin(euclid_dist_to_ideal_point)
        chosen_epsilons_list.append(epsilons[policy_index])
        chosen_policy_model =  weighted_models[policy_index]        
        grid_value_cp, price_value_cp = get_objective_values_1m(chosen_policy_model, self_suff)
        chosen_policy_grid.append(grid_value_cp)
        chosen_policy_price.append(price_value_cp)
        models.append(chosen_policy_model)

        # model_t is the model in the timestep before so we get the variables from this model for the next round
        model_t = chosen_policy_model
        old_time = new_time
        counter = counter+1
    
    plot_3d_pareto_fronts(grid_values_array, price_values_array, chosen_policy_grid, chosen_policy_price, hours_list, grid_costs_list_intra, ss_costs_list_intra)
    plot_2d_slices(chosen_policy_grid, chosen_policy_price, hours_list, grid_costs_list_intra, ss_costs_list_intra, epsilon_boundaries_low, epsilon_boundaries_high, grid_boundaries_low, grid_boundaries_high)
    #plt.show()
    return chosen_epsilons_list

 
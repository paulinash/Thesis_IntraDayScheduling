import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from intraday_solve import solve_intra_day_problems
import numpy as np

# TODO this whole class only works for 1 model

def get_objective_values_1m(model, self_suff=True):
    # obtains a model and returns the sum of values related to grid uncertainty in obj function and related to self sufficiency
    pg_nom = np.array(list(model.model.pg_nom.get_values().values()))
    DiS_Schedule = np.array(list(model.day_ahead_schedule.values()))
    schedule_list = (pg_nom - DiS_Schedule)**2

    prob_low = np.array(list(model.model.prob_low.get_values().values()))
    prob_high = np.array(list(model.model.prob_high.get_values().values()))
    exp_pg_low = np.array(list(model.model.exp_pg_low.get_values().values()))
    exp_pg_high = np.array(list(model.model.exp_pg_high.get_values().values()))
    prob_list = -prob_low*exp_pg_low + prob_high*exp_pg_high

    # List that contains all values of objective function that consider grid uncertainty
    grid_list = schedule_list + prob_list
    sum_grid = sum(grid_list)

    pg_nom_plus = np.array(list(model.model.pg_nom_plus.get_values().values()))
    pg_nom_minus = np.array(list(model.model.pg_nom_minus.get_values().values()))

    # List that containts all values of objective function that consider self sufficiency
    if self_suff:
        price_list = pg_nom_plus**2 + pg_nom_minus**2
    else: # Promoting cost efficiency
        price_list = pg_nom_plus**2 - pg_nom_minus**2
    sum_price = sum(price_list)
    return sum_grid, sum_price


def plot_pareto_front(x,y):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6)) 
    plt.scatter(x, y)
    plt.title('Pareto front of grid vs. price')
    plt.show()

    
def calculate_pareto_front_by_scalarisation(model, forecasts, params, time_slots, timeframe, self_suff, number_scalarisations):
    weights_1 = np.linspace(0,1,number_scalarisations)
    weights_2 = [1-w for w in weights_1]

    grid_values = []
    price_values = []
    for i in range(len(weights_1)):
        models = solve_intra_day_problems(model, forecasts, params, time_slots, timeframe, weights_1[i], weights_2[i], self_suff)
        
        grid_value, price_value = get_objective_values_1m(models[1], self_suff)
        grid_values.append(grid_value)
        price_values.append(price_value)
    plot_pareto_front(grid_values, price_values)
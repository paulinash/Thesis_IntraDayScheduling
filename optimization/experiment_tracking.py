''' This module is used to log data and results of the optimization in mlflow. '''
import mlflow
import os
from pyomo.environ import value
import pandas as pd


def start_experiment(experiment_name):
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def end_experiment():
    mlflow.end_run()


def log_data(data):
    ''' Log the input data in mlflow to ensure reproducibility.'''
    for key, value in data.items():
        mlflow.log_param(key, value)


def log_results(model):
    ''' Log the results of the optimization in mlflow. Execute "mlflow server" in terminal to see results. '''
    mlflow.log_metric('Objective Value', value(model.objective))
    
    # create a dataframe to store the results
    df = pd.DataFrame(
        {
            'x_low': model.x_low.get_values(), 
            'x_high': model.x_high.get_values(), 
            'prob_low': model.prob_low.get_values(), 
            'prob_high': model.prob_high.get_values(), 
            'e_nominal': model.e_nom.get_values(), 
            'pg': model.pg_nom.get_values(),
            'pb_nom': model.pb_nom.get_values(),
            'e_exp': model.e_exp.get_values(),
            'e_max': model.e_max.get_values(),
            'e_min': model.e_min.get_values()
            },  index=model.time_e)
    df.index.name = 'Time'

    temp_folder = os.path.join(os.getcwd(), 'temporary_folder')
    os.makedirs(temp_folder, exist_ok=True)

    results_path = os.path.join(temp_folder, 'results.csv')
    battery_evolution_path = os.path.join(temp_folder, 'battery_evolution.png')
    dispatch_schedule_path = os.path.join(temp_folder, 'dispatch_schedule.png')
    power_exchange_path = os.path.join(temp_folder, 'power_exchange.png')

    df.to_csv(results_path, index=True)
    
    mlflow.log_artifact(results_path)
    mlflow.log_artifact(battery_evolution_path)
    mlflow.log_artifact(dispatch_schedule_path)
    mlflow.log_artifact(power_exchange_path)

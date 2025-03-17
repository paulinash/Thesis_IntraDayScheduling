import pandas as pd
import json
import sys

def load_forecasts(folder_path, timeframe=None):
    ''' Load the expected value forecast and the pdf weights from the specified folder path. '''

    fc_exp_path = folder_path + 'expected_value_forecast.csv'
    fc_weights_path = folder_path + 'cdf_weights.csv'

    fc_exp = pd.read_csv(fc_exp_path, index_col=0)
    fc_exp.index = pd.to_datetime(fc_exp.index)

    if timeframe is not None:
        fc_exp = fc_exp.loc[timeframe[0]:timeframe[1]]

    if fc_exp.index.freq is None:
        try:
            fc_exp.index.freq = pd.infer_freq(fc_exp.index)
        except ValueError:
            print('ValueError: Specified timeframe is either not part of the forecasts or too short to infer frequency. Adjust the timeframe to match the specified forecasts.')
            sys.exit(1)

    fc_exp = fc_exp[fc_exp.columns[0]]
    
    fc_weights = pd.read_csv(fc_weights_path, index_col=0)
    fc_weights.index = pd.to_datetime(fc_weights.index)

    if timeframe is not None:
        fc_weights = fc_weights.loc[timeframe[0]:timeframe[1]]

    if fc_weights.index.freq is None:
        fc_weights.index.freq = pd.infer_freq(fc_weights.index)
        
    # combine the columns of fc_weights into a single column with a list of weights
    fc_weights = fc_weights.apply(lambda x: x.tolist(), axis=1)

    return {'fc_exp': fc_exp, 'fc_weights': fc_weights}

def load_costs(folder_path, timeframe=None):
    fc_costs_path = folder_path + 'electricity_prices_2017_germany.csv'
    fc_costs = pd.read_csv(fc_costs_path, sep=';',index_col=0, parse_dates=True, skiprows=1)
    fc_costs.index = pd.to_datetime(fc_costs.index, errors='coerce')
    if timeframe is not None:
        fc_costs = fc_costs.loc[timeframe[0]:timeframe[1]]

    if fc_costs.index.freq is None:
        try:
            fc_costs.index.freq = pd.infer_freq(fc_costs.index)
        except ValueError:
            print('ValueError: Specified timeframe is either not part of the forecasts or too short to infer frequency. Adjust the timeframe to match the specified forecasts.')
            sys.exit(1)

    fc_costs = fc_costs.iloc[:,0]
    
    return [fc_costs]

def load_params(path):
    ''' Load the parameters from the specified json file. '''
    with open(path, 'r') as file:
        params = json.load(file)
    
    return params
    

def preprocess_data(fc, params):
    input_data = fc.copy()
    input_data.update(params)

    return input_data

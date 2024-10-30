import pandas as pd
import json

def load_forecasts(folder_path, timeframe=None):
    ''' Load the expected value forecast and the pdf weights from the specified folder path. '''

    fc_exp_path = folder_path + 'expected_value_forecast.csv'
    fc_weights_path = folder_path + 'cdf_weights.csv'

    fc_exp = pd.read_csv(fc_exp_path, index_col=0)
    fc_exp.index = pd.to_datetime(fc_exp.index)

    if timeframe is not None:
        fc_exp = fc_exp.loc[timeframe[0]:timeframe[1]]

    if fc_exp.index.freq is None:
        fc_exp.index.freq = pd.infer_freq(fc_exp.index)

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


def load_params(path):
    ''' Load the parameters from the specified json file. '''
    with open(path, 'r') as file:
        params = json.load(file)
    
    return params
    

def preprocess_data(fc, params):
    input_data = fc.copy()
    input_data.update(params)

    return input_data

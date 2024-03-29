import numpy as np
from sklearn.metrics import mean_squared_error
from pwn_function import predict_with_network_one_step

def get_slope(input_data,target,weights):
    # Calculate the predictions: preds
    preds = (weights * input_data).sum()

    # Calculate the error: error
    error = preds - target
    
    # Calculate the slope: slope
    slope = 2 * input_data * error
    
    return slope

def get_mse(input_data, target, weights):
    model_array = []
    target_array=[]
    model_output = predict_with_network_one_step(input_data, weights)   #THIS IS A FLOAT NOT AN ARRAY!
    model_array.append(model_output)
    target_array.append(target)
    return mean_squared_error(target_array, model_array)
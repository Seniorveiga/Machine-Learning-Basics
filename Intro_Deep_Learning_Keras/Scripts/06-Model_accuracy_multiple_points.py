import numpy as np
from sklearn.metrics import mean_squared_error

from pwn_function import predict_with_network
"""
From now in advance we will call pwn_function.py to import predict_with_network so we do not have to rewrite it.
"""

#Variables: How does the weight of the nodes affect the result?
weights_0 = {'node_0': np.array([2, 1]), 
        'node_1': np.array([1, 2]),
        'output': np.array([1, 1])
        }

weights_1 = {'node_0': np.array([2, 1]),
 'node_1': np.array([1. , 1.5]),
 'output': np.array([1. , 1.5])}
#Input data
input_data = [np.array([0, 3]),np.array([1, 2]),np.array([-1, -2]),np.array([4, 0])]
target_actuals = [1,3,5,7]

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))


# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals,model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals,model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)
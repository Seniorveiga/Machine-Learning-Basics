import numpy as np
input_data = np.array([1,2,3])
target = 0
weights = np.array([0,2,1])

# Set the learning rate: learning_rate
learning_rate = 0.01
# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

"""
Slope = Pendiente || Slope helps us to calculate the new weights so they are updating constantly in order to approach 
the target value for our model.

Fíjate que aquí al ser un array de 3 elementos tanto los pesos como los input, significa que en los valores de entrada
hay 3 "bolitas", que pasan a una sola bolita final, que da primero como resultado 7, luego 5.04 etc...
"""

# Update the weights: weights_updated
weights_updated = weights - (learning_rate*slope)

# Get updated predictions: preds_updated
preds_updated = (weights_updated*input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)


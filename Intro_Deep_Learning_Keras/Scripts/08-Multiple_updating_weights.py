import numpy as np
import matplotlib.pyplot as plt
from getters_multiple_updating_weights import get_slope, get_mse

n_updates = 20
mse_hist = []

#Entering data
input_data = np.array([1, 2, 3])
target = 0
weights = np.array([-0.49929916,  1.00140168, -0.49789747])


# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

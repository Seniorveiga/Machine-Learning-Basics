# Import necessary modules
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

predictors = pd.read_csv("hourly_wages.csv", header=0)
n_cols = predictors.shape[1]
target = predictors.values[:,3]

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer = "adam",loss = "mean_squared_error")

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
target = np.array(target, dtype=np.float32)
target = np.array(target, dtype=np.float32)
print(predictors.shape)
print(target.shape)

model.fit(predictors, target)

"""
IMPORTANTE: When we use this model, all the data inside it shouldbe float32 type, not string, not booleans, just float32.
"""

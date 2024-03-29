# Import necessary modules
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
predictors = pd.read_csv("titanic_all_numeric.csv", header=0)
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer, second layer and output layer.
model.add(Dense(50, activation="relu", input_shape=(n_cols,))) #1st layer so we should introduce input_shape
model.add(Dense(32, activation="relu"))                        #2nd layer
model.add(Dense(1))                                            #Output layer: See that it only has 1 node.

"""
FORMATO ANTERIOR, YA NO SE USA.
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
"""
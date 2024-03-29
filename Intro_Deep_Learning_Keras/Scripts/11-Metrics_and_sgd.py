# Import necessary modules
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
#Traer df
df = pd.read_csv("titanic_all_numeric.csv", header=0)
predictors = df.drop(["survived"], axis = 1).values
predictors = np.array(predictors, dtype=np.float32)
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df["survived"])
print(n_cols)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation = "relu", input_shape = (n_cols,)))

# Add the output layer
model.add(Dense(2, activation = "softmax"))

# Compile the model
model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Fit the model
model.fit(predictors, target)
"""
IMPORTANT: La información que se le pasa a model.fit requiere que todos los datos no sólo sean numéricos, sino que sean
floats o ints exclusivamente.
"""
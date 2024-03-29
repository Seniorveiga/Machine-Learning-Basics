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
target = to_categorical(df["survived"])

n_cols = predictors.shape[1]

# Save the number of columns in predictors: n_cols
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Fit the model
hist = model.fit(predictors, target, validation_split = 0.3)

"""
We have created a validation split for each epoch, that is 30% of the sample.
"""
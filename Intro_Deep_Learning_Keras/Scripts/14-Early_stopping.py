# Import EarlyStopping
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

#Traer df
df = pd.read_csv("titanic_all_numeric.csv", header=0)
predictors = df.drop(["survived"], axis = 1).values
predictors = np.array(predictors, dtype=np.float32)
target = to_categorical(df["survived"])

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ["accuracy"])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, epochs = 30, validation_split = 0.3, callbacks = [early_stopping_monitor])

""" Cuando se emplea EalyStopping, es una función que hae que según el parámetro que pongamos en "patience" 
se pare antes o despues. Si patience = 2, significa que si tras 2 turnos no se mejora la precisión se 
detiene el modelo.
 """
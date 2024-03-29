import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

#Traer df
df = pd.read_csv("mnist.csv")
y = df.iloc[:,0:10]
X = df.drop(df.columns[0],axis=1).values
print(y.shape)
# Save the number of columns in predictors: n_cols

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation = "relu", input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation = "relu"))

# Add the output layer
model.add(Dense(10, activation = "softmax"))

# Compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Fit the model
model.fit(X,y,validation_split=0.3, epochs = 10)
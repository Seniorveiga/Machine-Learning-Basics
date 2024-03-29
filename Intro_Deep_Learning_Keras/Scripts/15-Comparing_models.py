import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2. Add the first and second layers.
model_1 = Sequential()
model_1.add(Dense(10, activation="relu", input_shape=input_shape))
model_1.add(Dense(10, activation='relu'))
# Add the output layer
model_1.add(Dense(2, activation='softmax'))

# Create the new model: model_2. Add the first and second layers.
model_2 = Sequential()
model_2.add(Dense(100, activation="relu", input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))
# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer="adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

"""
Notice the keyword argument verbose=False in model.fit(): 
This prints out fewer updates, since you'll be evaluating the models graphically instead of through text.
"""
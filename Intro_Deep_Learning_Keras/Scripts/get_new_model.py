from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

def get_new_model(input_shape):
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(input_shape,)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model

"""
Notas: Para pasar el argumento recuerda que sólo puedes pasar o enteros o floats, nada de en medio.
"""
# Import the SGD optimizer
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.optimizers import SGD
from get_new_model import get_new_model
from keras.utils import to_categorical


#Traer df
df = pd.read_csv("titanic_all_numeric.csv", header=0)
predictors = df.drop(["survived"], axis = 1).values
predictors = np.array(predictors, dtype=np.float32)
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df["survived"])

#-----------------------
# Create list of learning rates: lr_to_test
lr_to_test = [.000001,0.01,1.0]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model(input_shape = predictors.shape[1])
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(learning_rate = lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer, loss = "categorical_crossentropy")
    
    # Fit the model
    model.fit(predictors, target)
"""
We use centering and scaling for the differences when using differentes measures in a model.

For example, when you use two parameters in KNN model, distance and booleans have different weights so you 
should normalize it 
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
#For the second part
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

music_df = pd.read_csv("music_clean.csv")

# Training set and test sets.
X = music_df.drop("loudness",axis=1).values
y = music_df["loudness"].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
print(pipeline.score(X_test, y_test))
#This last part is to know the precision of the process. Quitting the StandardScaler(), you lose about 35% of precision.
"""
Building a mix between scaling and a Grid Search Cross_validation
"""
# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)

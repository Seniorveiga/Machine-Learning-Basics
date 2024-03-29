"""
The previous .py are made to evaluate the model. We will explore how to optimize it.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

diabetes_prev = pd.read_csv("diabetes_clean.csv")
diabetes = pd.DataFrame(diabetes_prev)

print(diabetes.columns)
X = diabetes.drop("diabetes",axis=1).values
y = diabetes["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#GRID SEARCH CV

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
lasso = Lasso(alpha = 0.3)

# Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))



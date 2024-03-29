import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

music_df = pd.read_csv("music_clean.csv")
"""
The point is to tranform some variables that are categoricals(Strings) into something that we can use in scikit-learn.
"""
# Create music_dummies
music_dummies = pd.get_dummies(data = music_df, drop_first = True)

# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))

# Create X and y
X = music_dummies.drop("popularity",axis=1).values
y = music_dummies["popularity"].values

# Create a KFold object
kf = KFold(n_splits=6, shuffle = True, random_state=5)

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE// Real mean squared error
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))
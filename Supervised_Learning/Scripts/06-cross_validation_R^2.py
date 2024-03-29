import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

# 1º) Shaping for LINEAR REGRESSION
# CONTINUES DOWN

ads_and_sales_preview = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = pd.DataFrame(ads_and_sales_preview)

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

# 2º) Use of Cross Validation with 6 folders: It trains the model with all the available data.
"""
Using a Cross Validation is like separating the set that you are working with in several sections and do a 
calculation of the R^2 based on each folder, which is done through "cross_val_scores" 
"""
# Create a KFold object
kf = KFold(n_splits=6, shuffle = True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

# Print the 95% confidence interval
print(np.quantile(cv_scores, [0.025, 0.975]))
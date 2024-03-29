import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1º) Shaping for LINEAR REGRESSION
# CONTINUES DOWN

ads_and_sales_preview = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = pd.DataFrame(ads_and_sales_preview)

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train,y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

"""
R^2_error and Mean squared Error.
"""

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared // Score is the same as R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_pred, y_test, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

"""
The features explain 99.9% of the variance in sales values! Looks like this company's advertising strategy is working well!
"""
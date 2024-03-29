import numpy as np
import pandas as pd

# 1º) Shaping for LINEAR REGRESSION

ads_preview = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = pd.DataFrame(ads_preview)

# Create X from the radio column's values
X = sales_df["radio"].values

# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X
X = X.reshape(-1,1)

# Check the shape of the features and targets
print(X.shape,y.shape)
"""
Prediction
"""
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

# Make predictions: Notice that the predictions are based on the values of "radio" column, not "sales"
predictions = reg.predict(X)

print(predictions[:5])

# Visualization of a linear regression model

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
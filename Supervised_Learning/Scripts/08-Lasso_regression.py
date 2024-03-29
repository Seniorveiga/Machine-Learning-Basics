import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Import the necessary modules

# 1º) Shaping for LINEAR REGRESSION
# CONTINUES DOWN

ads_and_sales_preview = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = pd.DataFrame(ads_and_sales_preview)

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values
sales_columns = sales_df.columns
#Create the Train group and test group

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size =0.3, random_state=42)

# 2º)Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
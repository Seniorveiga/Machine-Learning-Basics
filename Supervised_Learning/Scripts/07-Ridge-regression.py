import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
# Import the necessary modules

# 1º) Shaping for LINEAR REGRESSION
# CONTINUES DOWN

ads_and_sales_preview = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = pd.DataFrame(ads_and_sales_preview)

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values
#Create the Train group and test group

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size =0.3, random_state=42)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  
  # Create a Ridge regression model
  ridge = Ridge(alpha = alpha)
  
  # Fit the data
  ridge.fit(X_train,y_train)
  
  # Obtain R-squared
  score = ridge.score(X_test,y_test)
  ridge_scores.append(score)
print(ridge_scores)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#2nd part
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


music_df = pd.read_csv("music_clean.csv")

X = music_df.drop("energy",axis=1).values
y = music_df["energy"].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  
  # Append the results
  results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()

"""
In the last exercise, linear regression and ridge appeared to produce similar results. 
It would be appropriate to select either of those models; however, you can check predictive performance 
on the test set to see if either one can outperform the other.
"""
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.fit_transform(X_test)

for name, model in models.items():
  
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)
  
  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)
  
  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))
  
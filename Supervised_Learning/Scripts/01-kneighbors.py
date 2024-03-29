import pandas as pd
import numpy as np

churn_preview = pd.read_csv("telecom_churn_clean.csv")
churn_df = pd.DataFrame(churn_preview)

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
print(knn.fit(X, y))
"""
Once it is fit for the data it cna be used to predict the label for new data points.
"""

#--Prediction--------
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions for X_new
print("Predictions: {}".format(y_pred)) 
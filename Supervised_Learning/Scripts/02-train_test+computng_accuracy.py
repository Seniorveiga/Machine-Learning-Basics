import pandas as pd
import numpy as np

churn_preview = pd.read_csv("telecom_churn_clean.csv")
churn_df = pd.DataFrame(churn_preview)

#Consider that X and y target features and values in different variables: That´s why we divide in X and y.

# Import the module
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
print(X_train.shape,y_train.shape)
knn.fit(X_train,y_train)

"""
We cannot use fit to generalize data. Train set is used to fit the classifier.
Then we calculate the accuracy against the test sets labels.
Example: If churn appears 10% of observations, it sould appear a 10% in test and train split.

"""

# Print the accuracy
print(knn.score(X_test, y_test))

#------Overfit and underfit----------

# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test,y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
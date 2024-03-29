import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Import confusion matrix
from sklearn.metrics import confusion_matrix,classification_report

diabetes_prev = pd.read_csv("diabetes_clean.csv")
diabetes = pd.DataFrame(diabetes_prev)

print(diabetes.columns)
X = diabetes.drop("diabetes",axis=1).values
y = diabetes["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

knn = KNeighborsClassifier(n_neighbors=6)

print(X.shape, y.shape)
print(X_train.shape, y_train.shape)

# Fit the model to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""
Logistic Regression: Model that calculates the probability, p, that an observation belongs to a binary class.
"""
from sklearn.linear_model import LogisticRegression
# Import roc_curve
from sklearn.metrics import roc_curve
#ROC AUC Values
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

diabetes_prev = pd.read_csv("diabetes_clean.csv")
diabetes = pd.DataFrame(diabetes_prev)

print(diabetes.columns)
X = diabetes.drop("diabetes",axis=1).values
y = diabetes["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train,y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
"""
The predict() method is used to predict a category for a set of input features. 
It returns a discrete value that can be directly assigned to each input feature.

On the other hand, the predict_proba() method returns the predicted probabilities of the input features 
belonging to each category. The method, instead of returning a discrete class, returns the probabilities 
associated with each class. 
This is useful when not only do we want to know the category of the input features, but we also want to 
know the model's confidence in its prediction.
"""
print(y_pred_probs[:10])

#ROC Curve
"""
It is a curve to visualize how the true positive rate and false positive rate vary as the decision threshold changes.
"""

# Generate ROC curve values: false positive rates, true positive rates, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()

"""
As the curve is over the line, that means is much better than a random model predicting the class.
"""
#ROC AUC Values
y_pred = logreg.predict(X_test)

# Calculate roc_auc_score
print("ROC AUC Score: Area under the ROC Curve")
print(roc_auc_score(y_test, y_pred_probs))

# Calculate the confusion matrix
print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print("Classification Report")
print(classification_report(y_test, y_pred))
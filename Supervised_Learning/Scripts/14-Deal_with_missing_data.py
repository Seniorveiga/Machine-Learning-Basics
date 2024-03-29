import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


"""
NEEDS FIXING IN X_train 
"""
music = pd.read_csv("music_clean.csv")
music_df = pd.DataFrame(music)

# Print missing values for each column
print(music_df.isna().sum().sort_values())

# Remove values where less than 5% are missing
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])

print(music_df["genre"].value_counts())

"""
# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)
"""

print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))

# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Build steps for the pipeline
steps = [("imputer", imputer), 
         ("knn", knn)]
# Create the pipeline
pipeline = Pipeline(steps)
"""
The pipeline o "tubería" is the set of operations that our data set is going to do, in that particular order, in this case:
1ª) Imputer
2ª) KNN
"""

#-----------

"""
NOTA: If we want to impute more than one column, we should separate X and y in categorical and numerical values, and 
after that use append. Sample:

X_cat = music_df["genre"].values.reshape(-1,1)
X_num = music_df.drop([]"genre","popularity"],axis=1).values
y = music_df["popularity"].values

X_train_cat,X_test_cat,y_train,y_test = train_test_split(X_cat,y, test_size = 0.2, random_state = 42)
X_train_num,X_test_num,y_train,y_test = train_test_split(X_num,y, test_size = 0.2, random_state = 42)

imp_cat = SimpleImputer(strategy= "most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_train_cat)

    # Another one for the num data
imp_cat = SimpleImputer()
X_train_num = imp_cat.fit_transform(X_train_num)
X_test_num = imp_cat.transform(X_test_num)

X_train = np.append(X_train_cat, X_train_num, axis = 1)
X_test = np.append(X_test_cat, X_test_num, axis = 1)

"""
# Training set and test sets.
X = music_df.drop("genre",axis=1).values
y = music_df["genre"].values

kf = KFold(n_splits=6, shuffle = True, random_state=5)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Fit the pipeline to the training data
pipeline = pipeline.fit(X_train,y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

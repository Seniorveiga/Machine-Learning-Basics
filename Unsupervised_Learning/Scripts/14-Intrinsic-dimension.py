"""
Some data can be aproximated using only a part of their measures. FOr example, in the iris dataset (flowers) the petal 
width does not make a big difference between different species. Knowing that the dataset only has 3 variables, and one 
of them does not help us to separate the flowers into different species, we can say it has intrinsic dimension 2.

Intrinsic dimension = Number of PCA with significant variance.

In this case, from 6 dimensions, the intrinsic ones are 2.
"""

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt

fish_csv = pd.read_csv("fish.csv", header=None)
fish = pd.DataFrame(fish_csv)

#------Separation-Values--------------
print(fish)
fish_species = fish[0]
fish_measures = fish.drop(0,axis=1)

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(fish_measures)

# Plot the explained variances
features = range(pca.n_components_)         #We have 6 features thanks to the fit method in pipeline.
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
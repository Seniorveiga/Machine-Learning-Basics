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

#1st-----Apply-Standard_scaler------

scaler = StandardScaler()
scaler.fit(fish_measures)
measures_scaled = scaler.transform(fish_measures)

#2nd------PCA_for_dimensionality_reduction--------
 
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(measures_scaled)

# Transform the scaled samples: pca_features
pca_features = pca.transform(measures_scaled)

# Print the shape of pca_features
print(pca_features.shape)
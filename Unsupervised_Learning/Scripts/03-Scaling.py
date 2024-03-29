"""
Sometimes data are going to give weird classifications because of their variability. In this case, fishes have really 
different lengths which have much more weight than a categorical value. We can solve this by using StandardScaler, which
allows us the "normalize" the data (Is not the same but pretty similar)
"""
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Import pandas
import pandas as pd

fishes_csv = pd.read_csv("fish.csv", header = None)
fishes = pd.DataFrame(fishes_csv)

# Create scaler: scaler
scaler = StandardScaler()   

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

#--------------How to separate string values-----------------------
#Separating species and measures
species = fishes[0]
print(species)
fishes_measures = fishes.drop(0,axis=1)

#Converting to numpy both of the DataFrames.
species = species.to_numpy()
fishes_measures = fishes_measures.to_numpy()
#-----------------------------------------------------------------

# Fit the pipeline to samples
pipeline.fit(fishes_measures)

# Calculate the cluster labels: labels
labels = pipeline.predict(fishes_measures)

# Create a DataFrame with labels and species as columns: df
dataf={
    "labels": labels,
    "species": species
}
df = pd.DataFrame(dataf)

# Create crosstab: ct
ct = pd.crosstab(df["labels"],df["species"])

# Display ct
print(ct)

"""
Notes: The error appeared in pipeline.fit, so that means the object that we were using was not appropiate due to the
strings it contained. We solve this separating both parts with strings, and we make a new df.

Supposing we have more strings such as the area the belong, we use a list with the different columns.
"""

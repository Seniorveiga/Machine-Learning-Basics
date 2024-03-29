"""
 Use the fcluster() function to extract the cluster labels for this intermediate clustering, 
 and compare the labels with the grain varieties using a cross-tabulation.
"""

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster


grain_csv = pd.read_csv("seeds.csv",header=None)
grain_data = pd.DataFrame(grain_csv)
print(grain_data)

#Separating species and measures
grain_species = grain_data[7]
print(grain_species)
grain_measures = grain_data.drop(7, axis = 1)

#Converting to numpy both of the DataFrames.
grain_species = grain_species.to_numpy()
grain_measures = grain_measures.to_numpy()

# Calculate the linkage: mergings
mergings = linkage(grain_measures,method = "complete")

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6, criterion = "distance" )

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': grain_species})

# Create crosstab: ct
ct = pd.crosstab(df["labels"],df["varieties"])

# Display ct
print(ct)
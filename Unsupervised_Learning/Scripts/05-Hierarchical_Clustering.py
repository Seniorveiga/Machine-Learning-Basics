"""
In the video, you learned that the SciPy linkage() function performs hierarchical clustering on an array of samples. 
Use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram() to visualize 
the result.
"""
import pandas as pd
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

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

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=grain_species,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
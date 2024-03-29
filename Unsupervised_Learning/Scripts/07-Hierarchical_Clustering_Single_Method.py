"""
In the video, you saw a hierarchical clustering of the voting countries 
at the Eurovision song contest using 'complete' linkage. 
Now, perform a hierarchical clustering of the voting countries with 'single' linkage, 
and compare the resulting dendrogram with the previous one.
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
mergings = linkage(grain_measures, method = "single")

# Plot the dendrogram
dendrogram(mergings,
            labels = grain_species,
            leaf_rotation = 90,
            leaf_font_size=6,
)
plt.show()
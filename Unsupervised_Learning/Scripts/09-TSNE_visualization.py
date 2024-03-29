"""
When we have lots of data together, how do we know if they are clustered properly? Maybe they could be 
subdivided more or less, but TSNE solves this.

It gives a scatter plot which separates the dots in different areas, but the metrics in the axises do not give
any kind of information, despite of the numbers.
"""
import pandas as pd
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
# Import TSNE
from sklearn.manifold import TSNE

grain_csv = pd.read_csv("seeds.csv",header=None)
grain_data = pd.DataFrame(grain_csv)
print(grain_data)

"""
FALTA NORMALIZARLOS
"""

#Separating species and measures
grain_species = grain_data[7]
print(grain_species)
grain_measures = grain_data.drop(7, axis = 1)

#Converting to numpy both of the DataFrames.
grain_species = grain_species.to_numpy()
grain_measures = grain_measures.to_numpy()

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(grain_data)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c= grain_species)
plt.show()
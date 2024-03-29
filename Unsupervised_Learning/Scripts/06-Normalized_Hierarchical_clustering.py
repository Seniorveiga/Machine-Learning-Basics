"""
In chapter 1, you used k-means clustering to cluster companies according to their stock price movements. 
Now, you'll perform hierarchical clustering of the companies.

Note: SciPy hierarchical clustering doesn't fit into a sklearn pipeline, 
so you'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
# Import normalize
from sklearn.preprocessing import normalize

stocks_csv = pd.read_csv("company-stock-movements-2010-2015-incl.csv")
stocks = pd.DataFrame(stocks_csv)

print(stocks)

#Separation
companies = stocks["Unnamed: 0"]
movements = stocks.drop("Unnamed: 0", axis=1)

companies = companies.to_numpy()
movements = movements.to_numpy()

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method = "complete")

# Plot the dendrogram
dendrogram(mergings,
            labels = companies,
            leaf_rotation = 90,
            leaf_font_size = 6,
)

plt.show()

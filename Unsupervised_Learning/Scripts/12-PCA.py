import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Import PCA
from sklearn.decomposition import PCA

grains_csv = pd.read_csv("seeds-width-vs-length.csv")
grains = pd.DataFrame(grains_csv)
grains = grains.to_numpy()

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)

"""
NOTES: PCA rotates the data so they are oriented with the axes.
Nevertheless, it decorrelates the data as the matrix transformation is not linearly correlated. But, to know it, we have
the Pearson correlation!

This gives us the linear correlation without the use of the graphic.
"""
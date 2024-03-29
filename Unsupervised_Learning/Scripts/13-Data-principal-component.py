import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Import PCA
from sklearn.decomposition import PCA

grains_csv = pd.read_csv("seeds-width-vs-length.csv")
grains = pd.DataFrame(grains_csv)
grains = grains.to_numpy()

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
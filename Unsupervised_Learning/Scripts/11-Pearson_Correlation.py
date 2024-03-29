import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

grains_csv = pd.read_csv("seeds-width-vs-length.csv")
grains = pd.DataFrame(grains_csv)
grains = grains.to_numpy()

# Assign the 0th column of grains: width
width = grains[:,0]
print(type(width))

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)

"""
NOTES: You can do this exercise in 2 ways: By transforming the Dataframe into a numpy array as we have already done.
This means that  it accepts both a numpy array and pandas series.

We can know this by printing the type of the "width" object.
"""

grains_csv = pd.read_csv("seeds-width-vs-length.csv")
grains = pd.DataFrame(grains_csv)

# Assign the 0th column of grains: width
width = grains.iloc[:,0]
print(type(width))

# Assign the 1st column of grains: length
length = grains.iloc[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)
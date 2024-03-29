import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

semillas = pd.read_csv("seeds.csv", header = None)
seeds = pd.DataFrame(semillas)

print(seeds)
print()

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(seeds)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

"""
Notes: the .fit() method accepts the whole dataFrame as a training set.

"""

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters = 3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(seeds)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': seeds.iloc[:,7].values})

# Create crosstab: ct
crosstab = pd.crosstab(df["labels"],df["varieties"])

# Display ct
print(crosstab)
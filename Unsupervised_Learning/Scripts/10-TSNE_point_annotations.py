import pandas as pd
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
# Import normalize
from sklearn.preprocessing import normalize
# Import TSNE
from sklearn.manifold import TSNE

stocks_csv = pd.read_csv("company-stock-movements-2010-2015-incl.csv")
stocks = pd.DataFrame(stocks_csv)

print(stocks)

#Separation
companies = stocks["Unnamed: 0"]
movements = stocks.drop("Unnamed: 0", axis=1)

companies = companies.to_numpy()
movements = movements.to_numpy()
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate = 50)

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha = 0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
"""
An NMF is a technique that allows us to make dimension reduction, with the advantage that the data are easy
to interpret.

However, not all the data can be used through this function (For example negative data are not allowed)
    - The number of components should be specified.
    - It works with NumPy arrays and csr_matrix.
"""
import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv('wikipedia-vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)

# Note: This should be seen in the course about "Preproccesing Data in Python", 
# that´s why it is already included in the doc.

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components = 6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index = titles)

# Print the row for 'Anne Hathaway"
print(df.loc["Anne Hathaway"])

#------Learning topics of documents-----
# Import pandas
import pandas as pd

#For the wikipedia words
with open("wikipedia-vocabulary-utf8.txt", "r") as mierdasdewikipedia:
	words = mierdasdewikipedia.readlines()

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns = words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())

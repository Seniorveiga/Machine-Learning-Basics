import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

from scipy.sparse import csr_matrix

"""
The object we have to make the import from does not have any data. This will be done smilar to "18-NMF.py" with 
wikipedia articles but with artists.
"""
"""
df = pd.read_csv("artists.csv", index_col=0)
artists = csr_matrix(df.transpose())
titles = list(df.columns)

print(artists)

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components = 20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)
"""
"""
Note that Normalizer() is different to StandardScaler(), which you used in the previous exercise. 

While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) 
by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, 
each company's stock price - independently of the other. 
"""
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

stocks_csv = pd.read_csv("company-stock-movements-2010-2015-incl.csv")
stocks = pd.DataFrame(stocks_csv)
print(stocks.columns)

#We can see a failure if we try directly, so we need to separate the company names from the movements

#--------------How to separate string values-----------------------
#Separating stocks and measures
companies = stocks.iloc[:,0]
print(companies)
movements = stocks.drop("Unnamed: 0",axis=1)

#Converting to numpy both of the DataFrames.
stocks = stocks.to_numpy()
movements = movements.to_numpy()
#------------------------------------------------------------------

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values("labels"))

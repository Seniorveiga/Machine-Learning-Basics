from random import uniform
import pandas as pd
import matplotlib.pyplot as plt

random_list_1 = []
for i in range(0,300):
    n = uniform(-2,-1)
    random_list_1.append(n)
for i in range(0,300):
    n = uniform(0,1)
    random_list_1.append(n)

random_list_2 = []
for i in range(0,300):
    n = uniform(-1.5,-.5)
    random_list_2.append(n)
for i in range(0,300):
    n = uniform(1,2)
    random_list_2.append(n)

#NEW POINTS
new_points_1 = []
new_points_2 = []
for i in range(0,150):
    n = uniform(-1.5,1)
    random_list_1.append(n)
    new_points_1.append(n)
for i in range(0,150):
    n = uniform(-1,2)
    random_list_2.append(n)
    new_points_2.append(n)

data_points = {
    0:random_list_1,
    1:random_list_2
}

data_new_points = {
    0:new_points_1,
    1:new_points_2
}

points = pd.DataFrame(data_points)
new_points = pd.DataFrame(data_new_points)

xs = points.iloc[:,0].values
ys = points.iloc[:,1].values
plt.scatter(xs,ys)
plt.show()

#--------------------KMeans---------------
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=2)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

#-----INSPECTION---------
# Assign the columns of new_points: xs and ys
xs = new_points.iloc[:,0].values
ys = new_points.iloc[:,1].values

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha = 0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker = 'D', s = 50)
plt.show()

"""
Some notes:

To generate random values for the DataFrame, you need to use uniform function for "random" package. 
This allows us to bring float values rather than ints, which can be done through a "for" loop and 
the "randrange()" function.

When you are working with a DataFrame, you should use .iloc for the columns, not just the brackets for the
DataFrame.
"""
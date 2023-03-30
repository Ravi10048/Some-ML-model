import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

wcss = [] #wcss- with cluster sum of square

for i in range(1,11):  # tends to find distance between each data point to a centroid (cluster point) and each circle have each own centroid
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) # 
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)  # give and append wcss value

# We assume first 2 data as centroid then find the distance between them by all centroid then update the centroid then again continue
import seaborn as sns
# plot elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title("Elbow point graph")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS value")
plt.show()  # we will consider k value we gives us sharp significant drop in value

'''Optimum Number of Clusters = 3'''

# Initialize the KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3,init='k-means++', random_state=42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Make predictions on the data
y_pred = kmeans.predict(X)
print(y_pred)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='orange', label='Centroids')
plt.title('K-Means Clustering Results')
plt.show()


# confusion matrix is not useful or not used in unsupervised learning
kmeans_cm = confusion_matrix(y, y_pred)
print(kmeans_cm)












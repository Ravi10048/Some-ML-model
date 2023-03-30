import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Load Iris dataset
iris = load_iris()
X=iris.data
y=iris.target
# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
iris_reduced = pca.fit_transform(iris.data)

print(X)
print(iris_reduced)

# Set hyperparameters for FCM
n_clusters = 3
fcm = FCM(n_clusters=n_clusters, max_iter=1000, m=2)

# Fit FCM to the data
fcm.fit(iris_reduced)

# Get predicted clusters for the data
predicted_clusters = fcm.predict(iris_reduced)
centers=fcm._centers
print(centers)

import matplotlib.pyplot as plt
y_pred=predicted_clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.scatter(centers[0], centers[1],s=100,c='orange', label='')
plt.title('K-Medoids Clustering Results')
plt.show()




# Evaluate clustering performance using Adjusted Rand Score
true_labels = iris.target
ari = adjusted_rand_score(true_labels, predicted_clusters)
print(f"Adjusted Rand Score: {ari}")

# confusion matrix is not useful or not used in unsupervised learning
kmeans_cm = confusion_matrix(y, y_pred)
print(kmeans_cm)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import load_iris
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.samples.definitions import SIMPLE_SAMPLES

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Define the distance metric
metric = distance_metric(type_metric.EUCLIDEAN)

# Initialize the KMedoids object with 3 clusters
kmedoids_instance = kmedoids(X, initial_index_medoids=[0, 50, 100], metric=metric)

# Run the KMedoids clustering algorithm
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

print("cluster :" ,clusters)
print("medoid :" ,medoids)

# Map the cluster labels to the target variable
y_pred = [-1] * len(X)
for i, cluster in enumerate(clusters):
    for index in cluster:
        y_pred[index] = i

print('y_pred :',y_pred)
# Plot the results

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(X[medoids[:], 0], X[medoids[:], 1], s=100,c='orange', label='Medoid')
plt.title('K-Medoids Clustering Results')
plt.show()

# confusion matrix is not useful or not used in unsupervised learning
# kmedoid_cm = confusion_matrix(y, y_pred)
# print(kmedoid_cm)





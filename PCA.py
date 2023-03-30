import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

X=iris.data
y=iris.target

# Perform PCA with two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X)
print(X_pca)
# Plot the PCA results
import matplotlib.pyplot as plt
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

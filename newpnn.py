# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load iris dataset
iris = load_iris()

# create dataframes for features and target
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(data=iris.target, columns=['species'])

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the newpnn model
model = NearestCentroid(metric='euclidean', shrink_threshold=None)

# fit the model to the training data
model.fit(X_train, y_train.values.ravel())

# make predictions on the test data
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# generate the classification report
cr = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Classification Report:')
print(cr)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



''' In this code, we first load the iris dataset and split it into training and testing sets. We then define a newpnn model with the NearestCentroid class from scikit-learn, and fit it to the training data. We make predictions on the test data, and calculate the accuracy, confusion matrix, and classification report using the accuracy_score, confusion_matrix, and classification_report functions from scikit-learn.'''


'''NewPNN is a simple supervised learning algorithm used for classification tasks.
It is a variant of the k-Nearest Neighbors (k-NN) algorithm.
NewPNN calculates the distances between a new input and the prototype of each class, instead of using distances to the k nearest neighbors.
The prototype of each class is the centroid or mean of the feature values of all the training examples in that class.
The class of the new input is then assigned to the class with the closest prototype.
NewPNN is a simple and computationally efficient algorithm that works well on datasets with low dimensionality and well-separated classes.
However, it may not perform well on datasets with high dimensionality and overlapping classes.'''
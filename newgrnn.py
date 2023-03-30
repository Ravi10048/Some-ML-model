from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define the New GRNN model
class NewGRNN:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            similarities = np.exp(-np.sum((self.X_train - x)**2, axis=1) / (2 * self.sigma**2))
            y_pred.append(np.argmax(np.bincount(self.y_train, weights=similarities)))
        return y_pred

# Train the New GRNN model with the training set
model = NewGRNN(sigma=1)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

print(y_pred)
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Calculate the loss
loss = np.sum(y_test != y_pred) / len(y_test)
print("Loss:", loss)

# Calculate the accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)
print("Accuracy:", accuracy)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



'''NewGRNN is a type of supervised neural network model that can be used for classification or regression tasks. It works by calculating the similarities between the input data points and the training data points using a Gaussian kernel. The similarities are then used to compute a weighted average of the output values of the training data points to obtain the predicted output value for the input data point.

Compared to other neural network models, NewGRNN is computationally efficient and has a low risk of overfitting the training data. However, it may not perform as well as more complex models on complex datasets with nonlinear relationships between the input and output variables.'''

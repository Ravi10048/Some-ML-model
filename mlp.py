import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# Train the MLP classifier on the training set
mlp.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = mlp.predict(X_test)

# Compute the confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Compute the accuracy and specificity
accuracy = np.mean(y_pred == y_test)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

# Compute the loss
loss = mlp.loss_

# Print the confusion matrix, classification report, accuracy, specificity, and loss
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', cr)
print('specificity ',specificity)
print('accuracy ',accuracy)
print('loss ',loss)



import seaborn as sns
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


'''The MLP is a type of artificial neural network that consists of an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next layer, and the network is trained using backpropagation to adjust the weights and biases of the neurons.
'''
'''This code trains an MLP classifier with two hidden layers of 10 neurons each, using the Iris dataset.'''



''' newrb stands for "new radial basis network" and it creates a type of neural network called a radial basis function network, which is a feedforward network that uses radial basis functions as activation functions in the hidden layer.

newff stands for "new feedforward network" and it creates a basic feedforward neural network with fully connected layers, which can be trained using various optimization algorithms to solve regression or classification problems.

While these functions can be useful for certain tasks, they are not considered to be part of the deep learning paradigm, which generally involves the use of neural networks with many layers and complex architectures.'''
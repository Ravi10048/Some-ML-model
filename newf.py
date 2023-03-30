# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Define the model architecture
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(4,)),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print the test accuracy and loss
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert probabilities to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:')
print(cm)
cr=classification_report(y_test, y_pred_classes)
print('Classification Report:')
print(cr)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('specificity ',specificity)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()







'''This code defines a simple feedforward neural network with one hidden layer of 10 neurons and 
an output layer of 3 neurons for the three Iris species. The model is compiled with the Adam optimizer 
and the sparse categorical crossentropy loss function. The model is trained on the training set for 100 epochs with a validation split of 0.2. The test accuracy, loss, confusion matrix, and classification report are printed at the end.'''
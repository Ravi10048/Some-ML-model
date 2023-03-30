from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the MLPClassifier
rb = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='lbfgs', max_iter=1000, random_state=42)
rb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rb.predict(X_test)

# Compute the confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Compute the accuracy and loss
accuracy = rb.score(X_test, y_test)
loss = rb.loss_

print("\nAccuracy:", accuracy)
print("Loss:", loss)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

'''In this code, we first load the Iris dataset using the scikit-learn library. We then split the dataset into training and testing sets, and standardize the data using the StandardScaler function. We train the Radial Basis Networks model using the MLPClassifier function and predict the test set results using the predict function. We then print the confusion matrix and classification report '''

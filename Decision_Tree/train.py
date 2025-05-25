from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DescisionTree import DecisionTree

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train decision tree
clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Accuracy function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Print accuracy
acc = accuracy(y_test, predictions)
print("Accuracy:", acc)

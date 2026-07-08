import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Target names:", iris.target_names)
print("Feature names:", iris.feature_names)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("First training sample:", X_train[0])
print("First training label:", y_train[0])

def distance(a, b):
    difference = a - b
    squared_difference = difference ** 2
    sum_squared_difference = np.sum(squared_difference)
    return np.sqrt(sum_squared_difference)

def find_neighbors(test_sample, X_train, y_train, k):
    distances = []

    for i in range(len(X_train)):
        train_sample = X_train[i]
        train_label = y_train[i]

        d = distance(test_sample, train_sample)
        distances.append((d, train_label))

    distances.sort(key=lambda item: item[0])

    neighbors = distances[:k]
    return neighbors

test_sample = X_test[0]
k = 5

neighbors = find_neighbors(test_sample, X_train, y_train, k)

print("Test sample:", test_sample)
print("True label:", y_test[0])
print("Nearest neighbors:")
print(neighbors)

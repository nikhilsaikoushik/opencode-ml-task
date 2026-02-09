import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

TRAIN_SAMPLES = 5000
TEST_SAMPLES = 500

x_train = x_train[:TRAIN_SAMPLES]
y_train = y_train[:TRAIN_SAMPLES]
x_test = x_test[:TEST_SAMPLES]
y_test = y_test[:TEST_SAMPLES]


x_train = x_train.reshape(TRAIN_SAMPLES, -1)
x_test = x_test.reshape(TEST_SAMPLES, -1)

x_train = x_train / 255.0
x_test = x_test / 255.0

print("Dataset loaded and preprocessed")

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(x_train, y_train, test_sample, k, metric="euclidean"):
    distances = []

    for i in range(len(x_train)):
        if metric == "euclidean":
            dist = euclidean_distance(x_train[i], test_sample)
        elif metric == "manhattan":
            dist = manhattan_distance(x_train[i], test_sample)
        else:
            raise ValueError("Unsupported distance metric")

        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_nearest = distances[:k]

    labels = [label for _, label in k_nearest]
    prediction = max(set(labels), key=labels.count)

    return prediction

def evaluate_knn(k, metric):
    correct = 0
    misclassified = []

    for i in range(len(x_test)):
        pred = knn_predict(x_train, y_train, x_test[i], k, metric)

        if pred == y_test[i]:
            correct += 1
        else:
            misclassified.append((i, y_test[i], pred))

    accuracy = correct / len(x_test)
    return accuracy, misclassified

k_values = [1, 3, 5]

for metric in ["euclidean", "manhattan"]:
    print(f"\nResults using {metric.capitalize()} Distance:")
    for k in k_values:
        acc, mis = evaluate_knn(k, metric)
        print(f"K = {k} | Accuracy = {acc:.4f} | Misclassified = {len(mis)}")

print("\nSample misclassified cases (index, true label, predicted label):")
for m in mis[:10]:
    print(m)
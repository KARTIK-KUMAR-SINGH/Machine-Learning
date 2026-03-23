import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w = 0
b = 0
alpha = 0.1
iterations = 1000

for _ in range(iterations):
    z = x_train * w + b
    y_hat = sigmoid(z)

    dw = (1/len(x_train)) * np.sum((y_hat - y_train) * x_train)
    db = (1/len(x_train)) * np.sum(y_hat - y_train)

    w -= alpha * dw
    b -= alpha * db

predictions = (sigmoid(x_train * w + b) >= 0.5).astype(int)
print("Predictions:", predictions)
print("Actual:     ", y_train)
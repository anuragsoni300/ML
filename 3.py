import numpy as np
np.set_printoptions(precision=2, suppress=False)

# 1. Input data
X = np.array([[1], [2], [3], [4], [5]])  # shape: (5, 1)
y = np.array([[2], [4], [6], [8], [10]])  # shape: (5, 1)

X_b = np.c_[np.ones((X.shape[0], 1)), X]
W = np.zeros((2,1))

learning_rate = 0.02
epochs = 10000

for epochs in range(epochs):
    y_pred = X_b @ W
    error = y_pred - y
    gradient = 2 / X_b.shape[0] * X_b.T @ error
    W = W - learning_rate * gradient

print("Final weight\n",W)

x_new = np.array([[1, 6]])
y_new_pred = x_new @ W
print(y_new_pred[0][0])

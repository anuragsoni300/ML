import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])
n = len(X)

# Reshape for matrix operations
X_b = np.c_[np.ones((n, 1)), X.reshape(-1, 1)]
Y = Y.reshape(-1, 1)

# Initial weights
W = np.random.randn(2, 1)
print(W)
learning_rate = 0.01
epochs = 100

# For plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='red', label='Data')

# Training loop
for epoch in range(epochs):
    gradients = 2 / n * X_b.T @ (X_b @ W - Y)
    W -= learning_rate * gradients

    if epoch % 10 == 0 or epoch == epochs - 1:
        Y_pred = X_b @ W
        plt.plot(X, Y_pred, label=f'Epoch {epoch}')

# Plot final result
plt.title("Gradient Descent Learning Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

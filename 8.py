import numpy as np

# XOR Input and Output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Dimensions
np.random.seed(42)
input_dim = 2
hidden_dim = 2
output_dim = 1

# Initialize parameters
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# Forward pass
def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Loss function
def compute_loss(y, a2):
    eps = 1e-8
    return -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))

# Backward pass
def backward(X, y, z1, a1, z2, a2, W1, b1, W2, b2, learning_rate=0.1):
    m = X.shape[0]

    dz2 = a2 - y                           # (4,1)
    dW2 = (a1.T @ dz2) / m                 # (2,1)
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  # (1,1)

    da1 = dz2 @ W2.T                       # (4,2)
    dz1 = da1 * sigmoid_derivative(z1)     # (4,2)
    dW1 = (X.T @ dz1) / m                  # (2,2)
    db1 = np.sum(dz1, axis=0, keepdims=True) / m  # (1,2)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

# Training loop
for epoch in range(10000):
    z1, a1, z2, a2 = forward(X)
    loss = compute_loss(y, a2)
    W1, b1, W2, b2 = backward(X, y, z1, a1, z2, a2, W1, b1, W2, b2)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# Final output
print("Final predictions (rounded):")
_, _, _, final_output = forward(X)
print(np.round(final_output))
print("Actual output:")
print(y)

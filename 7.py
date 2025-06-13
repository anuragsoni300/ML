import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Dataset: 4 samples
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  # shape: (4, 2)

# Labels: only 1+1 = 1 (simple AND logic)
Y = np.array([[0], [0], [0], [1]])  # shape: (4, 1)
# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # x = sigmoid(z)
    return x * (1 - x)

# Initialize weights
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training
lr = 0.1
epochs = 10000

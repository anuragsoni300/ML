import numpy as np

# Input features: [Hours studied]
X = np.array([[1], [2], [3], [4], [5], [6]])
# Labels: 0 = Fail, 1 = Pass
Y = np.array([[0], [0], [0], [1], [1], [1]])
X_b = np.c_[np.ones((X.shape[0],1)),X]
W = np.zeros((2, 1))
def sigmoid(z):
    return 1/(1+np.exp(-z))
def compute_loss(Y, Y_pred):
    return -np.mean(Y*np.log(Y_pred) + (1-Y)*np.log(1-Y_pred))
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    z = X_b@W
    Y_pred = sigmoid(z)
    loss = compute_loss(Y, Y_pred)
    # print(f"Epoch {epoch}, Loss: {loss}")
    gradien = X_b.T @ (Y_pred-Y)/Y.shape[0]
    W -= learning_rate * gradien

print("Final weights:\n", W)

X_new = np.array([[1, 4.5]])
Y_new = sigmoid(X_new @ W)

print(Y_new)

import numpy as np

# Features: [Hours, Classes, Sleep]
X = np.array([
    [2, 3, 6],
    [3, 4, 7],
    [4, 4, 5],
    [5, 5, 8],
    [6, 6, 9]
])

# Output: Exam score
Y = np.array([55, 65, 70, 85, 95])
X_b = np.c_[np.ones((X.shape[0], 1)), X]

W = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
print(W)

X_new = np.array([[1, 4, 4, 6]])
Y_pred = X_new @ W

print(Y_pred)
print(np.round(np.linalg.inv(X_b.T @ X_b) @ X_b.T))

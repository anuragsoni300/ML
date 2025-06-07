import numpy as np

# Create arrays
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# Element-wise operations
# print("a + b:", a + b)
# print("Dot product:", np.dot(a, b))
# print("Cross product:", np.cross(a, b))
# print(np.shape(a))

# a = np.array([[1, 2, 3],[4, 5, 6]])
# print(a)
# a = np.array([[3,2], [7,4]])
# b = np.array([16,36])
# print(np.linalg.solve(a, b))
# x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# print("25th percentile:", np.percentile(x, 25))
# print("50th percentile (median):", np.percentile(x, 50))
# print("75th percentile:", np.percentile(x, 90))
# x = np.array([10, 20, 30, 40, 50])
# mean = np.mean(x)
# std = np.std(x)

# z_scores = (x - mean) / std
# print("Z-scores:", z_scores)
import numpy as np
import matplotlib.pyplot as plt

# Input (Hours studied)
X = np.array([1, 2, 3, 4, 5])

# Output (Exam scores)
Y = np.array([40, 50, 65, 70, 80])

mx = np.mean(X)
my = np.mean(Y)

slope = np.sum((X-mx)*(Y-my))/np.sum((X-mx)**2)
b = my - slope * mx

Y_pred = slope*X + b

# Step 4: Plot
# plt.scatter(X, Y, color='blue', label='Actual Data')
# plt.plot(X, Y_pred, color='red', label='Prediction Line')
# plt.xlabel("Hours Studied")
# plt.ylabel("Exam Score")
# plt.legend()
# plt.grid()
# plt.show()
# Predict for a new input

print(slope * 6 + b)
print(np.mean(Y-Y_pred)**2)

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.array([-5, 0, 1, 5])
print(sigmoid(z_values))

y = 1
y_hat = 0.9

loss = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
print(loss)  # should be small, because prediction is good

import matplotlib.pyplot as plt

# Prediction probabilities from 0.001 to 0.999
y_hat = np.linspace(0.001, 0.999, 100)

# Loss when true label is 1
loss_y1 = -np.log(y_hat)

# Loss when true label is 0
loss_y0 = -np.log(1 - y_hat)

# Plotting
plt.plot(y_hat, loss_y1, label="y = 1", color='green')
plt.plot(y_hat, loss_y0, label="y = 0", color='red')
plt.title("Binary Cross Entropy Loss")
plt.xlabel("Predicted Probability (ŷ)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

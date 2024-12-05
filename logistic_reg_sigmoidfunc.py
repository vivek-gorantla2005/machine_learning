import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate random X values between 0 and 100
X = np.array([])

# Populate X with 100 random integers between 0 and 100
for i in range(100):
    X = np.append(X, random.randint(0, 100))

# Reshape X to be 2D (100, 1)
X = X.reshape(-1, 1)

# Create y where elements are 0 if corresponding element in X is < 30, otherwise 1
y = np.array([[0 if i < 30 else 1] for i in X])

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get model parameters
bo = model.intercept_[0]  # Flatten intercept (1D value)
b1 = model.coef_[0][0]  # Flatten coefficient (1D value)

print("intercept", bo)
print("slope:", b1)


# Sigmoid function for the logistic regression model
def sigmoid_curve(x):
    return 1 / (1 + np.exp(-(bo + b1 * x)))

# Generate an array of X values to plot the continuous sigmoid curve
X_plot = np.linspace(0, 100, 100).reshape(-1, 1)


# Plot the sigmoid curve
plt.plot(X_plot, sigmoid_curve(X_plot), color='red', label='Sigmoid curve')
plt.scatter(X, y, c=y, cmap='viridis')  # Scatter plot of the data
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

pred = model.predict_proba([[66]])
print(pred)
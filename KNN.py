import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Points for blue and red
x_blue = [0.3, 0.5, 1.4, 1.7, 2]
y_blue = [1, 4.5, 2.3, 1.9, 8.9]

x_red = [3.3, 3.5, 4, 4.4, 5.7, 6]
y_red = [7, 1.5, 6.3, 1.9, 2.9, 7.1]

# Combine data points
X = np.array(
    [[0.3, 1], [0.5, 4.5], [1.4, 2.3], [1.7, 1.9], [2, 8.9],
     [3.3, 7], [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]]
)
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) 

# Plot the points
plt.scatter(x_blue, y_blue, color='blue', label='Class 0 (Blue)')
plt.scatter(x_red, y_red, color='red', label='Class 1 (Red)')
plt.scatter(1, 5, color='green', s=100, label='New Point (Green)')
plt.axis([-0.5, 10, -0.5, 10])
plt.legend()
plt.show()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,Y)

# Predict the class of the new point
predict = classifier.predict(np.array([[1,5]]))
print(predict)


import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

# Blue points
x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

# Red points
x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

# Manually creating the feature matrix (X) and labels (Y)
X = [
    [0.3, 1], [0.5, 4.5], [1, 2.3], [1.4, 1.9], [1.7, 8.9], [2, 4.1],  # Blue points
    [3.3, 7], [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]   # Red points
]
Y = [0, 0, 0, 0, 0, 0,  # Blue points labeled as 0
     1, 1, 1, 1, 1, 1]  # Red points labeled as 1

X = np.array(X)  # Convert to NumPy array
Y = np.array(Y)  # Convert to NumPy array

plt.plot(x_blue,y_blue,'ro',color='blue')
plt.plot(x_red,y_red,'ro',color='red')
plt.plot(2.5,4.5,'ro',color = 'green')
# plt.show()

classifier = svm.SVC()
classifier.fit(X,Y)
print(classifier.predict([[2.5,4.5]]))
plot_decision_regions(X,Y,clf=classifier,legend=2)
plt.axis([-0.5,10,-0.5,10])
plt.show()
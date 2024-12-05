import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

# Load the iris dataset
iris_data = datasets.load_iris()
features = iris_data.data
targets = iris_data.target

# Define parameter grid for max_depth
param_grid = {'max_depth': np.arange(1, 10)}

# Split the dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.3)

# Perform GridSearchCV
model = DecisionTreeClassifier()
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(feature_train, target_train)

# Display the best parameter
print('Best parameter:', tree.best_params_)

# Use the best model to make predictions on the test set
grid_predictions = tree.predict(feature_test)

# Print confusion matrix and accuracy score
print("Confusion Matrix:")
print(confusion_matrix(target_test, grid_predictions))

print("Accuracy Score:", accuracy_score(target_test, grid_predictions))


predicted = cross_validate(model,features,targets,cv=10)
print(np.mean(predicted['test_score']))



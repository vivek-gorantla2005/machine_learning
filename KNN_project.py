import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing

# Load the dataset
data = pd.read_csv("credit_data.csv")

# Prepare features and target
features = data[["income", "age", "loan"]]
target = data.default

# Convert to numpy arrays
X = np.array(features).reshape(-1, 3)
y = np.array(target)

# Normalize features using MinMaxScaler to scale the values between 0 and 1 for making the knn more accurtate
X = preprocessing.MinMaxScaler().fit_transform(X)

# Split the dataset into training and test sets

features_train, features_test, target_train, target_test = train_test_split(X, y, test_size=0.3)


# Compute cross-validation scores for different k values
cross_valid_scores = []
for k in range(1, 101):  # Start k from 1 instead of 0
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())

print('optimal k value is :', np.argmax(cross_valid_scores) )


# Train a KNN model
model = KNeighborsClassifier(n_neighbors=32)
fitted_model = model.fit(features_train, target_train)

# Make predictions on the test set
predictions = fitted_model.predict(features_test)


# Print confusion matrix and accuracy score
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

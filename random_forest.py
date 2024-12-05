from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import datasets

# Load the iris dataset
iris_data = datasets.load_iris()
features = iris_data.data
targets = iris_data.target

# Split the dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.3)

model = RandomForestClassifier(n_estimators=1000,max_features='sqrt')
fitted_model = model.fit(feature_train,target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))

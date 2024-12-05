from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

digit_data = datasets.load_digits()
imgfeatures = digit_data.images.reshape((len(digit_data.images), -1))
img_target = digit_data.target

model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

feature_train, feature_test, target_train, target_test = train_test_split(
    imgfeatures, img_target, test_size=0.3, random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    "max_depth": [5, 10],
    "min_samples_leaf": [1, 4]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(feature_train, target_train)
print("Best Parameters:", random_search.best_params_)

random_predictions = random_search.predict(feature_test)
print("Accuracy:", accuracy_score(target_test, random_predictions))

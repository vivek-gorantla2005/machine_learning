import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv("credit_data.csv")

# Prepare features and target
features = data[["income", "age", "loan"]]
target = data.default

# Convert to numpy arrays
X = np.array(features).reshape(-1, 3)
y = np.array(target)

feature_train, feature_test,target_train,target_test = train_test_split(X,y,test_size=0.3)

model = GaussianNB()
fittedModel = model.fit(feature_train,target_train)

predictions = fittedModel.predict(feature_test)

print(confusion_matrix)
print(accuracy_score(target_test,predictions))



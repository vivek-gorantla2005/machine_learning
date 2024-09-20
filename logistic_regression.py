import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd

def predict_flood(model, rainfall, river_level, soil_moisture):
    new_data = np.array([[rainfall, river_level, soil_moisture]])
    prediction = model.predict(new_data)
    return 'Flood' if prediction[0] == 1 else 'No_Flood'

data = {
    'rainfall' :[100, 200, 150, 80, 120, 300],
    'riverlevel': [5, 6, 7, 5, 4, 3],
    'soil_moisture': [30, 25, 45, 40, 35, 32],
    'flood': [0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
x = df[['rainfall', 'riverlevel', 'soil_moisture']].values
y = df['flood']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# Evaluate the model
accuracy = np.mean(y_predict == y_test)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Example prediction
rainfall = 110
river_level = 8
soil_moisture = 24
print(predict_flood(model, rainfall, river_level, soil_moisture))
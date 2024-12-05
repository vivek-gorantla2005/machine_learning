import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (replace 'data.csv' with your actual file path)
data = pd.read_csv('heart.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Basic information about the dataset
print("\nDataset info:")
data.info()

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Correlation heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# # Distribution of numerical features
# numerical_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
# for column in numerical_columns:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(data[column], kde=True, bins=30)
#     plt.title(f'Distribution of {column}')
#     plt.show()

# # Count plot for categorical features
# categorical_columns = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']
# for column in categorical_columns:
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x=data[column])
#     plt.title(f'Count Plot of {column}')
#     plt.show()

# Load the data (replace 'data.csv' with your actual file path)

# Preprocessing
# Handling missing values (drop or impute)
data.dropna(inplace=True)  # Dropping rows with missing values for simplicity

# Features and target
X = data.drop(columns=['TenYearCHD'])  # Independent variables
y = data['TenYearCHD']  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

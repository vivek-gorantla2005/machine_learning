import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
uri = "D:/Vivek/machine_learning/Iris.csv"
dataset = pd.read_csv(uri)

# Print some basic information about the dataset
print("First few rows of the dataset:\n", dataset.head())
print("Count of samples by Species:\n", dataset.groupby('Species').size())

# Drop the 'Id' column (assuming it exists in the dataset)
dataset = dataset.drop(['Id'], axis=1)

# Plot histograms for all numeric columns
dataset.hist()
plt.show()

# Plot scatter matrix
scatter_matrix(dataset, c=dataset['Species'].astype('category').cat.codes, figsize=(10, 10), marker='o', alpha=0.8)
plt.show()

# Separate features and target variable
X = dataset.drop('Species', axis=1).values
y = dataset['Species'].astype('category').cat.codes.values

# Split the dataset into training and test sets
validationsize = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=validationsize, random_state=seed)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate models
for name, model in models.items():
    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Print model performance
    print(f"{name} Accuracy: {accuracy:.2f}")

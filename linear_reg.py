import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Load the dataset
df = pd.read_csv("ecommerce_dataset.csv")

# Features (Price and Discount) and Target (Final Price)
X = df[['Price (Rs.)', 'Discount (%)']]  # Independent variables
y = df['Final_Price(Rs.)']  # Dependent variable

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict the Final Price using the trained model
df['Predicted_Final_Price(Rs.)'] = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, df['Predicted_Final_Price(Rs.)'])
rmse = math.sqrt(mse)

# Print model coefficients and RMSE
print("Model Coefficients (Weights):", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error (RMSE):", rmse)
print('R squared value',model.score(X,y))

# Compare actual vs predicted final prices
print(df[['Price (Rs.)', 'Discount (%)', 'Final_Price(Rs.)', 'Predicted_Final_Price(Rs.)']].head())

# Plot the actual vs predicted prices for visualization
plt.figure(figsize=(10, 6))
plt.scatter(y, df['Predicted_Final_Price(Rs.)'], alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Final Price')
plt.xlabel('Actual Final Price (Rs.)')
plt.ylabel('Predicted Final Price (Rs.)')
plt.grid()
plt.show()

print(model.predict([[500,10]]))
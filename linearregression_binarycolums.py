import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from sklearn.linear_model import LinearRegression

# Retrieve and load the dataset
medical_charges_url = "https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
urlretrieve(medical_charges_url, "medical.csv")
medical_df = pd.read_csv("medical.csv")

# Map smoker column to numerical values
smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_codes'] = medical_df['smoker'].map(smoker_codes)

# Calculate correlation
correlation = medical_df['charges'].corr(medical_df['smoker_codes'])
print("Correlation between charges and smoker codes:", correlation)

# Plot scatter plot
# plt.scatter(medical_df['smoker_codes'], medical_df['charges'], alpha=0.8)
# plt.xlabel("Smoker Codes")
# plt.ylabel("Charges")
# plt.title("Scatter Plot of Charges vs. Smoker Codes")
# plt.show()

# Filter data for smokers and non-smokers
smokers = medical_df[medical_df['smoker'] == 'yes']
non_smokers = medical_df[medical_df['smoker'] == 'no']

# Plot histograms
# plt.hist(smokers['charges'], bins=30, alpha=0.7, label='Smokers')
# plt.hist(non_smokers['charges'], bins=30, alpha=0.7, label='Non-Smokers')
# plt.xlabel('Charges')
# plt.ylabel('Frequency')
# plt.title('Histogram of Medical Charges for Smokers and Non-Smokers')
# plt.legend()
# plt.show()

# Prepare inputs and targets for linear regression
inputs = medical_df[['age', 'smoker_codes']]
targets = medical_df['charges']

# Train the linear regression model
model = LinearRegression()
model.fit(inputs, targets)

# Prepare new data for predictions
age = int(input("enter your age"))
smoker =int(input("Do you smoke"))

# Create a DataFrame for new data
# new_data = pd.DataFrame({
#     'age': [age],
#     'smoker_codes': [smoker]
# })

# Make predictions
predicted_charges = model.predict([[age,smoker]])
print(f'Amount predicted for age and smoker status pairs {age}  is {predicted_charges}')


#do one hot encoding to convert region cloumns to indvidual 0 and 1 groups
# from sklearn import preprocessing
# enc = preprocessing.OneHotEncoder()
# enc.fit(medical_df[['region']])

# one_hot = enc.transform(medical_df[['region']]).toarray()
# medical_df[['northeast','nortwest','southeast','southwest']] = one_hot
# print(medical_df)

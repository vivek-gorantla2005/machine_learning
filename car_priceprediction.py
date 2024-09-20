#import the libraies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#load the dataset
car_dataset = pd.read_csv("D:/Vivek/machine_learning/car data.csv")

#checking the first 5rows and columns of the dataset
# print(car_dataset.head())

#getting the number of rows and columns of the dataset
# print(car_dataset.shape)

#getting information about the dataset 
# print(car_dataset.info())
# checing for the null values in the dataset
# print(car_dataset.isnull().sum())

#now checking the categorical data feilds 
# print(car_dataset.Fuel_Type.value_counts())
# print(car_dataset.Seller_Type.value_counts())
# print(car_dataset.Transmission.value_counts())

#now its very important to encode the categorical data feids to make the model effictient 
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_dataset.replace({'Seller_Type':{'Individual':0,'Dealer':1}},inplace=True)
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
# print(car_dataset.head())


#split data into train and  testdata sets
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)  #axis = 1 for removing a column axis = 0 fro removing a row
Y = car_dataset['Selling_Price']
print(X)
# print(Y)

#splitting data Training and Testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1 ,random_state=2)

#model training 
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

#model evaluation (prediction on trainging data)
training_data_prediction = lin_reg_model.predict(X_train)

#rmse error
error_score = metrics.r2_score(Y_train,training_data_prediction)
print(error_score)

def get_user_input():
    present_price = float(input("Enter the present price of the car: "))
    kms_driven = float(input("Enter the number of kilometers driven: "))
    owner = int(input("Enter the number of owners (0 for no previous owners, 1 for one previous owner, etc.): "))
    age = int(input("Enter the age of the car in years: "))
    fuel_type = input("Enter the fuel type (Petrol, Diesel, CNG): ")
    seller_type = input("Enter the seller type (Individual, Dealer): ")
    transmission = input("Enter the transmission type (Manual, Automatic): ")

    # Encoding the categorical variables
    fuel_type_encoded = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}.get(fuel_type, 0)
    seller_type_encoded = {'Individual': 0, 'Dealer': 1}.get(seller_type, 0)
    transmission_encoded = {'Manual': 0, 'Automatic': 1}.get(transmission, 0)

    user_data = [[present_price, kms_driven, owner, age, fuel_type_encoded, seller_type_encoded, transmission_encoded]]
    return user_data

user_input = get_user_input()

# Predicting the selling price based on user input
predicted_price = lin_reg_model.predict(user_input)
print(f"Predicted Selling Price: {predicted_price[0]}")


#plotting graph of actual prices vs predicted prices
# Plotting graph of actual prices vs predicted prices
plt.scatter(Y_train, training_data_prediction, color='blue', label='Training data')
plt.scatter([0], predicted_price[0], color='red', label='User prediction', marker='o')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.legend()
plt.show()





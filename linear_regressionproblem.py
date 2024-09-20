from urllib.request import urlretrieve
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

medical_charges_url = "https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
urlretrieve(medical_charges_url,"medical.csv")
medical_df = pd.read_csv("medical.csv")
# print(medical_df.info())
# print(medical_df.describe())
# sns.set_style('darkgrid')
# matplotlib.rcParams['font.size'] =14
# matplotlib.rcParams['figure.figsize'] =(10,6)
# matplotlib.rcParams['figure.facecolor'] ="#00000000"

# medical_df.age.describe()

# fig =px.histogram(medical_df,marginal='box',x="age",nbins=47,title="Distributon of age")
# fig.update_layout(bargap=0.1)
# fig.show()

# fig = px.histogram(medical_df,marginal='box',color_discrete_sequence=['red'],x="bmi",nbins=47,title="Distribution of BMI")
# fig.update_layout(bargap=0.1)
# fig.show()

# fig = px.histogram(medical_df,marginal='box',x="smoker",color='sex',color_discrete_sequence=['pink','gray'],title="Annual medical charges")
# fig.update_layout(bargap=0.1)
# fig.show()

#visulaze data based on age and charges
# fig = px.scatter(medical_df,x='age',y='charges',color="smoker",opacity=0.8,hover_data=['sex'],title="Age vs charges")
# fig.update_traces(marker_size = 5)

# fig.show()



numeric_df = medical_df.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# ..................................
# Display the correlation matrix
# print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="Reds", annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# ....................................

#now lets plot the relation between non smoker age and charges as age and charges have highest correlation
non_smoker_df = medical_df[medical_df.smoker == 'no']
print(non_smoker_df)
plt.title("Age vs Charges")
fig = px.scatter(non_smoker_df, x = 'age', y = 'charges',color='age')
fig.show()

def estimate_chareges(age,w,b):
    return w * age + b
 
def try_parameters(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    estimated_charges = estimate_chareges(ages,w,b)
    plt.plot(ages,estimated_charges,'r',alpha=0.9)
    plt.scatter(ages,target,alpha=0.8)
    plt.xlabel('ages')
    plt.ylabel('charges')
    plt.legend(['Estimatedcharges', 'Actual charges'])
    plt.show()

def rmse(actual_charges, predicted_charges):
    return np.sqrt(np.mean((actual_charges - predicted_charges)**2))

try_parameters(300,-4000)

# predicted_charges = estimate_chareges(non_smoker_df.age,350,-4000)
# actual_charges = non_smoker_df.charges
# print("RMSE:", rmse(actual_charges, predicted_charges))

# ......................
#using Scikit learn for linear regression
#the SDERegressor class from scikitlearn is used to train the  model using stochastic gradient descent technique
# ..................
inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
age = np.array([20,25,30,45,60]).reshape(-1,1)
model = LinearRegression()
model.fit(inputs,targets)
predections = model.predict(inputs)
predicted_charges = model.predict(age)
print(f'amount predicted for {age} is {predicted_charges}')
plt.plot(age,predicted_charges,'r',alpha=0.9)
plt.scatter(inputs,targets,alpha=0.8)
plt.xlabel('ages')
plt.ylabel('charges')
plt.legend(['Estimatedcharges', 'Actual charges'])
plt.show()
print(rmse(targets,predections))
# displays the slope of the line(x)
# print(model.coef_)  
# displays the  intersept of the line(c)
# print(model.intercept_) 


# plt.scatter(non_smoker_df.bmi,non_smoker_df.charges,alpha=0.9)
# plt.xlabel('BMI')
# plt.ylabel('charges')
# plt.show()


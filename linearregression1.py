import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

toss = np.array([10,20,30,40,50,60,70,80]).reshape(-1,1)
heads = np.array([5,17,4,12,2,30,37,60])

model = LinearRegression()

model.fit(toss,heads)

num = 340
predict_heads = model.predict([[num]]);
print(predict_heads);


plt.scatter(toss,heads,color= "black");
plt.plot(toss,model.predict(toss),color = "red")
plt.title("Heads vs Tails")
plt.xlabel('Toss')
plt.ylabel('Heads')
plt.show()

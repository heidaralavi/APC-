"""
By H.Alavi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# reading data from excel
mydata=pd.read_excel('myexceldata.xlsx')
y=mydata.values[:,4]
x = mydata.values[:,0:4]
print(mydata)

# plotting matx of data
pd.plotting.scatter_matrix(mydata, c=y,s=150,figsize=[11,11])

# training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(x_test)
print(y_predict)
print(y_test)
plt.scatter(y_test,y_predict)
plt.xlabel('Real Value')
plt.ylabel('predictet Value')
plt.show()

# new data and answer
sample=np.array([[6.3,2.7,4.9,2.5]])
reg.fit(x,y)
y_predict=reg.predict(sample)
print(y_predict)
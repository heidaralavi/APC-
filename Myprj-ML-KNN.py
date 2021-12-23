"""
By H.Alavi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# reading data from excel
mydata=pd.read_excel('myexceldata.xlsx')
y=mydata.values[:,4]
x = mydata.values[:,0:4]
print(mydata)

# plotting matrix of data
#pd.plotting.scatter_matrix(mydata, c=y,s=150,figsize=[11,11])

# training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)
knn=KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=2)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print(x_test,y_predict,y_test)
plt.scatter(y_test,y_predict)
plt.xlabel('Real Value')
plt.ylabel('predictet Value')
plt.show()

# new data and answer
sample=np.array([[5,3,1,0.2]])
knn.fit(x,y)
y_predict=knn.predict(sample)
print(y_predict)

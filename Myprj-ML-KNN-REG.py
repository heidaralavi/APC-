"""
By H.Alavi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score

# reading data from CSV
inputdata=pd.read_csv(r'C:\Users\h.alavi\Documents\GitHub\APC-PRJ\input-data.csv')
outputdata=pd.read_csv(r'C:\Users\h.alavi\Documents\GitHub\APC-PRJ\output-data.csv')

x= inputdata.iloc[:,:].values
y=outputdata.iloc[:,9].values
#print(outputdata)


# plotting matrix of data
#pd.plotting.scatter_matrix(inputdata, c='blue',s=25,figsize=[25,25])
#pd.plotting.scatter_matrix(outputdata, c='red',s=25,figsize=[25,25])


corr = outputdata.corr() #OR
#corr = inputdata.corr()
print(corr.shape)
print(corr)
plt.figure(figsize=(11,11))
sns.heatmap(corr, cbar=False,square= True, fmt='.2f', annot=True, annot_kws={'size':10}, cmap='Greens')



# training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)
knn=KNeighborsRegressor(n_neighbors=7,metric='minkowski',p=2)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
y_predict_train=knn.predict(x_train)
#print(x_test,y_predict,y_test)
plt.scatter(y_test,y_predict)
plt.xlabel('Real Value')
plt.ylabel('predictet Value')
plt.show()

print("The Predict Score is ", (r2_score(y_predict, y_test)))
print("The Train Score is: ", (r2_score(y_predict_train, y_train)))


'''
# new data and answer
#sample=np.array([[5,3,1,0.2]])
#knn.fit(x,y)
#y_predict=knn.predict(sample)
#print(y_predict)
'''
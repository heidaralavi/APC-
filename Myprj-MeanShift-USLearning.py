"""
By H.Alavi
"""
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
y=mydata.values[:,26]
x = mydata.values[:,0:27]
print(mydata)

#reading header 
col=mydata.columns.ravel()

#clustering data
ms=MeanShift(bandwidth=40)
ms.fit(x)
labels=ms.labels_
#labels=ms.predict(x)    #OR
print(labels)

#ploting one sample of data with center of cluster
center=ms.cluster_centers_
for jj in range(10,27):
    plt.figure(figsize=(20,20),dpi=85)
    for kk in range(10):
        first=kk
        secound=jj
        plt.subplot(4,4,kk+1)
        plt.scatter(x[:,first],x[:,secound],c=labels)
        plt.xlabel(col[first])
        plt.ylabel(col[secound])
        plt.scatter(center[:,first],center[:,secound],marker='x',c='red',s=50)
        plt.title(col[secound])
    plt.show()
plt.show()

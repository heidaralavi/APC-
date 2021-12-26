"""
By H.Alavi
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# reading data from excel
mydata=pd.read_excel('data-norm.xlsx')
y=mydata.values[:,9]
x = mydata.values[:,0:27]
print(x)
'''
# plotting matrix of data
plt.figure(figsize=(20,20),dpi=85)
pd.plotting.scatter_matrix(mydata, c='blue',alpha=0.35, s=55,figsize=[26,26])
plt.show()
'''
#reading header 
col=mydata.columns.ravel()

#clustering data
kmn=KMeans(n_clusters=4)
kmn.fit(x)
labels=kmn.predict(x)
print(labels)

#ploting one sample of data with center of cluster
center=kmn.cluster_centers_
'''
for jj in range(10,27):
    plt.figure(figsize=(20,20),dpi=85)
    for kk in range(10):
        first=kk
        secound=jj
        plt.subplot(3,4,kk+1)
        plt.scatter(x[:,first],x[:,secound],c=labels)
        plt.xlabel(col[first])
        plt.ylabel(col[secound])
        plt.scatter(center[:,first],center[:,secound],marker='x',c='red',s=150)
        plt.title(col[secound])
    plt.show()
plt.show()
'''

plt.scatter(x[:,0],x[:,10],c='blue')
plt.xlabel(col[0])
plt.ylabel(col[10])
plt.show()



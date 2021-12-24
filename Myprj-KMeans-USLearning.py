"""
By H.Alavi
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
y=mydata.values[:,26]
x = mydata.values[:,0:26]
print(mydata)

#reading header 
col=mydata.columns.ravel()

#clustering data
kmn=KMeans(n_clusters=5)
kmn.fit(x)
labels=kmn.predict(x)
print(labels)

#ploting one sample of data with center of cluster
center=kmn.cluster_centers_

first=0
secound=10
plt.scatter(x[:,first],x[:,secound],c=labels)
plt.xlabel(col[first])
plt.ylabel(col[secound])
plt.scatter(center[:,first],center[:,secound],marker='x',c='red',s=50)
plt.show()

"""
#calculating inertia of cluster 
inery=[]
for k in np.arange(1,16):
    kmn=KMeans(n_clusters=k)
    kmn.fit(x)
    inery.append(kmn.inertia_)

#ploting inertia
plt.plot(np.arange(1,16),inery,'o-')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.show()
"""
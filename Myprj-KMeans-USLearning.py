"""
By H.Alavi
"""
import pandas as pd
import numpy as np
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
kmn=KMeans(n_clusters=4)
kmn.fit(x)
labels=kmn.predict(x)
print(labels)

#ploting one sample of data with center of cluster
center=kmn.cluster_centers_

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
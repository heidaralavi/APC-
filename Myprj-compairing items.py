"""
By H.Alavi
"""

import pandas as pd

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
y=mydata.values[:,26]
x = mydata.values[:,0:26]
print(mydata)

# plotting matrix of data
pd.plotting.scatter_matrix(mydata, c='blue',alpha=0.1, s=25,figsize=[25,25])

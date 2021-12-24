"""
By H.Alavi
"""

import pandas as pd
from sklearn.preprocessing import scale , normalize, minmax_scale

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
y=mydata.values[:,26]
x = mydata.values[:,0:27]
print(mydata)

col=mydata.columns.ravel()

mydata_all=mydata.values
mydata_normal=normalize(mydata_all,norm='l1',axis=0)

mydata=pd.DataFrame(mydata_normal,columns=col)
print(mydata)

mydata.to_excel("data-norm.xlsx",sheet_name='Sheet_name_1')
"""
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

"""
"""
By H.Alavi
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
y=mydata.values[:,26]
x = mydata.values[:,0:26]
print(mydata)

# plotting matrix of data
#pd.plotting.scatter_matrix(mydata, c='blue',alpha=0.35, s=25,figsize=[26,26])

#Plotting Correlation Coefficient 
corr = mydata.corr()
print(corr.shape)
print(corr)
plt.figure(figsize=(11,11))
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens')
"""
By H.Alavi
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# reading data from excel
mydata=pd.read_excel('data-feo.xlsx')
#print(mydata.columns)
x = mydata.iloc[:, :-1].values
y = mydata.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.transform(y_train)
#print(X_test)



classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

'''
y=mydata.values[:,26]
x = mydata.values[:,0:26]
print(mydata)
'''




'''
# plotting matrix of data
#pd.plotting.scatter_matrix(mydata, c='blue',alpha=0.35, s=25,figsize=[26,26])

#Plotting Correlation Coefficient 
corr = mydata.corr()
print(corr.shape)
print(corr)
plt.figure(figsize=(11,11))
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens')
'''
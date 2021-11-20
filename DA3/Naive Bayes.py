import numpy as np
import pandas as pd
from sklearn import preprocessing

balance_data = pd.read_csv('dataNB.csv',sep= ',', header = None, skiprows=1, names=['Outlook','Temperature','Humidity','Wind','PlayTennis'])
print(balance_data.head())

balance_data.columns

le = preprocessing.LabelEncoder()

list = []
X = balance_data.loc[:, ['Outlook', 'Temperature', 'Humidity', 'Wind']]
for column in X:
    X[column] = le.fit_transform(X[column])
    #print(balance_data[column])
    list.append(dict(zip(le.classes_, range(len(le.classes_)))))

print(X)

Y = balance_data.loc[:, ['PlayTennis']]
Y = Y.apply(le.fit_transform)
list.append(dict(zip(le.classes_, range(len(le.classes_)))))
print("\n\nthis is Y:\n",Y,"\n")

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X,Y)

x_test = [[list[0]['Sunny'],list[1]['Cool'],list[2]['High'],list[3]['Strong']]] #Getting label encodings for Sunny, Cool, High, Strong parameters

testdata = pd.DataFrame(columns=['Outlook', 'Temperature', 'Humidity', 'Wind'], data=x_test)

print(testdata)

y_pred = model.predict(testdata)

print("PlayTennis? ",le.inverse_transform(y_pred))
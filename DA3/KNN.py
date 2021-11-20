import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

balance_data = pd.read_csv('dataKNN.csv',sep= ',', header = None, skiprows=1, names=['P1','P2','Class'])
print(balance_data.head())

X = balance_data.iloc[:, :-1].values
y = balance_data.iloc[:, -1].values
print(X,"\n")
print(y)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

y_pred = print("prediction for p1=3, and p2=7: ", classifier.predict([[3,7]]))

print(y_pred)
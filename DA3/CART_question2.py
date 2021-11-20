import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import classification_report

balance_data = pd.read_csv('Cartdata.csv',sep= ',', header = None, skiprows=1, names=['Age','Job','House','Credit','Loan Approved'])
print(balance_data.head())

balance_data.columns

le = preprocessing.LabelEncoder()

list = []
X = balance_data.loc[:, ['Age', 'Job', 'House', 'Credit']]
for column in X:
    X[column] = le.fit_transform(X[column])
    list.append(dict(zip(le.classes_, range(len(le.classes_)))))

print(X)

Y = balance_data.loc[:, ['Loan Approved']]
Y = Y.apply(le.fit_transform)
list.append(dict(zip(le.classes_, range(len(le.classes_)))))
print("\n\nthis is Y:\n",Y,"\n")

clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 12,max_depth=3, min_samples_leaf=3)

clf_gini.fit(X, Y)

x_test = [[list[0]['Young'],list[1][False],list[2]['No'],list[3]['Good']]] #Getting label encodings for Young, FALSE, No, Good parameters

testdata = pd.DataFrame(columns=['Age', 'Job', 'House', 'Credit'], data=x_test)

print(testdata)

y_pred = clf_gini.predict(testdata)
print("Prediction for Age=Young, Job=False, House=No, Credit=Good: ",le.inverse_transform(y_pred))



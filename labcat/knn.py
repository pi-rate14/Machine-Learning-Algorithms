import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

df1 = pd.read_csv('data.csv')

X=df1.drop(labels='Class',axis=1)

y=df1['Class']



Xs = scale(X)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)
knn_model_2 = knn.fit(Xs_train, y_train)
print('k-NN score for test set: %f' % knn_model_2.score(Xs_test, y_test))
print('k-NN score for training set: %f' % knn_model_2.score(Xs_train, y_train))
y_true, y_pred = y_test, knn_model_2.predict(Xs_test)
print(classification_report(y_true, y_pred))
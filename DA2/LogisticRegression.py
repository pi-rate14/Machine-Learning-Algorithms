import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("exam_results.csv")
print("Data head: ")
print(df.head())

plt.scatter(df.hours, df.result, marker="+", color="red")

x_train, x_test, y_train, y_test = train_test_split(df[['hours']],df.result, test_size=.30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, 
fit_intercept=True, intercept_scaling=1, l1_ratio=None, 
max_iter=100, multi_class='auto', n_jobs=None, penalty='12', 
random_state=None, solver='lbfgs', tol=0.0001, verbose=0, 
warm_start=False)


y_predicted = model.predict(x_test)

print("model coefficient", model.coef_)

print("model intercept", model.intercept_)

#DOING MANUAL CALCULATION

import math
def sigmoid(x):
    return 1/( 1 + math.exp(-x) )

def prediction_function(hours):
    z = model.coef_* hours + model.intercept_
    # z = mx+c
    y = sigmoid(z)
    return y

hours = 33
print("probability of student passing after 33 hours of study: ", prediction_function(hours))
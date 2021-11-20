import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Glucose_Data.csv')
x = data['Age']
y = data['Glucose']
print(data.head())
def linear_regression(x, y):    
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
   
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
   
    B0 = y_mean - (B1*x_mean)
   
    reg_line = 'y = {} + {}x'.format(B0, round(B1, 3))
   
    return (B0, B1, reg_line)
N = len(x)
x_mean = x.mean()
y_mean = y.mean()
B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den
B0 = y_mean - (B1 * x_mean)
def corr_coef(x, y):
    N = len(x)
   
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R
B0, B1, reg_line = linear_regression(x, y)
print("Intercept: ", B1)
print("Slope: ", B0)
print('Regression Line: ', reg_line)
def predict(new_x):
    y = B0 + B1 * new_x
    return y
print("Glucose at age 55 years: ", predict(55))
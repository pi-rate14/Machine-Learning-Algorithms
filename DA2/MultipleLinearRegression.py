import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("homeprices.csv")

print("Median: ", df.bedrooms.median())

median_bedrooms = math.floor(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(median_bedrooms)
#null values in df are filled with median

reg = linear_model.LinearRegression()
#fit used to train the model
reg.fit(df[['area','bedrooms','age']],df.price)

print("Regression Coefficients: ", reg.coef_)
print("Regression Intercept: ", reg.intercept_)

# 3000 sqr ft area, 3 bedrooms, 40 year old
print("Price of 3000 sqr ft area, 3 bedrooms, 40 year old House: ", reg.predict([[3000,3,40]]))
# 2500 sqr ft area, 4 bedrooms, 5 year old
print("Price of 2500 sqr ft area, 4 bedrooms, 5 year old House:", reg.predict([[2500,4,5]]))
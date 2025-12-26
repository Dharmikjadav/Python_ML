import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("room.csv")
print(df)

a = int(df.bedrooms.median())

df.bedrooms.fillna(a,inplace=True)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']] , df.price)
print(reg.coef_)
print(reg.predict([[4200,5.0,8]]))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("hotel_bookings.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df[['price']])

plt.scatter(df.area , df.price)
plt.plot(df.area , reg.predict(df[['area']]))
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

df1 = pd.read_csv("home.csv")
p = reg.predict(df1)
df1['price'] = p

df1.to_csv("home.csv" , index=False)

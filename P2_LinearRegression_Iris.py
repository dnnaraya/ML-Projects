#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:16:53 2024

@author: dhivyanarayanan
"""

#iris dataset
# It includes 3 iris species with 50 samples each
# 4 features
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm

#Find the regression b/w petal length and petal width of class 2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import datasets

iris_data = datasets.load_iris()

# as it is for class 2 - samples from 50 to 100 has been given for input and the column 2 is petallength
# output variable - column 3 - petalwidth

X = iris_data.data[50:100, 2:3]
y = iris_data.data[50:100, 3]

# as the samples are less, no need to do train test split

reg = LinearRegression()
reg.fit(X,y)


yPred = reg.predict(X)

print("Intercept : ", reg.intercept_)
print("Coefficients : ", reg.coef_)

mse = mean_squared_error(y, yPred)
r2 = r2_score(y, yPred)

print("MSE = ", mse)
print("R2 score = ", r2)

plt.figure()
plt.scatter(y, yPred, color = 'blue', alpha = 0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted values - For Iris Species 2")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.grid()
plt.show()




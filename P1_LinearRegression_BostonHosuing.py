#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:55:40 2024

@author: dhivyanarayanan
"""

#BostonHousing

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Boston Housing Dataset
# 11 Features (column - 0 to 10)
# Target - medv (column - 13)

bosData = pd.read_csv("BostonHousing.csv")

X = bosData.iloc[:,0:11] # all samples/rows from col 0 to 10
Y = bosData.iloc[:,13] # all samples from col 13

# train test split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=5)

reg = LinearRegression()
reg.fit(Xtrain, Ytrain)

Ytrain_pred = reg.predict(Xtrain)

mse_train = mean_squared_error(Ytrain, Ytrain_pred )
r2_train = r2_score(Ytrain, Ytrain_pred )

print("Train MSE = ", mse_train)
print("Train R2 score = ", r2_train)


# Plot the train values

plt.figure()
plt.scatter(Ytrain, Ytrain_pred, color='blue', alpha = 0.6)
plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Train values")
plt.xlabel("Actual train values")
plt.ylabel("Predicted train values")
plt.grid()
plt.show()



Ytest_pred = reg.predict(Xtest)

mse_test = mean_squared_error(Ytest, Ytest_pred )
r2_test = r2_score(Ytest, Ytest_pred )

print("Test MSE = ", mse_test)
print("Test R2 score = ", r2_test)


# Plot the train values

plt.figure()
plt.scatter(Ytest, Ytest_pred, color='blue', alpha = 0.6)
plt.plot([Ytest.min(), Ytest.max()], [Ytest.min(), Ytest.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Test values")
plt.xlabel("Actual test values")
plt.ylabel("Predicted test values")
plt.grid()
plt.show()



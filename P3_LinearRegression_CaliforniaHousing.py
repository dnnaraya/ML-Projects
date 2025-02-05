#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:29:06 2024

@author: dhivyanarayanan
"""

# California Housing - Fetch from sklearn.datasets
# Has 8 Features
# Target - MedHouseVal
# Total no.of pts = 20640



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import datasets


california_data = datasets.fetch_california_housing()

california_feature_names = california_data.feature_names
california_target = california_data.target_names

print("Feature names : ", california_feature_names)
print("Target name : ", california_target)

X = california_data.data
y = california_data.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=5)

reg = LinearRegression()
reg.fit(Xtrain, ytrain)

ytrain_Pred = reg.predict(Xtrain)

mse_train = mean_squared_error(ytrain,ytrain_Pred)
r2_train = r2_score(ytrain, ytrain_Pred)

print("Train MSE = ", mse_train)
print("Train R2 = ", r2_train)


ytest_Pred = reg.predict(Xtest)

mse_test = mean_squared_error(ytest,ytest_Pred)
r2_test = r2_score(ytest,ytest_Pred)

print("Test MSE = ", mse_test)
print("Test R2 = ", r2_test)

plt.figure()
plt.scatter(ytest, ytest_Pred, color='blue', alpha = 0.6)
plt.plot([ytest.min(), ytest.max()],[ytest.min(), ytest.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Test values")
plt.xlabel("Actual MedHouseVal - Test")
plt.ylabel("Predicted MedHouseVal - Test")
plt.grid()
plt.show()





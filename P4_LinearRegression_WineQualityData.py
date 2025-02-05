#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:55:56 2024

@author: dhivyanarayanan
"""

#Wine quality dataset 
# 11 Features
# Target - Quality - col 11
# 4898 instances

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Fetch a dataset by id

wineQualityData = fetch_ucirepo(id=186)

# access variable info in tabular format
#print(wineQualityData.variables)



#access data (as pandas dataframes)

X = wineQualityData.data.features
y = wineQualityData.data.targets

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.2, random_state = 2)

reg = LinearRegression()
reg.fit(Xtrain, ytrain)

ytrain_pred = reg.predict(Xtrain)

mse_train = mean_squared_error(ytrain, ytrain_pred)
r2_train = r2_score(ytrain, ytrain_pred)

print("Train MSE = ", mse_train)
print("Train R2 score = ", r2_train)

ytest_pred = reg.predict(Xtest)

mse_test = mean_squared_error(ytest, ytest_pred)
r2_test = r2_score(ytest, ytest_pred)

print("Test MSE = ", mse_test)
print("Test R2 score = ", r2_test)


plt.figure()
plt.scatter(ytest, ytest_pred, color="blue", alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color="red", linestyle = '--')
plt.title("Actual vs Predicted Wine Quality")
plt.xlabel("Actual wine quality - Test")
plt.ylabel("Predicted wine Quality - test")
plt.grid()
plt.show()






# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:28:26 2020

@author: Tanvi Kulkarni
"""

import numpy as np  #work with arrays and maths
import matplotlib.pyplot as plt #plyplot is one module of matplotlib, is used to graph data
import pandas as pd #import dataset

dataset = pd.read_csv('SkinCancerData.csv') #dataframe
dependentVar = dataset.iloc[:,2].values  #The value to be predicted/calculated
independentVar = dataset.iloc[:,1:2].values #Matrix of features.

'''splitting the data into training and test set'''

from sklearn.model_selection import train_test_split
iv_train, iv_test, dv_train, dv_test = train_test_split(independentVar, dependentVar, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(iv_train, dv_train)

'''predict funtion expects a 2D array as input, therefore if we want to predict a singular value
we pass it as [[value]] , this represents a 2D array.'''
'''
Also we can get the coefficient of regression and intercept with the following functions : regressor.coef_, regressor.intercept
'''

predicted_values = regressor.predict(iv_test)

'''
plt.scatter(iv_train, dv_train, color = 'red')
plt.plot(iv_train, regressor.predict(iv_train), color = 'blue')
plt.title('Salary vs Exp - Training set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

plt.scatter(iv_test, dv_test, color = 'red')
plt.plot(iv_test, predicted_values, color = 'blue')
plt.title('Salary vs Exp - Test set')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()
'''

plt.scatter(iv_train, dv_train, color = 'red')
plt.plot(iv_train, regressor.predict(iv_train), color = 'blue')
plt.title('Latitude vs Mortality - Training set')
plt.xlabel('Latitude')
plt.ylabel('Mortality')
plt.show()

plt.scatter(iv_test, dv_test, color = 'red')
plt.plot(iv_test, predicted_values, color = 'blue')
plt.title('Latitude vs Mortality - Test set')
plt.xlabel('Latitude')
plt.ylabel('Mortality')
plt.show()
import numpy as np  #work with arrays and maths
import matplotlib.pyplot as plt #plyplot is one module of matplotlib, is used to graph data
import pandas as pd #import dataset

dataset = pd.read_csv('Data.csv') #dataframe
dependentVar = dataset.iloc[:,3].values  #The value to be predicted/calculated
independentVar = dataset.iloc[:,:-1].values #Matrix of features.

#In python [:,:] specifies range so, [rows,columns]. No number specifies all the rows/columns

print(dependentVar)
print(independentVar)

#Taking care of missing data
#Missing data causes errors when training you ML model
#1st way - delete the missing data rows. (if the amount of missing data is small.)
#2nd way - Average of the column, Median of the column

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#Fit method will find the mean of the columns of missing values
#upper bound of a range in python is excluded
imputer.fit(independentVar[:,1:3])
#transform method will actually replace all the missing values.
#transform method returns the updated matrix of features
independentVar[:, 1 : 3] = imputer.transform(independentVar[:, 1 : 3])

print(dependentVar)
print(independentVar)

#encoding categorical data
#Two types of encoding - 
#1) Integer encoding - Here the variables have a natural ordering between them - Ordinal variables
#2) one-hot encoding - here the variables dont have a order between them. If we use integer encoding here the model may assume
#a natural ordering resulting in wrong output. Therefore binary representations are used.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#transformers arguments - type of transformation, class of transformation, columns to which transformation is to be applied
#remainder = keep the columns to which transformation is not applied
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
independentVar = np.array(ct.fit_transform(independentVar))

print(independentVar)

#encode dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dependentVar = le.fit_transform(dependentVar)

print(dependentVar)

#Feature Scaling
#Putting numerical values in the same range
#If some features havemuch higher values in comparison with other features - this can create bias in correlation computations.
#Linear regression automatically compensates for higher values of features
#Logistic regression and SVR requires feature scaling
#Two types of scaling 1)standard and 2)normalization - https://en.wikipedia.org/wiki/Feature_scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
independentVar = sc.fit_transform(independentVar)

print(independentVar)

#splitting the data into training and test set
from sklearn.model_selection import train_test_split
#train_test_split is a function -- it returns four elements - matrix of features of training set, matrix of features test set, dependent var of training set and dependent var of test set.
iv_train, iv_test, dv_train, dv_test = train_test_split(independentVar, dependentVar, test_size = 0.2, random_state = 0)
print('Independent var training - ',iv_train)
print('Independent var test - ',iv_test)
print('Dependent var training - ',dv_train)
print('Dependent var test - ',dv_test)








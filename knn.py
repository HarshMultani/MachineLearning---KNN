# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:14:10 2019

@author: 138709
"""

# K - Nearest Neighbors

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values


# One - Hot Encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

gender = X[:,1]
gender = gender.reshape(-1,1)


from sklearn.preprocessing import OneHotEncoder
onehotEncoder = OneHotEncoder(sparse = False)
gender = onehotEncoder.fit_transform(gender)
X = np.append(X, gender, axis = 1)
X = X[:,[2,3,4]]


# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)


 # Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Fitting KNN Model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)


# Predicting the model
Y_pred = model.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


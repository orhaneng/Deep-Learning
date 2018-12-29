#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:41:56 2018

@author: omerorhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#missing data
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:,1] =labelencoder_X_1.fit_transform(x[:,1])
labelencoder_X_2 = LabelEncoder()
x[:,2] =labelencoder_X_2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1] )
x =onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test =train_test_split(x,y,test_size=0.2, random_state=0)

'''
feature scalling
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Part 2 now lets make the ANN
#importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#initializing the ANN
classifier = Sequential()
#adding the input layer and first hidden layer
#independent variable =11 
#11+1/2 = 6 ß 
#first layer we have to identify the input_dim
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu',input_dim=11))
#adding the second hidden later
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu'))
#adding the output layer
classifier.add(Dense(output_dim = 1,init='uniform',activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann to the traning set
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
#predicting the test results
y_pred = classifier.predict(x_test)
y_pred =(y_pred >0.5)


#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

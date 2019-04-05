#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:23:42 2019

@author: abhijeet
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Defining Training and Testing set !!

directory = '/home/abhijeet/abhijeet/kaggle/MNIST/digit-recognizer/'
train_df = pd.read_csv(directory +'train.csv')
test_df = pd.read_csv(directory + 'test.csv')

train = train_df.values
test = test_df.values

X_train = train[: , 1:]
y_train = train[:, :1].ravel()
X_test = test

print(X_test.shape)
print(X_train.shape)

# Training of the features !!
# Using Keras Neural Network !!
# Keras
from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim = 784))
#Second  Hidden Layer
#classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(10, activation='softmax', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='RMSprop',loss='sparse_categorical_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=20, epochs= 100 )


#y_hat_xgb = model_xgb.predict(X_test)
y_hat_keras = classifier.predict(X_test)
sample_submission = []
for i in range(len(y_hat_keras)):
    sample_submission.append((i+1 , np.argmax(y_hat_keras[i])))
    
df = pd.DataFrame(columns = ['ImageId' , 'Label'] , data =  sample_submission)


df.to_csv(directory + 'sample_submission_Keras_sequential.csv', index = None)
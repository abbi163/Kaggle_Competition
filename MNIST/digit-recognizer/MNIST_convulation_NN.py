#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:03:25 2019

@author: abhijeet
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
import os
import matplotlib.pyplot as plt
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# mnist = keras.datasets.mnist

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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

# Convulation neural network..
X_train = X_train.reshape(42000 , 28, 28, 1)
X_test  = X_test.reshape(28000 , 28, 28, 1)
y_train = y_train.reshape(42000 , 1)

# One hot encoding !!

from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Create Model 

model = Sequential()
# add more layers !

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model !

#train the model
model.fit(X_train, y_train,  epochs=10)

y_hat_conv = model.predict(X_test)
sample_submission = []
for i in range(len(y_hat_conv)):
    sample_submission.append((i+1 , np.argmax(y_hat_conv[i])))
    
df = pd.DataFrame(columns = ['ImageId' , 'Label'] , data =  sample_submission)

df.to_csv(directory + 'sample_submission_Keras_conv.csv', index = None)

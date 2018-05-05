#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:23:58 2018

@author: chenrui
"""
from __future__ import print_function
import csv
import pandas as pd


import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation
from sklearn import preprocessing

data = pd.read_csv("text_traindata.csv")
test = pd.read_csv("test.csv")

train_data = data['data'][:int(len(data)*0.8)]
train_label = data['Labels'][:int(len(data)*0.8)]

test_data = data['data'][int(len(data)*0.8):]
test_label = data['Labels'][int(len(data)*0.8):]
################# hyperparameters
batch_size = 2048
num_labels = 5
################# create the feature vectors
vocab_size = 1000
tokenize = Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_data)

x_train = tokenize.texts_to_matrix(train_data)
x_test = tokenize.texts_to_matrix(test_data)
################ make one-hot labels
encoder = preprocessing.LabelBinarizer()
encoder.fit(train_label)
y_train = encoder.transform(train_label)
y_test = encoder.transform(test_label)
################ define the model
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
############### train and evaluate
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=10, 
                    verbose=1, 
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, 
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
############## predict on our data
test_data = pd.read_csv("test_data_text.csv")
x_test = test_data['data']
# change to 1000 vector
x_test = tokenize.texts_to_matrix(x_test)
result = []
for i in range(len(x_test)):    
    prediction = model.predict(np.array([x_test[i]]))
    text_labels = encoder.classes_ 
    predicted_label = text_labels[np.argmax(prediction[0])]
    result.append(predicted_label)
test['Score'] = result
test.to_csv("Summary_cnn_1024.csv", sep=',',index=False)

from keras.utils import plot_model
import pydot

plot_model(model, to_file='model.png')
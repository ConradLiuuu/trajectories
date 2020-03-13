#!/usr/bin/env python3
## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding
import tensorflow as tf

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# read dataset
dataset = pd.read_csv('/home/lab606a/ML/trajectories/fixed/classification/dataset20200311_5balls.csv', header=None)
dataset = np.array(dataset)

# set x train
x_train = dataset[:,1:]
row = int(x_train.shape[1]/3)
col = 3
x_train = x_train.reshape(x_train.shape[0],row,col)
x_train = x_train.astype('float32')

#set y train
y_train = dataset[:,0]
y_train = y_train.astype('int')

# 4 direction balls
n_classes = 4

# onehot encoding
y_train = np_utils.to_categorical(y_train, 4)

# define model
model = Sequential()
model.add(LSTM(units=512, activation='tanh', input_shape=(row,col) , unroll=True, return_sequences=True))
model.add(LSTM(units=256, activation='tanh', unroll=True, return_sequences=True))
model.add(LSTM(units=128, activation='tanh', unroll=True, return_sequences=True))
model.add(LSTM(units=64, activation='tanh', unroll=True, return_sequences=True))
model.add(LSTM(units=32, activation='tanh', unroll=True, return_sequences=True))
model.add(LSTM(units=16, activation='tanh', unroll=True, return_sequences=True))
model.add(LSTM(units=8, activation='tanh', unroll=True))
model.add(Dense(units=n_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
batch_size = 400
training_iters = 500
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=training_iters, shuffle=True)

# plot accuracy history
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./training history/5balls_accuracy.png')

# clear plot
plt.clf()

# plot loss history
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./training history/5balls_loss.png')

# saved model
model.save('./saved model/classifier_fixed_5balls')


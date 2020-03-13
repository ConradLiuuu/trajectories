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

num_of_ball = 11

# read dataset
dataset_path = '/home/lab606a/ML/trajectories/not_fixed/classification/dataset20200311_' + str(num_of_ball) + 'balls.csv'
dataset = pd.read_csv(dataset_path, header=None)
dataset = np.array(dataset)

# set x train
x_train = dataset[:,1:]
x_train = x_train.astype('float32')

#set y train
y_train = dataset[:,0]
y_train = y_train.astype('int')

maxlen = x_train.shape[1]

# 4 direction balls
n_classes = 4

# onehot encoding
y_train = np_utils.to_categorical(y_train, 4)

# define model
model = Sequential()
model.add(Embedding(1000, 1, input_length=maxlen))
model.add(LSTM(units=512, activation='tanh', unroll=True, return_sequences=True))
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
training_iters = 2000
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=training_iters, shuffle=True)

# plot accuracy history
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
acc_png = './training history/' + str(num_of_ball) + 'balls_accuracy.png'
plt.savefig(acc_png)

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
loss_png = './training history/' + str(num_of_ball) + 'balls_loss.png'
plt.savefig(loss_png)

# saved model
model_path = './saved model/classifier_fixed_' + str(num_of_ball) + 'balls'
model.save(model_path)


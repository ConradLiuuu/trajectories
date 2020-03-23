#!/usr/bin/env python3
## import libraries
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, TimeDistributed, RepeatVector
import matplotlib.pyplot as plt
#import os
import tensorflow as tf

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

n_step = 5

def split(data, depth):
    dataset = data
    depth = depth
    X = np.zeros([int(depth), n_step, 3])
    Y = np.zeros([int(depth), n_step, 3])
    c = 0
    d = 0
    
    for i in range(int(depth)):
        for j in range(n_step):
            if d < dataset.shape[0]:
                X[i,j,:] = dataset[d, c:c+3]
                Y[i,j,:] = dataset[d, (c+3*n_step):(c+3*n_step+3)]
                
                if ((c+3*n_step+3) != (dataset.shape[1])):
                    c +=3
                else:
                    c = 0
                    d += 1
        if (c-3) > 0:
            c = (c - 3*n_step + 3)
        else:
            c = c
            
    return X, Y

file_name = 'back_speed6'

dataset = pd.read_csv('./datasets/back_speed6_20200304.csv', header=None)
dataset = dataset.fillna(0)
dataset = np.array(dataset)

maxlen_train = dataset.shape[1]+12
dataset = sequence.pad_sequences(dataset, maxlen=maxlen_train, padding='post', dtype='float32')

depth_train = (int(dataset.shape[1]/3)+1-n_step-n_step)*dataset.shape[0] # (all_balls + 1 - input_balls - output_balls)*n_rows

x_train, y_train = split(data=dataset, depth=depth_train)

model = Sequential()
model.add(LSTM(256, activation='linear', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(RepeatVector(x_train.shape[1]))
model.add(LSTM(256, activation='linear', return_sequences=True))
model.add(TimeDistributed(Dense(3)))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
#model.summary()

history = model.fit(x_train, y_train, batch_size=500, epochs=300)

# plot accuracy history
# summarize history for accuracy 
plt.plot(history.history['acc'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
acc_png = './training history/' + file_name + '_accuracy.png'
plt.savefig(acc_png)
        
# saved model
model_path = './saved model/prediction_' + file_name 
model.save(model_path)


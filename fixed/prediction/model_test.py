#!/usr/bin/env python3
## import libraries
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

n_step = 5

file_name = sys.argv[1]

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

model_path = './saved model/prediction_' + file_name
model = load_model(model_path)

dataset_test = pd.read_csv('./test/'+file_name+'_20200304_test.csv', header=None)
dataset_test = dataset_test.fillna(0)
dataset_test = np.array(dataset_test)

maxlen_test = dataset_test.shape[1]+12
dataset_test = sequence.pad_sequences(dataset_test, maxlen=maxlen_test, padding='post', dtype='float32')

depth_test = (int(dataset_test.shape[1]/3)+1-n_step-n_step)*dataset_test.shape[0] # (all_balls + 1 - input_balls - output_balls)*n_rows

x_test, y_test = split(data=dataset_test, depth=depth_test)
        
pred = model.predict(x_test)

error = y_test-pred

error = error.reshape(error.shape[0]*error.shape[1], error.shape[2])
y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
pred = pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2])
tmp = np.hstack((y_test, pred))
tmp = np.hstack((tmp, error))

df = pd.DataFrame(tmp)
path = './test result/prediction_'+file_name+'.csv'
df.to_csv(path,index=0 ,header=0)

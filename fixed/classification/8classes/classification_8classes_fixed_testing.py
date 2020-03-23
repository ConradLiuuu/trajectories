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

num_of_ball = sys.argv[1]

# load model
model_path = './saved model/classifier_8classes_fixed_' + num_of_ball + 'balls'
model = load_model(model_path)

def cal_acc(num, direction, speed):
    num1 = num
    num = int(num)
    speed = str(speed)
    file = '/home/lab606a/ML/trajectories/fixed/classification/8classes/test/' + num1 + 'balls/' + direction + '_speed' +  speed + '_20200304_test.csv'
    test_up2 = pd.read_csv(file, header=None)
    test_up2 = np.array(test_up2)
    #print(test_up2.shape)
    test_up2 = test_up2.reshape(test_up2.shape[0],num,3)
    cnt = np.array([0,0,0,0,0,0,0,0])
    pred = model.predict(test_up2)
    for i in range (pred.shape[0]):
        for j in range (8):
            if max(pred[i,:]) == pred[i,j]:
                cnt[j] += 1
            #else:
                #print(i)
        #print("i = ",i+1)
        #print("\n",cnt)
    #print('------------------------------')
    return cnt

top5_accuracy = cal_acc(num_of_ball, 'top', 5)
top6_accuracy = cal_acc(num_of_ball, 'top', 6)
left5_accuracy = cal_acc(num_of_ball, 'left', 5)
left6_accuracy = cal_acc(num_of_ball, 'left', 6)
right5_accuracy = cal_acc(num_of_ball, 'right', 5)
right6_accuracy = cal_acc(num_of_ball, 'right', 6)
back5_accuracy = cal_acc(num_of_ball, 'back', 5)
back6_accuracy = cal_acc(num_of_ball, 'back', 6)

accuarcy = np.zeros([8,8])
accuarcy[0,:] = top5_accuracy
accuarcy[1,:] = top6_accuracy
accuarcy[2,:] = left5_accuracy
accuarcy[3,:] = left6_accuracy
accuarcy[4,:] = right5_accuracy
accuarcy[5,:] = right6_accuracy
accuarcy[6,:] = back5_accuracy
accuarcy[7,:] = back6_accuracy

df = pd.DataFrame(data=accuarcy,
                  index=["top speed 5 classification result", "top speed 6 classification result",
                        "left speed 5 classification result", "left speed 6 classification result",
                        "right speed 5 classification result", "right speed 6 classification result",
                        "back speed 5 classification result", "back speed 6 classification result"],
                 columns=["top spin 5","top spin 6", "left spin 5", "left spin 6", "right spin 5", "right spin 6", "back spin 5", "back spin 6"])

csv_path = '/home/lab606a/ML/trajectories/fixed/classification/8classes/test result/' + num_of_ball + 'balls.csv'
df.to_csv(csv_path)


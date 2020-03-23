#!/usr/bin/env python
import os

for i in range(5,7):
    for j in range(4):
        if j == 0:
            direction = 'top_speed'+str(i)
        if j == 1:
            direction = 'left_speed'+str(i)
        if j == 2:
            direction = 'right_speed'+str(i)
        if j == 3:
            direction = 'back_speed'+str(i)

        cmd = 'python3 model_test.py '+direction
        os.system(cmd)


        

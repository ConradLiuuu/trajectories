#!/usr/bin/env python
import os

for i in range(5,17):
    cmd = "python3 classification_" + str(i) + "balls_test.py"
    os.system(cmd)

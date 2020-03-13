#!/usr/bin/env python
import os

for i in range(5,17):
    cmd = "python3 classification_fixed_" + str(i) + "balls.py"
    os.system(cmd)

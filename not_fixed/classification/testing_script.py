#!/usr/bin/env python
import os

for i in range(5,17):
    cmd = "python3 classification_not_fixed_testing.py " + str(i)
    os.system(cmd)

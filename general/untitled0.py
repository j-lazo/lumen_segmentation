#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:11:47 2020

@author: nearlab
"""

import cv2
import numpy as np
import time

dir_1 = 'sample_img.png'
dir_2 = 'sample_img.npy' 

start_time = time.time()
array = np.load(dir_2)
gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
end_time = time.time()
print(start_time - end_time)

start_time = time.time()
array = cv2.imread(dir_1)
gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
end_time = time.time()
print(start_time - end_time) 

start_time = time.time()
array = cv2.imread(dir_1, cv2.COLOR_BGR2GRAY)
end_time = time.time()
print(start_time - end_time) 
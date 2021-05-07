#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:18:55 2020

@author: jlazo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:18:22 2020

@author: jlazo
"""

import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from os import listdir
from matplotlib import pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import os.path
from os import path
from PIL import Image
from os import listdir
from os.path import isfile, join
from datetime import datetime
import csv

project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
                 'data/lumen_data/'
general_model = 'ResUNet'
model_to_test = 'ResUnet_lr_1e-05_bs_16_rgb_27_04_2021_20_10'
folder_to_test = 'phantom_001_pt1'


def calculate_area_and_circunference(dir_folder):
    mask_list = sorted(os.listdir(dir_folder))

    list_areas = []
    list_circuference = []
    list_circularities = []

    size_x = []
    size_y = []

    for mask in mask_list[:]:
        name_mask = ''.join([dir_folder, mask])

        arc_len, area = findArc(name_mask)
        if area != 0:
            circulatiry = 1.0*(arc_len**2)/(4*np.pi*area)
            list_circularities.append(circulatiry)


        list_areas.append(area)
        list_circuference.append(arc_len)

        #size_x.append(np.amax(list_x_pixels) - np.amin(list_x_pixels))
        #size_y.append(np.amax(list_y_pixels) - np.amin(list_y_pixels))

    return list_areas, list_circuference, list_circularities


def calculateDistance(x1, y1, X, Y):

    dist_vector = []
    for index, x2, in enumerate(X):
        y2 = Y[index]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_vector.append(dist)

    return dist_vector


def findArc(image, th=200):
    img = cv2.imread(image)
    res = img.copy()
    ## convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## threshold the gray
    th, threshed = cv2.threshold(gray, th, 255,  cv2.THRESH_BINARY)
    ## Find contours on the binary threshed image
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]


    ## calcualte
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        cv2.drawContours(res, [cnt], -1, (0,255,0), 3, cv2.LINE_AA)
        #print("Length: {:.3f}\nArea: {:.3f}".format(arclen, area))

    cnt = cnts[0]
    pnts_x = [point[0][0] for point in cnt]
    pnts_y = [point[0][1] for point in cnt]

    moments = cv2.moments(cnt)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    distances = calculateDistance(cx, cy, pnts_x, pnts_y)
    fig, ax = plt.subplots()
    ax.plot(cx, cy, 'ro')
    ax.add_artist(plt.Circle((cx, cy), np.min(distances), color='g', fill=False))
    ax.add_artist(plt.Circle((cx, cy), np.max(distances), color='b', fill=False))

    return arclen, area


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))
        
        return dice_values


def read_results_csv_str(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(row[row_id])
        
        return dice_values


def calculate_rates(image_1, image_2):

    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)

    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    accuracy_value = accuracy_score(image_1, image_2)

    if (np.unique(image_1) == [False]).all() and (np.unique(image_1) == [False]).all():
        recall_value = 1.
        precision_value = 1.

    else:
        recall_value = recall_score(image_1, image_2)
        precision_value = average_precision_score(image_1, image_2)

    return precision_value, recall_value, accuracy_value
        

def dice(im1, im2,smooth=.001):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
        
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)


def read_img(dir_image):
    original_img = cv2.imread(dir_image)
    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    img = (img > 0.9) * 1.0
    return img

# ------ --- aqui empieza lo mero bueno -----
# ---------- save the resutls of the validation dataset in a CSV file


ground_truth_imgs_dir = project_folder + 'test/' + folder_to_test + '/label/'
result_mask_dir = ''.join([project_folder, 'results/', general_model, '/',
                           model_to_test, '/predictions/', folder_to_test,
                           '/'])

ground_truth_image_list = sorted([file for file in listdir(ground_truth_imgs_dir) if isfile(join(ground_truth_imgs_dir, file))])
results_image_list = sorted([file for file in listdir(result_mask_dir) if isfile(join(result_mask_dir, file))])

name_images = []
results_dice = []
results_sensitivity = []
results_specificity = []
results_accuracy = []

for image in ground_truth_image_list[:]:
    #print(image in results_image_list)
    #result_image = [name for name in results_image_list if image[-12:] == name[-12:]][0]

    if image in results_image_list:
        result_image = image
        print(result_image)
        original_mask = read_img(''.join([ground_truth_imgs_dir, image]))
        predicted_mask = read_img(''.join([result_mask_dir, result_image]))
        dice_val = dice(original_mask, predicted_mask)
        name_images.append(result_image)
        results_dice.append(dice_val)
        
        sensitivity, specificity, accuracy = calculate_rates(original_mask, predicted_mask)
        results_sensitivity.append(sensitivity)
        results_specificity.append(specificity)
        results_accuracy.append(accuracy)
                    
        #print(sensitivity, specificity)
    else:
        print(image, 'not found in results list')


ground_truth_image_list = [file for file in listdir(ground_truth_imgs_dir) if isfile(join(ground_truth_imgs_dir, file))]
results_image_list = [file for file in listdir(result_mask_dir) if isfile(join(result_mask_dir, file))]

now = datetime.now()
name_test_csv_file = ''.join([project_folder, 'results/', general_model,
                              '/', model_to_test, '/',
                              'results_evaluation_',
                              folder_to_test, '_',
                              model_to_test,
                              '_new.csv'])
print('saved in :', name_test_csv_file)

with open(name_test_csv_file, mode='w') as results_file:
    results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, file in enumerate(name_images):
        results_file_writer.writerow([str(i), file, 
                                      results_dice[i], 
                                      results_sensitivity[i], 
                                      results_specificity[i], 
                                      results_accuracy[i]])
                                     #arclen, circunference])
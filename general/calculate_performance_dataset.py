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

project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/'
folder_to_test = 'test_02'


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


def calculae_rates(image_1, image_2):

    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)

    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    precision_value = average_precision_score(image_1, image_2)
    recall_value = recall_score(image_1, image_2)
    accuracy_value = accuracy_score(image_1, image_2)

    return precision_value, recall_value, accuracy_value
        
    

def dice(im1, im2):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
        
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    
     
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def read_img(dir_image):
    original_img = cv2.imread(dir_image)
    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    img = (img > 0.9) * 1.0
    return img

# save the resutls of the validation dataset in a CSV file
    

model_to_test = 'ResUnet_lr_0.001_bs_16_grayscale_21_10_2020 19_48'
ground_truth_imgs_dir= project_folder + 'test/' + folder_to_test + '/label/'
result_mask_dir = project_folder + 'results/' + model_to_test + '/predictions/' + folder_to_test + '/'

ground_truth_image_list = [file for file in listdir(ground_truth_imgs_dir) if isfile(join(ground_truth_imgs_dir, file))]
results_image_list = [file for file in listdir(result_mask_dir) if isfile(join(result_mask_dir, file))]

results_dice = []
results_sensitivity = []
results_specificity = []
results_accuracy = []

for image in ground_truth_image_list[:]:
    print(image)
    print(results_image_list)
    result_image = [name for name in results_image_list if image[-12:] == name[-12:]][0]
    if result_image is not None:
        original_mask = read_img(''.join([ground_truth_imgs_dir, image]))
        predicted_mask = read_img(''.join([result_mask_dir, result_image]))
        dice_val = dice(original_mask, predicted_mask) 
        
        #print(dice_val, 1-dice_val)
        results_dice.append(dice_val)
        
        sensitivity, specificity, accuracy = calculae_rates(original_mask, predicted_mask)
        results_sensitivity.append(sensitivity)
        results_specificity.append(specificity)
        results_accuracy.append(accuracy)
                    
        #print(sensitivity, specificity)
    else:
        print(image, 'not found in results list')
             
"""pre_image_list = read_results_csv_str(csv_path, 1)
pre_calc_dsc = read_results_csv(csv_path, 2)        
pre_calc_prec = read_results_csv(csv_path, 3)
pre_calc_rec = read_results_csv(csv_path, 4)"""
        
now = datetime.now()

ground_truth_image_list = [file for file in listdir(ground_truth_imgs_dir) if isfile(join(ground_truth_imgs_dir, file))]
results_image_list = [file for file in listdir(result_mask_dir) if isfile(join(result_mask_dir, file))]


now = datetime.now()
name_test_csv_file = ''.join([project_folder, 'results_test_', '.csv']) 
with open(name_test_csv_file, mode='w') as results_file:
    results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, file in enumerate(ground_truth_image_list):
        results_file_writer.writerow([str(i), file, 
                                      results_dice[i], 
                                      results_sensitivity[i], 
                                      results_specificity[i], 
                                      results_accuracy[i]])
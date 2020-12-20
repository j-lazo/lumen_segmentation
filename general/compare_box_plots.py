#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:45:37 2020

@author: jlazo
"""

import pandas as pd
import os.path
from os import path
from PIL import Image
from os import listdir
from os.path import isfile, join
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import gridspec

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"


def get_mcc(groundtruth_list, predicted_list):
    """Return mcc covering edge cases"""

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = -1

    elif _mcc_denominator_zero(tn, fp, fn, tp) is True:
        mcc = -1

    # Finally calculate MCC
    else:
        mcc = ((tp * tn) - (fp * fn)) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return mcc


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image, dtype=np.float32)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image, dtype=np.float32)
        # mask_rgb = mask_rgb.astype(int)
        size_x, size_y, channels = np.shape(mask)
        plt.figure()
        plt.title(label)
        plt.imshow(mask.astype(np.float32))

        for x_index in range(size_x):
            for y_index in range(size_y):
                if mask[
                    x_index, y_index, 0] != 0:  # and mask[x_index, y_index, 1] == 0 and mask[x_index, y_index, 2] == 0:
                    mask_rgb[x_index, y_index, :] = color
                    # print(mask_rgb[x_index, y_index, :])

        color_mask += mask_rgb
        plt.close()

        """for label, mask in masks.items():
                color = colors[label]
                mask_rgb = np.zeros_like(image)
                mask_rgb[mask != 0] = color
                color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)"""

    return color_mask.astype(np.float32)  # cv2.addWeighted(image, 0.1, color_mask, 0.5, 0)


def compare_boxplots(data, labels, Title):

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array

    fig, ax1 = plt.subplots(figsize=(11, 7))
    fig.canvas.set_window_title('Boxplot Comparison')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.85, bottom=0.25)
    
    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    #ax1.set_title(Title)
    #ax1.set_xlabel('Model')
    ax1.set_ylabel(Title, fontsize=15)
    
    # Now fill the boxes with desired colors
    #box_colors = ['darkkhaki', 'royalblue']
    box_colors = ['royalblue', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    averages = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        averages[i] = np.average(data[i])
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')
    
    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.1
    bottom = -0.1
    ax1.set_ylim(bottom, top)

    ax1.set_xticklabels(labels, fontsize=15, weight='bold')
    #ax1.set_xticklabels(np.repeat(labels, 2), fontsize=15, weight='bold')
    
    # set stars and lines
    y_max = 1.1
    y_min = 0.5
    #ax1.set_ylabel(fontsize=15, weight='bold')
    """ax1.annotate("", xy=(1, y_max), 
                 xycoords='data',
                 xytext=(3, y_max), textcoords='data',
                 arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                            connectionstyle="bar,fraction=0.04"))
    
    ax1.annotate("", xy=(1, y_max), 
                 xycoords='data',
                 xytext=(5, y_max), textcoords='data',
                 arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                            connectionstyle="bar,fraction=0.06"))
    
    ax1.text(1.9, y_max + (y_max - y_min)*0.06, stars(0.01),
        horizontalalignment='center',
        verticalalignment='center', 
        size='x-small') 
    
    ax1.text(3, y_max + (y_max - y_min)*0.12, stars(0.01),
        horizontalalignment='center',
        verticalalignment='center', 
        size='x-small') 
    
    ax1.text(4, y_max + (y_max - y_min)*0.24, stars(0.01),
        horizontalalignment='center',
        verticalalignment='center',
        size='x-small') 
    
    ax1.annotate("", xy=(2, y_max), 
             xycoords='data',
             xytext=(4, y_max), textcoords='data',
             arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                        connectionstyle="bar,fraction=0.08"))
    
    ax1.annotate("", xy=(2, y_max), 
             xycoords='data',
             xytext=(6, y_max), textcoords='data',
             arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                        connectionstyle="bar,fraction=0.08"))
    
    ax1.annotate("", xy=(5, y_max), 
         xycoords='data',
         xytext=(7, y_max), textcoords='data',
         arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                    connectionstyle="bar,fraction=0.04"))"""



    
    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], 0.95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='large',
                 weight=weights[k], color=box_colors[k])
    
    # Finally, add a basic legend

    fig.text(0.10, 0.1, '---',
             backgroundcolor=box_colors[1], color='black', weight='roman',
             size='large')

    fig.text(0.10, 0.045, '--',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='large')

    """fig.text(0.10, 0.1, 'Grayscale dataset',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='large')

    fig.text(0.10, 0.045, 'Color dataset',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='large')"""
    
    fig.text(0.10, 0.005, '*', color='white', backgroundcolor='silver',
             weight='roman', size='large')
    
    fig.text(0.115, 0.003, 'Average Value', color='black', weight='roman',
             size='large')
    
    plt.show()


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))
        
        return dice_values


def read_img_results(dir_image):
    original_img = cv2.imread(dir_image, cv2.COLOR_BGRA2RGBA)

    if original_img is None:
        print('Could not open or find the image:', args.input)
        exit(0)


def main():

    path_file_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_test_02_ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08_.csv'

    path_file_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_test_02_3D_ResUnet_lr_0.0001_bs_8_rgb_29_11_2020_20_15_new.csv'

    path_file_3 = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
              'data/' \
              'lumen_data/results/' \
               'compare_3Dvs2D/' \
              'results_evaluation_test_02_MaskRCNN_thershold_0.8_grayscale_.csv'

    path_file_4 = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
              'data/lumen_data/' \
              'results/compare_3Dvs2D/' \
              'results_evaluation_test_02_3DConv_results_rgb_lr_1e-3_.csv'

    path_file_5 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_test_02_ensemble_3_new.csv'

    path_file_6 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_test_02_ensemble_3.5_new.csv'

    path_file_7 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_test_02_ensemble_3_new.csv'

    path_file_8 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  'lumen_data/results/' \
                  'compare_3Dvs2D/' \
                  'results_evaluation_ensemble_average_2.csv'




    #SAN network
    #path_file_7 = '/home/jlazo/Desktop/current_work/ICPR2020/data/enlarged_dataset/ensembles'
    #restore the one bellow

    #unet bn

    labels = ['A', 'B', 'C', 'D', 'E']

    data_experiment_1 = read_results_csv(path_file_1, 2)
    data_experiment_2 = read_results_csv(path_file_2, 2)
    data_experiment_3 = read_results_csv(path_file_3, 2)
    data_experiment_4 = read_results_csv(path_file_4, 2)
    data_experiment_5 = read_results_csv(path_file_5, 2)
    data_experiment_6 = read_results_csv(path_file_6, 2)
    data_experiment_7 = read_results_csv(path_file_7, 2)
    data_experiment_8 = read_results_csv(path_file_8, 2)
        
    data = [data_experiment_1, data_experiment_2, 
                    data_experiment_3, 
                    data_experiment_4,
                    data_experiment_5]
                    #data_experiment_6,
                    #data_experiment_7,
                    #data_experiment_8]

    compare_boxplots(data, labels, 'DSC')
    
    data_experiment_1 = read_results_csv(path_file_1, 3)
    data_experiment_2 = read_results_csv(path_file_2, 3)
    data_experiment_3 = read_results_csv(path_file_3, 3)
    data_experiment_4 = read_results_csv(path_file_4, 3)
    data_experiment_5 = read_results_csv(path_file_5, 3)
    data_experiment_6 = read_results_csv(path_file_6, 3)
    data_experiment_7 = read_results_csv(path_file_7, 3)
    data_experiment_8 = read_results_csv(path_file_8, 3)
    
    data = [data_experiment_1, data_experiment_2, 
                    data_experiment_3, 
                    data_experiment_4,
                    data_experiment_5]
                    #data_experiment_6,
                    #data_experiment_7,
                    #data_experiment_8]

    compare_boxplots(data, labels, 'Prec')
    
    data_experiment_1 = read_results_csv(path_file_1, 4)
    data_experiment_2 = read_results_csv(path_file_2, 4)
    data_experiment_3 = read_results_csv(path_file_3, 4)
    data_experiment_4 = read_results_csv(path_file_4, 4)
    data_experiment_5 = read_results_csv(path_file_5, 4)
    data_experiment_6 = read_results_csv(path_file_6, 4)
    data_experiment_7 = read_results_csv(path_file_7, 4)
    data_experiment_8 = read_results_csv(path_file_8, 4)
    

    data = [data_experiment_1, data_experiment_2, 
                    data_experiment_3, 
                    data_experiment_4, 
                    data_experiment_5]
                    #data_experiment_6,
                    #data_experiment_7,
                    #data_experiment_8]

    compare_boxplots(data, labels, 'Rec')
    
   
    
if __name__ == "__main__":
    main()


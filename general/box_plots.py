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



def compare_boxplots(data, labels, Title):

    fig, ax1 = plt.subplots(figsize=(3*4, 10))
    fig.canvas.set_window_title('Boxplot Comparison')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.85, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgray',
                   alpha=0.9)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_ylabel(Title, fontsize=20)

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    box_colors = ['khaki', 'darkseagreen', 'darkkhaki', 'plum', 'royalblue']
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
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 5]))
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

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'bold', 'bold', 'bold', 'bold']
    box_colors2 = ['gray', 'gray', 'gray', 'gray', 'gray']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 5
        ax1.text(pos[tick], 0.95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='large',
                 weight=weights[k], color=box_colors2[k])

    # Finally, add a basic legend

    """fig.text(0.10, 0.1, '---',
             backgroundcolor=box_colors[1], color='black', weight='roman',
             size='large')

    fig.text(0.10, 0.045, '--',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='large')

    fig.text(0.10, 0.005, '*', color='white', backgroundcolor='silver',
             weight='roman', size='large')

    fig.text(0.115, 0.003, 'Average Value', color='black', weight='roman',
             size='large')"""

    plt.show()


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))

        return dice_values



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

    # SAN network
    # path_file_7 = '/home/jlazo/Desktop/current_work/ICPR2020/data/enlarged_dataset/ensembles'
    # restore the one bellow

    # unet bn

    labels = ['', '', '$Prec$', '', '',
              '', '', '$Rec$', '', '',
              '', '', '$DSC$', '', '']

    data_experiment_1 = read_results_csv(path_file_1, 3)
    data_experiment_2 = read_results_csv(path_file_2, 3)
    data_experiment_3 = read_results_csv(path_file_3, 3)
    data_experiment_11 = read_results_csv(path_file_4, 3)
    data_experiment_21 = read_results_csv(path_file_5, 3)

    data_experiment_4 = read_results_csv(path_file_4, 3)

    #data_experiment_11 = read_results_csv(path_file_2, 3)
    #data_experiment_21 = read_results_csv(path_file_2, 2)
    data_experiment_31 = read_results_csv(path_file_1, 4)
    data_experiment_12 = read_results_csv(path_file_2, 4)
    data_experiment_22 = read_results_csv(path_file_3, 4)
    data_experiment_32 = read_results_csv(path_file_4, 4)
    data_experiment_13 = read_results_csv(path_file_5, 4)

    data_experiment_41 = read_results_csv(path_file_2, 5)
    data_experiment_42 = read_results_csv(path_file_3, 5)

    #data_experiment_13 = read_results_csv(path_file_4, 3)

    data_experiment_23 = read_results_csv(path_file_1, 2)
    data_experiment_33 = read_results_csv(path_file_2, 2)
    data_experiment_14 = read_results_csv(path_file_3, 2)
    data_experiment_24 = read_results_csv(path_file_4, 2)
    data_experiment_34 = read_results_csv(path_file_5, 2)

    data_experiment_43 = read_results_csv(path_file_4, 5)
    data_experiment_44 = read_results_csv(path_file_5, 5)

    data = [data_experiment_1, data_experiment_2,
            data_experiment_3,
            data_experiment_11, data_experiment_21,
            data_experiment_31,
            data_experiment_12, data_experiment_22,
            data_experiment_32,
            data_experiment_13, data_experiment_23,
            data_experiment_33,
            data_experiment_14, data_experiment_24,
            data_experiment_34,
            ]

    # data_experiment_6,
    # data_experiment_7,
    # data_experiment_8]

    compare_boxplots(data, labels, '')

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
            data_experiment_4]
    # data_experiment_6,
    # data_experiment_7,
    # data_experiment_8]

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
            data_experiment_4]
    # data_experiment_6,
    # data_experiment_7,
    # data_experiment_8]

    compare_boxplots(data, labels, 'Rec')


if __name__ == "__main__":
    main()


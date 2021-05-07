#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:24:17 2020

@author: jlazo
"""
import sys
import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split


from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

import os.path
from os import path
from PIL import Image
from os import listdir
from os.path import isfile, join
from datetime import datetime
import csv
import matplotlib.pyplot as plt

def get_mcc(groundtruth_list, predicted_list):
    """Return mcc covering edge cases"""   

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    
    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True :
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
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image, dtype=np.float32)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image, dtype=np.float32)
        #mask_rgb = mask_rgb.astype(int) 
        size_x, size_y, channels = np.shape(mask)
        plt.figure()
        plt.title(label)
        plt.imshow(mask.astype(np.float32))
        
        for x_index in range(size_x):
            for y_index in range(size_y):
                if mask[x_index, y_index, 0] != 0: #and mask[x_index, y_index, 1] == 0 and mask[x_index, y_index, 2] == 0:
                    mask_rgb[x_index, y_index, :] = color
                    #print(mask_rgb[x_index, y_index, :])
        

        color_mask += mask_rgb
        plt.close()
        
        """for label, mask in masks.items():
                color = colors[label]
                mask_rgb = np.zeros_like(image)
                mask_rgb[mask != 0] = color
                color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)"""
        
        
    return color_mask.astype(np.float32)#cv2.addWeighted(image, 0.1, color_mask, 0.5, 0)


def calculae_rates(image_1, image_2):

    image_1 = np.asarray(image_1).astype(np.bool)
    image_2 = np.asarray(image_2).astype(np.bool)
    
    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    if image_1.shape != image_2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    precision_value = average_precision_score(image_1, image_2)
    recall_value = recall_score(image_1, image_2)

    print(precision_value, recall_value)

    return precision_value, recall_value
        
    

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
    img = (img > 0.9) * 1.0
    return img

# save the resutls of the validation dataset in a CSV file


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))
        
        return dice_values
    
def read_img_results(dir_image):

    #original_img = cv2.imread(dir_image, cv2.COLOR_BGRA2RGBA)
    #original_img = cv2.imread(dir_image, cv2.COLOR_BGR2RGB)
    original_img = cv2.imread(dir_image)
    new_im = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    if original_img is None:
        print('Could not open or find the image:', dir_image)
        exit(0)

    img = cv2.resize(new_im, (256, 256))

    return img

def read_mask(dir_image):
    original_img = cv2.imread(dir_image)
    
    if original_img is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    
    height, width, depth = original_img.shape
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    img = (img > 0.9) * 1.0 
    return img

def read_results_csv_plot(file_path):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append([row[1], row[2]])
        
        return dice_values
    

def print_box_plots(name_test_csv_file, name_validation_csv_file, save_directory):

    path_file_1= name_test_csv_file
    path_file_2 = name_validation_csv_file
    
    list_dice_values_file_1 = read_results_csv(path_file_1, 2)
    list_dice_values_file_2 = read_results_csv(path_file_2, 2)
    data_dice = [list_dice_values_file_1, list_dice_values_file_2]

    list_precision_values_file_1 = read_results_csv(path_file_1, 3)
    list_precision_values_file_2 = read_results_csv(path_file_2, 3)
    data_precision_values = [list_precision_values_file_1, list_precision_values_file_2]
    
    list_recall_file_1 = read_results_csv(path_file_1, 4)
    list_recall_file_2 = read_results_csv(path_file_2, 4)
    data_recall = [list_recall_file_1, list_recall_file_2]
    
    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(131)
    ax1.boxplot(data_dice[0], 1, 'gD')
    ax2 = fig1.add_subplot(132)
    ax2.boxplot(data_precision_values[0], 1, 'gD')
    ax3 = fig1.add_subplot(133)
    ax3.boxplot(data_recall[0], 1, 'gD')
    ax1.title.set_text('Dice Coeff')
    ax2.title.set_text('Precision')
    ax3.title.set_text('Recall')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)

    plt.savefig(save_directory + 'results_test.svg')    
    plt.close()
    
    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(131)
    ax1.boxplot(data_dice[1], 1, 'gD')
    ax2 = fig2.add_subplot(132)
    ax2.boxplot(data_precision_values[1], 1, 'gD')
    ax3 = fig2.add_subplot(133)
    ax3.boxplot(data_recall[1], 1, 'gD')
    ax1.title.set_text('Dice Coeff')
    ax2.title.set_text('Precision')
    ax3.title.set_text('Recall')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    plt.savefig(save_directory + 'results_val.svg')
    plt.close()


def compare_results(dir_groundtruth, directories_prediction_1,
                    dir_csv_file_1,
                    directories_prediction_2,
                    dir_csv_file_2,
                    directories_prediction_3,
                    dir_csv_file_3,
                    directories_prediction_4,
                    dir_csv_file_4,
                    directories_prediction_5,
                    dir_csv_file_5,
                    save_directory):
    alpha=0.5
    confusion_matrix_colors = {
        'tp': (50, 100, 0),  # cyan
        'fp': (125, 0, 125),  # magenta
        'fn': (0, 100, 50),  # blue
        'tn': (0, 0, 0)  # black
    }

    path_images_folder = dir_groundtruth + 'image/rgb/'
    path_masks_folder = dir_groundtruth + 'label/'
    
    list_dice_values_1 = read_results_csv_plot(dir_csv_file_1)
    list_dice_values_2 = read_results_csv_plot(dir_csv_file_2)
    list_dice_values_3 = read_results_csv_plot(dir_csv_file_3)
    list_dice_values_4 = read_results_csv_plot(dir_csv_file_4)
    list_dice_values_5 = read_results_csv_plot(dir_csv_file_5)

    image_list = [f for f in listdir(path_images_folder) if isfile(join(path_images_folder, f))]
    mask_list = sorted([f for f in listdir(path_masks_folder) if isfile(join(path_masks_folder, f))])

    predicted_masks_1 = [f for f in listdir(directories_prediction_1) if isfile(join(directories_prediction_1, f))]
    predicted_masks_2 = [f for f in listdir(directories_prediction_2) if isfile(join(directories_prediction_2, f))]
    predicted_masks_3 = [f for f in listdir(directories_prediction_3) if isfile(join(directories_prediction_3, f))]
    predicted_masks_4 = [f for f in listdir(directories_prediction_4) if isfile(join(directories_prediction_4, f))]
    predicted_masks_5 = [f for f in listdir(directories_prediction_5) if isfile(join(directories_prediction_5, f))]

    for image in predicted_masks_1[:]:
        result_image = [name for name in mask_list if (image[:] == name[:])][0]
        if result_image is not None:

                path_image = ''.join([path_images_folder, image]) 
                path_mask = ''.join([path_masks_folder, image]) 
                path_predicted_1 = ''.join([directories_prediction_1, image]) 
                path_predicted_2 = ''.join([directories_prediction_2, image])
                path_predicted_3 = ''.join([directories_prediction_3, image])
                path_predicted_4 = ''.join([directories_prediction_4, image])
                path_predicted_5 = ''.join([directories_prediction_5, image])

                image_frame = read_img_results(path_image)
                mask_image = read_mask(path_mask)                             
                
                for counter, element in enumerate(list_dice_values_1):
                    if image == element[0]:
                        dice_value_1 = float(element[1])
                        predicted_mask_1 = read_mask(path_predicted_1)
                        dice_value_1 = float("{:.3f}".format(dice_value_1))
                        overlay_1 = get_confusion_matrix_overlaid_mask(image_frame, mask_image, predicted_mask_1, alpha,
                                                                     confusion_matrix_colors)
                
                for counter, element in enumerate(list_dice_values_2):
                    if image == element[0]:
                        dice_value_2 = float(element[1])
                        predicted_mask_2 = read_mask(path_predicted_2)
                        dice_value_2 = float("{:.3f}".format(dice_value_2))
                        overlay_2 = get_confusion_matrix_overlaid_mask(image_frame, mask_image, predicted_mask_2, alpha,
                                                                       confusion_matrix_colors)
                        
                for counter, element in enumerate(list_dice_values_3):
                    if image == element[0]:
                        dice_value_3 = float(element[1])
                        predicted_mask_3 = read_mask(path_predicted_3)
                        dice_value_3 = float("{:.3f}".format(dice_value_3))
                        overlay_3 = get_confusion_matrix_overlaid_mask(image_frame, mask_image, predicted_mask_3, alpha,
                                                                       confusion_matrix_colors)

                for counter, element in enumerate(list_dice_values_4):
                    if image == element[0]:
                        dice_value_4 = float(element[1])
                        predicted_mask_4 = read_mask(path_predicted_4)
                        dice_value_4 = float("{:.3f}".format(dice_value_4))
                        overlay_4 = get_confusion_matrix_overlaid_mask(image_frame, mask_image, predicted_mask_4, alpha,
                                                                       confusion_matrix_colors)

                for counter, element in enumerate(list_dice_values_5):
                    if image == element[0]:
                        dice_value_5 = float(element[1])
                        predicted_mask_5 = read_mask(path_predicted_5)
                        dice_value_5 = float("{:.3f}".format(dice_value_5))
                        overlay_5 = get_confusion_matrix_overlaid_mask(image_frame, mask_image,
                                                                       predicted_mask_5, alpha,
                                                                       confusion_matrix_colors)
                        my_dpi = 96
                        plt.figure(3, figsize=(640/my_dpi, 480/my_dpi), dpi=my_dpi)
                        print('img:', image[12:-4])
                        plt.subplot(171)
                        #plt.title('img:', image[12:-4])
                        #plt.title('Frame')
                        plt.imshow(image_frame)
                        plt.axis('off')
                        
                        plt.subplot(172)
                        #plt.title('Mask')
                        plt.imshow(mask_image)
                        plt.axis('off')

                        plt.subplot(173)
                        title_1 = 'DSC: ' + str(dice_value_1)
                        #title_1 = str(dice_value_1)
                        plt.title(title_1)
                        plt.imshow(overlay_1)
                        plt.axis('off')
                        
                        plt.subplot(174)
                        title_2 = str(dice_value_2)
                        plt.title(title_2)
                        plt.imshow(overlay_2)
                        plt.axis('off')

                        plt.subplot(175)
                        title_3 = str(dice_value_3)
                        plt.title(title_3)
                        plt.imshow(overlay_3)
                        plt.axis('off')

                        plt.subplot(176)
                        title_3 = str(dice_value_4)
                        plt.title(title_3)
                        plt.imshow(overlay_4)
                        plt.axis('off')

                        plt.subplot(177)
                        title_3 = str(dice_value_5)
                        plt.title(title_3)
                        plt.imshow(overlay_5)
                        plt.axis('off')

                        plt.savefig(''.join([save_directory,  image]))
                        plt.close()


                
"""    for image in predicted_masks_2[:]:
        result_image = [name for name in mask_list if (image[:] == name[:])][0]
        if result_image is not None:

                path_image = ''.join([path_images_folder, image]) 
                path_mask = ''.join([path_masks_folder, image]) 
                path_predicted_2 = ''.join([directories_prediction_2, image]) 
                                
                for counter, element in enumerate(list_dice_values_2):
                    if image == element[0]:
                        dice_value_2 = float(element[1])
                        predicted_mask_2 = read_mask(path_predicted_2)
                        dice_value_2 = float("{:.3f}".format(dice_value_2))

                                                
                        
    for image in predicted_masks_3[:]:
        result_image = [name for name in mask_list if (image[:] == name[:])][0]
        if result_image is not None:

                path_predicted_3 = ''.join([directories_prediction_3, image]) 
                                
                for counter, element in enumerate(list_dice_values_3):
                    if image == element[0]:
                        dice_value_3 = float(element[1])
                        predicted_mask_3 = read_mask(path_predicted_3)
                        dice_value_3 = float("{:.3f}".format(dice_value_3))
                        
                        
                        my_dpi = 96
                        plt.figure(3, figsize=(640/my_dpi, 480/my_dpi), dpi=my_dpi)
                        
                        plt.subplot(151)
                        plt.title(image)
                        plt.imshow(image_frame)
                        plt.axis('off')
                        
                        plt.subplot(152)
                        plt.title('Mask')
                        plt.imshow(mask_image)
                        plt.axis('off')
                                              
                        plt.subplot(153)
                        title_1 = 'DSC: ' + str(dice_value_1)
                        plt.title(title_1)
                        plt.imshow(predicted_mask_1)
                        plt.axis('off')
                        
                        plt.subplot(154)
                        title_2 = 'DSC: ' + str(dice_value_2)
                        plt.title(title_2)
                        plt.imshow(predicted_mask_2)
                        plt.axis('off')
                        
                        plt.subplot(155)
                        title_3 = 'DSC: ' + str(dice_value_3)
                        plt.title(title_3)
                        plt.imshow(predicted_mask_3)
                        plt.axis('off')

                        plt.savefig(''.join([save_directory,  image, '_',str(counter),'_.png']))
                        plt.close()"""
                        
def crop_images(image_directory, roi, string_to_add):

    image_list = [f for f in listdir(image_directory) if isfile(join(image_directory, f))]
    
    for image in image_list:
        print('resizing', image)
        path_image = ''.join([image_directory, image]) 
        
        original_img = cv2.imread(path_image)
        croped_img = original_img[roi[1]:roi[3], roi[0]:roi[2]]
        new_name = ''.join([image_directory, string_to_add, image])
        cv2.imwrite(new_name, croped_img)
    

def main():
    """
    test_dataset = 'test_03'
    project_folder = '/home/nearlab/Jorge/current_work/' \
                     'lumen_segmentation/data/lumen_data/'
    test_directory = ''.join([project_folder, 'test/', test_dataset, '/'])

    predictions_test_directory_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                                   'old_lumen_data/results/' \
                                   'ResUnet_lr_0.001_bs_16_rgb_05_11_2020_16_32/' \
                                   'predictions/' + test_dataset + '/'

    name_test_csv_file_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                           'old_lumen_data/results/' \
                           'ResUnet_lr_0.001_bs_16_rgb_05_11_2020_16_32/' \
                           'results_evaluation_' + test_dataset + '_ResUnet_lr_0.001_bs_16_rgb_05_11_2020_16_32_.csv'

    predictions_test_directory_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                                   'old_lumen_data/results/' \
                                   'MaskRCNN_threshold_0.8_grayscale/' \
                                   'predictions/' + test_dataset + '/'

    name_test_csv_file_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                                   'old_lumen_data/results/' \
                                   'MaskRCNN_threshold_0.8_rgb/' \
                                   'results_evaluation_' + test_dataset + '_MaskRCNN_thershold_0.8_rgb_.csv'
    
    save_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'old_lumen_data/results/' \
                     'comparison_MaskRCNN_vs_ResUnet_color/' + test_dataset + '/'"""

    """test_directory = project_folder + 'test_03/'
        predictions_test_directory_1 = project_folder + 'predictions/new_data/'
        predictions_test_directory_2 = project_folder + 'predictions/old_data/'


        name_test_csv_file_1 = project_folder + 'predictions/results_new_data.csv'
        name_test_csv_file_2 = project_folder + 'predictions/results_old_data.csv'
        save_directory = project_folder + 'results_comparison/'"""

    test_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/test_02/'

    predictions_test_directory_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08/'
    predictions_test_directory_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/MaskRCNN_thershold_0.8_grayscale_/'
    predictions_test_directory_3 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/3D_ResUnet_lr_0.001_bs_16_grayscale_16_11_2020_19_37/'
    predictions_test_directory_4 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/3DMaskRCNN_2/'
    predictions_test_directory_5 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/ensemble_all_data/average/'

    name_test_csv_file_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/results_evaluation_test_02_ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08_.csv'
    name_test_csv_file_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/results_evaluation_test_02_MaskRCNN_thershold_0.8_grayscale_.csv'
    name_test_csv_file_3 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/results_evaluation_test_02_3D_ResUnet_lr_0.001_bs_16_grayscale_16_11_2020_19_37_.csv'
    name_test_csv_file_4 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/results_evaluation_test_02_3DMaskRCNN_2_.csv'
    name_test_csv_file_5 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/results_evaluation_test_02_ensemble_all_data_average.csv'

    save_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/results/compare_3Dvs2D/comparisons/'
    compare_results(test_directory,
                    predictions_test_directory_1, name_test_csv_file_1,
                    predictions_test_directory_2, name_test_csv_file_2,
                    predictions_test_directory_3, name_test_csv_file_3,
                    predictions_test_directory_4, name_test_csv_file_4,
                    predictions_test_directory_5, name_test_csv_file_5,
                    save_directory)

    roi = [76, 160, 580, 300]
    crop_images(save_directory, roi, 'test_')
    #crop_images(save_directory_val, roi,'val_')


if __name__ == "__main__":
    main()


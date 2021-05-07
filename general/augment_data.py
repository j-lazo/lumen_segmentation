#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: j-lazo
"""

import cv2
import os
import random
import numpy as np
from scipy.ndimage import zoom

def clipped_zoom(img, zoom_factor, **kwargs):

    """
    :param img:
    :param zoom_factor:
    :param kwargs:
    :return:
    """

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        #trim_top = ((out.shape[0] - h) // 2)
        #trim_left = ((out.shape[1] - w) // 2)
        #out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def adjust_brightness(image, gamma=1.0):
    """

    :param image:
    :param gamma:
    :return:
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)


def augment_data(files_path):
    
    files = os.listdir(files_path + 'image/')
    masks = os.listdir(files_path + 'label/')
    
    for i, element in enumerate(files[:]):
        if element not in masks:
            print(element, 'has no pair')
        print(1.0*i/len(files), element)
        img = cv2.imread("".join([files_path, 'image/', element]))
        mask = cv2.imread("".join([files_path, 'label/', element]))

        rows, cols, channels = img.shape
        # define the rotation matrixes
        rot1 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rot2 = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
        rot3 = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
        # rotate the images
        im_rot1 = cv2.warpAffine(img, rot1, (cols, rows))
        im_rot2 = cv2.warpAffine(img, rot2, (cols, rows))
        im_rot3 = cv2.warpAffine(img, rot3, (cols, rows))
        #rotate the masks 
        mask_rot1 = cv2.warpAffine(mask, rot1, (cols, rows))
        mask_rot2 = cv2.warpAffine(mask, rot2, (cols, rows))
        mask_rot3 = cv2.warpAffine(mask, rot3, (cols, rows))
        
        # flip images 
        horizontal_img = cv2.flip(img, 0)
        vertical_img = cv2.flip(img, 1)
        #flip masks
        horizontal_mask = cv2.flip(mask, 0)
        vertical_mask = cv2.flip(mask, 1)
       
        # save the images 
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_1', '.png']), im_rot1)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_2', '.png']), im_rot2)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_3', '.png']), im_rot3)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_4', '.png']), horizontal_img)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_5', '.png']), vertical_img)
        
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_1', '.png']), mask_rot1)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_2', '.png']), mask_rot2)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_3', '.png']), mask_rot3)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_4', '.png']), horizontal_mask)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_5', '.png']), vertical_mask)
    
        # change brightness
        list_of_images = [img, im_rot1, im_rot2, im_rot3, horizontal_img, vertical_img]
        list_of_masks = [mask, mask_rot1, mask_rot2, mask_rot3, horizontal_mask, vertical_mask]
        gammas = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

        for i in range(4):
            index = random.randint(0,len(list_of_images)-1)
            img_choice = list_of_images[index]
            mask_choice = list_of_masks[index]
            image_brg = adjust_brightness(img_choice, random.choice(gammas))
            cv2.imwrite("".join([files_path, 'image/', element[:-4], '_', str(i+6), '.png']), image_brg)
            cv2.imwrite("".join([files_path, 'label/', element[:-4], '_', str(i+6), '.png']), mask_choice)
                      
        index_2 = random.randint(0,len(list_of_images)-1)
        img_choice_2 = list_of_images[index_2]
        mask_choice_2 = list_of_masks[index_2]
        zoom_in_img = clipped_zoom(img_choice_2, 1.2)
        zoom_in_mask = clipped_zoom(mask_choice_2, 1.2)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_10_.png']), zoom_in_img)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_10_.png']), zoom_in_mask)
        
        index_3 = random.randint(0,len(list_of_images)-1)
        img_choice_3 = list_of_images[index_3]
        mask_choice_3 = list_of_masks[index_3]
        zoom_out_img = clipped_zoom(img_choice_3, 0.8)
        zoom_out_mask = clipped_zoom(mask_choice_3, 0.8)
        cv2.imwrite("".join([files_path, 'image/', element[:-4], '_11_.png']), zoom_out_img)
        cv2.imwrite("".join([files_path, 'label/', element[:-4], '_11_.png']), zoom_out_mask)


def main():
    
    path_directory = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'lumen_data/test/phantom_001_pt2/augmented_data/'
    augment_data(path_directory)


if __name__ == "__main__":
    main()


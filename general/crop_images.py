#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:27:14 2020

@author: jlazo
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def crop_imgs(dir_folder, destination_folder, min_x, max_x, min_y, max_y):
    images_folder = sorted(os.listdir(dir_folder))
    print(len(images_folder))
    images_folder = sorted([image for image in images_folder
                   if np.random.rand() <= 0.15])
    total_img = len(images_folder)
    print(total_img)
    # images_folder.sort()

    for i, image in enumerate(images_folder):
        print(i, total_img, image)
        name_img = ''.join([dir_folder, '/', image])
        img = cv2.imread(name_img)
        croped_img = img[min_y:max_y, min_x:max_x, :]
        name_cropped_img = destination_folder + image
        cv2.imwrite(name_cropped_img, croped_img)


if __name__ == '__main__':
    directory = '/home/nearlab/Jorge/DATASETS/lesion_classification/urs/' \
                'urs_case_010/frames_selected/lesion/'

    destination_folder = '/home/nearlab/Jorge/DATASETS/' \
                         'lesion_classification/urs/urs_case_010/lesion/'

    min_x = 240
    min_y = 175
    max_x = 460
    max_y = 420

    crop_imgs(directory, destination_folder,
              min_x, max_x, min_y, max_y)
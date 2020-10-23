#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:29:27 2020

@author: nearlab
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


def calculate_size_maks(dir_folder):

    mask_list = sorted(os.listdir(dir_folder))
    list_x_pixels = []    
    list_y_pixels = []
    
    size_x = []
    size_y = []
    
    for mask in mask_list[:]: 
        name_mask = ''.join([dir_folder, mask])
        
        mask = cv2.imread(name_mask)
        shape_mask = np.shape(mask)
        
        if np.amax(mask) !=0: 
            del(list_x_pixels[:])
            del(list_y_pixels[:])
            #print(np.amax(mask[:,:,0]), np.amax(mask[:,:,1]), np.amax(mask[:,:,2]))
            #print(np.amin(mask[:,:,0]), np.amin(mask[:,:,1]), np.amin(mask[:,:,2]))    
            i,j = np.where(mask[:,:,2]==255)
            
            list_x_pixels.append(i) 
            list_y_pixels.append(j)
            
            size_x.append(np.amax(list_x_pixels)-np.amin(list_x_pixels))  
            size_y.append(np.amax(list_y_pixels)-np.amin(list_y_pixels))  
           
                            
    return size_x, size_y
             
        
def main():   

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_001_pt1/label/'
   
    size_x, size_y = calculate_size_maks(directory)  
    plt.figure()
    plt.plot(size_x,size_y, 'ro')
    
    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_001_pt2/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x,size_y, 'ro', label='patient 1')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_006_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)
    plt.plot(size_x, size_y, 'g*', label='patient 6')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x,size_y, 'bo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt2/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x,size_y, 'bo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_003_pt3/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x,size_y, 'bo', label ='patient 3')
    plt.legend(loc='best')
    plt.xlabel('horizontal size (pixels)')
    plt.ylabel('vertical size (pixels)')

    plt.figure()
    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_004_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)
    plt.plot(size_x, size_y, 'r*', label='patient 4')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_002_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)
    plt.plot(size_x,size_y, 'bo', label='patient 2')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_005_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'g*')
    
    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_005_pt2/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'g*', label = 'patient 5')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt1/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'yo')
    
    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt2/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'yo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt3/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'yo')

    directory = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy/p_007_pt4/label/'
    size_x, size_y = calculate_size_maks(directory)  
    plt.plot(size_x, size_y, 'yo', label='patient 7')

    plt.legend(loc='best')
    plt.xlabel('horizontal size (pixels)')
    plt.ylabel('vertical size (pixels)')
    plt.show()
        
if __name__ == '__main__':
    main()
    

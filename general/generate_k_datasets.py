#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:18:43 2020

@author: jlazo
"""

import os
import random
import shutil


def make_directories(path):
    if os.path.isdir(path) is False:
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s" % path)
        

def generate_k_subsets(k, current_directory):
    
    files_path_images = "".join([current_directory, 'all/image/'])
    files_path_labels = "".join([current_directory, 'all/label/'])
    
    original_images = os.listdir(files_path_images)
    label_images = os.listdir(files_path_labels)
    new_directories = []
    
    for i in range(k):
        new_dir = ''.join(['k_', str(i), '/']) 
        new_directories.append(current_directory + new_dir)
        
        new_sub_dir_imgs = ''.join([current_directory, new_dir, '/image/']) 
        new_sub_dir_maks = ''.join([current_directory, new_dir, '/label/']) 
        make_directories(new_sub_dir_imgs)
        make_directories(new_sub_dir_maks)
        
    len_dir = len(original_images)
    for index, chosen_file in enumerate(original_images):
        name_dir = random.choice(new_directories)
        print(index,len_dir, name_dir, chosen_file)
        if chosen_file in label_images:
            print('coping: ', chosen_file)
            shutil.copy(files_path_images + chosen_file, "".join([name_dir, 'image/', chosen_file]))
            shutil.copy(files_path_labels + chosen_file, "".join([name_dir, 'label/', chosen_file]))
            
        else:
            print('pair not found')
            
        
def main():
    current_directory = '/home/nearlab/Jorge/ICPR2020/data/k_fold_rgb/'
    k = 5
    generate_k_subsets(k, current_directory)


if __name__ == '__main__':
    main()

import os
import shutil
import cv2
import numpy as np
import os.path
from os import path



def select_no_empty_frames(current_directory):

    files_path_images = "".join([current_directory, 'frame/'])
    files_path_labels = "".join([current_directory, 'mask/'])
    
    original_images = os.listdir(files_path_images)
    label_images = os.listdir(files_path_labels)
    
    directory_new_images = current_directory + 'image/'
    directory_new_labels = current_directory + 'label/'
    
    if not(path.exists(directory_new_images)):
        os.mkdir(directory_new_images)

    if not(path.exists(directory_new_labels)):
        os.mkdir(directory_new_labels)

       
    print(files_path_images)
    print(files_path_labels)
    counter = 0
    
    for count_i, image in enumerate(original_images):
        if image in label_images:
            img = cv2.imread(''.join([files_path_labels, image]), 0)
            
            if np.count_nonzero(img)>1:
                counter+=1
                shutil.copy(files_path_images + image, "".join([directory_new_images, image]))
                shutil.copy(files_path_labels + image, "".join([directory_new_labels, image]))


def prepare_data(directory):
    select_no_empty_frames(directory)
    
            
def main():
    current_directory = '/home/nearlab/Jorge/Data_IEO/P_003/useful_frames/converted/P_003-pt4-converted/'
    prepare_data(current_directory)


if __name__ == '__main__':
    main()

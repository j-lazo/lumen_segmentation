import os
import cv2
import numpy as np

def convert_mask(dir_folder):

    masks_folder_dir = dir_folder
    masks_folder = os.listdir(masks_folder_dir)

    for mask in masks_folder[:]:
        print(dir_folder, mask)
        img = cv2.imread(''.join([dir_folder, mask]))
        #w,h,d = np.shape(img)

        new_image = np.zeros(np.shape(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray/255

        gray[gray >= 0.5] = 255
        gray[gray < 0.5] = 0

        new_image[:, :, 0] = gray
        new_image[:, :, 1] = gray
        new_image[:, :, 2] = gray

        print(np.shape(new_image))
        cv2.imwrite(''.join([dir_folder, mask]), new_image)

def main():

    folder = '/home/nearlab/Jorge/current_work/artifact_detection/data/results/results_mask_rcnn_ResNet50/'

    convert_mask(folder)


if __name__ == "__main__":
    main()
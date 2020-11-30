import os
import numpy as np
import cv2


def dilate(folder, output_folder, kernel_size=3):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_list = sorted(os.listdir(folder))

    if not(os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    for j, image in enumerate(img_list[:]):
        print(j, image)
        img = cv2.imread(os.path.join(folder, image), 1)
        dilation = cv2.dilate(img, kernel, iterations=1)
        new_name = ''.join([output_folder, image])
        cv2.imwrite(new_name, dilation)


def erode(folder, output_folder, kernel_size=3):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_list = sorted(os.listdir(folder))

    if not(os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    for j, image in enumerate(img_list[:]):
        print(j, image)
        img = cv2.imread(os.path.join(folder, image), 1)
        erosion = cv2.erode(img, kernel, iterations=1)
        new_name = ''.join([output_folder, image])
        cv2.imwrite(new_name, erosion)


def main():

    path_directory = '/home/nearlab/Jorge/current_work/' \
                     'lumen_segmentation/data/lumen_data/' \
                     'test/test_02/label/'

    ouput_folder = '/home/nearlab/Jorge/current_work/' \
                   'lumen_segmentation/data/' \
                   'lumen_data/test/test_02/label_dilate_3/'

    dilate(path_directory, ouput_folder, kernel_size=3)


if __name__ == "__main__":
    main()
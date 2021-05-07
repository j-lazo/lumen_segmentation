import cv2
import os
import numpy as np

def convert_format(dir_folder, destination_foler):

    imgs_folder_dir = dir_folder
    list_images = sorted(os.listdir(imgs_folder_dir))

    for image in list_images:
        print(dir_folder, image)
        img = cv2.imread(''.join([dir_folder, image]))
        img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
        cv2.imwrite(''.join([destination_foler, image[:-4], '.png']), img)


def main():

    base_dir = '/home/nearlab/Jorge/data/polyp_sundataset/case100/image/rgb/'

    destination_foler = '/home/nearlab/Jorge/data/polyp_sundataset/case100/image/grayscale/'

    convert_format(base_dir, destination_foler)


if __name__ == '__main__':
    main()
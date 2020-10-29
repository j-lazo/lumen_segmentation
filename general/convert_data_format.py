import os
import random

import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt


def convert_data_to_pickle(folder, new_folder):

    if not os.path.isdir(os.path.join(folder, new_folder)):
        os.mkdir(os.path.join(folder, new_folder))

    if not os.path.isdir(os.path.join(folder, new_folder, 'image')):
        os.mkdir(os.path.join(folder, new_folder, 'image'))

    """if not os.path.isdir(os.path.join(folder, new_folder, 'image', 'rgb')):
        os.mkdir(os.path.join(folder, new_folder, 'image', 'rgb'))

    if not os.path.isdir(os.path.join(folder, new_folder, 'image', 'grayscale')):
        os.mkdir(os.path.join(folder, new_folder, 'image', 'grayscale'))"""

    if not os.path.isdir(os.path.join(folder, new_folder, 'label')):
        os.mkdir(os.path.join(folder, new_folder, 'label'))

    # ----------------------------------------------------------------
    volume_size = 3

    video_set = os.listdir(folder)
    video_set.sort()
    img_clusters_number = []
    img_clusters_name = []
    img_size = 256

    img_list = os.listdir(os.path.join(folder, 'image'))
    img_list.sort()

    mask_list = os.listdir(os.path.join(folder,  'label'))
    mask_list.sort()

    for j, image in enumerate(img_list[:]):
        name_img = image[:-4]
        print(image)
        number_img = int(name_img.replace('p006_video_1_', ''))

        counter = number_img
        img_clusters_number.append(number_img)
        img_clusters_name.append(image)
        if len(img_clusters_number) > 1:
            if not(img_clusters_number[-2] == counter -1):
                img_clusters_number.clear()
                img_clusters_number.append(number_img)
                img_clusters_name.clear()
                img_clusters_name.append(image)

        if len(img_clusters_number) >= 3:
            print(img_clusters_name)
            frames_gray = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)
            frames = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)
            masks = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)

            for k, image_name in reversed(list(enumerate(img_clusters_name[-3:]))):

                frame = cv2.imread(os.path.join(folder, 'image', image_name), 1)
                frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)

                mask = cv2.imread(os.path.join(folder, 'label', image_name), 1)
                mask = cv2.resize(mask, (img_size, img_size), cv2.INTER_AREA)

                #frames_gray[k:, :, :] = frame_gray
                frames[k, :, :, :] = frame
                masks[k, :, :, :] = mask

            img_clusters_number.clear()
            img_clusters_name.clear()
            #frames = np.moveaxis(frames, 0, -2)
            #masks = np.moveaxis(masks, 0, -2)
            np.save(os.path.join(folder, new_folder, 'image', name_img + '.npy'), frames)
            np.save(os.path.join(folder, new_folder, 'label', name_img + '.npy'), masks)


def main():

    folder = '/home/nearlab/Jorge/DATASETS/lumen_ureteroscopy_3D_volumes/' \
             'p_006_pt1/'
    new_folder = '3D_volumes'
    convert_data_to_pickle(folder, new_folder)


if __name__ == "__main__":
    main()
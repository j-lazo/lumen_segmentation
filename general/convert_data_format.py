import os
import random

import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt


"""# Dataset selezionato"""
"""# Volumi e maschere corrispondenti a coppie"""


def conver_data_to_pickle(folder, new_folder):

    if not os.path.isdir(os.path.join(folder, new_folder)):
        os.mkdir(os.path.join(folder, new_folder))

    if not os.path.isdir(os.path.join(folder, new_folder, 'train')):
        os.mkdir(os.path.join(folder, new_folder, 'train'))

    if not os.path.isdir(os.path.join(folder, new_folder, 'train', 'image')):
        os.mkdir(os.path.join(folder, new_folder, 'train', 'image'))

    if not os.path.isdir(os.path.join(folder, new_folder, 'train', 'label')):
        os.mkdir(os.path.join(folder, new_folder, 'train', 'label'))

    img_size = 128


    # ----------------------------------------------------------------
    scan_size = 4

    # TRAINING:
    video_set = os.listdir(folder)

    video_set.sort()
    num_video = []
    img_clusters = []

    for z in tqdm(range(len(video_set))):
        img_list = os.listdir(os.path.join(folder, 'image'))
        img_list.sort()

        mask_list = os.listdir(os.path.join(folder,  'label'))
        mask_list.sort()


        #for j in tqdm(range(len(img_list) - scan_size)):
        flag = True
        for j, image in enumerate(img_list[:10]):
            print(image)
            print(img_clusters)
            name_img = image[:-4]
            number_img = int(name_img.replace('p007_video_4_', ''))
            img_clusters.append(number_img)
            counter = number_img
            print(counter, j, img_clusters)

            if img_clusters == []:
                img_clusters.append(number_img)
                flag = True

            if img_clusters[-1] != counter -1 and flag is False:
                img_clusters.clear()
                flag = False

            #print(name_img)

            #frames = np.zeros([4, img_size, img_size, 3], dtype=np.uint8)
            #masks = np.zeros([4, img_size, img_size, 3], dtype=np.uint8)
            #for k in range(1):
            #    frame = cv2.imread(os.path.join(folder, 'image', image), 1)
            #    frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)
            #    #frame = frame[:, :, :, np.newaxis]
            #    mask = cv2.imread(os.path.join(folder, 'label', image), 1)
            #    mask = cv2.resize(mask, (img_size, img_size), cv2.INTER_AREA)
            #    #mask = mask[:, :, :, np.newaxis]
            #    frames[k, :, :, :] = frame
            #    masks[k, :, :, :] = mask
            #frames = np.moveaxis(frames, 0, -2)
            #masks = np.moveaxis(masks, 0, -2)




            #np.save(os.path.join(folder, new_folder, 'train', 'image', name_img + '.npy'), frames)
            #np.save(os.path.join(folder, new_folder, 'train', 'label', name_img + '.npy'), masks)

    plt.figure()
    plt.plot(num_video, '*')
    plt.show()


def main():

    folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
             'quick_test/p_007_pt4/'
    new_folder = 'new_folder'
    conver_data_to_pickle(folder, new_folder)


if __name__ == "__main__":
    main()
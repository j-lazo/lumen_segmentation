import os
import random

import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
import shutil


def convert_continous_images_to_pickle(folder, subfolder,
                                       new_folder, volume_size):

    """
    This function gathers 2D imgs in blocks of n images and save them into npy array
    :param folder:
    :param subfolder:
    :param new_folder: directory folder where the image-volumes will be saved
    :return:
    """

    if not os.path.isdir(os.path.join(new_folder, subfolder)):
        os.mkdir(os.path.join(new_folder, subfolder))

    if not os.path.isdir(os.path.join(new_folder, subfolder, 'image')):
        os.mkdir(os.path.join(new_folder, subfolder, 'image'))

    if not os.path.isdir(os.path.join(new_folder, subfolder, 'label')):
        os.mkdir(os.path.join(new_folder, subfolder, 'label'))

    # ----------------------------------------------------------------

    img_clusters_number = []
    img_clusters_name = []
    img_size = 256

    img_list = sorted(os.listdir(os.path.join(folder, 'image')))
    mask_list = sorted(os.listdir(os.path.join(folder,  'label')))

    base_name_imgs = img_list[0][:-8]
    print('base_name_imgs:', base_name_imgs)

    for j, image_name in enumerate(img_list[:]):
        name_img = image_name[:-4]
        number_img = int(name_img.replace(base_name_imgs, ''))
        counter = number_img
        img_clusters_number.append(number_img)
        img_clusters_name.append(image_name)

        # when there are more than 2 images in the cluster check consistency in numbers
        if len(img_clusters_number) > 1:
            output_name = image_name[:-4]
            # check if the last number in the number list is continuous with the previous one
            # if not clear the cluster and append the last one in a new list
            if not(img_clusters_number[-2] == counter - 1):
                img_clusters_number.clear()
                img_clusters_name.clear()
                # clear the cluster and append the new image in a new one to restart the process
                img_clusters_number.append(number_img)
                img_clusters_name.append(image_name)

        if len(img_clusters_number) >= volume_size:
            print(img_clusters_name)
            frames = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)
            masks = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)

            for k, image_name in (list(enumerate(img_clusters_name[-3:]))):
                frame = cv2.imread(os.path.join(folder, 'image', image_name), 1)
                frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)
                print(os.path.isfile(os.path.join(folder, 'label', image_name)))
                mask = cv2.imread(os.path.join(folder, 'label', image_name), 1)
                #mask = cv2.resize(mask, (img_size, img_size), cv2.INTER_AREA)

                #frames_gray[k:, :, :] = frame_gray
                if np.shape(frame[:, :, 0]) == np.shape(frame[:, :, 1]) == np.shape(frame[:, :, 2]):
                    frames[k, :, :, :] = frame
                    #masks[k, :, :, :] = mask

            img_clusters_number.clear()
            img_clusters_name.clear()

            # save the mask
            np.save(os.path.join(new_folder, subfolder, 'image', output_name + '.npy'), frames)
            #np.save(os.path.join(new_folder, subfolder, 'label', output_name + '.npy'), masks)

            mask_out = cv2.imread(os.path.join(folder, 'label', output_name + '.png'), 1)
            mask_out = cv2.resize(mask_out, (img_size, img_size), cv2.INTER_AREA)
            cv2.imwrite(os.path.join(new_folder, subfolder, 'label', output_name + '.png'), mask_out)
            print(np.shape(frames))
            print(np.shape(masks))


def convert_images_to_pickle(folder, subfolder, new_folder,
                                       base_name_imgs, volume_size=3,
                                       img_size=256):

    """
    Function that gathers continuous frames into volume pickles, it
    only saves the pickles

    :param folder:
    :param subfolder:
    :param new_folder:
    :param base_name_imgs:
    :param volume_size:
    :param img_size:
    :return:
    """

    video_set = sorted(os.listdir(folder))
    img_clusters_name = []
    img_list = sorted(os.listdir(folder))

    # create a cluster of continous images
    for j in range(1, len(img_list[:])-1):
        name_img = img_list[j][:-4]
        # here you save the name of the images into a list...
        img_clusters_name.append(img_list[j-1])
        img_clusters_name.append(img_list[j])
        img_clusters_name.append(img_list[j+1])
        # create an empty volume
        frames = np.zeros([volume_size, img_size, img_size], dtype=np.uint8)
        print(img_clusters_name)

        for k, image_name in (list(enumerate(img_clusters_name[-3:]))):
            frame = cv2.imread(os.path.join(folder, image_name), 0)
            frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)
            frames[k, :, :] = frame
            print(np.shape(frames))

        output_name = img_clusters_name[-1][:-4]
        print(output_name)
        #np.save(os.path.join(new_folder, subfolder, output_name + '.npy'), frames)
        img_clusters_name.clear()
        #print(np.shape(frames))


def main():


    orig_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/phantom_lumen/all_cases/'
    new_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/phantom_lumen/volume_data/' \
                 '8_continuous_frames/all_cases/'

    subfolder = '??'
    folder = os.path.join(orig_folder, subfolder)
    volume_size = 8
    convert_continous_images_to_pickle(folder, subfolder, new_folder, volume_size)


if __name__ == "__main__":
    main()
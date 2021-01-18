import os
import random

import cv2
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
import shutil


def convert_data_to_pickle(folder, subfolder, new_folder, base_name_imgs):

    if not os.path.isdir(os.path.join(new_folder, subfolder)):
        os.mkdir(os.path.join(new_folder, subfolder))

    if not os.path.isdir(os.path.join(new_folder, subfolder, 'image')):
        os.mkdir(os.path.join(new_folder, subfolder, 'image'))

    """if not os.path.isdir(os.path.join(folder, new_folder, 'image', 'rgb')):
        os.mkdir(os.path.join(folder, new_folder, 'image', 'rgb'))

    if not os.path.isdir(os.path.join(folder, new_folder, 'image', 'grayscale')):
        os.mkdir(os.path.join(folder, new_folder, 'image', 'grayscale'))"""

    if not os.path.isdir(os.path.join(new_folder, subfolder, 'label')):
        os.mkdir(os.path.join(new_folder, subfolder, 'label'))

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
        number_img = int(name_img.replace(base_name_imgs, ''))

        counter = number_img
        img_clusters_number.append(number_img)
        img_clusters_name.append(image)
        if len(img_clusters_number) > 1:
            output_name = image[:-4]
            if not(img_clusters_number[-2] == counter -1):
                img_clusters_number.clear()
                img_clusters_number.append(number_img)
                img_clusters_name.clear()
                img_clusters_name.append(image)

        #output_name = img_clusters_name[-2]
        if len(img_clusters_number) >= 3:
            print(img_clusters_name)
            frames_gray = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)
            frames = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)
            masks = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)

            for k, image_name in reversed(list(enumerate(img_clusters_name[-3:]))):

                frame = cv2.imread(os.path.join(folder, 'image', image_name), 1)
                frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)
                print(image_name)
                print(os.path.isfile(os.path.join(folder, 'label', image_name)))
                mask = cv2.imread(os.path.join(folder, 'label', image_name), 1)
                mask = cv2.resize(mask, (img_size, img_size), cv2.INTER_AREA)

                #frames_gray[k:, :, :] = frame_gray
                if np.shape(frame[:, :, 0]) == np.shape(frame[:, :, 1]) == np.shape(frame[:, :, 2]):
                    frames[k, :, :, :] = frame
                    masks[k, :, :, :] = mask

            img_clusters_number.clear()
            img_clusters_name.clear()
            #frames = np.moveaxis(frames, 0, -3)
            #masks = np.moveaxis(masks, 0, -3)
            np.save(os.path.join(new_folder, subfolder, 'image', output_name + '.npy'), frames)
            np.save(os.path.join(new_folder, subfolder, 'label', output_name + '.npy'), masks)
            #shutil.copy(''.join([folder, 'label/', output_name, '.png']),
            #            ''.join([new_folder, subfolder, 'label/', output_name, '.png']))

            mask_out = cv2.imread(os.path.join(folder, 'label', output_name + '.png'), 1)
            mask_out = cv2.resize(mask_out, (img_size, img_size), cv2.INTER_AREA)
            #cv2.imwrite(os.path.join(new_folder, subfolder, 'label', output_name + '.png'), mask_out)
            print(np.shape(frames))
            print(np.shape(masks))

def convert_continous_images_to_pickle(folder, subfolder, new_folder, base_name_imgs):


    volume_size = 3
    video_set = os.listdir(folder)
    video_set.sort()
    img_clusters_number = []
    img_clusters_name = []
    img_size = 256

    img_list = os.listdir(folder)
    img_list.sort()

    for j, image in enumerate(img_list[:]):
        name_img = image[:-4]
        print(image)
        number_img = int(name_img.replace(base_name_imgs, ''))

        counter = number_img
        img_clusters_number.append(number_img)
        img_clusters_name.append(image)

        if len(img_clusters_number) > 1:
            output_name = image[:-4]
            if not(img_clusters_number[-2] == counter -1):
                img_clusters_number.clear()
                img_clusters_number.append(number_img)
                img_clusters_name.clear()
                img_clusters_name.append(image)

        #output_name = img_clusters_name[-2]
        if len(img_clusters_number) >= 3:
            print(img_clusters_name)
            frames = np.zeros([volume_size, img_size, img_size, 3], dtype=np.uint8)

            for k, image_name in reversed(list(enumerate(img_clusters_name[-3:]))):

                frame = cv2.imread(os.path.join(folder, image_name), 1)
                frame = cv2.resize(frame, (img_size, img_size), cv2.INTER_AREA)
                print(image_name)
                if np.shape(frame[:, :, 0]) == np.shape(frame[:, :, 1]) == np.shape(frame[:, :, 2]):
                    frames[k, :, :, :] = frame

            img_clusters_number.clear()
            img_clusters_name.clear()
            np.save(os.path.join(new_folder, subfolder, output_name + '.npy'), frames)
            print(np.shape(frames))


def main():

    orig_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/video_test/original_frames/'
    subfolder = ''
    folder = orig_folder + subfolder
    new_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/video_test/volume_frames/'
    base_name_imgs = 'P_006_pt1_frame_'
    convert_continous_images_to_pickle(folder, subfolder, new_folder, base_name_imgs)


if __name__ == "__main__":
    main()
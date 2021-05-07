import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def check_size(input_image, target_size):
    w, h = np.shape(input_image)

    if w != target_size[0] and h != target_size[1]:
        output_image = cv2.resize(input_image, (target_size[0], target_size[1]),
                                  interpolation=cv2.INTER_AREA)
    else:
        output_image = input_image

    return output_image


def merge_masks(mask_1, mask_2, target_size, opertaion):

    single_channel_mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
    single_channel_mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)
    single_channel_mask_1 = check_size(single_channel_mask_1, target_size)
    single_channel_mask_2 = check_size(single_channel_mask_2, target_size)

    if opertaion == 'AND':
        merged_mask = np.logical_and(single_channel_mask_1.astype(bool),
                                     single_channel_mask_2.astype(bool))
    elif opertaion == 'OR':
        merged_mask = np.logical_or(single_channel_mask_1.astype(bool),
                                    single_channel_mask_2.astype(bool))
    else:
        print('Operation ot recognized')

    return merged_mask


def main():

    dir_list_mask_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/label_2/'
    dir_list_mask_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/label_1/'
    dir_images = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/all_data/patient_cases/phantom_001_pt2/image/'
    save_dir = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/lumen_data/test/phantom_001_pt2/label/'
    list_mask_1 = sorted(os.listdir(dir_list_mask_1))
    list_mask_2 = sorted(os.listdir(dir_list_mask_2))
    list_images = sorted(os.listdir(dir_images))

    for i, image in enumerate(list_mask_1):
        if (image in list_mask_2) and (image in list_images):
            print(image, i/len(list_mask_1))
            dir_image = dir_images + image
            dir_mask_1 = dir_list_mask_1 + image
            dir_mask_2 = dir_list_mask_2 + image
            img = cv2.imread(dir_image)
            w, h, d = np.shape(img)
            mask_1 = cv2.imread(dir_mask_1)
            mask_2 = cv2.imread(dir_mask_2)
            merged_mask = merge_masks(mask_1, mask_2, [w, h], 'OR')
            zeros = np.zeros((w, h, d))
            zeros[:, :, 0] = merged_mask * 255
            zeros[:, :, 1] = merged_mask * 255
            zeros[:, :, 2] = merged_mask * 255

            cv2.imwrite(save_dir + image, zeros)


if __name__ == "__main__":
    main()